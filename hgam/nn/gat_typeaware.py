"""Type-aware graph attention layer (Phase C implementation of C3 claim).

This module implements the architectural commitment in the revised
paper's Claim 3: a GAT variant whose attention coefficients include a
learnable bias term indexed by the (source, target) node-type pair.

The bias is a small matrix ``b`` of shape ``(num_node_types,
num_node_types, heads)`` — for the binary MUAV/CUAV case that means
``2 * 2 * heads`` new parameters, an order of magnitude cheaper than
HGT-style per-relation linear maps but still enough to break the
permutation symmetry between intra-type and cross-type attention.

The class is intentionally *parametric* so that the four E-abl variants
can be toggled from config without code changes:

================== ============================ ===========================
``method_variant``  ``use_type_aware_bias``      effect on encoder
================== ============================ ===========================
hgam-full           True                         per-type encoders enabled
hgam-no-bias        False                        per-type encoders enabled
hgam-shared-enc     True                         single shared encoder
hgam-vanilla        False                        single shared encoder
================== ============================ ===========================

The ``share_encoder`` knob is read by :class:`~hgam.nn.encoding.EncoderByType`,
not by this class. This class only controls the bias.

Math (single head, omitted index for clarity):
    e_{vu} = LeakyReLU( a^T [W h_v || W h_u] + b_{tau(v), tau(u)} )
    alpha_{vu} = softmax_{u in N(v)}( e_{vu} )
    h'_v = sigma( sum_{u in N(v)} alpha_{vu} W h_u )

Reference: Vaswani-style scaled dot-product attention augmented with
relational bias terms; the bias scheme is analogous to RotaryHGAT's
"edge type embedding into attention logit" — but with a closed-form
2x2 matrix rather than a learnable embedding lookup.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax as scatter_softmax


class TypeAwareGATConv(MessagePassing):
    """A graph attention convolution with type-aware bias.

    Parameters
    ----------
    in_channels : int
        Input feature dimension per node.
    out_channels : int
        Output feature dimension per head.
    num_node_types : int
        How many distinct node types exist (e.g. 2 for MUAV/CUAV).
    heads : int, default=1
        Number of attention heads.
    use_type_aware_bias : bool, default=True
        If False, the layer behaves like vanilla GAT (no relational bias).
        The bias parameter is still constructed but kept frozen at zero
        to keep state_dicts checkpoint-compatible across variants.
    negative_slope : float, default=0.2
        LeakyReLU negative slope, matching the original GAT paper.
    dropout : float, default=0.0
        Attention dropout, applied after softmax.
    add_self_loops : bool, default=True
        Whether to inject self-edges in the propagated edge_index.
    bias : bool, default=True
        Whether the output linear projection has an additive bias.
    """

    _alpha: Optional[Tensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_node_types: int,
        heads: int = 1,
        use_type_aware_bias: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__(aggr="add", node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.heads = heads
        self.use_type_aware_bias = use_type_aware_bias
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        # Linear projection W (no bias in the projection itself; output
        # bias is separate below).
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)

        # Attention parameters a_src and a_dst, matching torch_geometric
        # GATConv's split formulation: alpha = (Wh_v . a_src) + (Wh_u . a_dst).
        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))

        # Type-aware bias: shape (T, T, heads). Indexed by (target_type, source_type).
        # We register it as a Parameter regardless of `use_type_aware_bias` so
        # that loading a state_dict from one variant into another stays clean;
        # when the flag is False the contribution is masked to zero in forward.
        self.type_bias = nn.Parameter(
            torch.zeros(num_node_types, num_node_types, heads)
        )

        if bias:
            self.output_bias = nn.Parameter(torch.zeros(heads * out_channels))
        else:
            self.register_parameter("output_bias", None)

        self._alpha = None  # stash of last attention coefficients for viz/E4

        self.reset_parameters()

    def reset_parameters(self) -> None:  # noqa: D401
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        # type_bias kept at zero initialization so we start in the same place
        # as vanilla GAT; the model has to *learn* a non-zero bias for it to
        # matter, which keeps the early training trajectory comparable
        # across the E-abl variants.
        nn.init.zeros_(self.type_bias)
        if self.output_bias is not None:
            nn.init.zeros_(self.output_bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        node_type: Tensor,
        return_attention_weights: bool = False,
    ):
        """Run one type-aware GAT layer.

        Parameters
        ----------
        x : Tensor, shape (N, in_channels)
            Node features.
        edge_index : Tensor, shape (2, E)
            Directed edges. Convention follows torch_geometric:
            ``edge_index[0]`` is the source, ``edge_index[1]`` is the target.
        node_type : Tensor, shape (N,), dtype long
            Type id in ``[0, num_node_types)`` for each node.
        return_attention_weights : bool, default=False
            If True, returns ``(out, (edge_index_after_self_loops, alpha))``.

        Returns
        -------
        out : Tensor, shape (N, heads * out_channels)
        """
        N = x.size(0)
        H, C = self.heads, self.out_channels

        if self.add_self_loops:
            loop_index = torch.arange(N, device=x.device)
            loop_index = loop_index.unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, loop_index], dim=1)

        # Linear projection: (N, in) -> (N, H, C)
        x_proj = self.lin(x).view(N, H, C)

        # Pre-compute the dot products with att_src and att_dst once.
        alpha_src = (x_proj * self.att_src).sum(dim=-1)  # (N, H)
        alpha_dst = (x_proj * self.att_dst).sum(dim=-1)  # (N, H)

        # Propagate. The `propagate` machinery will look up `_j` and `_i`
        # tensors automatically based on edge_index.
        out = self.propagate(
            edge_index,
            x=x_proj,
            alpha=(alpha_src, alpha_dst),
            node_type=node_type,
            size=None,
        )

        out = out.view(N, H * C)
        if self.output_bias is not None:
            out = out + self.output_bias

        if return_attention_weights:
            return out, (edge_index, self._alpha)
        return out

    # ------------------------------------------------------------------ #
    # MessagePassing hooks
    # ------------------------------------------------------------------ #

    def message(
        self,
        x_j: Tensor,
        alpha_j: Tensor,
        alpha_i: Tensor,
        node_type_i: Tensor,
        node_type_j: Tensor,
        index: Tensor,
        size_i: Optional[int],
    ) -> Tensor:
        # Standard GAT pre-softmax score, in additive form.
        alpha = alpha_j + alpha_i  # (E, H)

        # Type-aware additive bias: pick b[type(target), type(source), :]
        # for every edge. When the flag is off, mask the contribution out.
        if self.use_type_aware_bias:
            bias = self.type_bias[node_type_i, node_type_j]  # (E, H)
            alpha = alpha + bias

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = scatter_softmax(alpha, index, num_nodes=size_i)

        # Stash for visualization (E4 attention extraction).
        self._alpha = alpha.detach()

        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        return x_j * alpha.unsqueeze(-1)  # (E, H, C)


class TypeAwareGATModule(nn.Module):
    """Multi-layer wrapper around :class:`TypeAwareGATConv`, drop-in
    compatible with the original :class:`hgam.nn.gat.GATModule` interface
    plus a required ``node_type`` argument to :meth:`forward`.

    Use this when ``USE_TYPE_AWARE_BIAS`` is on. The existing
    :class:`~hgam.nn.gat.GATModule` is kept for the ablation variants
    that switch the bias off, since it routes through plain
    ``torch_geometric.nn.GATConv`` which is more battle-tested.

    Parameters mirror :class:`GATModule`:

    Parameters
    ----------
    input_size : int
        Number of features per input node.
    hidden_sizes : list of int
        Output dimension for each layer.
    num_node_types : int
        Number of distinct node types (e.g. 2 for MUAV/CUAV).
    activation : str
        Activation function name resolved by
        :func:`hgam.nn.activation.get_activation` for the
        between-layer non-linearity.
    device : torch.device
    full_receptive_field : bool, default=True
        If True, the output concatenates every layer's output (matches
        the original GATModule behavior); otherwise only the last layer.
    n_heads : int, default=2
    average_last : bool, default=False
        Whether to average the last layer's heads (else concatenate).
    dropout : float, default=0.0
    add_self_loops : bool, default=True
    use_bias : bool, default=True
        Whether ``TypeAwareGATConv`` applies the type-aware bias.
        Setting ``False`` produces an "almost-vanilla" GAT that still
        uses the same parameter shapes as the type-aware variant (useful
        for the E-abl ``no-bias`` variant — same architecture, just
        bias masked out).
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        num_node_types: int,
        activation: str,
        device,
        full_receptive_field: bool = True,
        n_heads: int = 2,
        average_last: bool = False,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        use_bias: bool = True,
    ) -> None:
        from hgam.nn.activation import get_activation
        super().__init__()
        self.device = device
        self.layers = nn.ModuleList()
        self.full_receptive_field = full_receptive_field
        self.activation = get_activation(activation)
        self.last_layer_attention = None
        self.attention_indices = None
        self.n_heads = n_heads
        self.num_node_types = num_node_types
        self.use_bias = use_bias
        self._sizes = [input_size] + list(hidden_sizes)

        self.out_features = hidden_sizes[-1] if average_last else hidden_sizes[-1] * n_heads
        if self.full_receptive_field:
            self.out_features += sum(h * n_heads for h in hidden_sizes[:-1])

        for i in range(len(self._sizes) - 1):
            in_size = self._sizes[i] if i == 0 else self._sizes[i] * n_heads
            out_size = self._sizes[i + 1]
            self.layers.append(
                TypeAwareGATConv(
                    in_channels=in_size,
                    out_channels=out_size,
                    num_node_types=num_node_types,
                    heads=n_heads,
                    use_type_aware_bias=use_bias,
                    dropout=dropout,
                    add_self_loops=add_self_loops,
                )
            )
        self.layers.to(self.device)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        node_type: Tensor,
    ) -> Tensor:
        final_x = []
        for i, layer in enumerate(self.layers):
            is_last = i == len(self.layers) - 1
            if is_last:
                x, (idx, alpha) = layer(
                    x, edge_index, node_type, return_attention_weights=True
                )
                self.attention_indices = idx
                self.last_layer_attention = alpha
            else:
                x = layer(x, edge_index, node_type)
            x = self.activation(x)
            if self.full_receptive_field or is_last:
                final_x.append(x)
        return torch.cat(final_x, dim=1)
