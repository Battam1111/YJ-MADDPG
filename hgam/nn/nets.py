from enum import Enum
from hgam.utils._legacy_utils import action_normalize
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
from yaml import load, Loader
import torch.nn.functional as F
# from ..encoding import isin
from hgam.nn.action_layers import CriticLayer, HeteroMAGNetActionLayer, ActorLayer
from hgam.nn.encoding import EncoderByType
from hgam.nn.gat import GATModule
from hgam.nn.gat_typeaware import TypeAwareGATModule
from hgam.nn.mixing import VDNMixer
import numpy as np

class MADDPGAgent(torch.nn.Module):
   def __init__(self,
                node_type:int,
                node_types:list,
                dim_obs_list:list,
                dim_act_list:list,
                n_agents:int,
                encoding_output_size:list,
                graph_hidden_size:list,
                action_hidden_size:list,
                share_encoding:bool,
                act_encoding,
                act_comms,
                act_action,
                device,
                full_receptive_field,
                gat_n_heads,
                gat_average_last,
                dropout,
                add_self_loops,
                # Phase C parametrization. Defaults preserve pre-Phase-C behavior.
                use_type_aware_bias: bool = False,
                num_node_types: int = 2,
                actor_local_node_types=None):
      super().__init__()
      dim_obs_act_list = [sum(x) for x in zip(dim_obs_list, dim_act_list)]

      def _make_actor():
         return Actor_graph(share_encoding,
                            dim_obs_list,
                            node_type,
                            dim_act_list[node_type],
                            encoding_output_size[1],
                            graph_hidden_size[1],
                            action_hidden_size[node_type],
                            act_encoding,
                            act_comms,
                            act_action,
                            device,
                            full_receptive_field,
                            gat_n_heads,
                            gat_average_last,
                            dropout,
                            add_self_loops,
                            use_type_aware_bias=use_type_aware_bias,
                            num_node_types=num_node_types,
                            local_node_types=actor_local_node_types)

      def _make_critic():
         return Critic(share_encoding,
                       dim_obs_act_list,
                       node_type,
                       node_types,
                       encoding_output_size[0],
                       graph_hidden_size[0],
                       action_hidden_size[node_type],
                       act_encoding,
                       act_comms,
                       act_action,
                       device,
                       full_receptive_field,
                       gat_n_heads,
                       gat_average_last,
                       dropout,
                       add_self_loops,
                       use_type_aware_bias=use_type_aware_bias,
                       num_node_types=num_node_types)

      self.actor = [_make_actor() for _ in range(n_agents)]
      self.critic = _make_critic()
      self.target_actor = [_make_actor() for _ in range(n_agents)]
      self.target_critic = _make_critic()

      for i in range(n_agents):
         self.target_actor[i].load_state_dict(self.actor[i].state_dict())
      self.target_critic.load_state_dict(self.critic.state_dict())
   
class Actor_graph(torch.nn.Module):
   def __init__(self,
                share_encoding: bool,
                features_by_node_class: list,
                node_type: int,
                dim_actions: int,
                encoding_output_size: int,
                graph_module_sizes: list,
                action_hidden_size: int,
                act_encoding: str,
                act_comms: str,
                act_action: str,
                device,
                full_receptive_field: bool = False,
                gat_n_heads: int = 1,
                gat_average_last: bool = False,
                dropout: int = 0,
                add_self_loops: bool = True,
                # Phase C parametrization for the E-abl ablation. Defaults
                # preserve the pre-Phase-C behavior so older callers stay green.
                use_type_aware_bias: bool = False,
                num_node_types: int = 2,
                local_node_types=None,
                ):
      super().__init__()

      self.device = device
      # NOTE this assumes all agents have the same number of actions
      self.dim_actions = dim_actions
      self.features_by_node_class = features_by_node_class
      self.node_type = node_type
      self.use_type_aware_bias = use_type_aware_bias
      # share_encoding came in as a positional bool above; cache it for forward.
      self.share_encoding = share_encoding
      self.num_node_types = num_node_types
      # local_node_types lists the type id of each node in this actor's
      # local graph, ordered [self, neighbor_1, neighbor_2, ...]. When
      # None (Phase B and earlier behavior), every node is treated as
      # type 0 and only the first encoder slot is used.
      self.local_node_types = local_node_types

      # Encoder construction: per-type when share_encoding=False, single
      # otherwise. When per-type, both classes' input dimensions must be
      # present in features_by_node_class.
      if share_encoding:
         encoder_inputs = [self.features_by_node_class[0]]
      else:
         encoder_inputs = list(self.features_by_node_class[:num_node_types])
      self.encoding_layer = EncoderByType(encoder_inputs,
                                          encoding_output_size,
                                          act_encoding,
                                          device)

      # GAT construction: type-aware variant when the bias is on; otherwise
      # the original GATModule (plain torch_geometric.nn.GATConv).
      if use_type_aware_bias:
         self.relational_layer = TypeAwareGATModule(
            self.encoding_layer.out_features,
            graph_module_sizes,
            num_node_types=num_node_types,
            activation=act_comms,
            device=device,
            full_receptive_field=full_receptive_field,
            n_heads=gat_n_heads,
            average_last=gat_average_last,
            dropout=dropout,
            add_self_loops=add_self_loops,
            use_bias=True,
         )
      else:
         self.relational_layer = GATModule(self.encoding_layer.out_features,
                                           graph_module_sizes,
                                           act_comms,
                                           device,
                                           full_receptive_field,
                                           gat_n_heads,
                                           gat_average_last,
                                           dropout,
                                           add_self_loops)

      self.action_layer = ActorLayer(self.relational_layer.out_features+self.encoding_layer.out_features,
                                     self.dim_actions,
                                     action_hidden_size,
                                     act_action,
                                     device)
         

      # self.mixer = None
      # if mixing == 'vdn':
      #    self.mixer = VDNMixer()

   def forward(self, x):
      # x: current agent + neighbors. Shape [bs, 1 + n_neighbors, obs_dim].
      # Phase-D optimization: replace torch_geometric Batch.from_data_list (which
      # iterated bs times in Python, ~7ms per call) with a direct vectorized
      # batched-graph construction. Equivalent numerically because every graph
      # in the batch shares the same topology (n_agents nodes, n_agents-1
      # directed edges from neighbours to node 0).
      bs, n_agents = x.shape[0], x.shape[1]
      device = self.device

      if bs == 1:
         # Single-graph fast path
         x = x.squeeze(0)  # (n_agents, obs_dim)
         edge_src = torch.arange(1, n_agents, device=device, dtype=torch.long)
         edge_dst = torch.zeros(n_agents - 1, device=device, dtype=torch.long)
         edge_index = torch.stack([edge_src, edge_dst], dim=0)
      else:
         # Batched: flatten to (bs * n_agents, obs_dim)
         x = x.reshape(bs * n_agents, -1)
         # Build batched edge_index: graph b has edges from
         # (1+b*n_agents, 2+b*n_agents, ..., n_agents-1+b*n_agents) -> 0+b*n_agents.
         src_template = torch.arange(1, n_agents, device=device, dtype=torch.long)
         graph_offsets = (
            torch.arange(bs, device=device, dtype=torch.long)
            .repeat_interleave(n_agents - 1) * n_agents
         )
         src = src_template.repeat(bs) + graph_offsets
         dst = graph_offsets  # all destinations are node 0 within each graph
         edge_index = torch.stack([src, dst], dim=0)

      # Build the per-node type tensor.
      # When self.local_node_types is provided (Phase C), tile it across
      # the batch so each subgraph's nodes carry their correct type id.
      # When it is None, fall back to the legacy "everything is type 0"
      # behavior so older callers stay green.
      if self.local_node_types is not None:
         per_graph_types = torch.tensor(
            self.local_node_types, device=self.device, dtype=torch.long
         )
         node_type = per_graph_types.repeat(bs)
      else:
         node_type = torch.zeros(bs * n_agents, device=self.device, dtype=torch.long)

      input_by_class = {}
      for nt in node_type.unique():
         node_mask = (node_type == nt)
         # Map node-type id to the corresponding encoder slot. When
         # share_encoding=True, every type maps to slot 0; otherwise the
         # encoder has one slot per type and we read each type's own
         # input width.
         enc_slot = 0 if self.share_encoding else int(nt)
         in_size = self.features_by_node_class[enc_slot]
         input_by_class[enc_slot] = x[node_mask, :in_size]
      del node_mask

      indices = torch.arange(0, x.shape[0], n_agents)
      # Encoder expects keys to be node-class ids. When share_encoding=True
      # all nodes are remapped to a single class id of 0 before encoding.
      encoder_node_type = (
         node_type
         if not self.share_encoding
         else torch.zeros_like(node_type)
      )
      y = self.encoding_layer(input_by_class, encoder_node_type)
      del input_by_class

      input = y[indices]

      if self.relational_layer is not None:
         if self.use_type_aware_bias:
            y = self.relational_layer(y, edge_index, node_type)
         else:
            y = self.relational_layer(y, edge_index)

      cur_y = y[indices]
      y = torch.cat((input, cur_y), dim=1)
      re = self.action_layer(y)
      re = action_normalize(re)

      return re
   
class Actor(torch.nn.Module):
    def __init__(self, dim_observation, dim_action, device, activation_fc=F.leaky_relu):
        super(Actor, self).__init__()
        self.device = device
        self.activation_fc = activation_fc
        self.FC1 = torch.nn.Linear(dim_observation, 128).to(self.device)
        self.FC2 = torch.nn.Linear(128, 64).to(self.device)
        self.FC3 = torch.nn.Linear(64, dim_action).to(self.device)

      #   def init_xavier(m):
      #    if type(m) in (torch.nn.Linear, torch.nn.LSTMCell):
      #       torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('sigmoid'))

      #   for nets in [self.FC1, self.FC2, self.FC3]:
      #      if nets is not None:
      #          nets.apply(init_xavier)

    def forward(self, obs):
        result = self.activation_fc(self.FC1(obs))
        result = self.activation_fc(self.FC2(result))
        result = action_normalize(torch.tanh(self.FC3(result)))
        return result
    
class Critic(torch.nn.Module):
   def __init__(self,
                share_encoding: bool,
                features_by_node_class: list,
                node_type: int,
                node_types: list,
                encoding_output_size: int,
                graph_module_sizes: list,
                action_hidden_size: int,
                act_encoding: str,
                act_comms: str,
                act_action: str,
                device,
                full_receptive_field: bool = True,
                gat_n_heads: int = 1,
                gat_average_last: bool = False,
                dropout: int = 0,
                add_self_loops: bool = True,
                # Phase C parametrization. Defaults preserve pre-Phase-C behavior.
                use_type_aware_bias: bool = False,
                num_node_types: int = 2,
                ):
      super().__init__()
      self.device = device
      self.n_agents = len(node_types)
      individual_node_types = torch.tensor(node_types, device=device, dtype=torch.long)
      shared_node_types = torch.zeros(len(node_types), device=device, dtype=torch.long)

      self.share_encoding = share_encoding
      self.use_type_aware_bias = use_type_aware_bias
      self.num_node_types = num_node_types

      # The encoder routing tensor: which encoder slot each agent maps to.
      # When sharing, everything maps to slot 0; otherwise to its own type id.
      if share_encoding:
         self.node_types_encoding = shared_node_types
      else:
         self.node_types_encoding = individual_node_types

      # The full per-node type tensor used for type-aware GAT bias lookup.
      # Always the individual ids regardless of share_encoding, since the
      # GAT bias and the encoder routing are independent ablation knobs.
      self.full_node_types = individual_node_types

      self.features_by_node_class = features_by_node_class
      self.node_type = node_type

      # Encoder construction: one slot per type when share_encoding=False,
      # a single slot otherwise.
      if share_encoding:
         encoder_inputs = [self.features_by_node_class[0]]
      else:
         encoder_inputs = list(self.features_by_node_class[:num_node_types])
      self.encoding_layer = EncoderByType(encoder_inputs,
                                          encoding_output_size,
                                          act_encoding,
                                          device)

      # GAT: type-aware variant when bias flag is on, else vanilla GATModule.
      if use_type_aware_bias:
         self.relational_layer = TypeAwareGATModule(
            self.encoding_layer.out_features,
            graph_module_sizes,
            num_node_types=num_node_types,
            activation=act_comms,
            device=device,
            full_receptive_field=full_receptive_field,
            n_heads=gat_n_heads,
            average_last=gat_average_last,
            dropout=dropout,
            add_self_loops=add_self_loops,
            use_bias=True,
         )
      else:
         self.relational_layer = GATModule(self.encoding_layer.out_features,
                                              graph_module_sizes,
                                              act_comms,
                                              device,
                                              full_receptive_field,
                                              gat_n_heads,
                                              gat_average_last,
                                              dropout,
                                              add_self_loops)

      self.action_layer = CriticLayer(self.relational_layer.out_features+self.encoding_layer.out_features,
                                       action_hidden_size,
                                       act_action,
                                       device)

      # self.mixer = None
      # if mixing == 'vdn':
      #    self.mixer = VDNMixer()

   def forward(self, state, action, index):
      # Phase-D optimization: replace torch_geometric Batch.from_data_list with
      # direct vectorized construction. Every graph in the batch has the same
      # topology (n_agents nodes, n_agents-1 edges from all-but-focal -> focal).
      x = torch.cat((state, action), 2)         # (bs, n_agents, obs+act)
      bs = x.shape[0]
      n_agents = self.n_agents
      device = self.device

      # Flatten: (bs * n_agents, obs+act)
      y = x.reshape(bs * n_agents, -1)

      # Build batched edge_index: in each graph b the edges go from
      # [0..n_agents) \ {index} -> {index}, offset by b * n_agents.
      src_template = torch.cat([
         torch.arange(0, index, device=device, dtype=torch.long),
         torch.arange(index + 1, n_agents, device=device, dtype=torch.long),
      ])  # (n_agents - 1,)
      graph_offsets = (
         torch.arange(bs, device=device, dtype=torch.long)
         .repeat_interleave(n_agents - 1) * n_agents
      )
      src = src_template.repeat(bs) + graph_offsets
      dst = graph_offsets + index
      edge_index_y = torch.stack([src, dst], dim=0)

      input_by_class = {}
      batch_node_types_encoding = self.node_types_encoding.repeat(bs)
      batch_full_node_types = self.full_node_types.repeat(bs)

      for nt in self.node_types_encoding.unique():
         node_mask = (batch_node_types_encoding == nt)
         # When share_encoding=True every node id maps to slot 0; otherwise
         # the encoder has one slot per type and we index by node type.
         in_size = self.features_by_node_class[int(nt)]
         input_by_class[int(nt)] = y[node_mask, :in_size]
      del node_mask

      indices = torch.arange(index, y.shape[0], self.n_agents)

      y = self.encoding_layer(input_by_class, batch_node_types_encoding)
      del input_by_class

      input = y[indices]

      if self.relational_layer is not None:
         if self.use_type_aware_bias:
            y = self.relational_layer(y, edge_index_y, batch_full_node_types)
         else:
            y = self.relational_layer(y, edge_index_y)

      cur_y = y[indices]
      cur_y = torch.cat((input, cur_y), 1)
      values = self.action_layer(cur_y)

      return values
   
class HeteroMAGNet(torch.nn.Module):

   def __init__(self,
                action_layer_type: HeteroMAGNetActionLayer.LayerType,
                share_encoding: bool,
                share_comms: bool,
                node_types: list, #与data.x的节点类型顺序对应
                features_by_node_class: list,#dim_input for each group
                dim_actions: int,
                encoding_output_size: list,
                graph_module_sizes: list,
                action_hidden_size: int,
                act_encoding: str,
                act_comms: str,
                act_action: str,
                device,
                full_receptive_field: bool = True,
                gat_n_heads: int = 1,
                gat_average_last: bool = False
                ):
      super().__init__()

      self.device = device
      
      self.action_layer: HeteroMAGNetActionLayer
      self.action_layer_type = action_layer_type
      # NOTE this assumes all agents have the same number of actions
      self.dim_actions = dim_actions
      
      self.n_agents = len(node_types)
      self.n_type = len(set(node_types))
      n_agents_by_net = [node_types.count(ai) for ai in sorted(set(node_types))] #node_types:[1,0,1,1...] n_agents 每种type的agent数量
      individual_node_types = torch.tensor(node_types, device=device)
      shared_node_types = torch.zeros(len(node_types), device=device, dtype=torch.int)

      if share_encoding:
         n_agents_by_net_encoding = [sum(n_agents_by_net)]
         self.node_types_encoding = shared_node_types

         assert len(set(features_by_node_class)) == 1, 'Number of inputs by node class must all be equal when sharing parameters from the encoding layer'
         self.features_by_node_class = list(set(features_by_node_class))
      else:
         n_agents_by_net_encoding = n_agents_by_net
         self.node_types_encoding = individual_node_types

         assert len(features_by_node_class) == len(n_agents_by_net_encoding), 'Number of node classes and number of inputs must be the same when not sharing parameters from the encoding layer'
         self.features_by_node_class = features_by_node_class

      self.node_types_comms = individual_node_types if not share_comms else shared_node_types

      # here we use relational_layer.out_features instead of relational_output_size
      # because the output size of some graph modules depend on more than the number of features,
      # like the GATModule
      # act_layer_input_size = self.relational_layer.out_features if self.relational_layer is not None else self.encoding_layer.out_features
      # act_layer_input_size = self.relational_layer.out_features

      if action_layer_type == HeteroMAGNetActionLayer.LayerType.CRITIC:
         self.encoding_layer = EncoderByType(self.features_by_node_class,
                                          encoding_output_size[0],
                                          act_encoding,
                                          device)
         
         self.relational_layer = GATModule(self.encoding_layer.out_features,
                                           graph_module_sizes,
                                           act_comms,
                                           device,
                                           full_receptive_field,
                                           gat_n_heads,
                                           gat_average_last)
         
         self.action_layer = CriticLayer(self.n_agents,
                                       self.relational_layer.out_features,
                                       action_hidden_size,
                                       act_action,
                                       device)
         
      elif action_layer_type == HeteroMAGNetActionLayer.LayerType.ACTOR:
         self.encoding_layer = EncoderByType(self.features_by_node_class,
                                          encoding_output_size[1],
                                          act_encoding,
                                          device)
         
         self.relational_layer = GATModule(self.encoding_layer.out_features,
                                           graph_module_sizes,
                                           act_comms,
                                           device,
                                           full_receptive_field,
                                           gat_n_heads,
                                           gat_average_last)
         
         self.action_layer = ActorLayer(self.relational_layer.out_features,
                                       self.dim_actions,
                                       action_hidden_size,
                                       act_action,
                                       device)
         

      # self.mixer = None
      # if mixing == 'vdn':
      #    self.mixer = VDNMixer()

   def forward(self, data):
      """"""
      # NOTE if data is a Batch object, x, edge_index and node_type will still
      # be low-dimensional Tensors, containing all graphs in a single disconnected graph.
      # So be careful when processing a batch not to confuse it with a single graph

      # get data
      x, edge_index = data.x, data.edge_index

      # get batch size (single state = batch of size 1)
      bs = 1
      if isinstance(data, Batch) and data.num_graphs > 1:
         bs = data.num_graphs  #optimize那里episodes*1（ts）

      batch_node_types_encoding = self.node_types_encoding.repeat(bs)#[0,0,1...]
      batch_node_types_comms = self.node_types_comms.repeat(bs)
      # batch_node_types_action = self.node_types_action.repeat(bs)#[0,0,1...]

      # separate input tensor into tensors pertaining to individual agent classes
      # according to what the encoding module expects
      node_type = []
      input_by_class = {}
      for node_feature in x:
         type = node_feature[-self.n_type:].argmax().item()
         node_type.append(type)
         # input_by_class[type] = torch.cat((input_by_class[type], node_feature[:-self.n_type].unsqueeze(0)), 0)

      for nt in self.node_types_encoding.unique():
         # grab nodes of the current class
         node_mask = (batch_node_types_encoding == nt)
         # grab features only of those nodes, remove padding
         in_size = self.features_by_node_class[int(nt)]
         input_by_class[int(nt)] = x[node_mask, :in_size]
      del node_mask

      # apply encoding layer, output is a single tensor of size n_agents x encoding_size
      x = self.encoding_layer(input_by_class, batch_node_types_encoding)
      del input_by_class

      # if the communication layer exists, apply it to the data
      # output is also a single tensor of size n_agents x comms_output_size
      if self.relational_layer is not None:
         x = self.relational_layer(x, edge_index, batch_node_types_comms)

      # NOTE this is where I used to filter agent nodes from non-agent nodes,
      # as well as their features, but, as of the time of this writing,
      # all nodes are agent nodes
      if self.action_layer_type == HeteroMAGNetActionLayer.LayerType.CRITIC:
         num = x.shape[0] / self.n_agents
         x_shape = [x[i*5:(i+1)*5].reshape(-1) for i in range(num)]
         x_concat = torch.cat(x_shape, dim = 0)
         re = self.action_layer(x_concat) # batch*(dim_obs+dim_act)*n_agents
      elif self.action_layer_type == HeteroMAGNetActionLayer.LayerType.ACTOR:
         re = self.action_layer(x[::self.n_agents]) #本地agent
      # obs_by_class = {}
      # for nt in self.node_types_action.unique():
      #    # grab nodes of that type
      #    agent_mask = (batch_node_types_action == nt)
      #    obs_by_class[int(nt)] = x[agent_mask]
      #    obs_by_class[nt] = torch.cat((input_by_class[nt], x[agent_indices]), dim=-1)

         # input for recurrent layers is
         # (number of sequences) (sequence size) (features of individual elements)
         # translated for this network, it should be
         # (episodes) (steps in episode) (individual steps)

         # episodes = my batch size, bs
         # steps in episode = total number of nodes / batch_size
         # batch_size = number of graphs in the Batch object, or 1 if Data
         # size of previous layer = self.relational_layer.out_features

         # by my interpretation, this should be the correct size
         # obs_by_class[nt] = obs_by_class[nt].view(
         #     agent_indices.size(0) // bs, bs, self.relational_layer.out_features)

         # however, it looks like it should be like this?!
         # obs_by_class[nt] = obs_by_class[nt].view(
         #  agent_indices.size(0) // bs, bs, self.relational_layer.out_features)
      # x = self.action_layer(obs_by_class, batch_node_types_action)

      return re #.view(bs, -1, self.n_actions)

