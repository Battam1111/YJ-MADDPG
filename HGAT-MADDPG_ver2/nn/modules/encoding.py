import torch
from torch.cuda import device as torch_device

from nn.activation import get_activation


class EncoderByType(torch.nn.Module):

   def __init__(self,
                n_inputs_by_node_class: list,#[195]
                encoding_size: int, 
                activation: str,
                device: torch_device):
      super().__init__()
      self.device = device
      self.out_features = encoding_size
      self.n_inputs_by_node_class = n_inputs_by_node_class
      self.activation = get_activation(activation)
      self.layer = torch.nn.ModuleList()

      for in_size in n_inputs_by_node_class: 
         self.layer.append(torch.nn.Linear(in_size, encoding_size))
      self.layer.to(self.device)
      
      def init_he(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

      def init_xavier(m):
         if type(m) in (torch.nn.Linear, torch.nn.LSTMCell):
            torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('sigmoid'))

      # for nets in [self.layer]:
      #    if nets is not None:
      #       for net in nets:
      #          net.apply(init_xavier)

   # def init_hidden(self, batch_size: int):
   #    assert isinstance(self.layer2[0], torch.nn.LSTMCell)

   #    self.hidden_states = []
   #    for i in range(len(self.n_agents_by_net)):
   #       hs = torch.zeros(self.n_agents_by_net[i] * batch_size,
   #                     self.layer2[i].hidden_size,
   #                     device=self.device)
   #       cs = torch.zeros(self.n_agents_by_net[i] * batch_size,
   #                     self.layer2[i].hidden_size,
   #                     device=self.device)
   #       self.hidden_states.append((hs, cs))

   def apply_net(self, x: torch.tensor, index: int):
      # remove singleton dim
      x.squeeze_()
      if x.ndim == 1:
         x = x.unsqueeze(0)
      assert x.ndim == 2, "only agent dim and feature dim here!"

      l1 = self.layer[index]
      x = self.activation(l1(x.float()))

      return x

   def forward(self, x: dict, node_type: torch.tensor):#x:{1:agent,features维度为2,2:features} node_type:
      """Encode node features

      :param x: Dictionary containing node classes as keys and tensors with their respective features as values
      :type x: dict
      :param node_type: tensor containing the class of each node
      :type node_type: torch.tensor
      :return: a tensor witht he encoded features of all nodes
      :rtype: torch.tensor
      """
      # create tensor to hold the encoded results
      X = torch.empty(node_type.size(0), self.out_features, device=self.device)#agent数量*features

      for nt in node_type.unique().tolist():
         node_mask = (node_type == nt)  # grab nodes of that type
         enc = self.apply_net(x[nt], nt)  # apply corresponding layer to input
         X[node_mask] = enc  # put outputs in their corresponding places

      return X
