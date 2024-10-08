"""
Label Encoders -- :mod:`hcanet.encoding`
*****************************************
"""

import numpy as np
import torch


def isin(a: torch.tensor, b: torch.tensor, mask: bool = False) -> torch.tensor:
   """Checks which values of the first tensor are present in the second tensor. Copied from
   `here <https://github.com/pytorch/pytorch/issues/3025#issuecomment-392601780>`_.

   :param a: values whose presence will be checked against the values in ``b``
   :type a: torch.tensor
   :param b: values to look for in ``a``
   :type b: torch.tensor
   :param mask: ``True`` returns boolean masks, ``False`` returns indices, defaults to False
   :type mask: bool, optional
   :return: boolean tensor of the same size of ``a`` informing which items of ``a`` were found in ``b``, or long tensor informing the positions in ``a`` that contained items present in ``b``
   :rtype: torch.tensor
   """
   c = (a.unsqueeze(-1) == b).any(-1)

   return c if mask else c.nonzero().squeeze()


class LabelEncoder:
   """A label encoder that mimics the functionality of :py:class:`sklearn.preprocessing.LabelEncoder`

   Encode target labels with value between ``0`` and ``n_classes - 1``.

   :param labels: list of original labels, defaults to None
   :type labels: list or numpy.ndarray or torch.tensor, optional
   """

   def __init__(self, labels=None):
      self.encoded_labels = None
      if labels is not None:
         self.fit(labels)

   @staticmethod
   def _tolist(thing):
      if type(thing) in [torch.tensor, np.ndarray]:
         if len(thing.shape) != 1:
            raise ValueError("Input value must be one-dimensional")
         thing = thing.tolist()

      if type(thing) != list:
         thing = list(thing)

      return thing

   def fit(self, labels):
      """Generates encoded labels for the given list of (not necessarily unique) labels

      :param labels: original labels
      :type labels: list or numpy.ndarray or torch.tensor
      :return: the object itself
      :rtype: LabelEncoder
      """
      labels = self._tolist(labels)

      self.encoded_labels = {}
      enc_label = 0
      for label in labels:
         if label not in self.encoded_labels:
            self.encoded_labels[label] = enc_label
            enc_label += 1

      return self

   @property
   def classes__(self):
      return list(self.encoded_labels.keys())

   def transform(self, labels):
      """Transforms a list of labels into their encoded counterparts, generated by the constructor or a previous call to :py:func:`fit()`

      :param labels: original labels
      :type labels: list or numpy.ndarray or torch.tensor
      :return: encoded labels
      :rtype: list
      """
      labels = self._tolist(labels)

      new_labels = [None] * len(labels)

      for i in range(len(labels)):
         new_labels[i] = self.encoded_labels[labels[i]]

      return new_labels

   def inverse_transform(self, labels):
      """Transforms a list of encoded labels into their original counterparts. This method has :math:`O(n^2)` complexity, so be careful

      :param labels: encoded labels
      :type labels: list or numpy.ndarray or torch.tensor
      :return: original labels
      :rtype: list
      """
      if type(labels) in [torch.tensor, np.ndarray]:
         labels = labels.tolist()

      if type(labels) != list:
         labels = list(labels)

      new_labels = [None] * len(labels)

      for index, l in enumerate(labels):
         for key, val in self.encoded_labels.items():
            if val == l:
               new_labels[index] = key
               break

      return new_labels
