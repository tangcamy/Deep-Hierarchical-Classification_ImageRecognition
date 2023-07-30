'''Hierarchical Loss Network
'''

import pickle
import torch
import torch.nn as nn


class HierarchicalLossNetwork:
    def __init__(self,device='cpu',total_level=1,alpha=1):
        print('HierarchicallLossNetwork')
        self.total_level = total_level
        self.alpha = alpha

    def calculate_lloss(self, predictions, true_labels):
        '''Calculates the layer loss.
        '''

        lloss = 0
        for l in range(self.total_level):

            lloss += nn.CrossEntropyLoss()(predictions[l], true_labels[l])

        return self.alpha * lloss



