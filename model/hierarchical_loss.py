'''Hierarchical Loss Network
'''

import pickle
import torch
import torch.nn as nn
from helper import read_meta


class HierarchicalLossNetwork:
    '''Logics to calculate the loss of the model.
    '''

    def __init__(self, metafile_path, hierarchical_labels_one,hierarchical_labels_two, device='cpu', total_level=2, alpha=1, beta=0.8, p_loss=3):
        '''Param init.
        '''
        self.total_level = total_level
        self.alpha = alpha
        self.beta = beta
        self.p_loss = p_loss
        self.device = device
        #self.level_one_labels, self.level_two_labels = read_meta(metafile=metafile_path)
        self.level_one_labels, self.level_two_labels , self.level_third_labels = read_meta(metafile_path) #Coarse_label_names / fine_label_names / third_lable_names
        self.hierarchical_one_labels = hierarchical_labels_one #原本 hierarchical_labels , Layer1_& Layer_2
        self.hierarchical_two_labels = hierarchical_labels_two # Layer2_& Layer_3

        self.numeric_hierarchy_one = self.words_to_indices(self.hierarchical_one_labels,self.level_one_labels, self.level_two_labels) #原本numeric_hierarchy
        self.numeric_hierarchy_two = self.words_to_indices(self.hierarchical_two_labels,self.level_two_labels , self.level_third_labels) 


    def words_to_indices(self,hierarchical_labels,level_ONE_labels,level_TWO_labels):
        '''Convert the classes from words to indices.
        '''
        numeric_hierarchy = {}
        for k, v in hierarchical_labels.items():
            numeric_hierarchy[level_ONE_labels.index(k)] = [level_TWO_labels.index(i) for i in v]

        return numeric_hierarchy


    def check_hierarchy(self, numeric_hierarchy,current_level, previous_level):
        '''Check if the predicted class at level l is a children of the class predicted at level l-1 for the entire batch.
        '''

        #check using the dictionary whether the current level's prediction belongs to the superclass (prediction from the prev layer)
        bool_tensor = [not current_level[i] in numeric_hierarchy[previous_level[i].item()] for i in range(previous_level.size()[0])]

        return torch.FloatTensor(bool_tensor).to(self.device)


    def calculate_lloss(self, predictions, true_labels):
        '''Calculates the layer loss.
        '''

        lloss = 0
        for l in range(self.total_level):

            lloss += nn.CrossEntropyLoss()(predictions[l], true_labels[l])

        return self.alpha * lloss

    def calculate_dloss(self, predictions, true_labels):
        '''Calculate the dependence loss.
        '''

        dloss = 0
        for l in range(1, self.total_level):

            current_lvl_pred = torch.argmax(nn.Softmax(dim=1)(predictions[l]), dim=1)# subClass_pred 
            prev_lvl_pred = torch.argmax(nn.Softmax(dim=1)(predictions[l-1]), dim=1)# superClass_pred

            if l == 1: # current[1]:subclass,prev[0]：superclass
                D_l = self.check_hierarchy(self.numeric_hierarchy_one,current_lvl_pred, prev_lvl_pred)
            else:# l==2 , current[2]:subtwoclass,prev[1]：subclass
                D_l = self.check_hierarchy(self.numeric_hierarchy_two,current_lvl_pred, prev_lvl_pred)

            #--torch.FloatTensor // float32 dtype
            l_prev = torch.where(prev_lvl_pred == true_labels[l-1], torch.FloatTensor([0]).to(self.device), torch.FloatTensor([1]).to(self.device))
            l_curr = torch.where(current_lvl_pred == true_labels[l], torch.FloatTensor([0]).to(self.device), torch.FloatTensor([1]).to(self.device))
            #--torch.pow(input,exp_value)=input*exp_value 
            dloss += torch.sum(torch.pow(self.p_loss, D_l*l_prev)*torch.pow(self.p_loss, D_l*l_curr) - 1)

        return self.beta * dloss
