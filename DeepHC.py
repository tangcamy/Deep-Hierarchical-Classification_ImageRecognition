# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 21:57:50 2023

@author: User
"""
import os
import sys
sys.path
import torch
print(torch.__version__)
print('Torch Cuda:',torch.cuda.is_available())
print('Torch Cuda Device counts:',torch.cuda.device_count())
current = '/home/orin/L5C_CellFMA/D3_Deep-Hierarchical-Classification_ImageRecognition/'
os.chdir(current)
# In[步驟]
'''
#1. 先到如下檔案下載照片- 需至檔案輪流註解 Test/Train
!python process_cifar100.py  /// for cifar100.py
!python process_dataset.py   /// for customer dataset cleaner.
'''

# In[train.py]
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchsummary import summary #pip install torchsummary
import torchvision.transforms as transforms

from level_dict import hierarchy,hierarchy_two
from runtime_args import args
from load_dataset import LoadDataset
from model import resnet50
from model.hierarchical_loss import HierarchicalLossNetwork
from helper import calculate_accuracy
from plot import plot_loss_acc

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')

if not os.path.exists(args.graphs_folder) : os.makedirs(args.graphs_folder)

print('image_size:'+str(args.img_size))

#----- dataset
train_dataset = LoadDataset(image_size=args.img_size, image_depth=args.img_depth, csv_path=args.train_csv,
                            cifar_metafile=args.metafile, transform=transforms.Compose([transforms.RandomAffine(40, scale=(.85, 1.15), shear=0),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomPerspective(distortion_scale=0.2),
                                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                                 transforms.ToTensor()]))

test_dataset = LoadDataset(image_size=args.img_size, image_depth=args.img_depth, csv_path=args.test_csv,
                            cifar_metafile=args.metafile, transform=transforms.ToTensor())

print('train_dataset:'+str(len(train_dataset)))
print('test_dataset:'+str(len(test_dataset)))


#----- generator
#train_generator = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.no_shuffle, num_workers=args.num_workers)
#test_generator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.no_shuffle, num_workers=args.num_workers)


train_generator = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
test_generator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

model = resnet50.ResNet50()
optimizer = Adam(model.parameters(), lr=args.learning_rate)

model = model.to(device)
HLN = HierarchicalLossNetwork(metafile_path=args.metafile, hierarchical_labels_one=hierarchy,hierarchical_labels_two=hierarchy_two, total_level=3,device=device)

train_epoch_loss = []
train_epoch_superclass_accuracy = []
train_epoch_subclass_accuracy = []
train_epoch_subtwoclass_accuracy = []

test_epoch_loss = []
test_epoch_superclass_accuracy = []
test_epoch_subclass_accuracy = []
test_epoch_subtwoclass_accuracy = []

#----- training 
for epoch_idx in range(args.epoch):

    i = 0

    epoch_loss = []
    epoch_superclass_accuracy = []
    epoch_subclass_accuracy = []
    epoch_subtwoclass_accuracy = []

    model.train()
    for i, sample in tqdm(enumerate(train_generator)):
        batch_x, batch_y1, batch_y2, batch_y3 = sample['image'].to(device), sample['label_1'].to(device), sample['label_2'].to(device),sample['label_3'].to(device)
        optimizer.zero_grad()
              
        superclass_pred,subclass_pred ,subtwoclass_pred = model(batch_x)
        prediction = [superclass_pred, subclass_pred,subtwoclass_pred] # add subtwoclass - layer3
    
        dloss = HLN.calculate_dloss(prediction, [batch_y1, batch_y2,batch_y3]) #depense loss
        lloss = HLN.calculate_lloss(prediction, [batch_y1, batch_y2,batch_y3]) # layer loss
        
        total_loss = lloss + dloss
        total_loss.backward()
        optimizer.step()
        epoch_loss.append(total_loss.item())
        epoch_superclass_accuracy.append(calculate_accuracy(predictions=prediction[0].detach(), labels=batch_y1))
        epoch_subclass_accuracy.append(calculate_accuracy(predictions=prediction[1].detach(), labels=batch_y2))
        epoch_subtwoclass_accuracy.append(calculate_accuracy(predictions=prediction[2].detach(), labels=batch_y3))


    train_epoch_loss.append(sum(epoch_loss)/(i+1))
    train_epoch_superclass_accuracy.append(sum(epoch_superclass_accuracy)/(i+1))
    train_epoch_subclass_accuracy.append(sum(epoch_subclass_accuracy)/(i+1))
    train_epoch_subtwoclass_accuracy.append(sum(epoch_subtwoclass_accuracy)/(i+1))



    print(f'Training Loss at epoch {epoch_idx} : {sum(epoch_loss)/(i+1)}')
    print(f'Training Superclass accuracy at epoch {epoch_idx} : {sum(epoch_superclass_accuracy)/(i+1)}')
    print(f'Training Subclass accuracy at epoch {epoch_idx} : {sum(epoch_subclass_accuracy)/(i+1)}')
    print(f'Training Subtwoclass accuracy at epoch {epoch_idx} : {sum(epoch_subtwoclass_accuracy)/(i+1)}')


    j = 0

    epoch_loss = []
    epoch_superclass_accuracy = []
    epoch_subclass_accuracy = []
    epoch_subtwoclass_accuracy = []

    model.eval()
    with torch.set_grad_enabled(False):
        #enumerate(建立批次迴圈計算時間)
        for j, sample in tqdm(enumerate(test_generator)):


            batch_x, batch_y1, batch_y2 ,batch_y3= sample['image'].to(device), sample['label_1'].to(device), sample['label_2'].to(device), sample['label_3'].to(device)
        
            superclass_pred,subclass_pred,subtwoclass_pred= model(batch_x)
            prediction = [superclass_pred,subclass_pred,subtwoclass_pred]

            dloss = HLN.calculate_dloss(prediction, [batch_y1, batch_y2,batch_y3])#depeense loss
            lloss = HLN.calculate_lloss(prediction, [batch_y1, batch_y2,batch_y3])#loss

            total_loss = lloss + dloss

            epoch_loss.append(total_loss.item())
            epoch_superclass_accuracy.append(calculate_accuracy(predictions=prediction[0], labels=batch_y1))
            epoch_subclass_accuracy.append(calculate_accuracy(predictions=prediction[1], labels=batch_y2))
            epoch_subtwoclass_accuracy.append(calculate_accuracy(predictions=prediction[2], labels=batch_y3))


    test_epoch_loss.append(sum(epoch_loss)/(j+1))
    test_epoch_superclass_accuracy.append(sum(epoch_superclass_accuracy)/(j+1))
    test_epoch_subclass_accuracy.append(sum(epoch_subclass_accuracy)/(j+1))
    test_epoch_subtwoclass_accuracy.append(sum(epoch_subtwoclass_accuracy)/(j+1))

    # #plot accuracy and loss graph - plot
    # plot_loss_acc(path=args.graphs_folder, num_epoch=epoch_idx, train_accuracies_superclass=train_epoch_superclass_accuracy,
    #                         train_accuracies_subclass=train_epoch_subclass_accuracy, train_losses=train_epoch_loss,
    #                         test_accuracies_superclass=test_epoch_superclass_accuracy, test_accuracies_subclass=test_epoch_subclass_accuracy,
    #                         test_losses=test_epoch_loss)




    print(f'Testing Loss at epoch {epoch_idx} : {sum(epoch_loss)/(j+1)}')
    print(f'Testing Superclass accuracy at epoch {epoch_idx} : {sum(epoch_superclass_accuracy)/(j+1)}')
    print(f'Testing Subclass accuracy at epoch {epoch_idx} : {sum(epoch_subclass_accuracy)/(j+1)}')
    print(f'Testing Subtwoclass accuracy at epoch {epoch_idx} : {sum(epoch_subtwoclass_accuracy)/(j+1)}')
    print('-------------------------------------------------------------------------------------------')

    torch.save(model.state_dict(), args.model_save_path+'FMA_3layer.pth')
    torch.save(model.state_dict(), args.model_save_path+'FMA_3layer.pt')
    print("Model saved!")
    
    
    
    #---- picture acc -----# plot error
    print('starte picture for acc')
    path=args.graphs_folder, 
    num_epoch=epoch_idx, 

    train_accuracies_superclass=train_epoch_superclass_accuracy,
    train_accuracies_subclass=train_epoch_subclass_accuracy, 
    train_accuracies_subtwoclass=train_epoch_subtwoclass_accuracy, 
    train_losses=train_epoch_loss,
    test_accuracies_superclass=test_epoch_superclass_accuracy, 
    test_accuracies_subclass=test_epoch_subclass_accuracy,
    test_accuracies_subtwoclass=test_epoch_subtwoclass_accuracy,
    test_losses=test_epoch_loss,#沒有逗號資料格式會變成list ，如果有就是tuple
    
    
    
    #epochs = [x for x in range(num_epoch+1)]
    epochs = [x for x in range(num_epoch[0]+1)]
    
    train_superclass_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":train_accuracies_superclass[0], "Mode":['train']*(num_epoch[0]+1)})
    train_subclass_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":train_accuracies_subclass[0], "Mode":['train']*(num_epoch[0]+1)})
    train_subtwoclass_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":train_accuracies_subtwoclass[0], "Mode":['train']*(num_epoch[0]+1)})
    test_superclass_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":test_accuracies_superclass[0], "Mode":['test']*(num_epoch[0]+1)})
    test_subclass_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":test_accuracies_subclass[0], "Mode":['test']*(num_epoch[0]+1)})
    test_subtwoclass_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":test_accuracies_subtwoclass[0], "Mode":['test']*(num_epoch[0]+1)})
    
    data_superclass = pd.concat([train_superclass_accuracy_df, test_superclass_accuracy_df])
    data_subclass = pd.concat([train_subclass_accuracy_df, test_subclass_accuracy_df])
    data_subtwoclass = pd.concat([train_subtwoclass_accuracy_df, test_subtwoclass_accuracy_df])
    
    sns.lineplot(data=data_superclass.reset_index(inplace=False), x='Epochs', y='Accuracy', hue='Mode')
    plt.title('Superclass Accuracy Graph')
    plt.savefig(path[0]+f'accuracy_superclass_epoch.png')
    #plt.show() 
    plt.clf()
    
    sns.lineplot(data=data_subclass.reset_index(inplace=False), x='Epochs', y='Accuracy', hue='Mode')
    plt.title('Subclass Accuracy Graph')
    plt.savefig(path[0]+f'accuracy_subclass_epoch.png')
    #plt.show()
    plt.clf()
    
       
    sns.lineplot(data=data_subtwoclass.reset_index(inplace=False), x='Epochs', y='Accuracy', hue='Mode')
    plt.title('Subclass Accuracy Graph')
    plt.savefig(path[0]+f'accuracy_subtwoclass_epoch.png')
    #plt.show()
    plt.clf()

    #--計算LOSS--#
    train_loss_df = pd.DataFrame({"Epochs":epochs, "Loss":train_losses[0], "Mode":['train']*(num_epoch[0]+1)})
    test_loss_df = pd.DataFrame({"Epochs":epochs, "Loss":test_losses[0], "Mode":['test']*(num_epoch[0]+1)})
    
    train_test_loss = pd.concat([train_loss_df, test_loss_df])
    
    sns.lineplot(data=train_test_loss.reset_index(inplace=False), x='Epochs', y='Loss', hue='Mode')
    plt.title('Loss Graph')
    plt.savefig(path[0]+f'loss_epoch.png')
    #plt.show()
    plt.clf()
    print('picture save done  start next epoch')







































