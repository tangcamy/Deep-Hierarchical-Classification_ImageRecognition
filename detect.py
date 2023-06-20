import os
import cv2 
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import resnet50
from runtime_args import args
from load_dataset import LoadDataset
from helper import read_meta
from urllib.request import urlopen


device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')

'''mdoelsave.pth location'''
modelroot = '/home/orin/L5C_CellFMA/Deep-Hierarchical-Classification_ImageRecognition/dataset/'
modelName = 'FMA_224224.pth'
os.chdir(modelroot)


model = resnet50.ResNet50()
model = model.to(device)
model.load_state_dict(torch.load(modelName))
model.eval()
#print(model)


'''data - loader'''
batch_size=1
epoch = 1
coarse_labels,fine_labels = read_meta(args.metafile)


datacsv ='/home/orin/L5C_CellFMA/Deep-Hierarchical-Classification_ImageRecognition/dataset/test.csv' #args.test_csv
detect_dataset = LoadDataset(image_size=args.img_size, image_depth=args.img_depth, csv_path=datacsv,
                            cifar_metafile=args.metafile, transform=transforms.ToTensor(),return_label=True) #return_label=Ture /for Training and Testing , False

detect_generator = DataLoader(detect_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

for e in range(epoch):
        for j, sample in tqdm(enumerate(detect_generator)): #detect_generator
                print('detect:'+str(j))
                batch_x ,batch_y1,batch_y2,imgname= sample['image'].to(device), sample['label_1'].to(device), sample['label_2'].to(device),sample['image_path']
                print(imgname,batch_y1,batch_y2)
                superclass_pred,subclass_pred = model(batch_x)

                print('-----------------Tensor-----------------')
                print(superclass_pred)
                print(subclass_pred)


                print('-----------------label index---------------')
                predicted_super = torch.argmax(superclass_pred, dim=1)
                predicted_sub = torch.argmax(subclass_pred, dim=1)
                print(predicted_super)
                print(predicted_sub)

                print('-----------------confidence-----------------') ## 有點奇怪
                probs_super = torch.nn.functional.softmax(superclass_pred, dim=1)
                conf,classes = torch.max(probs_super,1)
                imgclass= coarse_labels[(classes.item())]
                print('superclass',conf,imgclass)

                probs_sub = torch.nn.functional.softmax(subclass_pred, dim=1)
                conf_sub,classes_sub = torch.max(probs_sub,1)
                imgclass_sub= fine_labels[(classes_sub.item())]
                print('subclass',conf_sub,imgclass_sub)



