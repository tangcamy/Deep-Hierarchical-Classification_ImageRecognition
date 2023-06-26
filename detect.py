import os
import cv2 
import numpy as np
import pandas as pd
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

def makedirs(path):
    try:
        os.makedirs(path)
    except:
        return


device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')

'''mdoelsave.pth location'''
os.chdir(args.model_save_path)
modelName = 'FMA_224224.pth'

''' Mode read'''
model = resnet50.ResNet50()
model = model.to(device)
model.load_state_dict(torch.load(modelName))
model.eval()

''' predict csv'''
datacsv ='detect.csv' #args.test_csv現在路徑在args.model_save_path

'''data - loader'''
batch_size=1
epoch = 1
datadic={}
coarse_labels,fine_labels,third_labels = read_meta(args.metafile)

detect_dataset = LoadDataset(image_size=args.img_size, image_depth=args.img_depth, csv_path=datacsv,
                            cifar_metafile=args.metafile, transform=transforms.ToTensor(),return_label=True) #return_label=Ture /for Training and Testing , False

detect_generator = DataLoader(detect_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

dfsave = pd.DataFrame()

r=0
for e in range(epoch):
        for j, sample in tqdm(enumerate(detect_generator)): #detect_generator
                print("-----------Number-----:"+str(j))
                #----for test.csv testing----#
                batch_x ,batch_y1,batch_y2,batch_y3,imgpath= sample['image'].to(device), sample['label_1'].to(device), sample['label_2'].to(device), sample['label_3'].to(device),sample['image_path']
                print(imgpath,batch_y1,batch_y2,batch_y3)

#                print(imgpath)
                ''' Tensor balue'''
                superclass_pred,subclass_pred ,subtwoclass_pred= model(batch_x) 
                #predicted_super = torch.argmax(superclass_pred, dim=1)#tensor([1])
                #predicted_sub = torch.argmax(subclass_pred, dim=1)#tensor([9])
                #predicted_sub tow= torch.argmax(subtwoclass_pred, dim=1)#tensor([9])

                ''' confidence  & classes'''
                ''' - superclasses'''
                probs_super = torch.nn.functional.softmax(superclass_pred, dim=1) 
                super_value,super_index=torch.topk(probs_super,k=2,largest=True) #torch.topk(取出前幾大) , 2取出幾個        
                conf,classes = torch.max(probs_super,1) 
                imgclass= coarse_labels[(classes.item())]
                print('superclass',conf,imgclass)

                ''' - subclasses'''
                probs_sub = torch.nn.functional.softmax(subclass_pred, dim=1)
                sub_value,sub_index=torch.topk(probs_sub,k=4,largest=True)#torch.topk(取出前幾大) , 2取出幾個        
                conf_sub,classes_sub = torch.max(probs_sub,1)
                imgclass_sub= fine_labels[(classes_sub.item())]
                print('subclass',conf_sub,imgclass_sub)

                ''' - subtwoclasses'''
                probs_subtwo = torch.nn.functional.softmax(subtwoclass_pred, dim=1)
                subtwo_value,subtwo_index=torch.topk(probs_subtwo,k=5,largest=True)#torch.topk(取出前幾大) , 2取出幾個        
                conftwo_sub,classestwo_sub = torch.max(probs_subtwo,1)
                imgclasstwo_sub= third_labels[(classestwo_sub.item())]
                print('subclass',conftwo_sub,imgclasstwo_sub)
               
                ''' Get into datadic '''
                output_dic = {
                        'super_conf':[str(index)[:6] for index in super_value[0].tolist()],
                        'super_class':[coarse_labels[index] for index in super_index[0].tolist()],
                        'sub_conf':[str(index)[:6] for index in sub_value[0].tolist()],
                        'sub_class':[fine_labels[index] for index in sub_index[0].tolist()],
                        'subtwo_conf':[str(index)[:6] for index in subtwo_value[0].tolist()],
                        'subtwo_class':[third_labels[index] for index in subtwo_index[0].tolist()],
                        'Layer_1_ans':imgclass,
                        'Layer_1_conf':str(conf[0].tolist())[:6],
                        'Layer_2_ans':imgclass_sub,
                        'Layer_2_conf':str(conftwo_sub[0].tolist())[:6],
                        'Layer_3_ans':imgclasstwo_sub,
                        'Layer_3_conf':str(conf_sub[0].tolist())[:6],
                        'Layer_1_True':coarse_labels[(batch_y1.item())],
                        'Layer_2_True':fine_labels[(batch_y2.item())],
                        'Layer_3_True':third_labels[(batch_y3.item())]
                }
                ''' dataframe concat'''
                datadic[imgpath[0]] = output_dic
                df = pd.DataFrame(datadic)
                df = df.T

                if  len(dfsave) == 0 :
                        dfsave = df 
                else :
                        dfsave = pd.concat([df,dfsave],axis=0)

                #print(dfsave)
                ''' 暫時設定'''
                #r=r+1
                #if r == 2 :
                #        break

'''datasave cleaner'''
index_duplicates = dfsave.index.duplicated()
dfsave = dfsave.loc[~index_duplicates]
#dfsave.reset_index(drop=True,inplace=True)

makedirs(args.model_save_path+'result/')
dfsave.to_csv(args.model_save_path+'result/detect_predict.csv',index=True,index_label='ImagePath')
print('data_save:'+args.model_save_path+'detect_predict.csv')

