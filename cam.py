#coding: utf-8
import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from model import resnet50
import torch.nn as nn
import matplotlib.pyplot as plt

# 获取模型输出的feature/score
class_num = 14

modelName = 'FMA_3layer_50.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
''' Mode read'''
model = resnet50.ResNet50()
model_ft = model.to(device)
model_ft.load_state_dict(torch.load('FMA_3layer_50.pth', map_location=lambda  storage, loc:storage))  
model_features = nn.Sequential(*list(model_ft.children())[:-7])

class_ = {0:'CF REPAIR FAIL',1:'PI SPOT-WITH PAR',2:'POLYMER',3:'GLASS BROKEN',4:'PV-HOLE-T',5:'CF DEFECT',6:'CF PS DEFORMATION',7:'FIBER',8:'AS-RESIDUE-E',9:'LIGHT METAL',10:'GLASS CULLET',11:'ITO-RESIDUE-T',12:'M1-ABNORMAL',13:'ESD'}
model_ft.eval()
model_features.eval()

# Display all model layer weights
"""
for name, para in model_ft.named_parameters():
    print('{}: {}'.format(name, para.shape))
"""
fc_weights = model_ft.linear_lvl3.weight   #numpy数组取维度fc_weights[0].shape->(5,960)

data_transform = {
        "train": transforms.Compose([transforms.RandomAffine(40, scale=(.85, 1.15), shear=0),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomPerspective(distortion_scale=0.2),
                                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                                    transforms.ToTensor()]),
        "val": transforms.Compose([transforms.ToTensor()])}
""" 
#加載圖像
img_path = './dataset/images/CF PS DEFORMATION@NP@CF@CF_20220619_B5564PE-1-2.jpg'             #单张测试
_, img_name = os.path.split(img_path)
features_blobs = []
img = Image.open(img_path).convert('RGB')
img_tensor = (data_transform['val'](img)/255.0).unsqueeze(0).to(device) #[1,3,224,224]

features = model_features(img_tensor).detach()  #[1,2048,56,56]
logit = model_ft(img_tensor)  #[1,2] -> [ 3.3207, -2.9495]

h_x = torch.nn.functional.softmax(logit[2], dim=1).data.squeeze()  #tensor([0.9981, 0.0019])
probs, idx = h_x.sort(0, True)      #按概率从大到小排列
probs = probs.cpu().numpy()  #if tensor([0.0019,0.9981]) ->[0.9981, 0.0019]
idx = idx.cpu().numpy()  #[1, 0]

for i in range(class_num):
    print('{:.3f} -> {}'.format(probs[i], class_[idx[i]]))  #打印预测结果

def returnCAM(feature_conv, weight_softmax, class_idx):
    bz, nc, h, w = feature_conv.shape        #1,2048,56,56
    output_cam = []
    for idx in class_idx:  #只输出预测概率最大值结果不需要for循环
        feature_conv = feature_conv.reshape((nc, h*w))  # [2048,56*56]
        cam = torch.mm(weight_softmax[idx].unsqueeze(0),feature_conv)  #(1, 2048) * (2048, 56*56) -> (1, 56*56) （n,）是一个数组，既不是行向量也不是列向量
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  #Normalize
        cam_img = cam_img.detach().cpu().numpy()
        cam_img = np.uint8(255 * cam_img)                      #Format as CV_8UC1 (as applyColorMap required)
 
        #output_cam.append(cv2.resize(cam_img, size_upsample))  # Resize as image size
        output_cam.append(cam_img)
    return output_cam
 
# CAMs = returnCAM(features, fc_weights, [idx[0]])  #输出预测概率最大的特征图集对应的CAM
CAMs = returnCAM(features, fc_weights, idx)  #输出预测概率最大的特征图集对应的CAM
print(img_name + ' output for the top1 prediction: %s' % class_[idx[0]])

img = cv2.imread(img_path)
height, width, _ = img.shape  #读取输入图片的尺寸
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)  #CAM resize match input image size
result = heatmap * 0.6 + img * 0.4    #比例可以自己调节

text = '%s %.2f%%' % (class_[idx[0]], probs[0]*100) 				 #激活图结果上的文字显示
cv2.putText(result, text, (20, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
            color=(123, 222, 238), thickness=1, lineType=cv2.LINE_AA)
CAM_RESULT_PATH = './dataset/'  #CAM结果的存储地址
if not os.path.exists(CAM_RESULT_PATH):
    os.mkdir(CAM_RESULT_PATH)
image_name_ = 'output'
cv2.imwrite(CAM_RESULT_PATH + image_name_ + '_' + 'pred_vit_' + class_[idx[0]] + '.jpg', result)  #写入存储磁盘
""" 
imgpath = './dataset/detect_imgs/'

def returnCAM(feature_conv, weight_softmax, class_idx):
    bz, nc, h, w = feature_conv.shape        #1,2048,56,56
    output_cam = []
    for idx in class_idx:  #只输出预测概率最大值结果不需要for循环
        feature_conv = feature_conv.reshape((nc, h*w))  # [2048,56*56]
        cam = torch.mm(weight_softmax[idx].unsqueeze(0),feature_conv)  #(1, 2048) * (2048, 56*56) -> (1, 56*56) （n,）是一个数组，既不是行向量也不是列向量
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  #Normalize
        cam_img = cam_img.detach().cpu().numpy()
        cam_img = np.uint8(255 * cam_img)                      #Format as CV_8UC1 (as applyColorMap required)
 
        #output_cam.append(cv2.resize(cam_img, size_upsample))  # Resize as image size
        output_cam.append(cam_img)
    return output_cam

CAM_RESULT_PATH = './dataset/cam_pic/detect/cam_img/' 

for j,img in enumerate(os.listdir(imgpath)):
    bg = Image.new('RGB',(448,224), "#000000")
    print(j)
    images = Image.open(imgpath+img).convert('RGB')
    img_tensor = (data_transform['val'](images)/255.0).unsqueeze(0).to(device) #[1,3,224,224]
    features = model_features(img_tensor).detach()  #[1,2048,56,56]

    logit = model_ft(img_tensor)  #[1,2] -> [ 3.3207, -2.9495]
    h_x = torch.nn.functional.softmax(logit[2], dim=1).data.squeeze()  #tensor([0.9981, 0.0019])
    probs, idx = h_x.sort(0, True)      #按概率从大到小排列
    probs = probs.cpu().numpy()  #if tensor([0.0019,0.9981]) ->[0.9981, 0.0019]
    idx = idx.cpu().numpy()  #[1, 0]
        
    CAMs = returnCAM(features, fc_weights, idx)  #输出预测概率最大的特征图集对应的CAM
    print(img + ' output for the top1 prediction: %s' % class_[idx[0]])
        
    imgs = cv2.imread(imgpath+img)
    height, width, _ = imgs.shape  #读取输入图片的尺寸
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (height, width)), cv2.COLORMAP_JET)  #CAM resize match input image size
    result = heatmap * 0.5 + imgs * 0.5    #比例可以自己调节
    text = '%s %.2f%%' % (class_[idx[0]], probs[0]*100) 				 #激活图结果上的文字显示
    cv2.putText(result, text, (10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
            color=(123, 222, 238), thickness=1, lineType=cv2.LINE_AA)
    img_name = img.split('@')
    cv2.imwrite(CAM_RESULT_PATH + img[:-4] + '_campred.jpg', result)
    """ 
    x = j%3
    y = j//3
    """ 
    if img_name[0] != class_[idx[0]]:
        imgs = Image.open(imgpath+img)
        bg.paste(imgs,(0,0))
        im = Image.open(CAM_RESULT_PATH + img[:-4] + '_campred.jpg')
        bg.paste(im,(224,0))
        bg.save('./dataset/cam_pic/detect/cam_withorin_false/'+img[:-4]+'_cam.jpg')
    else:
        imgs = Image.open(imgpath+img)
        bg.paste(imgs,(0,0))
        im = Image.open(CAM_RESULT_PATH + img[:-4] + '_campred.jpg')
        bg.paste(im,(224,0))
        bg.save('./dataset/cam_pic/detect/cam_withorin_right/'+img[:-4]+'_cam.jpg')
  
#bg.save('./dataset/cbam_pic/output.jpg')
  
exit()