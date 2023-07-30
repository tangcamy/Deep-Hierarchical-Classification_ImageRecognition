import cv2
import os 
image_size =224

Traintype =True

if Traintype:
    image_root ='D:/Auo/Deep_Hierarchical_Classification-main/Deep_1/dataset/images/'
    root = image_root
else:
    detect_root ='D:/Auo/Deep_Hierarchical_Classification-main/Deep_1/dataset/detect_imgs/'
    root = detect_root


os.chdir(root)
for imgname in os.listdir(root):
    image = cv2.imread(imgname)
    imageresize = cv2.resize(image, (image_size,image_size))
    cv2.imwrite(imgname,imageresize)
    print(imgname)