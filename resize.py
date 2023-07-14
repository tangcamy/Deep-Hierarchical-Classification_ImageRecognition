import cv2
import os 
image_size =160

Traintype =False

if Traintype:
    image_root ='/home/orin/L5C_CellFMA/D3_Deep-Hierarchical-Classification_ImageRecognition/dataset/images/'
    root = image_root
else:
    detect_root ='/home/orin/L5C_CellFMA/D3_Deep-Hierarchical-Classification_ImageRecognition/dataset/detect_imgs/'
    root = detect_root


os.chdir(root)
for imgname in os.listdir(root):
    image = cv2.imread(imgname)
    imageresize = cv2.resize(image, (image_size,image_size))
    cv2.imwrite(imgname,imageresize)
    print(imgname)