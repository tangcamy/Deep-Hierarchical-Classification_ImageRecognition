import cv2
import os 
image_size =224

#image_root ='/home/orin/L5C_CellFMA/Deep-Hierarchical-Classification_ImageRecognition/dataset/images/'
detect_root ='/home/orin/L5C_CellFMA/Deep-Hierarchical-Classification_ImageRecognition/dataset/detect_imgs/'

#os.chdir(image_root)
os.chdir(detect_root)

for imgname in os.listdir(detect_root):
    image = cv2.imread(imgname)
    imageresize = cv2.resize(image, (image_size,image_size))
    cv2.imwrite(imgname,imageresize)
    print(imgname)