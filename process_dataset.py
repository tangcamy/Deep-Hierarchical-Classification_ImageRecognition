'''Pre-processing script to read Cifar-100 dataset and write the images onto disk with the corresponding labels recorded in a csv file.
'''

import os
import pickle
import numpy as np
import pandas as pd
import imageio
import cv2
from tqdm import tqdm
from helper import unpickle, read_meta


class Preprocess:
    '''Process the pickle files.
    '''
    def __init__(self, meta_filename='/home/orin/L5C_CellFMA/D3_Deep-Hierarchical-Classification_ImageRecognition/dataset/pickle_files/meta', train_file='/home/orin/L5C_CellFMA/D3_Deep-Hierarchical-Classification_ImageRecognition/dataset/pickle_files/train', test_file='/home/orin/L5C_CellFMA/D3_Deep-Hierarchical-Classification_ImageRecognition/dataset/pickle_files/test',
                        image_write_dir='/home/orin/L5C_CellFMA/D3_Deep-Hierarchical-Classification_ImageRecognition/dataset/images/', csv_write_dir='/home/orin/L5C_CellFMA/D3_Deep-Hierarchical-Classification_ImageRecognition/dataset/', train_csv_filename='train.csv', test_csv_filename='test.csv'):
        '''Init params.
        '''
        self.meta_filename = meta_filename
        self.train_file = train_file
        self.test_file = test_file
        self.image_write_dir = image_write_dir
        self.csv_write_dir = csv_write_dir
        self.train_csv_filename = train_csv_filename
        self.test_csv_filename = test_csv_filename

        if not os.path.exists(self.image_write_dir):
            os.makedirs(self.image_write_dir)

        if not os.path.exists(self.csv_write_dir):
            os.makedirs(self.csv_write_dir)
            
        #--原本的會有問題--#
        #self.coarse_label_names, self.fine_label_names = read_meta(meta_filename=self.meta_filename)
        self.coarse_label_names, self.fine_label_names , self.third_label_names= read_meta(self.meta_filename)



    def process_data(self, train=True):
        '''Read the train/test data and write the image array and its corresponding label into the disk and a csv file respectively.
        '''

        if train:
            pickle_file = unpickle(self.train_file)
        else:
            pickle_file = unpickle(self.test_file)

        filenames = pickle_file['filenames']#[t.decode('utf8') for t in pickle_file[b'filenames']]
        fine_labels = pickle_file['fine_labels']#pickle_file[b'fine_labels']
        coarse_labels = pickle_file['coarse_labels']#pickle_file[b'coarse_labels']
        third_labels = pickle_file['third_labels']#pickle_file[b'third_labels']
        #data = pickle_file[b'data']
        '''
        images = []
        for d in data:
            image = np.zeros((32,32,3), dtype=np.uint8)
            image[:,:,0] = np.reshape(d[:1024], (32,32))
            image[:,:,1] = np.reshape(d[1024:2048], (32,32))
            image[:,:,2] = np.reshape(d[2048:], (32,32))
            images.append(image)
        '''

        if train:
            csv_filename = self.train_csv_filename
        else:
            csv_filename = self.test_csv_filename
        c=0
        with open(f'{self.csv_write_dir}/{csv_filename}', 'w+') as f:
            for i, image in enumerate(filenames):
                filename = filenames[i]
                coarse_label = self.coarse_label_names[coarse_labels[i]]
                fine_label = self.fine_label_names[fine_labels[i]]
                third_label = self.third_label_names[third_labels[i]]
                #imageio.imsave(f'{self.image_write_dir}{filename}', image)
                c=c+1
                #print('count:'+str(c))
                print(f'{self.image_write_dir}{filename}')
                f.write(f'{self.image_write_dir}{filename}, {coarse_label}, {fine_label},{third_label}\n')



p = Preprocess()

##-----train-----##
p.process_data(train=True) #process the training set
print('train download ok')

##-----test-----##
p.process_data(train=False) #process the testing set
print('test download ok')









