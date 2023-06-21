import pickle
import os 
import pandas as pd
import pickle 
import shutil

# 注意：使用 pickle 的時候，檔名不可以命名成 pickle.py 
'''
data root
'''

rootPic = '/home/orin/L5C_CellFMA/CELL_FMADefect_C2/original_pic/detectPicture/'
projectroot = '/home/orin/L5C_CellFMA/Deep-Hierarchical-Classification_ImageRecognition/'
root = projectroot+'dataPickle_Transform/'
pickle_root = root + 'pickel_files/'
pickleDataset_root = root + 'preimages/'
pickledetect_root = pickleDataset_root + 'detect/'

'''
data save: 
'''

# In[0] create root
def makedirs(path):
    try:
        os.makedirs(path)
    except:
        return
makedirs(root)
makedirs(pickle_root)
makedirs(pickleDataset_root)
makedirs(pickledetect_root)

csvtotalname = 'detectdata'

def NameListCheck(value):
    a=[value]
    a=a[0].split('@')
    FileNmae = a[0]
    LocationFlag = a[1]
    DefectLocate = a[2]
    site = a[3]
    return FileNmae,LocationFlag,DefectLocate,site

def imgDataClean(fileName,dataset,dstroot,typeset,datalen):
    for img in dataset:
        shutil.copyfile(rootPic+i+'/'+img, dstroot+str(fileName)+'_'+img)
        print(dstroot+str(fileName)+'_'+img)
        a=[fileName]
        a=a[0].split('@')
        df=pd.DataFrame()
        df['FileName']=[fileName]
        df['dataNumber']=str(datalen)
        df['locationFlag']=a[1]
        df['dataType']=typeset
        df['picName']=str(fileName)+'_'+img
        datasave(root,df,csvtotalname)
        
def DataLess(value,fileListData):
    df=pd.DataFrame()
    df['FileName']=[value]
    df['dataNumber']=len(fileListData)
    df['locationFlag']='NAN'
    df['dataType']='NAN'
    df['picName']='NAN'
    datasave(root,df,csvtotalname)
            
def datasave(SAVEROOT,FINALDF,FILENAME):
    Data_name=FILENAME+'.csv' #檔名使用機台名稱
    his = list(filter(lambda x: x[0: len(Data_name)]==Data_name, os.listdir(SAVEROOT)))
    os.chdir(SAVEROOT)  
    if len(his)==1:
        Hisdata=pd.read_csv(FILENAME+'.csv')
        #Hisdata['DATETIME']=pd.to_datetime(Hisdata['DATETIME'])
        Condata=pd.concat([Hisdata,FINALDF],axis=0,ignore_index=False)
        #Condata.drop_duplicates(['fileName','FileName','dataNumber'],keep='first',inplace=True)
        Condata.to_csv(FILENAME+'.csv',index=False)
    else:
        FINALDF.to_csv(FILENAME+'.csv',index=False)


def Nameselect(value):
    '''imagename split slelect'''
    b=value.split('@')
    DefectName = b[0]
    LocationFlag = b[1]
    DefectLocate = b[2]
    site = b[3]
    return DefectName,LocationFlag,DefectLocate,site


def unpickle(file):
    '''Unpickle the given file
    '''
    with open(file, 'rb') as f:
        res = pickle.load(f, encoding='bytes')
    return res


# In[]
''' load meta data '''
meta_data = unpickle(pickle_root+'meta')
fine_label_names = meta_data['fine_label_names']
coarse_label_names = meta_data['coarse_label_names']

''' rootPic : rowdata來源( file/ filename)'''
for i in os.listdir(rootPic):
    try:
        #i='GLASS CULLET@UCT@TFT@Cell'
        fileList = list(filter(lambda x: x[-4:]=='.jpg', os.listdir(rootPic+i)))
        imgDataClean(i,fileList,pickledetect_root,'detect',len(fileList))

    except:
        print('------------Error:'+str(i))
        print('error'+str(i))


detect_filenames_list=[]
detect_coarselabels_list=[]
detect_finelabels_list=[]

for name in os.listdir(pickledetect_root):
    a = name.split('_') #a[0]=CF REPAIR FAIL@NP@CF@CF ; a[1]= 20220816 ; a[2]=B76V2XE-1-3.jpg
    imgclassification = a[0]
    b = Nameselect(imgclassification) 
    detect_filenames_list.append(name)
    detect_coarselabels_list.append(coarse_label_names.index(b[1])) #locationflag
    detect_finelabels_list.append(fine_label_names.index(b[0])) #defectname
    print(name)

'''# 存資料'''
detect = {
'filenames': detect_filenames_list,
'fine_labels':detect_finelabels_list,
'coarse_labels':detect_coarselabels_list
}
os.chdir(pickle_root)
with open('detect','wb') as file:# 'meta.pickle'
    pickle.dump(detect, file) # 使用 dump 把 data 倒進去 file 裡面


# In[] part 2


import os
import pickle
import numpy as np
import pandas as pd
import imageio
import cv2
from tqdm import tqdm
from helper import unpickle, read_meta


class Preprocess_detect:
    '''Process the detedt pickle files.
        image_write_dir : dataset/detect_imgs
    '''
    def __init__(self, meta_filename='/home/orin/L5C_CellFMA/Deep-Hierarchical-Classification_ImageRecognition/dataset/pickle_files/meta',detect_file='/home/orin/L5C_CellFMA/Deep-Hierarchical-Classification_ImageRecognition/dataset/pickle_files/detect',
                        image_write_dir='/home/orin/L5C_CellFMA/Deep-Hierarchical-Classification_ImageRecognition/dataset/detect_imgs/', csv_write_dir='/home/orin/L5C_CellFMA/Deep-Hierarchical-Classification_ImageRecognition/dataset/', detect_csv_filename='detect.csv'):
        '''Init params.
        '''
        self.meta_filename = meta_filename
        self.detect_file = detect_file
        self.image_write_dir = image_write_dir
        self.csv_write_dir = csv_write_dir
        self.detect_csv_filename = detect_csv_filename

        if not os.path.exists(self.image_write_dir):
            os.makedirs(self.image_write_dir)

        if not os.path.exists(self.csv_write_dir):
            os.makedirs(self.csv_write_dir)
            
        #--原本的會有問題--#
        #self.coarse_label_names, self.fine_label_names = read_meta(meta_filename=self.meta_filename)
        self.coarse_label_names, self.fine_label_names = read_meta(self.meta_filename)



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


        csv_filename = self.detect_csv_filename

        c=0
        with open(f'{self.csv_write_dir}/{csv_filename}', 'w+') as f:
            for i, image in enumerate(filenames):
                filename = filenames[i]
                coarse_label = self.coarse_label_names[coarse_labels[i]]
                fine_label = self.fine_label_names[fine_labels[i]]
                #imageio.imsave(f'{self.image_write_dir}{filename}', image)
                c=c+1
                #print('count:'+str(c))
                print(f'{self.image_write_dir}{filename}')
                f.write(f'{self.image_write_dir}{filename}, {coarse_label}, {fine_label}\n')



p = Preprocess_detect()
##-----detect-----##
p.process_data() #process the testing set
print('detect download ok')


