import pickle
import os 
import pandas as pd
import pickle 
import shutil

# 注意：使用 pickle 的時候，檔名不可以命名成 pickle.py 
'''
data root
'''

rootPic = '/home/orin/L5C_CellFMA/CELL_FMADefect_C2/original_pic/TestPicture_2/'#detectPicture_0621
projectroot = '/home/orin/L5C_CellFMA/D3_Deep-Hierarchical-Classification_ImageRecognition/'
root = projectroot+'dataPickle_Transform/'
pickle_root = root + 'pickel_files/' # save detect pickle
pickleDataset_root = root + 'preimages/' # save detect preimages
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
        df['defectlocate'] = a[2]
        df['locationFlag']=a[1]
        df['FMAdefect']= a[0]
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
coarse_label_names = meta_data['coarse_label_names']
fine_label_names = meta_data['fine_label_names']
third_label_names = meta_data['third_label_names']

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
detect_thirdlabels_list=[]

for name in os.listdir(pickledetect_root):
    a = name.split('_') #a[0]=CF REPAIR FAIL@NP@CF@CF ; a[1]= 20220816 ; a[2]=B76V2XE-1-3.jpg
    imgclassification = a[0]
    b = Nameselect(imgclassification) 
    detect_filenames_list.append(name)
    detect_coarselabels_list.append(coarse_label_names.index(b[2])) #defectlocate ,TFT&CF 
    detect_finelabels_list.append(fine_label_names.index(b[1])) #LocateFlog, NP UP OP INT
    detect_thirdlabels_list.append(third_label_names.index(b[0]))# FMA Defect
    #print(name)

'''# 存資料'''
detect = {
'filenames': detect_filenames_list,
'coarse_labels':detect_coarselabels_list,
'fine_labels':detect_finelabels_list,
'trhid_labels':detect_thirdlabels_list
}
os.chdir(pickle_root)
with open('detect','wb') as file:# 'meta.pickle'
    pickle.dump(detect, file) # 使用 dump 把 data 倒進去 file 裡面

