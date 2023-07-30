import pickle
import os 
import pandas as pd
import pickle 
import shutil

# 注意：使用 pickle 的時候，檔名不可以命名成 pickle.py 
'''
data root
'''

rootPic = 'D:/Auo/Deep_Hierarchical_Classification-main/zip/20_Training/20_BL/'
projectroot = 'D:/Auo/Deep_Hierarchical_Classification-main/Deep_1/'
root = projectroot+'dataPickle_Transform/'
pickle_root = root + 'pickel_files/'
pickleDataset_root = root + 'preimages/'
pickleTrain_root = pickleDataset_root + 'train/'
pickleTest_root = pickleDataset_root + 'test/'


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
makedirs(pickleTrain_root)
makedirs(pickleTest_root)
# In[1] section-1 < meta >: coarse_label_names,fine_label_names
data = {
'coarse_label_names': ['CF REPAIR FAIL','PI SPOT-WITH PAR','POLYMER','GLASS BROKEN','PV-HOLE-T','CF DEFECT','CF PS DEFORMATION','FIBER','AS-RESIDUE-E','LIGHT METAL','GLASS CULLET','ITO-RESIDUE-T','M1-ABNORMAL','ESD']
}
# 存資料
os.chdir(pickle_root)
with open('meta','wb') as file:# 'meta.pickle'
    pickle.dump(data, file) # 使用 dump 把 data 倒進去 file 裡面

os.chdir(root)

# In[2] setction-2 train / test >  create : filenames , fine_labels , coarse_labels
csvtotalname = 'pickleTotal'

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


''' rootPic : rowdata來源( file/ filename)'''
for i in os.listdir(rootPic):
    try:
        #i='GLASS CULLET@UCT@TFT@Cell'
        fileList = list(filter(lambda x: x[-4:]=='.jpg', os.listdir(rootPic+i)))
        if len(fileList)>=80:
            print('ok')
            print('------------Data:'+str(i))
            cutNumber = int(len(fileList)*0.9)
            trainData = fileList[0:cutNumber]
            testData = fileList[cutNumber:]

            FileNameCheck = NameListCheck(i)
            if FileNameCheck[2] == 'CF' or FileNameCheck[2] == 'TFT':
                imgDataClean(i,trainData,pickleTrain_root,'train',len(fileList))
                imgDataClean(i,testData,pickleTest_root,'test',len(fileList))
            else:
                DataLess(i,fileList)
        else:
            print('---------DataLess:'+str(i))
            DataLess(i,fileList)               
    except:
        print('------------Error:'+str(i))
        print('error'+str(i))


# In[3] section-3 < read meta dic for data transform index > & < train / test >  create : filenames , fine_labels , coarse_labels
'''
load meta for selection
'''
def unpickle(file):
    '''Unpickle the given file
    '''

    with open(file, 'rb') as f:
        res = pickle.load(f, encoding='bytes')
    return res

meta_data = unpickle(pickle_root+'meta')
#fine_label_names = meta_data['fine_label_names']
coarse_label_names = meta_data['coarse_label_names']


def Nameselect(value):
    b=value.split('@')
    DefectName = b[0]
    LocationFlag = b[1]
    DefectLocate = b[2]
    site = b[3]
    return DefectName,LocationFlag,DefectLocate,site

'''##-----train part -------##'''
print('-----------------------------------Train--------------------------------------')

train_filenames_list=[]
train_coarselabels_list=[]
for name in os.listdir(pickleTrain_root):
    a = name.split('_') #a[0]=CF REPAIR FAIL@NP@CF@CF ; a[1]= 20220816 ; a[2]=B76V2XE-1-3.jpg
    imgclassification = a[0]
    b = Nameselect(imgclassification) 
    train_filenames_list.append(name)
    train_coarselabels_list.append(coarse_label_names.index(b[0])) #defectname
    print(name)

'''# 存資料'''
train = {
'filenames': train_filenames_list,
'coarse_labels':train_coarselabels_list
}

os.chdir(pickle_root)
with open('train','wb') as file:# 'meta.pickle'
    pickle.dump(train, file) # 使用 dump 把 data 倒進去 file 裡面

'''##-----test part -------##'''
print('-----------------------------------Test--------------------------------------')
test_filenames_list=[]
test_coarselabels_list=[]

for name in os.listdir(pickleTest_root):
    a = name.split('_') #a[0]=CF REPAIR FAIL@NP@CF@CF ; a[1]= 20220816 ; a[2]=B76V2XE-1-3.jpg
    imgclassification = a[0]
    b = Nameselect(imgclassification) 
    test_filenames_list.append(name)
    test_coarselabels_list.append(coarse_label_names.index(b[0])) #defectname
    print(name)

'''# 存資料'''
test = {
'filenames': test_filenames_list,
'coarse_labels':test_coarselabels_list
}
os.chdir(pickle_root)
with open('test','wb') as file:# 'meta.pickle'
    pickle.dump(test, file) # 使用 dump 把 data 倒進去 file 裡面

print('ok')

