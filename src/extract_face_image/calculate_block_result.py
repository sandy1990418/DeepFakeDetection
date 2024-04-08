## location 
import os 
## progress bar
from tqdm import tqdm , tqdm_pandas
## parallel apply
import swifter
## save large data file
from joblib import dump,load
##################################################################
##################################################################

## computer vision package
import cv2 as cv
## data processing package
import numpy as np 
import pandas as pd 
from skimage.metrics import structural_similarity
from scipy.fft import fft, dct
import  scipy.stats
## visualization package
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling
## model construction package
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

## GLCM Co occurrence matrix 
from skimage.feature import graycomatrix


##################################################################
##################################################################

## MTCNN Facial detection
from facenet_pytorch import MTCNN ,InceptionResnetV1
#from mtcnn.mtcnn import MTCNN
import torch
import mmcv
from PIL import Image, ImageDraw

## release ram 
import gc

# ################################################################
# #################  extract face image  #########################
# ################################################################

class Viedeo2img_class:
    def __init__(self,file_location,store_location):
        self.file_location = file_location 
        self.store_location = store_location 
        self.video_list = os.listdir(self.file_location)
        # if self.file_location == 'deepfake_database\Celeb-DF\Celeb-DF-v1\Celeb-synthesis':
        #     self.video_list = os.listdir(self.file_location)[500:]

    def Video2img(self,file):
        #################################################################
        ##################### function of Video2img #####################
        #################################################################

        #######  file location  #######
        file_name = (file.split('\\')[-1]).split(".")[0]
        file_location = os.path.join(os.getcwd(),file)
        
        #######  store location  #######
        store_sub_location = file_location.split('deepfake_database')[-1].split(f'\\{file_name}')[0]
        store_location =self.store_location+store_sub_location
        store_location = os.path.join(os.getcwd(),store_location)

        ############## detect if GPU is availible ##############
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #print('Running on device: {}'.format(device))

        ############## object  ##############
        mtcnn = MTCNN(keep_all=True,device=device)

        ############## read video  ##############
        video = mmcv.VideoReader(file_location)
        frames = [frame for frame in video]

        frames_tracked = []
        for i in range(0,len(frames)):
            frame  = frames[i]
            #print('\rTracking frame: {}'.format(i + 1), end='')
            boxes, _ = mtcnn.detect(frame)
            if type(boxes).__module__ != 'numpy':
                continue
            boxes = boxes.tolist()[0]
            boxes = [round(i) for i in boxes]
            frame = frame[boxes[1]:boxes[3],boxes[0]:boxes[2],:]
            try :
                frame = cv.resize(frame,(128,128))
                if os.path.exists(store_location)!=True:
                    os.makedirs(store_location)
                cv.imwrite(f'{store_location}\\{file_name}_{i}.png',frame)
            except:
                continue
        #print('\nDone')

    def All_video2img(self):
        for i,file in enumerate(tqdm(self.video_list)):
            file = os.path.join(self.file_location,file)
            self.Video2img(file)
        print('All done')


######################################################
#######   video and file  in folder   ################
######################################################

class Video_file:
    def __init__(self,real_fold,fake_fold):
        self.real_fold  = real_fold       
        self.fake_fold  = fake_fold 
    ################################################################
    #################   video in folder    #########################
    ################################################################

    ## video in folder 
    def list_video_df(self):
        #################################################################
        ####################        Dataset         #####################
        '''
        1. Celeb-DF(v1、v2)
        2. DFGC(2022)
        '''
        #################################################################
        #################################################################
        ## read video in directory (fake and real)
        real_list = pd.DataFrame(columns=['group','label'])
        for i,multi_file in enumerate(self.real_fold):
            multi_real_list = (os.listdir(multi_file))
            multi_real_list = pd.DataFrame(multi_real_list,columns=['group'])
            multi_real_list['label'] = 1
            real_list=pd.concat([real_list,multi_real_list])
            del multi_real_list

        fake_list = os.listdir(self.fake_fold)
        fake_list = pd.DataFrame(fake_list,columns=['group'])
        fake_list['label'] = 0
        ## combine two list
        df = pd.concat([real_list,fake_list],ignore_index=True) ##fake_list.extend(real_list)
        df['group'] = [i.split('.mp4')[0] for i in df['group']] 
        del real_list,fake_list
        return df


    ################################################################
    #################   file in folder     #########################
    ################################################################
    def list_file_df(self):
        group_df = self.list_video_df()
        real_folder = [''.join(['deepfake_database/video2img/',i.split('deepfake_database/')[-1]]) for i in self.real_fold]
        ## read file in directory (fake and real)
        real_list = pd.DataFrame(columns=['file_name'])
        for i,multi_file in enumerate(real_folder):
            multi_real_list = os.listdir(os.path.join(os.getcwd(),multi_file))
            multi_real_list = pd.DataFrame(multi_real_list,columns=['file_name'])
            multi_real_list['file_location'] =multi_file
            real_list=pd.concat([real_list,multi_real_list])
            
            del multi_real_list
        

        fake_folder = ''.join(['deepfake_database/video2img/',self.fake_fold.split('deepfake_database/')[-1]]) 
        fake_list = os.listdir(os.path.join(os.getcwd(),fake_folder))
        fake_list = pd.DataFrame(fake_list,columns=['file_name'])
        fake_list['file_location'] = fake_folder

        ## combine two list
        df = pd.concat([real_list,fake_list],ignore_index=True) ##fake_list.extend(real_list)
        df['group']=["_".join(i.split("_")[:-1]) for i in df['file_name']]
        df['group_index']=[int(i.split("_")[-1].split('.')[0]) for i in df['file_name']]
        df = pd.merge(df,group_df,on = "group",how ='left') 
        df.sort_values(['group', 'group_index'], ascending=[True, True],ignore_index=True,inplace=True)
        df["file_name_next"]= df.groupby('group').shift(-1)[["file_name"]]
        df.dropna(axis=0,inplace =True)
        
        del real_list,fake_list,group_df
        
        df.to_csv('list_file_df.csv')
        del df 
        gc.collect()
        #return df



class Block_img:
    def __init__(self,df,split_number=128):
        self.img_location =  f'{df.file_location}/{df.file_name}'
        self.split_number = split_number
        
    ### block split
    def block_split(func):
        def wrapper(self): #(self,location)
            block_img = func(self) #func(self,location)
            img_shape = block_img.shape[0] -1
            pixel_result = np.sqrt((block_img[:img_shape,:img_shape] - block_img[:img_shape,1:])**2+(block_img[:img_shape,:img_shape] - block_img[1:,:img_shape])**2)
            ## 一階動差 [np.sqrt(((t1[y][x]-t1[y+1][x])**2)+((t1[y][x]-t1[y][x+1])**2)) for y in range(0,2) for x in range(0,2)]
            F_result = np.around(pixel_result[:,:pixel_result.shape[1]-1]-pixel_result[:,1:]).astype(int)
            F_result[np.where(F_result>=2)] = 4
            F_result[np.where(F_result==1)] = 3
            F_result[np.where(F_result==0)] = 2
            F_result[np.where(F_result==-1)] = 1
            F_result[np.where(F_result<=-2)] = 0
            
            gray_result=graycomatrix(F_result, [1,2], [0, np.pi/4, np.pi/2, 3*np.pi/4],levels=5,normed = True)
            gray_1step_result = np.mean(gray_result[:,:,0,:],axis = 2).ravel()
            gray_2step_result = np.mean(gray_result[:,:,1,:],axis = 2).ravel()
            del pixel_result,gray_result
            
            
            ## co - occurrence matrix  calculation 
            full_temp =np.full((F_result.shape[0]+4,F_result.shape[1]+4), np.nan)
            full_temp[2:-2,2:-2] = F_result
            gray_12step_result = np.zeros((5,5))
            for i in (range(2,full_temp.shape[0]-2)):
                for j in range(2,full_temp.shape[1]-2):
                    v_and_h = np.array([full_temp[i,j+1],full_temp[i,j-1],full_temp[i+1,j],full_temp[i-1,j]])
                    v_and_h=v_and_h[~np.isnan(v_and_h)].astype(int)
                    
                    
                    v_and_h2 = np.array([full_temp[i,j+2],full_temp[i,j-2],full_temp[i+2,j],full_temp[i-2,j]])
                    v_and_h2=v_and_h2[~np.isnan(v_and_h2)].astype(int)
                    
                    
                    if v_and_h2.size != 0 and v_and_h.size != 0:
                        for s1 in v_and_h.tolist():
                            for s2 in v_and_h2.tolist():
                                gray_12step_result[s1,s2]+=1
                          
    
            
            gray_12step_result = (gray_12step_result.astype(int)/np.sum(gray_12step_result)).ravel()   
            
            result = np.append(gray_1step_result,gray_2step_result)
            result = np.append(result,gray_12step_result)

            del full_temp,gray_12step_result
            
            return result
        return wrapper
    
    
    @block_split
    def B_input(self,image_size):
        img = cv.imread(self.img_location,cv.IMREAD_COLOR)
        img = cv.resize(img,(image_size,image_size))
        return img[:,:,0]

    @block_split
    def G_input(self,image_size):
        img = cv.imread(self.img_location,cv.IMREAD_COLOR)
        img = cv.resize(img,(image_size,image_size))
        return img[:,:,1]

    @block_split
    def R_input(self,image_size):
        img = cv.imread(self.img_location,cv.IMREAD_COLOR)
        img = cv.resize(img,(image_size,image_size))
        return img[:,:,2]

    @block_split
    def HSV_input(self,image_size):
        ## 只抓出V的部分
        img = cv.cvtColor(cv.imread(self.img_location,cv.IMREAD_COLOR),cv.COLOR_BGR2HSV)
        img = cv.resize(img,(image_size,image_size))
        return img[:,:,2]
    
    @block_split
    def YCrCb_input(self,image_size):
        ## 只抓出Y的部分
        img = cv.cvtColor(cv.imread(self.img_location,cv.IMREAD_COLOR) ,cv.COLOR_BGR2YCrCb)
        img = cv.resize(img,(image_size,image_size))
        return img[:,:,0]
