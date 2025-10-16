
"""
Created on Tue Apr 22 13:44:04 2025

@author: Junxuan Ma
"""
import glob
import czifile
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import pickle
import csv
from scipy.signal import find_peaks
from bisect import bisect
from scipy.signal import butter, lfilter, freqz,sosfreqz, sosfilt
from PIL import Image


#Read confocal files
"""Note the read_image function needs to be adapted based on the format of image
For the following analysis imgseg3 need to have time in 0th dimention
and melted pixel in 1st dimention"""    
def get_filename(tail='*.czi'):
    filenames = glob.glob(tail)
    return filenames

def read_image(filenames, index=0):#Custermize depending on the format
    name= filenames[index].replace('.czi', '')
    calcium_name= filenames[index].replace('_IF.czi', '.czi')
    
    IF = czifile.imread(filenames[index])
    IF2= np.squeeze(IF)
    TU= IF2[0,:,:]
    CGRP= IF2[1,:,:]
    
    Cal = czifile.imread(calcium_name)
    imgseg2 = np.squeeze(Cal)#remove not useful dimensions
    imgseg3 = np.reshape(imgseg2,(imgseg2.shape[0],imgseg2.shape[1]*imgseg2.shape[2]))#flatten the space dimension
    #IF_im = Image.open(filenames[index].replace('.czi', '_IF.tif'))
    
    return imgseg3,  TU,CGRP, name, imgseg2.shape[2],imgseg2.shape[1]

def plot(matrix):
    plt.imshow(matrix)

"""Calcium imaging part"""
"""Initial segmentation, remove area without KCL response based on KCL response mask
image is already in the format of (Time,Melted space)
Start is at least 10 frames before the 1st KCl response
End is the closest frame that KCl response finish
1stThreshold is to remove image part that is too dark
2ndThreshold is the minimum difference between base line and max KCl
"""
#Plot to decide brightness threshold
def Seg_plot_Brithreshold(image, start,  thre_min, n_test, nH, nW):#thre_min is an integel 
    plt.figure(figsize=(24,10))#for space series
    plt.subplots_adjust(hspace=0.12)
    for i in range(thre_min, thre_min + n_test):
        panel = plt.subplot(nH ,nW ,i- thre_min+1)
        imgseg4 = image[ start-100:start,: ].copy()
        imgseg_peak = np.max(imgseg4[:,:],     axis = 0)
        imgsegBr = imgseg_peak >  i
        mask1= np.reshape((imgsegBr),(h,w)).astype(int)
        plt.imshow( mask1)
        plt.title('Threshold1 ='+str( i))

#Plot to decide KCl response threshold
def Seg_plot_KClthreshold(image, start, end, threshold1, n_max, nH, nW):#threshold1 is an integel 
    plt.figure(figsize=(24,15))#for space series
    plt.subplots_adjust(hspace=0.12)
    for i in range(0, n_max):
        panel = plt.subplot(nH ,nW ,i+1)    
        imgseg4 = image[ start-100:start,: ].copy()
        imgseg_peak = np.max(imgseg4,     axis = 0)
        imgsegBr = imgseg_peak > threshold1
                
        imgseg_peak2 = np.mean(image[(end-10):end,:],     axis = 0)
        imgseg_base = np.mean(image[start:(start+10),:], axis = 0)
        imgsegKCl= (imgseg_peak2-imgseg_base)< i#select regions thresholded out and set this region 0
        #imgsegKCl[imgsegBr] = False#regions already removed by Threshold1 will not be removed again by Threshold2
        imgsegBr[imgsegKCl] = False
        mask1= np.reshape((imgsegBr),(h,w)).astype(int)
        
        plt.imshow( mask1 )
        plt.title('Threshold1 ='+str(threshold1)+', Threshold2 ='+str(i))

def Check_all_threshold(image, start, end, thre1_min=40, thre1_max=60, step=2, n_test_KCl=20):#threshold1 is an integel 
    plt.figure(figsize=(30,15))#for space series
    plt.subplots_adjust(hspace=0.2)
    for i in range(thre1_min, thre1_max,step):
        for j in range (0,n_test_KCl):
            nh=int((thre1_max-thre1_min)/step)
            panel = plt.subplot(n_test_KCl ,nh ,int((i-thre1_min)/step)+j*nh+1)    
            imgseg4 = image[ start-100:start,: ].copy()
            imgseg_peak = np.max(imgseg4,     axis = 0)
            imgsegBr = imgseg_peak > i
                    
            imgseg_peak2 = np.mean(image[(end-10):end,:],     axis = 0)
            imgseg_base = np.mean(image[start:(start+10),:], axis = 0)
            imgsegKCl= (imgseg_peak2-imgseg_base)< j
            imgsegBr[imgsegKCl] = False
            mask1= np.reshape((imgsegBr),(h,w)).astype(int)
            
            plt.imshow( mask1 )
            plt.title('Threshold1 ='+str(i)+', Threshold2 ='+str(j))
        
      
def set_pix_to_zero(Timeseries, mask):
    maskseries = mask [None,:].copy()
    maskseries = np.repeat (maskseries, Timeseries.shape[0] ,axis=0)
    Timeseries2 = Timeseries.copy()#add time dimension in the mask
    Timeseries2[maskseries] = 0#add time dimension in the mask
    return(Timeseries2)
    
def KCl_segmentaion(image, start, end, Threshold1, Threshold2):
    imgseg4 = image[start-100:start,: ].copy()
    imgseg_peak = np.max(imgseg4, axis = 0)
    imgsegBr = imgseg_peak <= Threshold1
    image2 = set_pix_to_zero(image, imgsegBr)
    
    imgseg_peak2 = np.mean(image[(end-10):end,:],     axis = 0)
    imgseg_base = np.mean(image[start:(start+10),:], axis = 0)
    imgsegKCl= (imgseg_peak2-imgseg_base) < Threshold2
    image2 = set_pix_to_zero(image2, imgsegKCl)

    return image2

"""Image transformation (flatten/unflatten, matrix/df) and visualization"""
def flatten(image_seg, pix_h=292, pix_w=384):
    return np.reshape(image_seg, shape=(pix_h,pix_w))

def visual_timepoint_flatten(image_seg, time=0, pix_h=292, pix_w=384):
    plot= flatten(image_seg[time,:], pix_h, pix_w)
    plt.imshow(plot)

#when it is not a time series but a single frame
def visual_frame_flatten(imgsegBr, pix_h=292, pix_w=384):
    plt.imshow(flatten(imgsegBr, pix_h, pix_w))

#PCA
def PCA_space(image, time_start, time_end, n_PCA_components):
    data= image.astype(np.int8)[time_start:time_end,:]
    pca =PCA(n_components= n_PCA_components)
    data_PCA = pca.fit(data).transform(data)
    return pca.components_, pca.explained_variance_, pca.explained_variance_ratio_, pca, data_PCA
"""Here time_start and time_end should expand over the timepoint where KCl is added"""

def PCA_time(image, time_start, time_end, n_PCA_components):
    data_t= image.astype(np.int8)[time_start:time_end,:]
    data_t = np.transpose(data_t, (1,0))#(1,0) is to put 1sth dimension into new 0th dimension
    pca =PCA(n_components= n_PCA_components)
    data_t_PCA = pca.fit(data_t).transform(data_t)
    return pca.components_, pca.explained_variance_, pca.explained_variance_ratio_, pca, data_t_PCA


#Plot all components in a panel: space
def PCA_plot_space(PCA_comp, PCA_eivalue, PCA_ratio, n_panel_width, n_panel_height):
    plt.figure(figsize=(24,15))#for space series
    plt.subplots_adjust(hspace=0.09)
    for i in range(0,PCA_comp.shape[0]):
        ci = PCA_comp[i, :]
        eigenvalue =PCA_eivalue[i]
        ratio = PCA_ratio[i]
        panel = plt.subplot(n_panel_height,n_panel_width,i+1)
        panel.imshow(np.reshape(ci,shape=(h,w)))
        plt.title(str(i)+'\nEigenvalue ='+str(eigenvalue)+'\nRatio ='+str(ratio))
        plt.rcParams.update({'font.size':8})
        
def PCA_plot_space_binary(threshold, PCA_comp, PCA_eivalue, PCA_ratio, n_panel_width, n_panel_height):
    plt.figure(figsize=(24,15))#for space series
    plt.subplots_adjust(hspace=0.09)
    for i in range(0,PCA_comp.shape[0]):
        ci = PCA_comp[i, :].copy()#without copy(), the visualization will ruin the data
        eigenvalue =PCA_eivalue[i]
        ratio = PCA_ratio[i]
        panel = plt.subplot(n_panel_height,n_panel_width,i+1)
        ci[ci>threshold]=1
        ci[ci<=threshold]=0
        panel.imshow(np.reshape(ci,shape=(h,w)))
        plt.title(str(i)+'\nEigenvalue ='+str(eigenvalue)+'\nRatio ='+str(ratio))
        plt.rcParams.update({'font.size':8})

#Plot all components in a panel: time        
def PCA_plot_time(PCA_comp, PCA_eivalue, PCA_ratio, n_panel_width, n_panel_height):
    plt.figure(figsize=(24,15))#for time series
    plt.subplots_adjust(hspace=0.3)
    for i in range(0,PCA_comp.shape[0]):
        ci = PCA_comp[i, :]
        eigenvalue = PCA_eivalue[i]
        ratio = PCA_ratio[i]
        panel = plt.subplot(n_panel_height,n_panel_width,i+1)
        panel.plot(np.array(range(0,ci.shape[0])),ci)
        plt.title(str(i)+'\nEigenvalue ='+str(eigenvalue)+'\nRatio ='+str(ratio))
        plt.rcParams.update({'font.size':8})    


#Dimension reduction
"""pca_data and pca are output of "PCA_time" function
#index_range imput in the form: "list(range(10,19))+list(range(0,2))"
#n_PCA_components for reverse PCA in dimension_reduction should be the same as initial PCA
#Is done after "PCA_time", the pca input here is the pca output of "PCA_time"
"""
def PCA_dimension_reduction(pca_data, pca, index_range):
    transformed = pca_data.copy()
    transformed[:,index_range]=0
    data1 = pca.inverse_transform(transformed)
    return data1

#No dimension reduction
def PCA_no_dimension_reduction(pca_data, pca):
    transformed = pca_data.copy()
    data1 = pca.inverse_transform(transformed)
    return data1

#ICA
def ICA_time(PCA_data,  n_ICA_components):#a long lasting step
    data2= PCA_data.astype(np.int8)[:,:]
    ica =FastICA(n_components= n_ICA_components, max_iter=10000).fit(data2)
    return ica

def ICA_time_plot(ICA_all, n_ICA_components, n_panel_width, n_panel_height):
    plt.figure(figsize=(24,15))#for time series
    plt.subplots_adjust(hspace=0.3)
    for i in range(0,n_ICA_components):
        ci = ICA_all.components_[i, :]
        panel = plt.subplot(n_panel_height,n_panel_width,i+1)
        panel.plot(np.array(range(1,ci.shape[0]+1)),ci)
        plt.rcParams.update({'font.size':8})

#ICA_space is based on component number
def ICA_space(PCA_data, n_ICA_components):
    data2= PCA_data.astype(np.int8)[:,:]
    #data2_t = np.transpose(PCA_data, (1,0))#(1,0)
    ica =FastICA(n_components= n_ICA_components).fit(data2)  
    return ica

#ICA_space2 is based on iteration number, not on component number
def ICA_space2(PCA_data):
    data2_t = np.transpose(PCA_data, (1,0))#(1,0)
    ica =FastICA(max_iter=500).fit(data2_t)  
    return ica

def ICA_space_plot(ICA_all, n_ICA_components, n_panel_width, n_panel_height):
    plt.figure(figsize=(24,15))#for time series
    plt.subplots_adjust(hspace=0.3)
    for i in range(0,n_ICA_components):
        ci = ICA_all.components_[i, :]
        panel = plt.subplot(n_panel_height,n_panel_width,i+1)
        panel.imshow(np.reshape(ci,shape=(h,w)))
        plt.rcParams.update({'font.size':8})
        plt.title(str(i))
        
def ICA_space_plot_binary(ICA_all, n_ICA_components, threshold, n_panel_width, n_panel_height):
    plt.figure(figsize=(24,15))#for time series
    plt.subplots_adjust(hspace=0.3)
    for i in range(0,n_ICA_components):
        ci = ICA_all.components_[i, :].copy()
        ci[ci>threshold]=1
        ci[ci<=threshold]=0
        panel = plt.subplot(n_panel_height,n_panel_width,i+1)
        panel.imshow(np.reshape(ci,shape=(h,w)))
        plt.rcParams.update({'font.size':8})
        plt.title(str(i))


#Save ICA components
def ICA_save(name, ica_data):
    with open('ica'+name+'.pckl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(ica_data.components_, f)
        f.close()

#Open ICA components
def ICA_open(i):#i is the index of files
    filenames_ICA = glob.glob('*.pckl')
    with open(filenames_ICA[i],'rb') as g:  # Python 3: open(..., 'rb')
        obj = pickle.load(g)
    return obj, filenames_ICA[i]

"""#Plot to check thresholds
The input threhold is how many times of the standard deviation from mean value"""
def seg_threshold(obj, n_ICA_components, threshold, n_panel_width, n_panel_height):
    plt.figure(figsize=(20,15))#for time series
    plt.subplots_adjust(hspace=0.3)
    for i in range(0,n_ICA_components):
        ci = np.reshape(obj[i, :],(292,384)).copy()
        Filter1_mask=ci.copy()
        Filter1_mask[Filter1_mask>np.std(ci)* threshold]=1
        Filter1_mask[Filter1_mask<=np.std(ci)* threshold]=0
        panel = plt.subplot(n_panel_height, n_panel_width, i+1)
        panel.imshow(Filter1_mask)
        plt.rcParams.update({'font.size':8})

"""The input should be one single ICA or PCA component in space
This compnent's shape is (1,112128), and needs to be changed to (292,384)
Input threhsold is to determine how to change component to binary
"""
"""The "ci" input of "matrix_to_df" is a matrix from PCA or ICA component pixel flatterned
The output is a dataframe with X, Y, value(pixel intensity) in its columns
The Loc (mapped to combinations of X and Y indices) is the index that can be mapped to calcium imaging time series"""

"""Switching between dataframe form and binary image matrix form for Position"""
def matrix_to_df(ci, dim_h=292, dim_w=384):#The value of pixel is preserved
    ci1 = np.reshape(ci,(dim_h,dim_w)).copy()
    Filter1=pd.DataFrame(ci1)
    Filter1['Y'] = range(0,len(Filter1))
    filmelt1 = pd.melt(Filter1, id_vars="Y",var_name='X',value_vars=Filter1.columns)
    filmelt1["Loc"] = (filmelt1["Y"]*dim_w)+filmelt1["X"]#"Loc" is the absolute location in raw image not thresholded image
    return filmelt1

def df_to_matrix (Position,  height= 292 ,width= 384): #the df's XY colums are used to construct a mask.
    matrix= np.zeros(shape=(height,width), dtype=np.int8)
    for i in range(Position.shape[0]):
        matrix[Position.iloc[i].Y,Position.iloc[i].X]=1
    return matrix

def DBSCAN_cluter(filmelt1,eps, min_persample, to_binary_threshold, dim_h=292, dim_w=384):
    Filter1_mask = filmelt1.loc[ filmelt1.value>to_binary_threshold,['X','Y','Loc','value'] ]
    cluster1 = DBSCAN(eps=eps,min_samples=min_persample).fit(Filter1_mask.loc[:,['X','Y']])
    #neuron defined by the "cluster1.labels_" (keeping all pixels)
    Filter1_mask['Cluster'] = cluster1.labels_
    df_mask2 = Filter1_mask.copy()[Filter1_mask['Cluster']!=-1]
    df_mask2 = df_mask2.reset_index(drop=True)  
    #neuron defined by the "cluster1.core_sample_indices_" (keeping only cores)
    core= Filter1_mask.iloc[cluster1.core_sample_indices_].reset_index(drop=True)
    index= ((core["Y"]*dim_w)+core["X"])
    df_mask_core =df_mask2[df_mask2["Loc"].isin(index)].reset_index(drop=True)
    return df_mask2, df_mask_core

"""Here input "ICA_component" can also be "PCA_component" but it has to be in space domain
"merge_sizefilted_clusters" function merge segmentations after size filter.
    This will avoid big wrong ROI that masks real neuron regions
    "merge_sizefilted_clusters" input is the ICA/PCA components selected, threshold to creat the mask of segmentation
    dim_h and dim_w is the pixel size of calcium imaging images
    eps (distance in pixel threshold to define density peak) and min_persample is the clutering parameters
    The min_size_threshold and max_size_threshold are used to filter out ROIs too small or large

    The output df_Pos is the dataframe, each row is a segmented pixel.
    We term this a mask for segmentation

"neuron_plot_cluster_df" is to visualize the mask with df_Pos as imput

df_Pos will be used as the mask to segment calcium imaging performed in "apply_mask_for_seg"
    Another input is the raw calcium imaging time series"""
def get_cluster_from_a_comp(ICA_components, index_ICA, threshold, dim_h, dim_w, eps, min_persample):
    ci1 = ICA_components[index_ICA, :]
    filmelt1 = matrix_to_df(ci1, dim_h, dim_w)
    df_mask2, df_mask_core= DBSCAN_cluter(filmelt1, eps, min_persample, threshold, dim_h, dim_w)
    return df_mask2#Here choose core or not core for final output

def size_filter(df_mask_add, min_size_threshold, max_size_threshold):
    df_mask_add["Size"]=df_mask_add.groupby("Cluster")["Cluster"].transform('count')
    df_mask_add2= df_mask_add[df_mask_add.Size > min_size_threshold]
    df_mask_add3= df_mask_add2[df_mask_add2.Size < max_size_threshold]
    return df_mask_add3

"""Optimization of clustering parameters"""
#This funcion is used to optimize eps and min_persample
#First get the dataframe for DBSCAN analysis
def get_mask(ICA_components, threshold, dim_h, dim_w,  min_size_threshold, max_size_threshold):
    i = 0
    matrix= np.zeros(shape=(dim_h,dim_w), dtype=np.int8)
    while i <= (ICA_components.shape[0]-1):#componentwise clustering and size-filtering
        #Transform to df, Cluster and size filter
        filmelt1 = matrix_to_df(ICA_components[i], dim_h, dim_w)
        df_mask_add= get_cluster_from_a_comp(ICA_components, i, threshold, dim_h, dim_w, eps, min_persample)  
        df_mask_add_filt=  size_filter(df_mask_add, min_size_threshold, max_size_threshold) 
        #Transform back to matrix, topup to the empty matrix,
        filter_matrix= df_to_matrix (df_mask_add_filt,  dim_h, dim_w)
        matrix= matrix+ filter_matrix#This interation is like a z projection
        i+= 1
    df0 = matrix_to_df(matrix, dim_h, dim_w) 
    mask = df0.loc[ df0.value>threshold,['X','Y','Loc','value'] ]
    return mask

def plot_DBSCAN_parameters_core(mask, eps_min, eps_n, min_min, min_n):#Tricky for the core reduce the soma size like a shrinking function
    plt.figure(figsize=(eps_n*10,min_n*10))#for space series
    #plt.subplots_adjust(hspace=0.12)
    for i in range(eps_min, eps_min+eps_n):
        for j in range (int(min_min/10), int(min_min/10)+min_n):
            panel = plt.subplot(min_n, eps_n, (min_n)*(i-eps_min)+(j-int(min_min/10))+1)
            cluster1 = DBSCAN(eps=i,min_samples=j*10).fit(mask.loc[:,['X','Y']])
            #neuron defined by the "cluster1.labels_" (keeping all pixels)
            mask['Cluster'] = cluster1.labels_
            #neuron defined by the "cluster1.core_sample_indices_" (keeping only cores)
            core= mask.iloc[cluster1.core_sample_indices_].reset_index(drop=True)
            index= ((core["Y"]*dim_w)+core["X"])
            Position =mask[mask["Loc"].isin(index)].reset_index(drop=True)
            Position= Position.rename(columns= {'Cluster':'Neuron_indice'})
            
            plot=sns.scatterplot( x="X", y="Y", data= Position,
                           hue='Neuron_indice',  palette='dark', legend=False, s=5)
            plot.set(xlim= (0,dim_w), ylim= (0,dim_h))
            plot.invert_yaxis()
            plt.title('eps ='+str(i)+', Min_size'+str(j*10))
            #i's effect on panel should be amplituted by min_n
def plot_DBSCAN_parameters(mask, eps_min, eps_n, min_min, min_n):#Not using the core keeps ROI of neurons without shrinkage
    plt.figure(figsize=(eps_n*10,min_n*10))#for space series
    #plt.subplots_adjust(hspace=0.12)
    for i in range(eps_min, eps_min+eps_n):
        for j in range (int(min_min/10), int(min_min/10)+min_n):
            panel = plt.subplot(min_n, eps_n, (min_n)*(i-eps_min)+(j-int(min_min/10))+1)
            cluster1 = DBSCAN(eps=i,min_samples=j*10).fit(mask.loc[:,['X','Y']])
            #neuron defined by the "cluster1.labels_" (keeping all pixels)
            mask['Cluster'] = cluster1.labels_
            df_mask2 = mask.copy()[mask['Cluster']!=-1]
            df_mask2 = df_mask2.reset_index(drop=True) 
            Position= df_mask2.rename(columns= {'Cluster':'Neuron_indice'})
            
            plot=sns.scatterplot( x="X", y="Y", data= Position,
                           hue='Neuron_indice',  palette='dark', legend=False, s=5)
            plot.set(xlim= (0,dim_w), ylim= (0,dim_h))
            plot.invert_yaxis()
            plt.title('eps ='+str(i)+', Min_size'+str(j*10))
            #i's effect on panel should be amplituted by min_n           


"""Implementation of the separation among neurons"""
#This function can be used to subdevide pixels in each neuron into different responsive profiles 
def merge_sizefiltered_clusters(ICA_components, threshold, dim_h, dim_w, eps, min_persample, min_size_threshold, max_size_threshold):
    i = 0
    df_Pos=pd.DataFrame()
    Cluster0=0
    while i <= (ICA_components.shape[0]-1):#componentwise clustering and size-filtering
        df_mask_add= get_cluster_from_a_comp(ICA_components, i, threshold, dim_h, dim_w, eps, min_persample)  
        #size filter on an individual component
        #size filter applied first in case big error mask excludes true neuron ROI
        #Because of the filter, the cluster index may not be consequtive
        df_mask_add_filt=  size_filter(df_mask_add, min_size_threshold, max_size_threshold) 
        if len(df_mask_add_filt.Cluster)>0:#If this component has a ROI
            Cluster0= max(df_mask_add_filt.Cluster) + Cluster0 +1 #to avoid overlapse of cluster index           
        df_Pos= pd.concat([df_Pos,df_mask_add_filt])
        i+= 1
    df_Pos= df_Pos.sort_values('Size')
    df_Pos=df_Pos.drop_duplicates("Loc",keep='last')#only larger ROI is kept, when the same neuron appears in both components
        
    Position= df_Pos.rename(columns= {'Cluster':'Neuron_indice'})
    return Position

#This function will not subdevide pixels in the same neuron
def merge_sizefiltered_neurons(ICA_components, threshold, dim_h, dim_w, eps, min_persample, min_size_threshold, max_size_threshold):
    i = 0
    matrix= np.zeros(shape=(dim_h,dim_w), dtype=np.int8)
    while i <= (ICA_components.shape[0]-1):#componentwise clustering and size-filtering
        #Transform to df, Cluster and size filter
        filmelt1 = matrix_to_df(ICA_components[i], dim_h, dim_w)
        df_mask_add= get_cluster_from_a_comp(ICA_components, i, threshold, dim_h, dim_w, eps, min_persample)  
        df_mask_add_filt=  size_filter(df_mask_add, min_size_threshold, max_size_threshold) 
        #Transform back to matrix, topup to the empty matrix,
        filter_matrix= df_to_matrix (df_mask_add_filt,  dim_h, dim_w)
        matrix= matrix+ filter_matrix#This interation is like a z projection
        i+= 1
    df = matrix_to_df(matrix, dim_h, dim_w)
    if np.max(matrix)>0:
        df_cluster,df_core= DBSCAN_cluter(df,eps, min_persample, threshold, dim_h, dim_w)
        Position= df_cluster.rename(columns= {'Cluster':'Neuron_indice'})#change this if core is not used
    else: 
        Position=pd.DataFrame() 
    return Position
    
     

    
    


   
def neuron_plot_df(Position):
     plt.figure(figsize=(24,15))
     plot=sns.scatterplot( x="X", y="Y", data= Position,
                      legend=False, s=5)
     plot.set(xlim= (0,383), ylim= (0,291))
     plot.invert_yaxis()    

def neuron_plot_cluster_df(Position):
     plt.figure(figsize=(24,15))
     plot=sns.scatterplot( x="X", y="Y", data= Position,
                    hue='Neuron_indice',  palette='dark', legend=False, s=5)
     plot.set(xlim= (0,383), ylim= (0,291))
     plot.invert_yaxis()

def apply_mask_for_seg(Calciumraw, Position):
    Neuron =[]
    for i in pd.unique(Position.Neuron_indice):
        ID = Position[Position.Neuron_indice==i].Loc
        Neuron_add = Calciumraw[:,ID.astype(np.int32)]
        Neuron.append(Neuron_add)          
    return Neuron     
        
"""Save time serie list"""
#Save ICA components
def Neuron_all_save(name, Neuron_all):
    with open('Array_'+name+'.pckl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(Neuron_all, f)
        f.close()







