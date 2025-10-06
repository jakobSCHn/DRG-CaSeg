

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 08:50:02 2025

@author: Junxuan Ma
"""

import numpy as np
from scipy.signal import butter, lfilter, freqz,sosfreqz,sosfilt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, deconvolve
from scipy import stats
from bisect import bisect
import pandas as pd
import math

"""0. Take a subset of pixels for analysis"""
def matrix_to_df(ci, dim_h=292, dim_w=384):
    ci1 = np.reshape(ci,(dim_h,dim_w)).copy()
    Filter1=pd.DataFrame(ci1)
    Filter1['Y'] = range(0,len(Filter1))
    filmelt1 = pd.melt(Filter1, id_vars="Y",var_name='X',value_vars=Filter1.columns)
    filmelt1["Loc"] = (filmelt1["Y"]*dim_w)+filmelt1["X"]#"Loc" is the absolute location in raw image not thresholded image
    return filmelt1

def df_to_matrix (Position,  height= 292 ,width= 384):
    matrix= np.zeros(shape=(height,width), dtype=np.int8)
    for i in range(Position.shape[0]):
        matrix[int(Position.iloc[i].Y), int(Position.iloc[i].X)]=1
    return matrix

"""Select boarder/ inner pixels for averaging
#Position is the dataframe of XY infor of pixels. Times determines the thickness of boarder
#the output is a matrix 
#The function shows that calcium list element is in the same order of "Neuron_indice" order
"""
def boarder_aver(Position, Neuron_all, times):
    j=0
    Pos_out=pd.DataFrame()
    Pos_in=pd.DataFrame()
    Neuron_indice=pd.DataFrame()
    Out_cal=[]
    In_cal=[]
    for cell in pd.unique(Position['Neuron_indice']):
        #Choose the cell
        Pos_ind=Position[Position['Neuron_indice']==cell]
        #Find outlines. Fist layer
        Pos_matrix= df_to_matrix(Pos_ind, h, w)
        out2= np.zeros((h, w), dtype=bool)#Empty bool matrix to fill in
        
        for i in range(0,times):
            out1=skimage.segmentation.find_boundaries(Pos_matrix, connectivity=1, mode='inner', background=0)
            #second layer
            Pos_matrix[out1]=0
            out2[out1]=True#out2 is the boarder region of a single neuron (index as cell)
        out_df=matrix_to_df(out2)
        out_df_pos= out_df[out_df['value']==True].Loc
        Pos_bool_out=Pos_ind['Loc'].isin(out_df_pos)
        
        Pos_out= pd.concat([Pos_out, Pos_ind[Pos_ind['Loc'].isin(out_df_pos)] ])
        Pos_in= pd.concat([Pos_in, Pos_ind[-Pos_ind['Loc'].isin(out_df_pos)] ])
        Neuron_indice= pd.concat([Neuron_indice, pd.Series([cell])])
        Out_cal.append(Neuron_all[j][:,Pos_bool_out])
        In_cal.append(Neuron_all[j][:,-Pos_bool_out])
        j+=1
    return Pos_out,Out_cal,Pos_in,In_cal,Neuron_indice



'''1.Functions to plot curves'''
def plot_curve(x):
    time = np.array(range(0,x.shape[0]))
    plt.figure(figsize=(30,5))
    plt.plot(time, x, 'r-', linewidth=0.5)

#in this array, each row is a curve to be plotted
#figure size in the form of tuple (30,5)
def plot_curves(array):
    plt.figure(figsize=(30,5))
    plt.plot(np.transpose(array))
    
'''2. normalization method, either by rolling window or by baseline normalization'''

#Absolute defined baseline for normalization， It is influecd by photobleaching
def normalization(x,def_base_start,def_base_end):
    nor= (x-np.mean(x[def_base_start:def_base_end]))/np.mean(x[def_base_start:def_base_end])
    return nor
def mean_smooth(x, roling_size = 300):
    roling = np.ones(roling_size) / roling_size
    data_convolved = np.convolve(x, roling, mode='same')
    return data_convolved


#Each row is a time series, number of neurons is the 0th dimension size
#function is what's done in rows, kwargs are other parameters to be put in fuction
#kwargs in the form of: kwargs = {'def_base_start': 0, 'def_base_end': 1000}
def do_in_rowwise(array, function, kwargs):
    array_done= np.apply_along_axis(function, 1, arr= array, **kwargs)
    return array_done

def nor_rowwise(array, def_base_start, def_base_end):
    kwargs = {'def_base_start': def_base_start, 'def_base_end': def_base_end}
    array_nor= do_in_rowwise(array, normalization, kwargs)
    return array_nor


"""3. LOW PASS BUTTER FILTER
"""
#fs: sample frequency
def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', output= 'sos', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=4):
    a = butter_lowpass(cutoff, fs, order=order)
    y = sosfilt(a, data)
    return y

def plot_butter(x,cutoff,fs,order,seg_start=0,seg_end=900):
    time=range(seg_start,seg_end)
    sig=x[seg_start:seg_end,]
    y = butter_lowpass_filter(sig, cutoff, fs, order)
    plt.figure(figsize=(30,5))
    plt.plot(time, sig, 'b-', label='data')
    plt.plot(time, y, 'r-', linewidth=1, label='filtered data')
    plt.xlabel('Time frame [n.]')
    plt.grid()
    plt.legend()
    #plt.ylim(-0.003, 0.005)

def plot_butter_opt(x,fs,seg_start,seg_end,cut_low,cut_high,cut_step,max_order):
    h=range(1,max_order+1)
    w=range(cut_low,cut_high+cut_step,cut_step)
    sig=x[seg_start:seg_end,]
    time=range(seg_start,seg_end)
    fig,axs=plt.subplots(int(len(w)),int(len(h)),figsize=(len(h)*10,len(w)*5))
    for i in range(0,len(w)):#i as w as cutoff
        for j in range(0,len(h)):#j as h as order
            y = butter_lowpass_filter(sig, w[i], fs, h[j])
            panel=plt.subplot(int(len(w)),int(len(h)), j+i*len(h)+1)
            panel.plot(plotsize=(30,5))
            panel.plot(time, sig, 'b-', label='data')
            panel.plot(time, y, 'r-', linewidth=2, label='filtered data')
            plt.title('order'+ str(h[j])+'_cutoff'+ str(w[i]) )

def plot_fre_response(order = 4,fs=30,cutoff=4):
    # Filter requirements.
    #fs = 30      # sample rate, Hz
    """The filter works with the normalized frequency cutoff which is the real frequency divided by sample frequency.
    This makes sure that the cutoff frequency makes sense
    https://electronics.stackexchange.com/questions/534198/how-is-the-filter-response-affected-by-the-sampling-rate
    """
    #cutoff = 8 # desired cutoff frequency of the filter, Hz
    #The cutoff cannot be lower than fs*0.5 which is called the Nyquist frequency
    # Get the filter coefficients so we can check its frequency response.
    sos = butter_lowpass(cutoff, fs, order)
    # from coefficients to frequency response.
    w, h = sosfreqz(sos, fs=fs)
    """
    The function freqz use parameters generated from the filter design to generate a frequency response
    w is the frequencies, h is the frequency response as complex numbers, 
    that's why to plot it, np.sqrt is needed.
    When the designed filter's output is sos use sosfreqz which compute the frequency response of a digital filter in SOS format.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfreqz.html#scipy.signal.sosfreqz
    """
    plt.subplot(2, 1, 1)
    plt.plot(w, np.abs(h), 'b')
    plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5*fs)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()

#x is unfiltered or non-deconvolved curve that can be compared with the fitlered and deconfolved curve for peak finding
def plot_find_peak_compare_curves(x, filt_x, threshold, seg_start,seg_end):#input as a single row of normalized time series
    time=range(seg_start,seg_end)
    y = filt_x[seg_start:seg_end,]
    #Here defines the threshold
    peaks,heights = find_peaks(y,height=threshold)
    plt.figure(figsize=(30,5))
    plt.plot(time, x[seg_start:seg_end,], 'b-', label='data')
    plt.plot(time, y, 'r-', linewidth=2, label='filtered data')
    plt.plot(peaks+seg_start, y[peaks],"o",linewidth=3,color = "purple")
    plt.xlabel('Time [frames]')
    plt.grid()
    
def plot_find_peak(y, threshold):#input as a single row of normalized time series
    time=range(0,len(y))
    #Here defines the threshold
    peaks,heights = find_peaks(y,height=threshold)
    plt.figure(figsize=(30,5))
    plt.plot(time, y, 'r-', linewidth=1.5, label='filtered data')
    plt.plot(peaks, y[peaks],"o",linewidth=3,color = "purple")
    plt.xlabel('Time [frames]')
    plt.grid()

def butter_array(Nor_array, cutoff, fs, order):
    kwargs = { 'cutoff': cutoff, 'fs': fs, 'order': order }
    filt_array = do_in_rowwise(Nor_array, butter_lowpass_filter, kwargs)
    return filt_array


"""4. Correction of photobleaching using linear fitting of baseline level"""
def regression_fun(x,slope,intercept):
    return slope * x + intercept

def subtract_regresssion(filt_x,base_start,base_end):
    time= range(0,filt_x.shape[0])
    base_time= range(base_start,base_end)
    base_y = filt_x[base_start:base_end,].copy()
    slope, intercept, r, p, std_err = stats.linregress(base_time, base_y)
    background= regression_fun(time,slope,intercept)
    return  background, filt_x-background

def add_regresssion(filt_x,base_start,base_end):
    time= range(0,filt_x.shape[0])
    base_time= range(base_start,base_end)
    base_y = filt_x[base_start:base_end,].copy()
    slope, intercept, r, p, std_err = stats.linregress(base_time, base_y)
    background= regression_fun(time,slope,intercept)
    return  filt_x-background+filt_x[0]
    
def plot_linear_correction(filt_x,base_start,base_end,show_start,show_end):
    time= range(show_start,show_end)
    base_y = filt_x[base_start:base_end,].copy()
    bg, y_=subtract_regresssion(filt_x, base_start,base_end)
    
    panel=plt.subplot(2,1, 1)
    panel.plot(plotsize=(30,5))
    panel.plot(time, filt_x[show_start:show_end,], 'r-', linewidth=0.5)
    panel.plot(time, bg[show_start:show_end,], 'b-', linewidth=0.5)
    panel.axhline(y=0,color='g')
    
    panel=plt.subplot(2,1, 2)
    panel.plot(plotsize=(30,5))
    panel.plot(time, y_[show_start:show_end,], 'r-', linewidth=0.5)
    panel.axhline(y=0,color='g')

def subtract_regre_array    (filt_array,base_start,base_end):
    kwargs2 = { 'base_start':base_start, 'base_end': base_end}
    filt_cor_array = do_in_rowwise(filt_array, subtract_regresssion, kwargs2)       
    return filt_cor_array[:,1,:]#as a row of number, each number is a neuron    
        
def add_regre_array    (filt_array,base_start,base_end):
    kwargs2 = { 'base_start':base_start, 'base_end': base_end}
    filt_cor_array = do_in_rowwise(filt_array, add_regresssion, kwargs2)       
    return filt_cor_array#as a row of number, each number is a neuron    


"""5. Spike averaged morphology to decide kernel of deconvolution""" 
#Use probability to remove noise and neuronwise delineat the shape of spike    
def around_spike_seg_row (filt_cor_x, threshold, seg_start, seg_end, seg_width):
    y = filt_cor_x[seg_start:seg_end,]
    peaks,heights = find_peaks( y[int(seg_width/2):(y.shape[0]+1-int(seg_width/2))], height= threshold )
    left,right= peaks, peaks+int(seg_width)
    seg = np.array( [filt_cor_x[left[i]:right[i]] for i in range(0,len(left))] )
    mean_seg = np.apply_along_axis(np.mean, 0, arr= seg)
    return mean_seg#as a number

def around_spike_seg_array(filt_cor_array, threshold, seg_start, seg_end, seg_width):
    kwargs2 = { 'threshold': threshold, 'seg_start':seg_start, 'seg_end': seg_end, 'seg_width': seg_width}
    mean_seg_array=do_in_rowwise(filt_cor_array, around_spike_seg_row , kwargs2)
    return mean_seg_array


'''6. Deconvolution for the purpuse of neuronal response to certain stimuli'''
#The rationale is that by check the maximum value of swaying of calcium curves, this is confounded by an internal release of calcium
#The solution is to calculate the spike frequency ratio compareing baseline and stimulated pieces
#Deconvolution can translate the infromation of swaying up into an increased spiking frequency.
#The key is to set the correct tao value for the kernel to restore the impulse response back to baseline

#From the shape of curves, one can define the kernel for deconvolution
#Kernel construction, output as a list
def get_exp_kernel(length,tao):
    time=list(range(0,length) )
    kernel=[]
    for val in time:
        add = math.exp(-(val/tao))
        kernel.append(add)
    return time,kernel

def plot_kernels(len_low,len_test_n,len_step,tao_low,tao_test_n,tao_step):
    w= range(0,len_test_n)
    h= range(0,tao_test_n)
    fig,axs=plt.subplots(int(len(w)),int(len(h)),figsize=(len(h)*10,len(w)*5))
    for i in range(0,len(w)):#i as w as cutoff
        for j in range(0,len(h)):#j as h as order
            kernel_length= len_low + i*len_step
            kernel_tao= tao_low + j*tao_step
            time,kernel = get_exp_kernel(kernel_length,kernel_tao) 
            panel=plt.subplot(int(len(w)),int(len(h)), j+i*len(h)+1)
            panel.plot(plotsize=(10,5))
            panel.plot(time,kernel)
            plt.title('tao:'+ str(kernel_tao)+'_length:'+ str(kernel_length) )

def deconvolve_curve(filt_cor_x, kernel_length, kernel_tao):
    ker_time,kernel= get_exp_kernel (kernel_length,kernel_tao)
    deconv,remainder= deconvolve(filt_cor_x, kernel)
    time=range(0,len(deconv))
    return time,deconv,remainder

def deconvolve_curve_essential(filt_cor_x, kernel_length, kernel_tao):
    ker_time,kernel= get_exp_kernel (kernel_length,kernel_tao)
    deconv,remainder= deconvolve(filt_cor_x, kernel)
    time=range(0,len(deconv))
    return deconv
  
def plot_kernel_setting(filt_cor_x, len_low,len_test_n,len_step,tao_low,tao_test_n,tao_step):
    w= range(0,len_test_n)
    h= range(0,tao_test_n)
    fig,axs=plt.subplots(int(len(w)),int(len(h)),figsize=(len(h)*10,len(w)*5))
    for i in range(0,len(w)):#i as w as cutoff
        for j in range(0,len(h)):#j as h as order
            kernel_length= len_low + i*len_step
            kernel_tao= tao_low + j*tao_step
            time,deconv,remainder = deconvolve_curve(filt_cor_x, kernel_length, kernel_tao)
            panel=plt.subplot(int(len(w)),int(len(h)), j+i*len(h)+1)
            panel.plot(plotsize=(10,5))
            panel.plot(time, filt_cor_x[0:len(deconv)], 'b-', label='data')
            panel.plot(time, deconv, 'r-', linewidth=1, label='deconvolved data')
            panel.plot(time, remainder[0:len(deconv)], 'g-', linewidth=1, label='remainder data')
            plt.title('tao:'+ str(kernel_tao)+'_length:'+ str(kernel_length) )

def deconvolve_curves(filt_cor_array, kernel_length, kernel_tao):
    kwargs = { 'kernel_length': kernel_length, 'kernel_tao':kernel_tao}
    decon_array=do_in_rowwise(filt_cor_array, deconvolve_curve_essential , kwargs)
    return decon_array

"""  Testing for the deconvolution functions:
time,kernel=get_exp_kernel(10,20)
plt.plot(time,kernel)
time,deconv,remainder= deconvolve_curve(filt_cor_x, 10, 20)
plot_kernels(5,5,2,1,6,2)
plot_kernel_setting(filt_cor_x[2700:3200], 5,5,2,1,6,2) 
#tao=21, length=5 makes more sense
decon_x= deconvolve_curve(filt_cor_x, 5, 21)
decon_array= deconvolve_curves(filt_cor_array, 5, 21)
plot_curves(decon_array[1:,],figsize=(30,5))
"""

    
"""7. Smooth_Find pieces
peak for the input is one element of:  peaks,heights = find_peaks(y,height=threshold)
minimum is the array of: minimum,lows = find_peaks(-y)"""
def boarder(peak, minimum):
    i =bisect(minimum,peak)
    left = minimum[i-1]
    right = minimum[i]
    return left,right

def boarders(peaks, minimum):
    pieces=[boarder(peak, minimum) for peak in peaks]
    return pieces

def get_peak_seg(filt_x, threshold, seg_start, seg_end):#input as a single row of filtered time series
    y = filt_x[seg_start:seg_end,]
    peaks,heights = find_peaks(y,height=threshold)
    minimum,lows = find_peaks(-y)
    pieces= boarders(peaks, minimum)
    return pieces

def plot_peak_seg(filt_x, threshold, seg_start, seg_end):#input as a single row of filtered time series
    time=range(seg_start,seg_end)
    y = filt_x[seg_start:seg_end,]
    peaks,heights = find_peaks(y,height=threshold)
    pieces= get_peak_seg(filt_x, threshold, seg_start, seg_end)
    plt.figure(figsize=(30,5))
    plt.plot(time, y, 'r-', linewidth=1)
    plt.plot(peaks+seg_start, y[peaks],"x",color = "purple")
    for i in range(0,len(pieces)):
        plt.plot(pieces[i][0]+seg_start, y[pieces[i][0]],"x",color = "g")
        plt.plot(pieces[i][1]+seg_start, y[pieces[i][1]],"x",color = "g")
        plt.xlabel('Time [frames]')
        plt.grid()
        plt.legend()
        
"""Smoothing of peak intervals, input as a single row of time series"""
def smooth_peak_interval(filt_x, threshold, seg_start, seg_end, times):
    y = filt_x[seg_start:seg_end,].copy()
    pieces= get_peak_seg(filt_x, threshold, seg_start, seg_end)  
    for i in range(0,len(pieces)-1):
        interval = range(pieces[i][1],pieces[i+1][0])
        if len(interval)>3:
            for j in range(0,times):
                y[interval,] = mean_smooth(y[interval,], roling_size = 3)
                y[0:pieces[0][0],] = mean_smooth(y[0:pieces[0][0],], roling_size = 3)
                y[pieces[len(pieces)-1][1]:,] = mean_smooth(y[pieces[len(pieces)-1][1]:,], roling_size = 3)
    return y

def plot_smooth_peak_seg(filt_x, threshold, seg_start, seg_end, times):
    time=range(seg_start,seg_end)
    y = smooth_peak_interval(filt_x, threshold, seg_start, seg_end, times)
    peaks,heights = find_peaks(filt_x[seg_start:seg_end,],height=threshold)
    pieces= get_peak_seg(filt_x, threshold, seg_start, seg_end)
    plt.figure(figsize=(30,5))
    plt.plot(time, y, 'r-', linewidth=1)
    plt.plot(peaks+seg_start, y[peaks],"x",color = "purple")
    for i in range(0,len(pieces)):
        plt.plot(pieces[i][0]+seg_start, y[pieces[i][0]],"x",color = "g")
        plt.plot(pieces[i][1]+seg_start, y[pieces[i][1]],"x",color = "g")
        plt.xlabel('Time [frames]')
        plt.grid()
        plt.legend()


"""7. Time series spikewise quantification"""
#Default:cutoff=8, fs=30, order=2
def get_peak_Morp(filt_x, threshold, seg_start, seg_end):#input as a single row of filtered time series
    y = filt_x[seg_start:seg_end,]
    peaks,heights = find_peaks(y,height=threshold)
    Morp=pd.DataFrame(heights)
    Morp['Up_width'] = pd.Series(dtype='int')
    Morp['Down_width'] = pd.Series(dtype='int')
    Morp['Width'] = pd.Series(dtype='int')
    Morp['Peak_loc'] = pd.Series(dtype='int')
    Morp['Start'] = seg_start
    Morp['End'] = seg_end
    minimum,lows = find_peaks(-y)
    for i in range(0,len(peaks)):
        left,right=boarder(peaks[i], minimum)
        Morp.iloc[i,1] = peaks[i]-left
        Morp.iloc[i,2] = right-peaks[i]
        Morp.iloc[i,3] = right-left
        Morp.iloc[i,4] = peaks[i]+seg_start
    return Morp

def Morp_calculator(Nor_array, threshold, seg_start, seg_end, cutoff, fs, order):
    kwargs = { 'cutoff': cutoff, 'fs': fs, 'order': order }
    filt_array = do_in_rowwise(Nor_array, butter_lowpass_filter, kwargs)
    Morp_array =pd.DataFrame()
    for i in range(0,filt_array.shape[0]):
        y = filt_array[i,:]
        peaks,heights = find_peaks(y[seg_start:seg_end,], height= threshold)#peak height threshold
        Morp_add = get_peak_Morp(y, threshold, seg_start, seg_end)
        Morp_add['Neuron_index'] = i
        Morp_array = pd.concat([Morp_array, Morp_add])
        #fre=fre.append()is wrong because append function is "mutable"
        #this is a useful appending updating function in for loop        
    return Morp_array #spike morphorlogy out of an array of different neurons

#Note the input has already filtered
def freq_calculator_row (filt_x, threshold, seg_start, seg_end):
    y = filt_x[seg_start:seg_end,]
    peaks,heights = find_peaks(y, height= threshold)
    freq = len(peaks)/(y.shape[0] * 28.57)
    return freq#as a number

#Note that the input is not yet filtered, the function applies butter fitler
def freq_calculator_array(filt_array, threshold, seg_start, seg_end):
    kwargs2 = {'threshold': threshold, 'seg_start':seg_start, 'seg_end': seg_end}
    freqs = do_in_rowwise(filt_array, freq_calculator_row, kwargs2)       
    return freqs#as a row of number, each number is a neuron    

#Calculate responsive peak height    
def peak_height(x,def_base_start, def_base_end, def_sti_start, def_sti_end):
    PH = np.max(x[def_sti_start:def_sti_end])- np.mean(x[def_base_start:def_base_end])
    return PH       
    
"""
8 Fequency dynamic characterization
"""
#The "freq_dyanimc" function allows the reponse should be evaluated dynamically
#The input should be normalized, filtered and corrected time series
def freq_dynamic (decon_array, time_start, time_step, n_sample):
    dyn= np.empty(decon_array.shape[0])
    time=[]
    for i in range(0,n_sample):
        sti_freqs= freq_calculator_array (decon_array, threshold, time_start+i*time_step, time_start+(i+1)*time_step)
        dyn=np.vstack((dyn,sti_freqs))
        time.append((time_start+i*time_step+time_start+(i+1)*time_step)/2)
    return dyn[1:,:],time

"""The following function defines a time window to perform the freq_dynamic,
"""    
"""If the time series is very long, the photobleach is not simulated as linear
The analysis is broken into smaller pieces with a short base line before stimulation and after
the photobleach correction is only performed locally"""
def freq_dynamic_pieces (np_array, sti_start, piece, time_step, n_sample):#piece is a duration for normalization and correction
    array= np_array[(sti_start-piece): (sti_start+ n_sample*time_step),:]#make small piece to reduce calculation time
    Nor_array= nor_rowwise(array.T, 0, piece)
    filt_array = butter_array(Nor_array, cutoff, fs, order)
    filt_cor_array= subtract_regre_array(filt_array, 0, piece)
    decon_array= deconvolve_curves(filt_cor_array, length, tao)
    dyn= np.empty(np_array.T.shape[0])
    time=[]
    for i in range(0,n_sample):
        sti_freqs= freq_calculator_array (decon_array, threshold, i*time_step, (i+1)*time_step)
        dyn=np.vstack((dyn,sti_freqs))
        time.append((sti_start+i*time_step+sti_start+(i+1)*time_step)/2)
    return dyn[1:,:],time
        



#From the corrected frequency dynamics, you can calculate the averaging and fold-change in chemical responses
#Because of the correction, some negative value may be produced, frequncy difference instead of change should be calculated
def freq_acceleration(dyn_cor, time_step, base_start,base_end, sti_start1, sti_end1, 
                      sti_start2, sti_end2, sti_start3, sti_end3, sti_start4, sti_end4 ):#Notice the time of start or end should be rounded by time_step
    base= dyn_cor[:,int(base_start/time_step):int(base_end/time_step)]
    ave_base = np.apply_along_axis(np.mean, 1, arr= base)
    sti= dyn_cor[:,int(sti_start/time_step):int(sti_end/time_step)]
    ave_sti = np.apply_along_axis(np.mean, 1, arr= sti)
    comb=np.vstack((ave_base, ave_sti))
    freq_df= pd.DataFrame(comb.T, columns=['Base', 'Sti'])
    freq_df['Acceleration']=freq_df['Sti']-freq_df['Base']
    return freq_df



"""9. Explaination and example for useage of the functions

#Normalization of original fluorescent recordings
#Important that the dataformat is time series in rows
obj=Neuron_all[2].T
Nor_array= nor_rowwise(obj, def_base_start=0, def_base_end=2000)



#Plot either multiple rows or a single row, input is np.array
plot_curves(Nor_array[0:5],figsize=(30,5))
plot_curves(obj,figsize=(30,5))


#Butter filter, the frequency response shows how much each frequency is kept while others are filterred out.
    #Such a filter is independent of which time serie is analyzed, so no row array data needs to input.
    #By playing with different cutoff, one can see in the freq response 
    #which frequency is kept (close to 1 in y) and which is lost (close to 0 in y),
    #one can also see that freqs higher than the low pass cutoff are still partly kept
    #that the removal is smooth.
    #One can also see that by setting higher order, the removal will be stiffer,
    #since the slope is more tilted.
plot_fre_response(order = 14,fs=30,cutoff=8)

#If your data is a matrix or np.array, its name is obj here, each row of obj is a time serie   
y=obj[40,:]#This is to select the 3rd row as a single time serie
x= normalization(y,0,5000)
#A better way to decide on butter filter parameter is to try on your time serie and get intuition of all combinations of the parameters
plot_butter_opt(x,fs=30,seg_start=0,seg_end=400,cut_low=2,cut_high=14,cut_step=2,max_order=8)
#Once the parameters are decided, you can plot filter's effect using this parameters specifically
plot_butter(x,8, 30, 2,seg_start=0,seg_end=900)
#This is the function to implement the butter filter using the decided parameter
filt_x= butter_lowpass_filter(x, 8, 30, order=2)
#This is the function to implement the butter filter on multi-neuron array, each row is a time serie of a neuron
filt_array = butter_array(Nor_array, cutoff=8, fs=30, order=2)
   


#The photobleach will shift the normalized fluorescent level and is linear overtime
#Here is to check the effect of linear photobleach correction 
plot_linear_correction(filt_x,base_start= 0,base_end=2000,show_start=0,show_end=2000)    
background,filt_cor_x = subtract_regresssion(filt_x,base_start=0,base_end=2000)
#The following function correct photobleach in multi-neuron array, each row is a normalized and butter-filtered time series
filt_cor_array= subtract_regre_array(filt_array,base_start=0,base_end=2000)



plot_curve(filt_cor_x)
plot_curves(filt_cor_array,(30,5))
plot_curves(filt_array,(30,5))


#Optional: visualization of the spike shape, may infer the kernel for deconvolution
#visualize the distribution of peak height which is the output of 
peaks,heights = find_peaks(filt_cor_x[0:1000,],height=0.03)
plt.hist(list(heights.values()))
#When putting all the peaks from different curves together, you can visualize the shape of the spike
mean_seg_array= around_spike_seg_array(filt_cor_array,0.03,0,4000,50)
plot_curves(mean_seg_array, (10,10))
#deconvolution algorism
time,kernel=get_exp_kernel(10,20)#Estabilishing the Kernel
plt.plot(time,kernel)
#plots to optimize kernels, first is to visualize kernels, second is to see how the outcome is influenced by kernel setting
plot_kernels(5,5,2,1,6,2)#len_low,len_test_n,len_step,tao_low,tao_test_n,tao_step
plot_kernel_setting(filt_cor_x[2700:3200], 5,5,2,1,6,2) #len_low,len_test_n,len_step,tao_low,tao_test_n,tao_step
#tao=21, length=5 in the kernel give more sense on deconvolution
time,decon_x,remainder= deconvolve_curve(filt_cor_x, 5, 21)#Deconvolution of a single curve
decon_array= deconvolve_curves(filt_cor_array, 5, 21)#Deconvolution of an array of curves



plot_curves(decon_array[1:,],figsize=(30,5))
plot_curves(filt_cor_array[1:,],figsize=(30,5))
#The next step is peak finding, you may try different threshold and visualize what peaks are found
plot_find_peak(x, filt_x, 0.03, 0,1000)#visualize the effect of butter filter
plot_find_peak(filt_x, filt_cor_x, 0.03, 0,1000)#visualize the effect of photobleach correction
plot_find_peak(filt_cor_x, decon_x, 0.06, 0,5000)#visualize the effect of deconvolution

#Smoothing of inter-spike noise for representative images
#The following function shows you the start and ending of the spike that is used to define inter-spike space
plot_peak_seg(decon_x, 0.03, 0, 1000)
#If you want the start and ending of each peaks, this function put them into a list, each element of the list is represent the start and ending of this peak
pieces= get_peak_seg(decon_x, 0.03, 0, 1000)
#Functions to further remove inter-peak noise using an averaging kernel with a width of 3 time frames.
smooth = smooth_peak_interval(decon_x,0.03,0,1000,20)
plot_curve(smooth)
#You can jointly visualzie peaks and boarder of peaks in the smoothed curves for representative images or deconvolution purpose
plot_smooth_peak_seg(decon_x,0.03,0,1000,20)

#This function output a number of frequency with unit of Hz, if the input is a row.
    #Please notice that the time resolution of imaging is 28.57 ms, you need to change this number in the function if another imaging protocol is used.
freq_calculator_row (decon_x, 0.03,3000,4000)
    #This function output the dataframe of the width of rise and fall of each peak, if the input is a single row.
Morp= get_peak_Morp(decon_x,0.03,0,1000)

#You can also calculate frequency and peak morphology when inputting arrays (multiple rows).
    #The output is a single row np.array, each number in this array is the calcium frequency of the corresponding neuron.
freqs= freq_calculator_array (decon_array, 0.03, 0,100)
    #The output is a dataframe, each row is the width infor of one spike, a column is used to label which neuron this spike belongs.
Morp_array = Morp_calculator(obj, 0.001, 11700,14000,  8,30,2)

#Use the change of frequency to study the response to e.g., capsaicin
#frequency dynamics and decay correction
#From the following plot, you can see that frequency is again a linear decay, similar to photobleach:    
base_start,base_end=0,2000
cutoff,fs,order,length,tao,threshold = 8,30,2,5,21,0.03

dyn,time= freq_dynamic (np_array=obj, time_start=0, time_step=100, n_sample=127)
for i in range(0,dyn.shape[1]):
    plt.plot(time,dyn[:,i])

for i in range(0,dyn.shape[1]):
    plt.plot(dyn[:,i])


#A linear model is again used to substract the linear decay of frequency over time
dyn_cor= add_regre_array(dyn.T,0,8)#IMLEMENTATION
for i in range(0,dyn.shape[1]):
    plt.plot(time,dyn_cor[i,:])
 
    
acc_Cap=freq_acceleration(dyn_cor, 100, 0,2800, 2800,3700)
acc_KCl=freq_acceleration(dyn_cor, 100, 0,2800, 3900,4100)

output= sig_data_output(dyn.T,100,0,2800,2800,3700,3900,4100)
"""


