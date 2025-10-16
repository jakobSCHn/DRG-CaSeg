# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 13:20:27 2025

@author: junma7451
"""

name= get_filename(tail='*.czi')
Cal = czifile.imread(name[0])
imgseg2 = np.squeeze(Cal)#remove not useful dimensions
imgseg3 = np.reshape(imgseg2,(imgseg2.shape[0],imgseg2.shape[1]*imgseg2.shape[2]))

plot(imgseg2[17136,:])

h=imgseg2.shape[1]
w=imgseg2.shape[2]
Seg_plot_Brithreshold(imgseg3, 10000,  3, 20, 5, 4)

Seg_plot_KClthreshold(imgseg3, 7000, 11000, 11, 20, 5, 4)

image_seg= KCl_segmentaion(imgseg3, 7000, 11000, 11,0)

PCs, var, ratio, pca, data_PCA=PCA_time(image_seg, 7000, 11000, 20)
PCA_plot_time(PCs, var, ratio, n_panel_width=5, n_panel_height=4)

PCs, var, ratio, pca, data_PCA=PCA_space(image_seg, 7000, 11000, 20)
PCA_plot_space(PCs, var, ratio, n_panel_width=5, n_panel_height=4)
