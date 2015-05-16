# -*- coding: utf-8 -*-
"""
Created on Mon May 11 12:25:25 2015

@author: wtw
"""

new_db_config = open('/home/wtw/clothesDetection/data/DB_info')
for new_db_name in new_db_config:
    print new_db_name
    print type(new_db_name)
    a = new_db_name[:-1]
    
    
    
    len(a)


import scipy.io as sio
import PIL

im = PIL.Image.open('/home/wtw/clothesDetection/'+ \
'data/clothesDataset/train/4/image_50/00016_5494ffb8N2bc25059.jpg')

filename = '/home/wtw/clothesDetection/'+ \
'data/clothesDataset/train/11/boxes.mat'
box_list =[]
raw_data = sio.loadmat(filename)['boxes'].ravel()
for i in xrange(raw_data.shape[0]):
    box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)
    for i in xrange(10):        
        if i < 6 and i > 2:
            print i
