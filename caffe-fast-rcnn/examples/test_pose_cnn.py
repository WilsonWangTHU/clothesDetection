import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import os
if not os.path.isfile(caffe_root + \
    'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print("Unable to load the model of pose")
        

caffe.set_mode_gpu()
caffe.set_device(1)
net = caffe.Net(caffe_root + 'models/pose_cnn/lsp_deploy.prototxt',
                caffe_root + 'models/pose_cnn/lsp_iter_60000.caffemodel',
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.array([128,128,128])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

net.blobs['data'].reshape(1,3,36,36)
input_data = transformer.preprocess('data', caffe.io.load_image(caffe_root + 'examples/images/im0002.jpg'))
net.blobs['data'].data[...] = input_data
out = net.forward()