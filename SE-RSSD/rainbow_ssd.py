# coding: utf-8

# * First, Load necessary libs and set up caffe and caffe_root
# In[1]:
# import time
import cv2
import numpy as np
from google.protobuf import text_format
import scipy.io as sio
import sys
import os

caffe_root = '/home/gxkj/caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.append(caffe_root)
os.chdir(caffe_root)
sys.path.insert(0, 'python')
# open gpu before import caffe
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import caffe
from caffe.proto import caffe_pb2

caffe.set_device(0)
caffe.set_mode_gpu()

model_def = '/home/gxkj/deconv_test/deconv.prototxt'
model_weights = '/home/gxkj/deconv_test/deconv.caffemodel'


net_ssd = caffe.Net(model_def,  # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)
def deconv_update(net_ssd,image):
    transformer = caffe.io.Transformer({'data': net_ssd.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    # transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
    transformed_image = transformer.preprocess('data', image)
    net_ssd.blobs['data'].data[...] = transformed_image
    net_ssd.params['deconv'][0].data[:, :, :, :] = np.zeros((3, 256, 2, 2))
    net_ssd.params['deconv'][0].data[0, 0, 0, 0] = 1.0
    net_ssd.params['deconv'][0].data[0, 0, 0, 1] = 2.0
    net_ssd.params['deconv'][0].data[0, 0, 1, 0] = 3.0
    net_ssd.params['deconv'][0].data[0, 0, 1, 1] = 4.0

    temp = np.zeros((1, 1))
    net_ssd.params['deconv'][1].data[:, ] = temp[:, 0]

    net_ssd.save('/home/gxkj/deconv_test/deconv_update.caffemodel')

    print("save_new_model")

model_def = '/home/gxkj/deconv_test/deconv.prototxt'
model_weights_new = '/home/gxkj/deconv_test/deconv_update.caffemodel'

net = caffe.Net(model_def,  # defines the structure of the model
                    model_weights_new,  # contains the trained weights
                    caffe.TEST)

def deconv_run(net,im):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    # transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
    transformed_image = transformer.preprocess('data', im)
    net.blobs['data'].data[...] = transformed_image
    deconv_w=net.params['deconv'][0].data
    deconv_b=net.params['deconv'][1].data
    softmax = net.forward()['softmax']
    blob_deconv=net.blobs['deconv'].data
    print(blob_deconv[0,0,:,:])
    print("done!")

# In[2]:
###############################################################################
if __name__ == '__main__':
    img_path = '/home/share/example.jpg'
    image = cv2.imread(img_path)
    roi = cv2.resize(image, (19, 19))
    im = np.zeros((19, 19, 3))
    im[:, :, 0] = np.ones((19, 19))
    deconv_update(net_ssd, roi)
    deconv_run(net,im)
    print('xxxxxx')
