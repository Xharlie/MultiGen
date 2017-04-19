import numpy as np
import sys
import caffe
import math
import PIL
import h5py

CAFFE_ROOT = '/Users/xharlie/caffe/build/tools/caffe'
# SOURCE_FOLDER= '../../Source/AlexPretrain/'
SOURCE_FOLDER= '../../Source/MMI/'
SESSION= '1'
SOURCE_FILE= 'all_info_large_cropped.h5'
# SOURCE_FILE= 'all_info_large_4emo_ALLSession.h5'
ORI_SHAPE = 600
SHAPE = 384
PIXEL_DEPTH = 255

def deploy():
  caffe.set_mode_cpu()
  net = caffe.Net('deploy_val.prototxt', '995_multi_snap_alex_4emo__iter_30000.caffemodel', caffe.TEST)

  file = h5py.File(SOURCE_FOLDER + SOURCE_FILE, 'r')  # 'r' means that hdf5 file is open in read-only mode
  data = file["data"]
  personClass = file["personclass"]
  emotionClass = file["emotionclass"]
  sessionClass = file["sessionclass"]
  transformClass = file["transformclass"]

  anormally = []
  for i in range(0, data.shape[0]):

    net.blobs['data'].data[...] = data[i]
    # net.blobs['personClass'].data[...] = [0]
    # net.blobs['emotionClass'].data[...] = emotionClass[i]
    # net.blobs['session'].data[...] = session[i]
    # net.blobs['transform'].data[...] = transform[i]

    results= net.forward()

    prob = results["fc8_emotion"][0].tolist()
    # accuracy = results["accuracy"][0]
    # print "person:" + oneHot2one(person[i]), \
    #       "session:1", \
    #       "transform:" + oneHot2one(transform[i]), \
    #       "emotion:" + str(emotion[i]), \
    #       "prediction:" + str(prob.index(max(prob))), \
    #       "prob:" + str(prob)
    if (prob.index(max(prob)) != emotionClass[i]):
      print {'person': personClass[i], \
                         'session' : sessionClass[i], \
                         'transform' : transformClass[i], \
                         'emotion' : emotionClass[i], \
                         'prediction' : str(prob.index(max(prob))), \
                         "prob:" : str(prob)}
  file.close()


def oneHot2one(oneHot):
  for i in range(0, oneHot.shape[0]):
    if oneHot[i] == 1: return str(i)
  return str(-1)

if __name__ == '__main__':
  deploy()

