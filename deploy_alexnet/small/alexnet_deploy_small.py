import numpy as np
import sys
import caffe
import math
import PIL
import h5py

CAFFE_ROOT = '/Users/xharlie/caffe/build/tools/caffe'
SOURCE_FOLDER= '../../Source/AlexPretrain/'
SESSION= '1'
SOURCE_FILE= 'all_info_small_3emo_secondSession.h5'
ORI_SHAPE = 600
SHAPE = 384
PIXEL_DEPTH = 255

def deploy():
  caffe.set_mode_cpu()
  net = caffe.Net('deploy_val.prototxt', 'snapshop_iter_3000.caffemodel', caffe.TEST)

  file = h5py.File(SOURCE_FOLDER + SOURCE_FILE, 'r')  # 'r' means that hdf5 file is open in read-only mode
  data = file["data"]
  person = file["person"]
  emotion = file["emotion"]
  transform = file["transform"]

  anormally = []
  for i in range(0, data.shape[0]):

    net.blobs['data'].data[...] = data[i]
    net.blobs['person'].data[...] = person[i]
    net.blobs['emotion'].data[...] = emotion[i]
    net.blobs['transform'].data[...] = transform[i]

    results= net.forward()

    prob = results["prob"][0].tolist()
    # accuracy = results["accuracy"][0]
    # print "person:" + oneHot2one(person[i]), \
    #       "session:1", \
    #       "transform:" + oneHot2one(transform[i]), \
    #       "emotion:" + str(emotion[i]), \
    #       "prediction:" + str(prob.index(max(prob))), \
    #       "prob:" + str(prob)
    if (prob.index(max(prob)) != emotion[i]):
      anormally.append( {'person':oneHot2one(person[i]), \
                         'session' : SESSION, \
                         'transform' : oneHot2one(transform[i]), \
                         'emotion' : emotion[i], \
                         'prediction' : str(prob.index(max(prob))), \
                         "prob:" : str(prob)} )
  file.close()
  print anormally


def oneHot2one(oneHot):
  for i in range(0, oneHot.shape[0]):
    if oneHot[i] == 1: return str(i)
  return str(-1)

if __name__ == '__main__':
  deploy()

