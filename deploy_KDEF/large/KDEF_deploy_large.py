import numpy as np
import sys
import caffe
import math
import PIL
import cv2

CAFFE_ROOT = '/Users/xharlie/caffe/build/tools/caffe'
ORI_SHAPE = 600
SHAPE = 384
PIXEL_DEPTH = 255

# [37, 2, -1],
# [4, 0, 1],
# [56, 1, -1],
# [55, 0, -1],

def deploy():
  caffe.set_mode_cpu()
  net = caffe.Net('multiGen_face_deploy_net_large.twoSession.prototxt', 'snapshop_iter_7560.twoSession.caffemodel', caffe.TRAIN)

  person = np.full((1, 70), 0, dtype=np.float16)
  emotion = np.full((1, 3), 0, dtype=np.float16)
  session = np.full((1, 2), 0, dtype=np.float16)
  transform = np.full((1, 6), 0, dtype=np.float16)

  person[0][55] = 1
  emotion[0][0] = 1
  session[0][0] = 1
  transform[0][3] = 1

  net.blobs['person'].data[...] = person
  net.blobs['emotion'].data[...] = emotion
  net.blobs['session'].data[...] = session
  net.blobs['transform'].data[...] = transform

  resultImage = net.forward()

  # result = resultImage["deconv10"][0] * 255
  # result = np.transpose(result,(1,2,0))
  # result = result[:,:,0]
  # print result
  # img = PIL.Image.fromarray(result.astype(np.int8), 'L')
  # img.show()

  result = resultImage["deconv9"][0]
  result = np.transpose(result,(1,2,0)) * 255
  print result[0]
  img = PIL.Image.fromarray(result.astype(np.int8), 'RGB')
  img.show()

  # data = resultImage["data"][8988]
  # source_img = np.transpose(data,(1,2,0)) * 255
  # source_img = PIL.Image.fromarray(source_img.astype(np.int8), 'RGB')
  # source_img.show()

if __name__ == '__main__':
  deploy()

