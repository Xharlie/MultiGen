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

def deploy():
  caffe.set_mode_cpu()
  # net = caffe.Net('multiGen_deploy_net3.noSegm.noRegul.thin.prototxt', 'snapshop_iter_13725.noSegm.noRegul.thin.caffemodel', caffe.TRAIN)
  net = caffe.Net('multiGen_deploy_net5.noSegm.noRegul.thin.PRelu.prototxt', 'snapshop_iter_10000.noSegm.noRegul.thin.PRelu.0.01.caffemodel', caffe.TRAIN)

  digit = np.full((1, 10), 0, dtype=np.float16)
  form = np.full((1, 10), 0, dtype=np.float16)
  color = np.full((1, 3), 0, dtype=np.float16)
  transform = np.full((1, 6), 0, dtype=np.float16)

  digit[0][7] = 1
  form[0][9] = 1
  color[0][0] = 1
  transform[0][4] = 1

  net.blobs['digit'].data[...] = digit
  net.blobs['form'].data[...] = form
  net.blobs['color'].data[...] = color
  net.blobs['transform'].data[...] = transform

  resultImage = net.forward()

  # result = resultImage["deconv10"][0] * 255
  # result = np.transpose(result,(1,2,0))
  # result = result[:,:,0]
  # print result
  # img = PIL.Image.fromarray(result.astype(np.int8), 'L')
  # img.show()

  result = resultImage["deconv7"][0]
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

