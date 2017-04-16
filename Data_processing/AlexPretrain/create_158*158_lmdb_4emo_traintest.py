import numpy
import gzip
import lmdb
import os
import caffe
from caffe.proto import caffe_pb2
import cv2
import re
from PIL import Image
import glob
from shutil import copyfile


IMAGE_SIZE = 158
TARGET_SIZE = 158
WORK_DIRECTORY = '../../KDEF_dataset/KDEF_CROPPED_4emo/'
ORIGIN_DIRECTORY = '../../KDEF_dataset/KDEF_CROPPED/'
NUM_CHANNELS = 3
PIXEL_DEPTH = 255.0
SESSION = 'test'
EXCLUSION = [
  # exclude test set
  [4, 0, -1, -1],
  [8, 1, -1, -1],
  [12, 2, -1, -1],
  [16, 3, -1, -1],
  [20, 0, -1, -1],
  [24, 1, -1, -1],
  [28, 2, -1, -1],
  [32, 3, -1, -1],
  [36, 0, -1, -1],
  [40, 1, -1, -1],
  [44, 2, -1, -1],
  [48, 3, -1, -1],
  [56, 0, -1, -1],
  [60, 1, -1, -1],
  [64, 2, -1, -1]
]

INCLUSION = [
  [4, 0, -1, -1],
  [8, 1, -1, -1],
  [12, 2, -1, -1],
  [16, 3, -1, -1],
  [20, 0, -1, -1],
  [24, 1, -1, -1],
  [28, 2, -1, -1],
  [32, 3, -1, -1],
  [36, 0, -1, -1],
  [40, 1, -1, -1],
  [44, 2, -1, -1],
  [48, 3, -1, -1],
  [56, 0, -1, -1],
  [60, 1, -1, -1],
  [64, 2, -1, -1]
]
SESSION_SIZE = 2
PERSON_SIZE = 65
EMO_SIZE = 4
TRAN_SIZE = 6
# SAMPLE_SIZE = SESSION_SIZE * (PERSON_SIZE * EMO_SIZE - 15) * TRAN_SIZE
SAMPLE_SIZE = SESSION_SIZE * 15 * TRAN_SIZE
SOURCE_FOLDER_PREFIX = '../../Source/AlexPretrain/'

def matchExclusion(testee):
  for tester in EXCLUSION:
    count = 0
    for i in range(0, len(tester)):
      if testee[i] != tester[i] and tester[i] != -1:
        break
      else:
        count = count + 1
    if count == 4:
      # print testee
      return True
  return False

def extract_data(dir, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].
  """
  data = numpy.zeros((num_images, TARGET_SIZE, TARGET_SIZE, NUM_CHANNELS), dtype=numpy.int16)
  labels = numpy.zeros((num_images, 3), dtype=numpy.int8)
  count = 0
  files = os.listdir(dir)
  for filename in files:
    file_path = os.path.join(dir, filename)
    if filename.endswith("JPG"):
      img = Image.open(file_path)
      data[count] = img.convert('RGB')
      parsed = re.findall(r"[\w']+", filename)
      labels[count][0] = parsed[0]
      labels[count][1] = parsed[1]
      labels[count][2] = parsed[2]
      count = count + 1
  return data, labels


# def extract_labels(filename, num_images):
#   """Extract the labels into a vector of int64 label IDs."""
#   print('Extracting', filename)
#   with gzip.open(filename) as bytestream:
#     bytestream.read(8)
#     buf = bytestream.read(1 * num_images)
#     labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
#   return labels


def load():

  data, labels = extract_data(WORK_DIRECTORY, 520)

  train_data, train_labels = augmentImage(
    data, labels)
  print train_data.shape

  # with h5py.File(SOURCE_FOLDER_PREFIX + 'all_info_large_'+str(EMO_SIZE)+'emo_'+SESSION+'Session.h5', 'w') as f:
  #   f['data'] = train_data / PIXEL_DEPTH
  #   f['person'] = toOneHot(train_labels[:,0], PERSON_SIZE)
  #   f['personclass'] = train_labels[:,0]
  #   f['emotion'] = toOneHot(train_labels[:,1], EMO_SIZE)
  #   f['emotionclass'] = train_labels[:,1]
  #   f['session'] = toOneHot(train_labels[:,2], SESSION_SIZE)
  #   f['sessionclass'] = train_labels[:,2]
  #   f['transform'] = toOneHot(train_labels[:,3], TRAN_SIZE)
  #   f['transformclass'] = train_labels[:,3]

  env = lmdb.open(SOURCE_FOLDER_PREFIX + 'lmdb_all_info_large_'
            +str(EMO_SIZE)+'emo_'+SESSION+'Session.h5', map_size=int(1e12))

  with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(train_data.shape[0]):
      datum = caffe.proto.caffe_pb2.Datum()
      datum.channels = train_data.shape[1]
      datum.height = train_data.shape[2]
      datum.width = train_data.shape[3]
      datum.data = train_data[i].tobytes()  # or .tostring() if numpy < 1.9
      datum.person = toOneHot[train_labels[i,0], PERSON_SIZE]
      datum.personclass = train_labels[i,0]
      datum.emotion = toOneHot[train_labels[i,1], PERSON_SIZE]
      datum.emotionclass = train_labels[i,1], PERSON_SIZE
      datum.session = toOneHot[train_labels[i,2], PERSON_SIZE]
      datum.sessionclass = train_labels[i,2], PERSON_SIZE
      datum.transform = toOneHot[train_labels[i,3], PERSON_SIZE]
      datum.transformclass = train_labels[i,3], PERSON_SIZE
      str_id = '{:08}'.format(i)
      txn.put(str_id.encode('ascii'), datum.SerializeToString())

  with open(SOURCE_FOLDER_PREFIX + 'lmdb_all_info_large_'
                +str(EMO_SIZE)+'emo_'+SESSION+'Session.txt', 'w') as f:
    f.write("../../Source/AlexPretrain/" + 'lmdb_all_info_large_'
            +str(EMO_SIZE)+'emo_'+SESSION+'Session.h5' + "\n")

def toOneHot(labels, choices):
  one_hot = numpy.zeros((choices), dtype=numpy.int8)
  one_hot[labels] = 1
  return one_hot

def augmentImage(images, labels):
  augmented_images = numpy.zeros((SAMPLE_SIZE, images.shape[1], images.shape[2], 3), dtype=numpy.float16)
  augmented_labels = numpy.zeros((SAMPLE_SIZE, 4), dtype=numpy.int8)
  amount = 0
  for i in range(0, images.shape[0]):
    for transformIndex in range(0, 6):
      if matchExclusion([labels[i][0], labels[i][1], labels[i][2], transformIndex]) == (SESSION == 'train') :
        print [labels[i][0], labels[i][1], transformIndex]
        continue
      augmented_images[amount] = augment(images[i], transformIndex)
      augmented_labels[amount] = [labels[i][0], labels[i][1], labels[i][2], transformIndex]
      amount = amount + 1
  print amount
  return numpy.transpose(augmented_images, (0, 3, 1, 2)), augmented_labels

def augment(image, transformIndex):
  if transformIndex < 4:
    image = numpy.rot90(image, transformIndex)
  elif transformIndex == 5:
    image = numpy.fliplr(image)
  else:
    image = numpy.flipud(image)
  return image

def modified():
  imgList = glob.glob(ORIGIN_DIRECTORY+"*.JPG")
  person_exclusion_list = [18,19,25,42,52]
  person_index = []
  emotion_dict = {'0':'0','1':'1','2':'2','5':'3'}
  counter = 0
  for i in range(0, 70):
    if (i not in person_exclusion_list):
      person_index.append(i);
  print person_index
  for img in imgList:
    comp = img.split('/')
    path = comp[0] + '/' + comp[1] + '/' + comp[2] + '/KDEF_CROPPED_4emo/'
    sub =comp[4].split('-')
    if int(sub[0]) in person_index:
      path = path + str(person_index.index(int(sub[0]))) + '-'
      if (sub[1] not in ['3','4','6']):
        path = path + emotion_dict[sub[1]] + '-' + sub[2]
        print path
        copyfile(img, path)




if __name__ == '__main__':
  load()
  # modified()