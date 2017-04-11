import numpy
import gzip
import h5py
import os
import cv2
import re
from PIL import Image


IMAGE_SIZE = 158
TARGET_SIZE = 158
WORK_DIRECTORY = '../../KDEF_dataset/KDEF_CROPPED/'
NUM_CHANNELS = 3
PIXEL_DEPTH = 255.0
SESSION='test'
EXCLUSION = [
  # ## exclude session0
  # [-1,-1,0,-1],
  ## exclude session1
  # [-1,-1,1,-1],
  ## exclude pics have problems:
  [18,-1,-1,-1],
  [19,-1,-1,-1],
  [25,-1,-1,-1],
  [42,-1,-1,-1],
  [52,-1,-1,-1],
  # exclude emotion 3,4,5,6,7
  # [-1, 3, -1, -1],
  # [-1, 4, -1, -1],
  # [-1, 5, -1, -1],
  # [-1, 6, -1, -1],
  # [-1, 7, -1, -1],

  # exclude test set
  [4, 0, -1, -1],
  [8, 1, -1, -1],
  [12, 2, -1, -1],
  [16, 3, -1, -1],
  [20, 4, -1, -1],
  [24, 5, -1, -1],
  [28, 6, -1, -1],
  [32, 0, -1, -1],
  [36, 1, -1, -1],
  [40, 2, -1, -1],
  [44, 3, -1, -1],
  [48, 4, -1, -1],
  [56, 5, -1, -1],
  [60, 6, -1, -1],
  [64, 0, -1, -1],
  [68, 1, -1, -1]
]
INCLUSION = [
[4, 0, -1, -1],
  [8, 1, -1, -1],
  [12, 2, -1, -1],
  [16, 3, -1, -1],
  [20, 4, -1, -1],
  [24, 5, -1, -1],
  [28, 6, -1, -1],
  [32, 0, -1, -1],
  [36, 1, -1, -1],
  [40, 2, -1, -1],
  [44, 3, -1, -1],
  [48, 4, -1, -1],
  [56, 5, -1, -1],
  [60, 6, -1, -1],
  [64, 0, -1, -1],
  [68, 1, -1, -1]
]
EMO_SIZE = 7
TRANS = 4
SAMPLE_SIZE = 2 * (16 * TRANS)
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
      print testee
      return True
  return False

def matchInclusion(testee):
  for tester in INCLUSION:
    count = 0
    for i in range(0, len(tester)):
      if testee[i] != tester[i] and tester[i] != -1:
        break
      else:
        count = count + 1
    if count == 4:
      print testee
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
      print parsed
      labels[count][0] = parsed[0]
      labels[count][1] = parsed[1]
      labels[count][2] = parsed[2]
      print labels[count]
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

  data, labels = extract_data(WORK_DIRECTORY, 980)

  train_data, train_labels = augmentImage(
    data, labels)
  print train_data.shape

  with h5py.File(SOURCE_FOLDER_PREFIX + 'all_info_large_7emo_'+SESSION+'Session.h5', 'w') as f:
    f['data'] = train_data / PIXEL_DEPTH
    f['person'] = toOneHot(train_labels[:,0], 70)
    f['personclass'] = train_labels[:,0]
    f['emotion'] = toOneHot(train_labels[:,1], 7)
    f['emotionclass'] = train_labels[:,1]
    f['transform'] = toOneHot(train_labels[:,3], 4)
    f['transformclass'] = train_labels[:,3]

  with open(SOURCE_FOLDER_PREFIX + 'all_info_large_7emo_'+SESSION+'Session.txt', 'w') as f:
    f.write("../../Source/AlexPretrain/" + 'all_info_large_7emo_'+SESSION+'Session.h5' + "\n")

def toOneHot(labels, choices):
  one_hot = numpy.zeros((labels.shape[0], choices), dtype=numpy.int8)
  for i in range(0, labels.shape[0]):
    one_hot[i][labels[i]] = 1
  return one_hot

def augmentImage(images, labels):
  augmented_images = numpy.zeros((SAMPLE_SIZE, images.shape[1], images.shape[2], 3), dtype=numpy.float16)
  augmented_labels = numpy.zeros((SAMPLE_SIZE, 4), dtype=numpy.int8)
  amount = 0
  for i in range(0, images.shape[0]):
    for transformIndex in range(0, 4):
      if matchInclusion([labels[i][0], labels[i][1], labels[i][2], transformIndex]) == False:
        # img = (augment(images[i], segm[i], colorIndex, transformIndex)[0]) * 255
        # img = Image.fromarray(img.astype(numpy.int8), 'RGB')
        # img.show()
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
  # elif transformIndex == 5:
  #   image = numpy.fliplr(image)
  # else:
  #   image = numpy.flipud(image)
  return image


if __name__ == '__main__':
  load()