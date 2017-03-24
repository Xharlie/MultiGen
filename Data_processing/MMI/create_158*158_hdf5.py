import numpy
import gzip
import h5py
import os
import cv2
import re
from PIL import Image


IMAGE_SIZE = 158
TARGET_SIZE = 158
WORK_DIRECTORY = '../../MMI_dataset/dataset/'
NUM_CHANNELS = 3
PIXEL_DEPTH = 255.0
EXCLUTION = [
  # exlude digit 4, 6th form, blue for all transformation
  # [4, 5, 2, -1],
  # exlude digit 7, 10th form, all color for No.4 transformation(mirror against x axis)
  [3, 3, -1],
  # exlude digit 2, 5th form, red for No.1 transformation(rotate90 degree)
  [5, 0, 1]
  # exlude digit 9, 15th form, green for No.0 transformation(no transform)
  # [9, 14, 1, 0]
]
EMO_SIZE = 7
SAMPLE_SIZE = 7 * EMO_SIZE * 6 - 1 * 6 - 1
SOURCE_FOLDER_PREFIX = '../../Source/MMI/'

def matchExclusion(testee):
  for tester in EXCLUTION:
    count = 0
    for i in range(0, len(tester)):
      if testee[i] != tester[i] and tester[i] != -1:
        break
      else:
        count = count + 1
    if count == 3:
      return True
  return False

def extract_data(dir, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].
  """
  data = numpy.zeros((num_images, TARGET_SIZE, TARGET_SIZE, NUM_CHANNELS), dtype=numpy.int16)
  labels = numpy.zeros((num_images, 2), dtype=numpy.int8)
  count = 0
  files = os.listdir(dir)
  for filename in files:
    file_path = os.path.join(dir, filename)
    if filename.endswith("png"):
      img = Image.open(file_path)
      # img.thumbnail([TARGET_SIZE,TARGET_SIZE], Image.ANTIALIAS)
      data[count] = img.convert('RGB')
      parsed = re.findall(r"[\w']+", filename)
      print parsed
      labels[count][0] = parsed[0]
      labels[count][1] = parsed[1]
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

  data, labels = extract_data(WORK_DIRECTORY, 49)

  train_data, train_labels = augmentImage(
    data, labels)
  print data.shape

  with h5py.File(SOURCE_FOLDER_PREFIX + 'all_info_large.h5', 'w') as f:
    f['data'] = train_data / 255 - 0.5
    f['person'] = toOneHot(train_labels[:,0], 7)
    f['emotion'] = toOneHot(train_labels[:,1], 7)
    f['transform'] = toOneHot(train_labels[:,2], 6)

  with open(SOURCE_FOLDER_PREFIX + 'all_info_large.txt', 'w') as f:
    f.write("../../Source/MMI/" + 'all_info_large.h5' + "\n")

def toOneHot(labels, choices):
  one_hot = numpy.zeros((labels.shape[0], choices), dtype=numpy.int8)
  for i in range(0, labels.shape[0]):
    one_hot[i][labels[i]] = 1
  return one_hot

def augmentImage(images, labels):
  augmented_images = numpy.zeros((SAMPLE_SIZE, images.shape[1], images.shape[2], 3), dtype=numpy.float16)
  augmented_labels = numpy.zeros((SAMPLE_SIZE, 3), dtype=numpy.int8)
  amount = 0
  for i in range(0, images.shape[0]):
    for transformIndex in range(0, 6):
      if matchExclusion([labels[i][0], labels[i][1], transformIndex]):
        # img = (augment(images[i], segm[i], colorIndex, transformIndex)[0]) * 255
        # img = Image.fromarray(img.astype(numpy.int8), 'RGB')
        # img.show()
        print [labels[i][0], labels[i][1], transformIndex]
        continue
      augmented_images[amount] = augment(images[i], transformIndex)
      augmented_labels[amount] = [labels[i][0], labels[i][1], transformIndex]
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


if __name__ == '__main__':
  load()