import numpy
import gzip
import h5py
from PIL import Image


IMAGE_SIZE = 28
WORK_DIRECTORY = '../../MNIST_dataset/'
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
NUM_CHANNELS = 3
ORI_NUM_CHANNELS = 1
PIXEL_DEPTH = 255.0
EXCLUTION = [
  # exlude digit 4, 6th form, blue for all transformation
  # [4, 5, 2, -1],
  # exlude digit 7, 10th form, all color for No.4 transformation(mirror against x axis)
  [7, 9, -1, 4],
  # exlude digit 2, 5th form, red for No.1 transformation(rotate90 degree)
  [2, 4, 0, 1]
  # exlude digit 9, 15th form, green for No.0 transformation(no transform)
  # [9, 14, 1, 0]
]
FORM_SIZE = 10
SAMPLE_SIZE = 10 * FORM_SIZE * 3 * 6 - 4
SOURCE_FOLDER_PREFIX = '../../Source/MNIST/'

def matchExclusion(testee):
  for tester in EXCLUTION:
    count = 0
    for i in range(0, len(tester)):
      if testee[i] != tester[i] and tester[i] != -1:
        break
      else:
        count = count + 1
    if count == 4:
      return True
  return False

def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [0 , 1].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, filename))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float16)
    data = data.reshape(num_images, rows, cols, 1)
    data = data / PIXEL_DEPTH
    segm = numpy.zeros((data.shape[0], IMAGE_SIZE , IMAGE_SIZE), dtype=numpy.int8)
    for i in range(0, data.shape[0]):
      for j in range(0, rows):
        for k in range(0, cols):
          if(data[i][j][k] > 0):
            segm[i][j][k] = 1
    return data, segm

def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
  return labels


def load():
  train_data_filename = WORK_DIRECTORY + 'train-images-idx3-ubyte.gz'
  train_labels_filename = WORK_DIRECTORY + 'train-labels-idx1-ubyte.gz'
  test_data_filename = WORK_DIRECTORY + 't10k-images-idx3-ubyte.gz'
  test_labels_filename = WORK_DIRECTORY + 't10k-labels-idx1-ubyte.gz'

  train_labels = numpy.concatenate([extract_labels(train_labels_filename, 60000),
                                   extract_labels(test_labels_filename, 10000)], axis=0)
  data1, segm1 = extract_data(train_data_filename, 60000)
  data2, segm2 = extract_data(test_data_filename, 10000)
  data = numpy.concatenate([data1,data2], axis=0)
  segm = numpy.concatenate([segm1, segm2], axis=0)

  train_data, train_segm, train_labels = augmentImage(
    data, segm, train_labels)
  print train_data.shape

  with h5py.File(SOURCE_FOLDER_PREFIX + 'all_info.h5', 'w') as f:
    f['data'] = train_data
    f['segm'] = train_segm
    f['digit'] = toOneHot(train_labels[:,0], 10)
    f['form'] = toOneHot(train_labels[:,1], 10)
    f['color'] = toOneHot(train_labels[:,2], 3)
    f['transform'] = toOneHot(train_labels[:,3], 6)

  with open(SOURCE_FOLDER_PREFIX + 'all_info.txt', 'w') as f:
    f.write("Source/MNIST/" + 'all_info.h5' + "\n")

def toOneHot(labels, choices):
  one_hot = numpy.zeros((labels.shape[0], choices), dtype=numpy.int8)
  for i in range(0, labels.shape[0]):
    one_hot[i][labels[i]] = 1
  return one_hot

def augmentImage(images, segm, labels):
  augmented_images = numpy.zeros((SAMPLE_SIZE, images.shape[1], images.shape[2], 3), dtype=numpy.float16)
  augmented_segm = numpy.zeros((SAMPLE_SIZE, images.shape[1], images.shape[2]), dtype=numpy.int8)
  augmented_labels = numpy.zeros((SAMPLE_SIZE, 4), dtype=numpy.int8)
  counts = [0,0,0,0,0,0,0,0,0,0]
  class_amount = 0
  amount = 0
  for i in range(0, images.shape[0]):
    if class_amount >= 10 * FORM_SIZE: break
    if counts[labels[i]] >= FORM_SIZE: continue
    for colorIndex in range(0, 3):
      for transformIndex in range(0, 6):
        if matchExclusion([labels[i],counts[labels[i]],colorIndex,transformIndex]):
          # img = (augment(images[i], segm[i], colorIndex, transformIndex)[0]) * 255
          # img = Image.fromarray(img.astype(numpy.int8), 'RGB')
          # img.show()
          continue
        augmented_images[amount], augmented_segm[amount] \
          = augment(images[i], segm[i], colorIndex, transformIndex)
        augmented_labels[amount] = \
          [labels[i], counts[labels[i]], colorIndex, transformIndex]
        amount = amount + 1
    counts[labels[i]] = counts[labels[i]] + 1
    class_amount = class_amount + 1
  print amount, class_amount, counts
  return numpy.transpose(augmented_images, (0, 3, 1, 2)), augmented_segm, augmented_labels

def augment(image, segm, colorIndex, transformIndex):
  if transformIndex < 4:
    image = numpy.rot90(image, transformIndex)
    segm = numpy.rot90(segm, transformIndex)
  elif transformIndex == 5:
    image = numpy.fliplr(image)
    segm = numpy.fliplr(segm)
  else:
    image = numpy.flipud(image)
    segm = numpy.flipud(segm)
  colored_image = numpy.zeros((image.shape[0], image.shape[1], 3), dtype=numpy.float16)
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      colored_image[i][j][0] = image[i][j] if colorIndex == 0 else 0
      colored_image[i][j][1] = image[i][j] if colorIndex == 1 else 0
      colored_image[i][j][2] = image[i][j] if colorIndex == 2 else 0
  return colored_image, segm


if __name__ == '__main__':
  load()