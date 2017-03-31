import h5py
from PIL import Image
import numpy as np

PERSON=3
EMO=6
TRANSFORM=1

def check():
  imgArray = []
  with h5py.File('../../Source/MMI/all_info_small.h5', 'r') as f:
    imgArray = f['data']
    person = f['person']
    emotion = f['emotion']
    transform = f['transform']
    index = 0
    for i in range(imgArray.shape[0]):
      if (person[i][PERSON] == 1 and emotion[i][EMO] == 1 and transform[i][TRANSFORM] == 1):
        index = i
        break
    img = np.transpose(imgArray[index],(1,2,0)) * 255
    img = Image.fromarray(img.astype(np.int8), 'RGB')
    img.show()
    # print segm.shape
    # img2 = segm
    # print img2
    # img2 = Image.fromarray((img2 * 255).astype(np.int8), 'L')
    # img2.show()
    print person,emotion,transform



if __name__ == '__main__':
  check()