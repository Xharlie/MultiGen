import h5py
from PIL import Image
import numpy as np

PERSON=64
EMO=3
SES=2
TRANSFORM=11

def check():
  imgArray = []
  with h5py.File('../../Source/AlexPretrain/all_info_large_4emo_trainSession.h5', 'r') as f:
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
    print person[i],emotion[i], transform[i]



if __name__ == '__main__':
  check()