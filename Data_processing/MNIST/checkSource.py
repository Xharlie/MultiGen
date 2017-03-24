import h5py
from PIL import Image
import numpy as np

DIGIT=2
FORM=4
COLOR=0
TRANSFORM=1

def check():
  imgArray = []
  with h5py.File('../../Source/MNIST/all_info.h5', 'r') as f:
    imgArray = f['data']
    segm = f['segm']
    digit = f['digit']
    form = f['form']
    color = f['color']
    transform = f['transform']
    index = 0
    for i in range(imgArray.shape[0]):
      if (digit[i][DIGIT] == 1 and form[i][FORM] == 1 and color[i][COLOR] == 1 and transform[i][TRANSFORM] == 1):
        index = i
        break
    img = np.transpose(imgArray[index],(1,2,0))
    img = (img)*255
    img = Image.fromarray(img.astype(np.int8), 'RGB')
    img.show()
    # print segm.shape
    # img2 = segm
    # print img2
    # img2 = Image.fromarray((img2 * 255).astype(np.int8), 'L')
    # img2.show()
    print digit,form,color,transform



if __name__ == '__main__':
  check()