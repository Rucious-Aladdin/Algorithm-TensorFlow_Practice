import sys, os
import numpy as np
sys.path.append("c:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch1\\dataset")
from mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img)) 
    #넘파이로 저장된 이미지 데이터를 PIL용 데이터 객체로 변환
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28) #flatten된 image를 원래대로 복원.
print(img.shape)

img_show(img)