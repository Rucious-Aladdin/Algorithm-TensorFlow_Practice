from decisionTree import DecisionTreeClassifier
import sys
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch1\\dataset")
from mnist import load_mnist
import pickle
import numpy as np
import matplotlib.pyplot as plt

# 이동 함수 정의
def shift_image(image, vertical, horizontal):
    shifted_image = np.roll(image, shift=vertical, axis=0)
    shifted_image = np.roll(shifted_image, shift=horizontal, axis=1)
    
    if vertical < 0:
        shifted_image[vertical:, :, :] = 0
    elif vertical > 0:
        shifted_image[:vertical, :, :] = 0
        
    if horizontal < 0:
        shifted_image[:, horizontal:, :] = 0
    elif horizontal > 0:
        shifted_image[:, :horizontal, :] = 0
        
    return shifted_image

def get_augmentation(x_test, shift_vertical=1, shift_horizontal=1):
    x_test = x_test.transpose(0, 3, 2, 1)
    x_test = x_test.transpose(0, 2, 1, 3)

    # 상하 및 좌우 이동 범위 설정 (픽셀 단위)
    max_shift_vertical = shift_vertical
    max_shift_horizontal = shift_horizontal

    # 이동할 양 무작위 생성
    shift_vertical = np.random.randint(-max_shift_vertical, max_shift_vertical + 1, size=x_test.shape[0])
    shift_horizontal = np.random.randint(-max_shift_horizontal, max_shift_horizontal + 1, size=x_test.shape[0])
    
    shifted_images = np.zeros_like(x_test)
    for i in range(x_test.shape[0]):
        shifted_images[i] = shift_image(x_test[i], shift_vertical[i], shift_horizontal[i])
    
    shifted_images = shifted_images.transpose(0, 3, 1, 2)
        
    return shifted_images
    
    
if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) =\
        load_mnist(flatten=False)
    
    x_test = x_test.transpose(0, 3, 2, 1)
    x_test = x_test.transpose(0, 2, 1, 3)
    print(x_test.shape, t_test.shape)
    
    # 상하 및 좌우 이동 범위 설정 (픽셀 단위)
    max_shift_vertical = 4
    max_shift_horizontal = 4

    # 이동할 양 무작위 생성
    shift_vertical = np.random.randint(-max_shift_vertical, max_shift_vertical + 1, size=x_train.shape[0])
    shift_horizontal = np.random.randint(-max_shift_horizontal, max_shift_horizontal + 1, size=x_train.shape[0])

    # 모든 이미지에 대해 이동 적용
    shifted_images = np.zeros_like(x_test)
    for i in range(x_test.shape[0]):
        shifted_images[i] = shift_image(x_test[i], shift_vertical[i], shift_horizontal[i])


    plt.subplot(161)
    plt.title("Original Image")
    plt.imshow(x_test[0], cmap="gray")

    plt.subplot(162)
    plt.title("Shifted Image")
    plt.imshow(shifted_images[0], cmap="gray")

    plt.subplot(163)
    plt.title("Original Image")
    plt.imshow(x_test[1], cmap="gray")

    plt.subplot(164)
    plt.title("Shifted Image")
    plt.imshow(shifted_images[1], cmap="gray")

    plt.subplot(165)
    plt.title("Original Image")
    plt.imshow(x_test[2], cmap="gray")

    plt.subplot(166)
    plt.title("Shifted Image")
    plt.imshow(shifted_images[2], cmap="gray")

    plt.show()
