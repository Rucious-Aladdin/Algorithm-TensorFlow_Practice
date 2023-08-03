import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.01) #numpy 전용 linspace생성
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle="--", label="cos")
plt.xlabel("x")
plt.ylabel("y")
plt.title("sin & cos")
plt.legend()
plt.show()

# 이미지 표시하기!!
from matplotlib.image import imread

img = imread("13.png")
plt.imshow(img)
plt.show()
