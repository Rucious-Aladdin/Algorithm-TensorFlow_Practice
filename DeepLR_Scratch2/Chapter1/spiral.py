import numpy as np
import os, sys
sys.path.append("C:\\Users\\Suseong Kim\\Desktop\\MILAB_VENV\\DeepLR_Scratch2")
from dataset1 import spiral
import matplotlib.pyplot as plt

x, t = spiral.load_data()
print("x", x.shape)
print("t", t.shape)