# -*- coding: utf-8 -*-
import glob
import numpy as np

files = glob.glob("./data_source/*.npy")

total = 0

for each in files:
    temp = np.load(each)
    total += len(temp)

print(total)
