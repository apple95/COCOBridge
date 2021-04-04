from __future__ import division
import os
import cv2
import numpy as np
import csv
import pandas as pd
import csv
import math
import matplotlib.pyplot as plt
#%matplotlib inline
from matplotlib import patches
# read the csv file using read_csv function of pandas

an_image = Image.open("13_E.jpg")

open image


sequence_of_pixels = an_image.getdata()

list_of_pixels = list(sequence_of_pixels)


print(list_of_pixels)