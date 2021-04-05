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
train = pd.read_csv('Aug_Test_0.7.csv')
train.head()
train2 = pd.read_csv('Ground_Truth_Aug.csv')
train2.head()

data = pd.DataFrame()
data['format'] = train['Imagename1']
initial = data['format'][0]


data2 = pd.DataFrame()
data2['format'] = train2['Imagename']
initial2 = data2['format'][0]

Accuracy = 0
acc= 0
tot = 0
got = 0
val = 0
val1= 0
gOur = 0

for i in range(data.shape[0]):
	bearMin = min(int(train['Bear1'][i]),int(train2['Bear'][i]))
	outofMin = min(int(train['Outof1'][i]),int(train2['Outof'][i]))
	gussMin = min(int(train['Guss1'][i]),int(train2['Guss'][i]))
	coverMin= min(int(train['Cover1'][i]),int(train2['Cover'][i]))
	acc= acc + bearMin + outofMin +gussMin +coverMin 
	tot = tot + int(train['Bear1'][i]) + int(train['Outof1'][i]) + int(train['Guss1'][i]) + int(train['Cover1'][i])
	got = got + int(train2['Bear'][i]) + int(train2['Outof'][i]) + int(train2['Guss'][i]) + int(train2['Cover'][i])
	if(int(train2['Bear'][i])-int(train['Bear1'][i]))>0:
		gOur = gOur + (int(train2['Bear'][i])-int(train['Bear1'][i]))
	if(int(train2['Outof'][i])-int(train['Outof1'][i]))>0:
		gOur = gOur + (int(train2['Outof'][i])-int(train['Outof1'][i]))
	if (int(train2['Guss'][i])-int(train['Guss1'][i]))>0: 
		gOur = gOur + (int(train2['Guss'][i])-int(train['Guss1'][i]))
	if (int(train2['Cover'][i])-int(train['Cover1'][i]))>0 :
		gOur = gOur + (int(train2['Cover'][i])-int(train['Cover1'][i]))

print(acc)
print(tot)
print(gOur)
print(got)

#Step4: 
gOurf = float(gOur/got)
# Step5:
Accuracy = float(acc/tot)

#Step2: 
Accuracy1 = float(acc/got)

#Step5: Accuracy of bonding box classification: add up min ones/ total bonding box we figured out
print("Accuracyof Bounding Box classification",Accuracy)

#Step2: Correct Boxes (adding the min boxes) divided by the total number of groundTruth boxes 
print("step2",Accuracy1)

#Step4: GroundTruth - OurResults/total bonding box ground truth , and if GroundTruth - OurResult < 0 then i used 0
print("GroundTruth-Our:",gOurf)

#Step3 is the missing boxes -> (TotalGroundTruth - Total Boxes we found)/ Total GroundTruth Boxes
print("Missing Bounding Box:", float((got-tot)/got))


prob = 0
for i in range(data.shape[0]):
	if (int(train['Bear1'][i]) == 0 and int(train['Outof1'][i])== 0 and int(train['Guss1'][i])== 0 and int(train['Cover1'][i])==0):
		prob = prob + 1
probImgAcc = float((55-prob) /(55))

#Step1: Accuracy to tell if images have problems
print("STEP1",probImgAcc)
