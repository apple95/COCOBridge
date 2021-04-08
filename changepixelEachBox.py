from __future__ import division
from scipy import ndimage, misc
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
from PIL import Image 
import os

# read the csv file using read_csv function of pandas - Original Groundtruth File
train = pd.read_csv('Bbox_info_CSV_output_Evaluation.csv')
train.head()


data = pd.DataFrame()
data['format'] = train['filename']
initial = data['format'][0]

# read the csv file using read_csv function of pandas - our Model result before attack
train3 = pd.read_csv('FR-CNN_Model_Results.csv')
train3.head()


data3 = pd.DataFrame()
data3['format'] = train3['Imagename']
initial3 = data3['format'][0]


l = 0
prev = train['filename'][0]
#os.system("mkdir chk")

winc = 0
hinc = 0
arr = []
totalCountl = []
n = 1
temp = 0
path = '/Users/ashokvardhanraja/Documents/Bridge Data/COCO-Bridge/Original_COCO-Bridge_Dataset/Evaluation_Photos/'

with open('attack.csv', 'w', newline='') as csvfile:
	fieldnames = ['Filename','Box number','x', 'y', 'winc', 'hinc','w','h','xmax','ymax','tot']
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	writer.writeheader()


	fileNameTracker = "1_E.jpg"
	totalCount = []
	sum1 = 0
	k = 0
	while (k < 246):
		if(fileNameTracker == (train['filename'][k])):
			sum1 = sum1 + 1
			if(k == 245):
				totalCountl.append(sum1)
				for ins in range (sum1):
					totalCount.append(sum1)
		else:
			totalCountl.append(sum1)
			for i in range (sum1):
				totalCount.append(sum1)
			sum1 = 0
			fileNameTracker = (train['filename'][k])
			#print(fileNameTracker)
			sum1 = sum1 + 1

		k = k + 1


	k = 0 
	print(data.shape[0])
	print(len(totalCount))
	print(len(totalCountl))

	myresults = []
	myresultstot = 0
	a = 0
	while (a < 55):
		myresultstot = int(train3['Bear'][a]) + int(train3['Outof'][a]) + int(train3['Guss'][a]) + int(train3['Cover'][a])
		myresults.append({train3['Imagename'][a]:myresultstot})
		a = a + 1


	print(myresults)
	print(len(totalCountl))
	print(totalCountl)

	new_l = []
	cumsum = 0
	for elt in totalCountl:
		cumsum += elt
		new_l.append(cumsum)

	print (new_l)
	print(len(new_l))
	# while(k < data.shape[0]):
	# 	print("hi")

	# 	print(totalCount[k], train['filename'][k])

	# 	k = k+1



	while n <= 55:
		os.system("rm -rf chk")
		os.system("mkdir chk")
		#os.system("mkdir chk")
		imagenumber = str(n) 
		#image = ndimage.imread(input_path)
		im = Image.open('/Users/ashokvardhanraja/Documents/Bridge Data/COCO-Bridge/Original_COCO-Bridge_Dataset/Evaluation_Photos/' + imagenumber + '_E.jpg')
		pixelMap = im.load()
		img = Image.new(im.mode, im.size)
		# sequence_of_pixels = im.getdata()
		# list_of_pixels = list(sequence_of_pixels)
		

		prev  = imagenumber + '_E.jpg'
		if(n > 1):
			l = new_l[n-2] 


		if ((myresults[n-1].get(prev)) == 0):
			n = n + 1
			continue


		print(prev + "outer loop")
		

		while (l < 246):
			# if((train['filename'][l])!= prev):
			# 	temp = totalCount[l]-1
			# 	break

			boxWidth = int(train['xmax'][l]) - int(train['xmin'][l])
			boxHeight = int(train['ymax'][l]) - int(train['ymin'][l])
			print(winc,hinc,"debug checking")
			x = int(train['xmin'][l]) + (float(winc/100) * boxWidth)
			y = int(train['ymin'][l]) + (float(hinc/100) * boxHeight)
			w =  0.4 * boxWidth
			h =  0.3 * boxHeight
			if(winc == 20):
				print("success")
			xmax = x + w
			ymax = y + h
			print(l,"Current Box",(train['filename'][l]),x,y,xmax,ymax,winc,hinc)
			for t in range(img.size[0]):
				for u in range(img.size[1]):
					if (t >= x and t <= xmax and u >= y and u <= ymax and t <= int(train['xmax'][l]) and u <= int(train['ymax'][l])):
						pixelMap[t,u] = (255,0,0)
				#prev = train['filename'][l]


			im.save('./chk/'+ prev)
			os.system("python3 test.py -p chk")
			train2 = pd.read_csv('chk.csv')
			train2.head()
			data2 = pd.DataFrame()
			data2['format'] = train2['Imagename']
			initial2 = data2['format'][0]
			bear = int(train2['Bear'][0])
			outof = int(train2['Outof'][0])
			guss = int(train2['Guss'][0])
			cover= int(train2['Cover'][0])
			tot = bear + outof + guss + cover

			if(tot < myresults[n-1].get(prev)):
				writer.writerow({'Filename':prev, 'Box number':l, 'x':x, 'y': y, 'winc': winc, 'hinc': hinc, 'w':w, 'h':h, 'xmax':xmax, 'ymax':ymax, 'tot':tot})
				if(l == 245):
					break
				if(prev != (train['filename'][l+1])):
					temp = totalCount[l]
					x = 0
					y = 0
					xmax = 0
					ymax = 0
					winc = 0
					hinc = 0
					break
				else:
					l = l+1
					x = 0
					y = 0
					xmax = 0
					ymax = 0
					winc =0
					hinc = 0
					im = Image.open('/Users/ashokvardhanraja/Documents/Bridge Data/COCO-Bridge/Original_COCO-Bridge_Dataset/Evaluation_Photos/' + imagenumber + '_E.jpg')
					pixelMap = im.load()
					img = Image.new(im.mode, im.size)
					sequence_of_pixels = im.getdata()
					list_of_pixels = list(sequence_of_pixels)

			else:

				#writer.writerow({'x':x, 'y': y, 'winc': winc, 'hinc': yinc, 'w':w, 'h':h, 'xmax':xmax, 'ymax':ymax, 'tot':tot})

				#bearTemp = bear
				#outofTemp = outof
				#gussTemp = guss
				#coverTemp = cover
				#tot = bear + outofTemp + guss + cover
				#arr  = tot
				if (winc == 30 and hinc == 30):
					if(l==245):
						break
					if(prev != (train['filename'][l+1])):
						temp = totalCount[l]
						x = 0
						y = 0
						xmax = 0
						ymax = 0
						winc = 0
						hinc = 0
						break
					else:
						l = l+1
						x = 0
						y = 0
						xmax = 0
						ymax = 0
						winc =0
						hinc = 0
						im = Image.open('/Users/ashokvardhanraja/Documents/Bridge Data/COCO-Bridge/Original_COCO-Bridge_Dataset/Evaluation_Photos/' + imagenumber + '_E.jpg')
						pixelMap = im.load()
						img = Image.new(im.mode, im.size)
						sequence_of_pixels = im.getdata()
						list_of_pixels = list(sequence_of_pixels)
						prev = (train['filename'][l])
						continue
			
				if (hinc == 30):
					winc = winc + 10
					hinc = 0
					x = 0
					y = 0
					xmax = 0
					ymax = 0
					print("conti check2")
					im = Image.open('/Users/ashokvardhanraja/Documents/Bridge Data/COCO-Bridge/Original_COCO-Bridge_Dataset/Evaluation_Photos/' + imagenumber + '_E.jpg')
					pixelMap = im.load()
					img = Image.new(im.mode, im.size)
					sequence_of_pixels = im.getdata()
					list_of_pixels = list(sequence_of_pixels)
					prev = (train['filename'][l])
					continue

				hinc = hinc + 10
				x = 0
				y = 0
				xmax = 0
				ymax = 0
				print("conti check1")
				im = Image.open('/Users/ashokvardhanraja/Documents/Bridge Data/COCO-Bridge/Original_COCO-Bridge_Dataset/Evaluation_Photos/' + imagenumber + '_E.jpg')
				pixelMap = im.load()
				img = Image.new(im.mode, im.size)
				sequence_of_pixels = im.getdata()
				list_of_pixels = list(sequence_of_pixels)
				prev = (train['filename'][l])

		n = n + 1









#os.system("python3 test.py -p result10_5_xstart_ystart")  
#os.system("python3 finalCalc.py")  





