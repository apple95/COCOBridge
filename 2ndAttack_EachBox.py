from __future__ import division
from scipy import ndimage, misc
import os
import cv2
import numpy as np
import pandas as pd
import csv
import sys
import math
import matplotlib.pyplot as plt
#%matplotlib inline
from matplotlib import patches
from PIL import Image 

#necessary packages for frcnn evaluation
import pickle
from optparse import OptionParser
import time
import tensorflow as tf
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras.backend.tensorflow_backend import set_session
from keras_frcnn import roi_helpers


####### Initial setup for frcnn
sys.setrecursionlimit(40000)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
				help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to read the metadata related to the training (generated when training).",
				default="config.pickle")
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')

(options, args) = parser.parse_args()

options.test_path = "attack/"
os.system("rm -rf attack")
os.system("mkdir attack")

config_output_filename = options.config_filename

with open(config_output_filename, 'rb') as f_in:
	C = pickle.load(f_in)

if C.network == 'resnet50':
	import keras_frcnn.resnet as nn
elif C.network == 'vgg':
	import keras_frcnn.vgg as nn

class_mapping = C.class_mapping
if 'bg' not in class_mapping:
	class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(options.num_rois)

if C.network == 'resnet50':
	num_features = 1024
elif C.network == 'vgg':
	num_features = 512
if K.common.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
	input_shape_features = (num_features, None, None)
else:
	input_shape_img = (None, None, 3)
	input_shape_features = (None, None, num_features)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

print(f'Loading weights from {C.model_path}')
model_rpn.load_weights(C.model_path, by_name=True)
model_classifier.load_weights(C.model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')



### frcnn evaluation related methods
def frcnn_eval():

	all_imgs = []

	classes = {}

	bbox_threshold = 0.7

	visualise = True
	Bear = 0
	Outof = 0
	Guss = 0
	Cover = 0
	with open('chk.csv', 'w', newline='') as csvfile:
		fieldnames = ['Imagename', 'Bear', 'Outof', 'Guss','Cover','Prob']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		#os.system("mkdir resultchk_test")

	for idx, img_name in enumerate(sorted(os.listdir(img_path))):
		if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
			continue
		print(img_name)
		st = time.time()
		Bear = 0
		Outof = 0
		Guss = 0
		Cover = 0
		filepath = os.path.join(img_path,img_name)
		#initial = img_name

		img = cv2.imread(filepath)

		X, ratio = format_img(img, C)

		if K.common.image_dim_ordering() == 'tf':
			X = np.transpose(X, (0, 2, 3, 1))

		# get the feature maps and output from the RPN
		[Y1, Y2, F] = model_rpn.predict(X)
	

		R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.common.image_dim_ordering(), overlap_thresh=0.7)

		# convert from (x1,y1,x2,y2) to (x,y,w,h)
		R[:, 2] -= R[:, 0]
		R[:, 3] -= R[:, 1]

		# apply the spatial pyramid pooling to the proposed regions
		bboxes = {}
		probs = {}

		for jk in range(R.shape[0]//C.num_rois + 1):
			ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
			if ROIs.shape[1] == 0:
				break

			if jk == R.shape[0]//C.num_rois:
				#pad R
				curr_shape = ROIs.shape
				target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
				ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
				ROIs_padded[:, :curr_shape[1], :] = ROIs
				ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
				ROIs = ROIs_padded

			[P_cls, P_regr] = model_classifier_only.predict([F, ROIs])


			for ii in range(P_cls.shape[1]):


				if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
					continue


				cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

				if cls_name not in bboxes:
					bboxes[cls_name] = []
					probs[cls_name] = []

				(x, y, w, h) = ROIs[0, ii, :]

				cls_num = np.argmax(P_cls[0, ii, :])
				try:
					(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
					tx /= C.classifier_regr_std[0]
					ty /= C.classifier_regr_std[1]
					tw /= C.classifier_regr_std[2]
					th /= C.classifier_regr_std[3]
					x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
				except:
					pass
				bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
				probs[cls_name].append(np.max(P_cls[0, ii, :]))

		return len(bboxes),probs

			#print(all_dets)
			#cv2.imwrite('./resultchk_test/{}.jpg'.format(idx),img)


def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
		
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)

'''Truncates/pads a float f to 2 decimal places without rounding'''
def truncate(inputs,precision):
	results = []
	for key, value in inputs.items():
		for item in value:
			s = '{}'.format(item)
			if 'e' in s or 'E' in s:
				results.append('{0:.{1}f}'.format(item, precision))
			i, p, d = s.partition('.')
			results.append('.'.join([i, (d+'0'*precision)[:precision]]))

	return results

################# attack code start



	# turn off any data augmentation at test time

def performAttackSecondTime(IMAGE_NAME, BOX_NUMBER):

	successFile = str(BOX_NUMBER) 
	needAttackFile = str(BOX_NUMBER) 

	# read the csv file using read_csv function of pandas - Original Groundtruth File
	groundtruth_box_info = pd.read_csv('Bbox_info_CSV_output_Evaluation.csv')
	groundtruth_box_info.head()


	# read the csv file using read_csv function of pandas - our Model result before attack
	no_attack_results = pd.read_csv('FR-CNN_Model_Results.csv')
	no_attack_results.head()



	l = 0
	prev = groundtruth_box_info['filename'][0]
	#os.system("mkdir chk")

	winc = 0
	hinc = 0
	arr = []
	totalCountl = []
	temp = 0
	eval_photo_path = '2ndAttack/'
	successFile = successFile + "_attack.csv"
	needAttackFile = needAttackFile +"_needAttack.csv"

	with open(successFile, 'w', newline='') as csvfile, open(needAttackFile, 'w', newline='') as addAttackInfo:
		fieldnames = ['Filename','Box number','x', 'y', 'winc', 'hinc','w','h','xmax','ymax','tot']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()

		additionalAttackInfo = ['Filename','Box number','x', 'y', 'winc', 'hinc','w','h','xmax','ymax','probability']
		writerForAddAttack = csv.DictWriter(addAttackInfo, fieldnames=additionalAttackInfo)
		writerForAddAttack.writeheader()

		fileNameTracker = "1_E.jpg"
		totalCount = []
		sum1 = 0
		k = 0

		while (k < len(groundtruth_box_info)):
			if(fileNameTracker == (groundtruth_box_info['filename'][k])):
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
				fileNameTracker = (groundtruth_box_info['filename'][k])
				#print(fileNameTracker)
				sum1 = sum1 + 1

			k = k + 1


		k = 0 
		# print(data.shape[0])
		# print(len(totalCount))
		# print(len(totalCountl))

		no_attack_total_box = []

		for i in range(len(no_attack_results)):
			num_box_no_attack = int(no_attack_results['Bear'][i]) + int(no_attack_results['Outof'][i]) + int(no_attack_results['Guss'][i]) + int(no_attack_results['Cover'][i])
			no_attack_total_box.append({no_attack_results['Imagename'][i]:num_box_no_attack})


		# print(no_attack_total_box)
		# print(len(totalCountl))
		# print(totalCountl)

		new_line_indexes = []
		#the first one is at index 0
		new_line_indexes.append(0)
		cumsum = 0
		for elt in totalCountl:
			cumsum += elt
			new_line_indexes.append(cumsum)


		for i in range(len(no_attack_total_box)):
			os.system("rm -rf attack")
			os.system("mkdir attack")
			#os.system("mkdir chk")

			imagenumber = str(no_attack_total_box[i]).split("_E")[0].split("{'")[1]
			splitImagename = str(IMAGE_NAME).split("_E")[0]
			#quick test only, specify an image to test
			if (imagenumber!=splitImagename):
				continue
			#quick test only, specify a box to test
			new_line_indexes[i] = BOX_NUMBER

			#image = ndimage.imread(input_path)
			im = Image.open(eval_photo_path + imagenumber + '_E.jpg')
			pixelMap = im.load()
			img = Image.new(im.mode, im.size)
			# sequence_of_pixels = im.getdata()
			# list_of_pixels = list(sequence_of_pixels)
			

			filename  = imagenumber + '_E.jpg'
			
			l = new_line_indexes[i] 

			#skip the case for no box found in no-attack evaluation
			if ((no_attack_total_box[i].get(filename)) == 0):
				i = i + 1
				continue


			#print(filename + " ####outer loop")
			
			probsPrev = []
			probs = []
			probTarget = 1.0

			xTarget = 0
			yTarget = 0
			wincTarget =0
			hincTarget = 0
			wTarget =0
			hTarget = 0
			xmaxTarget = 0
			ymaxTarget = 0


			while (l <= BOX_NUMBER):
				print('\nground truth Box index', l)

				boxWidth = int(groundtruth_box_info['xmax'][l]) - int(groundtruth_box_info['xmin'][l])
				boxHeight = int(groundtruth_box_info['ymax'][l]) - int(groundtruth_box_info['ymin'][l])
				#print(winc,hinc,"debug checking")
				x = int(groundtruth_box_info['xmin'][l]) + (float(winc/100) * boxWidth)
				y = int(groundtruth_box_info['ymin'][l]) + (float(hinc/100) * boxHeight)
				w =  0.2 * boxWidth
				h =  0.2 * boxHeight
				# if(winc == 20):
				# 	print("success")
				xmax = x + w
				ymax = y + h
				#print(l,"Current Box",(groundtruth_box_info['filename'][l]),x,y,xmax,ymax,winc,hinc)
				for t in range(img.size[0]):
					for u in range(img.size[1]):
						if (t >= x and t <= xmax and u >= y and u <= ymax and t <= int(groundtruth_box_info['xmax'][l]) and u <= int(groundtruth_box_info['ymax'][l])):
							pixelMap[t,u] = (0,0,0)


				im.save('attack/'+ filename)
				
				box_total, probs = frcnn_eval()
				probs = truncate(probs,4)

				if(len(probsPrev)==0):
					probsPrev = probs

				print('prevProbs: ', probsPrev)
				print('Probs: ', probs)
				print('minimum prob found: ', probTarget)

				if(box_total < no_attack_total_box[i].get(filename)):

					print("attack success!!!!")
					# print(filename)
					# print(box_total)
					# print(no_attack_total_box[i].get(filename))
					writer.writerow({'Filename':filename, 'Box number':l, 'x':x, 'y': y, 'winc': winc, 'hinc': hinc, 'w':w, 'h':h, 'xmax':xmax, 'ymax':ymax, 'tot':box_total})
					#clean probability tracking when attack success
					probsPrev = []
					probs = []
					probTarget = 1.0
					if(l == len(groundtruth_box_info)-1):
						break
					if(filename != (groundtruth_box_info['filename'][l+1])):
						temp = totalCount[l]
						x = 0
						y = 0
						xmax = 0
						ymax = 0
						winc = 0
						hinc = 0
						break
					else:
						##once attack success shall move to the next box of this image
						l = l+1
						x = 0
						y = 0
						xmax = 0
						ymax = 0
						winc =0
						hinc = 0
						im = Image.open(eval_photo_path + imagenumber + '_E.jpg')
						pixelMap = im.load()
						img = Image.new(im.mode, im.size)
						sequence_of_pixels = im.getdata()
						list_of_pixels = list(sequence_of_pixels)

				#in this case, the attack is not success yet
				else:
					#track if box prob of the tracking one is reduced
					for item in probs:
						if (not (item in probsPrev) or probTarget ==1):
							if(float(item)<probTarget):
								probTarget = float(item)
								## Add code to log the stick coordinates info
								xTarget = x
								yTarget = y
								wincTarget =winc
								hincTarget = hinc
								wTarget =w
								hTarget = h
								xmaxTarget = xmax
								ymaxTarget = ymax
								break

					probsPrev = probs

					if (winc == 100 and hinc == 100):
						#move to the next box in the same image or next image, clean probability tracking info
						##add code to save coordinate info for best attack in this round, even not success
						print(filename, l, "best attack is:", probTarget)
						writerForAddAttack.writerow({'Filename':filename, 'Box number':l, 'x':xTarget, 'y': yTarget, 'winc': wincTarget, 'hinc': hincTarget, 'w':wTarget, 'h':hTarget, 'xmax':xmaxTarget, 'ymax':ymaxTarget, 'probability':probTarget})
						probsPrev = []
						probs = []
						probTarget = 1.0
						##shall add code to start round 2 attack here may be
						## we can also do it in another program

						if(l==245):
							break
						if(filename != (groundtruth_box_info['filename'][l+1])):
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
							im = Image.open(eval_photo_path + imagenumber + '_E.jpg')
							pixelMap = im.load()
							img = Image.new(im.mode, im.size)
							sequence_of_pixels = im.getdata()
							list_of_pixels = list(sequence_of_pixels)
							filename = (groundtruth_box_info['filename'][l])
							continue
				
					if (hinc == 100):
						winc = winc + 10
						hinc = 0
						x = 0
						y = 0
						xmax = 0
						ymax = 0
						#print("conti check2")
						im = Image.open(eval_photo_path + imagenumber + '_E.jpg')
						pixelMap = im.load()
						img = Image.new(im.mode, im.size)
						sequence_of_pixels = im.getdata()
						list_of_pixels = list(sequence_of_pixels)
						filename = (groundtruth_box_info['filename'][l])
						continue

					hinc = hinc + 10
					x = 0
					y = 0
					xmax = 0
					ymax = 0
					#print("conti check1")
					im = Image.open(eval_photo_path + imagenumber + '_E.jpg')
					pixelMap = im.load()
					img = Image.new(im.mode, im.size)
					sequence_of_pixels = im.getdata()
					list_of_pixels = list(sequence_of_pixels)
					filename = (groundtruth_box_info['filename'][l])


C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

img_path = options.test_path

detail2ndAttack = pd.read_csv('need2ndAttack_Original.csv')
detail2ndAttack.head()


data = pd.DataFrame()
data['format'] = detail2ndAttack['Filename']
initial = data['format'][0]

tempStr = ""
lInd = 0

groundtruth_box_info = pd.read_csv('Bbox_info_CSV_output_Evaluation.csv')
groundtruth_box_info.head()


while lInd < len(detail2ndAttack):
	os.system("rm -rf 2ndAttack")
	os.system("mkdir 2ndAttack")

	img_name = detail2ndAttack['Filename'][lInd]
	imagAct = '/Users/ashokvardhanraja/Documents/Bridge Data/COCO-Bridge/Original_COCO-Bridge_Dataset/Evaluation_Photos/' + img_name 

	im = Image.open(imagAct)
	pixelMap = im.load()
	img = Image.new(im.mode, im.size)

	x = int(detail2ndAttack['x'][lInd])
	y = int(detail2ndAttack['y'][lInd])
	w = int(detail2ndAttack['w'][lInd])
	h = int(detail2ndAttack['h'][lInd])
# if(winc == 20):
# 	print("success")
	xmax = int(detail2ndAttack['xmax'][lInd])
	ymax = int(detail2ndAttack['ymax'][lInd])
#print(l,"Current Box",(groundtruth_box_info['filename'][l]),x,y,xmax,ymax,winc,hinc)
	for t in range(img.size[0]):
		for u in range(img.size[1]):
			if (t >= x and t <= xmax and u >= y and u <= ymax and t <= int(groundtruth_box_info['xmax'][int(detail2ndAttack['Box number'][lInd])]) and u <= int(groundtruth_box_info['ymax'][int(detail2ndAttack['Box number'][lInd])])):
				pixelMap[t,u] = (0,0,0)

	im.save('2ndAttack/'+ img_name)
	performAttackSecondTime(img_name,int(detail2ndAttack['Box number'][lInd]))
	
	lInd = lInd+1


