# importing required libraries
import pandas as pd
import csv
import matplotlib.pyplot as plt
#%matplotlib inline
from matplotlib import patches
# read the csv file using read_csv function of pandas
train = pd.read_csv('Bbox_info_CSV_Evaluation.csv')
train.head()
# reading single image using imread function of matplotlib
data = pd.DataFrame()
data['format'] = train['filename']
initial = data['format'][0]

Bear = 0
Outof = 0
Guss = 0
Cover = 0
with open('names2.csv', 'w', newline='') as csvfile:
    fieldnames = ['Imagename', 'Bear', 'Outof', 'Guss','Cover']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(data.shape[0]):
        if (initial == data['format'][i]):
            if (train['class'][i] == 'Bearing'):
                Bear = Bear + 1
            elif(train['class'][i] == 'Out of Plane Stiffener'):
                Outof = Outof + 1
            elif(train['class'][i] == 'Gusset Plate Connection'):
                Guss = Guss + 1
            else:
                Cover = Cover + 1
        else:
            writer.writerow({'Imagename':initial, 'Bear': Bear, 'Outof': Outof, 'Guss': Guss,'Cover':Cover})
            Bear = 0
            Outof = 0
            Guss = 0
            Cover = 0
            initial =data['format'][i]
            if (train['class'][i] == 'Bearing'):
                Bear = Bear + 1
            elif(train['class'][i] == 'Out of Plane Stiffener'):
                Outof = Outof + 1
            elif(train['class'][i] == 'Gusset Plate Connection'):
                Guss = Guss + 1
            else:
                Cover = Cover + 1
    writer.writerow({'Imagename':initial, 'Bear': Bear, 'Outof': Outof, 'Guss': Guss,'Cover':Cover})
#    data['format'][i] = data['format'][i] + ',' + str(train['xmin'][i]) + ',' + str(train['ymin'][i]) + ',' + str(train['xmax'][i]) + ',' + str(train['ymax'][i]) + ',' + train['class'][i]
#
#data.to_csv('annotate9.txt', header=None, index=None, sep=' ')
