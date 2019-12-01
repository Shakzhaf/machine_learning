import os
import numpy as np
from PIL import Image
import matplotlib.pylab as plt

path='C:/Users/hp/machine_learning/cnn_with_keras/train_data/'
train_file=open('train_file2.csv','a')
folders=os.listdir(path)

for label in range(1,29):
    train_file.write(str(label)+',')
train_file.write('label'+'\n')
x_train=[]
y_train=[]
x_test=[]
y_test=[]

for folder in folders:
    images_in_folder=os.listdir(path+folder)
    for image in images_in_folder:
        img=Image.open(path+folder+'/'+image)
        image_array=np.array(img)
        x_train.append(image_array)
        y_train.append(folder)
        
print('data_loaded')
print(len(x_train))
print(x_train[0].shape)
print(len(y_train))
print(y_train[255])
plt.imshow(x_train[255])
plt.show()