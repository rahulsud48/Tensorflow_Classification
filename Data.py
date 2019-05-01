import os
import sys
import matplotlib.pyplot as plt
from skimage.color import rgb2lab
from skimage.transform import resize
from collections import namedtuple
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm, tqdm_notebook
np.random.seed(101)



class images:

    def __init__(self,path):
        # Defining the some constants
        self.n_classes = 43
        self.image_size = (32,32)
        self.path = path
        self.image_label = namedtuple('image_label', ['X','y'])
        return None

    def format_image(self,images):
        array_ = np.stack([img[:,:,np.newaxis] for img in images], axis = 0).astype(np.float32)
        return array_

    def read_images(self):
        images_ = []
        class_ = []
        for n in tqdm(range(self.n_classes)):
            path_ = self.path+'/'+format(n,'05d')+'/'
            files_list = os.listdir(path_)
            file_list_ = []
            for i in files_list:
                if i[-4:] == '.ppm':
                    file_list_.append(i)
            for file_name in file_list_:
                img_ = plt.imread(path_+'/'+file_name)
                img_ = rgb2lab(img_/255.0)[:,:,0]
                img_ = resize(img_, self.image_size, mode = 'reflect')
                label_ = np.zeros((self.n_classes,))
                label_[n] = 1
                images_.append(img_)
                class_.append(label_)
        images_ = self.format_image(images_)
        class_ = np.matrix(class_) 
        return self.image_label(X = images_, y = class_)


class Preprocessing:

    def __init__(self,dataset_tuple):
        self.dataset = dataset_tuple
        return None

    def train_test(self):
        idx_train, idx_test = train_test_split(range(self.dataset.X.shape[0]), test_size = 0.25, random_state= 101)
        X_train = self.dataset.X[idx_train,:,:,:]
        X_test = self.dataset.X[idx_test,:,:,:]
        y_train = self.dataset.y[idx_train,:]
        y_test = self.dataset.y[idx_test,:]
        return X_train,X_test,y_train,y_test


