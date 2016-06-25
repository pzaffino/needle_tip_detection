
# coding: utf-8

# In[1]:

from __future__ import division
import joblib
import glob
import os
import numpy as np
import nrrd
import numpy as np
from sklearn import datasets, svm, metrics, decomposition
from sklearn.externals import joblib
import time
from joblib import Parallel, delayed  
import multiprocessing
num_cores = multiprocessing.cpu_count()
USERPATH = os.path.expanduser("~")
print(USERPATH)
import six.moves.cPickle as pickle

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

tb = TensorBoard(log_dir='/tmp/tensorboard', histogram_freq=1, write_graph=True)

# In[2]:

# tipsPath = "/Projects/Tips/"
# notipsPath = "/Projects/RandomCubes/"
notipsPath = "/Dropbox/2016-projectweekdata/RandomCubes/"
tipsPath = "/Dropbox/2016-projectweekdata/Tips/"

# fix random seed for reproducibility
seed = 7
checkpointer = ModelCheckpoint(filepath="weights2d.hdf5", verbose=1, save_best_only=True)


def loadAllDataFromPath(path):
    # path in directorty
    cubeTipsPath = glob.glob(USERPATH + path + "*.nrrd")
    # number of samples
    N = len(cubeTipsPath)
    print('number of sample %d' %N)
    cubeTips = []
    for path_i in cubeTipsPath:
        cubeTips.append(nrrd.read(path_i))
    data = [[] for i in range(N)]
    for i in range(N):
        # c = np.array(cubeTips[i][0])  # for patches of size 20,20,20
        c = np.array(cubeTips[i][0][5:-5,5:-5,5:-5]) # for patches of size 10,10,10
        data[i] = c

    output = np.array(data)
    return output

####
## The data is saved to numpy array to speed-up the loading. Uncomment lines below to create a new dataset
####
# In[4]:
# #
# tips = loadAllDataFromPath(tipsPath)
# notips = loadAllDataFromPath(notipsPath)[:len(tips)]
#
# target_0 = [0 for i in range(len(notips))]
# target_1 = [1 for i in range(len(tips))]
# y_train = np.array(target_0 + target_1)
# print('target shape:', y_train.shape)
# X_train = np.array(list(notips)+list(tips))
#
# print('data shape:', X_train.shape)
#
#
# # now, we have **data**: 2D array of randomCubes then tips and **target** 2D array of 0, then 1
#
# # In[5]:

# f_Xtrain = open('X_data_n3.save', 'wb')
# f_ytrain = open('y_data_n3.save', 'wb')
#
# pickle.dump(X_train, f_Xtrain, protocol=pickle.HIGHEST_PROTOCOL)
# pickle.dump(y_train, f_ytrain, protocol=pickle.HIGHEST_PROTOCOL)
#
# f_Xtrain.close()
# f_ytrain.close()


# In[6]:

# Load the dataset
f_Xdata = open('X_data_n3.save', 'rb')
f_ydata = open('y_data_n3.save', 'rb')

X_data = pickle.load(f_Xdata)
X_data = X_data.astype('float32')

# normalize the raw data
X_data -= np.mean(X_data)
X_data /= np.std(X_data)

## second method for normalization
# X_data /= 255

y_data= pickle.load(f_ydata)
y_data_binary = to_categorical(y_data)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y_data)
y_data = encoder.transform(y_data)

print("Data shape and label shape")
print(X_data.shape, y_data.shape)

# In[7]:

def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def simple_test(y):
    x=[]
    for yi in y:
        if yi:
            x.append(np.ones(64000))
        else: x.append(np.zeros(64000))
    return np.array(x)

# init the global var
model = 0

def create_baseline():

    nb_classes = 1

    # create model
    global model
    model = Sequential()

    model.add(Convolution2D(10, 10, 2, border_mode='same',
                            input_shape=(10,10,10)))
    model.add(Activation('relu'))
    model.add(Convolution2D(10, 3, 3))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Convolution2D(40, 5, 3, border_mode='same' ))
    model.add(Activation('relu'))
    model.add(Convolution2D(40, 5, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(40, 5, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(480))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(480))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

# np.random.seed(seed)
estimators = []
# estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, nb_epoch=200,
                                          batch_size=64, verbose=1)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(y=y_data, n_folds=4, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X_data, y_data, cv=kfold)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

json_string = model.to_json()
x=8
model.save_weights('my_model_weights_2d_%d.h5'%x, overwrite=True)
open('my_model_architecture%d.json'%x, 'w').write(json_string)

