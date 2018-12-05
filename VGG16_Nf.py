# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 17:24:11 2018

@author: Weiyu_Lee
"""
import os
os.environ['KERAS_BACKEND']='tensorflow'

from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras import regularizers
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint 
from keras.callbacks import TensorBoard

import numpy as np
import random
from sklearn.metrics import f1_score, precision_score, recall_score

batch_size = 32
learning_rate = 1e-4
epoch = 500
Nf_input_root = "/data/wei/dataset/MDetection/Stem_cell/"
ckpt_root = "/data/wei/model/Stem_cell/"
tb_root = ckpt_root + 'log'
MODEL = 'ResNet50'

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(
            self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        
        print("[P, R, F1] = [{}, {}, {}]".format(_val_precision, _val_recall, _val_f1))
        
        return

def load_data(data_root, balance=False):
    
    if not os.path.exists(os.path.join(data_root, "normal_data.npy")):
        print(data_root + "normal_data.npy not exist!")
        return    

    print("Loading data [{}]...".format(data_root))
    f = open(os.path.join(data_root, "normal_data_.npy"), "rb")
    n_dict = np.load(f).item()
    f = open(os.path.join(data_root, "abnormal_data_.npy"), "rb")
    ab_dict = np.load(f).item()   

    if balance == True:
        n_dict['data'] = random.sample(n_dict['data'], len(ab_dict['data']))
        n_dict['label'] = random.sample(n_dict['label'], len(ab_dict['label']))

    print("Normal image: {}".format(np.array(n_dict['data']).shape))
    print("Abnormal image: {}".format(np.array(ab_dict['data']).shape))

    return n_dict['data'], n_dict['label'], ab_dict['data'], ab_dict['label']

#==============================================================================
# Load data
train_n_image, train_n_label, train_ab_image, train_ab_label = load_data(Nf_input_root + 'train_data')
test_n_image, test_n_label, test_ab_image, test_ab_label = load_data(Nf_input_root + 'test_data')

x_train = np.concatenate((train_n_image, train_ab_image)) / 255
y_train = np.concatenate((train_n_label, train_ab_label))
x_test = np.concatenate((test_n_image, test_ab_image)) / 255
y_test = np.concatenate((test_n_label, test_ab_label))

print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_test : ", x_test.shape)
print("y_test : ", y_test.shape)

# Setting the model
img_rows, img_cols, img_channel = x_train.shape[1], x_train.shape[2], x_train.shape[3]
if img_channel == 1:
    x_train = np.concatenate((x_train, x_train, x_train), axis=-1)
    x_test = np.concatenate((x_test, x_test, x_test), axis=-1)
    img_channel = 3
    print("==> x_train: ", x_train.shape)
    print("==> x_test : ", x_test.shape)
    
# Build model
if MODEL == 'VGG16':
    # VGG16    
    #==============================================================================
    model = Sequential()

    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channel))
    
    for i in range(0,19):
       model.add(base_model.layers[i])
    for i in range(0,18):
       model.get_layer(index=i).kernel_regularizer=regularizers.l2(0.01) 

    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))

    #==============================================================================
elif MODEL == 'ResNet50':
    # ResNet50
    #==============================================================================   
    base_model = applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(img_rows, img_cols, img_channel), pooling=None)
    
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    #==============================================================================
   
lr_decay = learning_rate / epoch

#model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=learning_rate, momentum=0.9), metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=lr_decay, amsgrad=False), metrics=['accuracy'])

model.summary()

metrics = Metrics()
ckptCallBack = ModelCheckpoint(ckpt_root + "weights.{epoch:02d}-{val_loss:.4f}.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
tbCallBack = TensorBoard(log_dir=tb_root, histogram_freq=0, write_graph=True, write_images=True)

model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, validation_data=(x_test, y_test), callbacks=[metrics, ckptCallBack, tbCallBack])
