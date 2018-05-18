# -*- coding: utf-8 -*-

import os

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Flatten
from keras.layers import Convolution1D, MaxPool1D, Convolution2D, MaxPool2D

from data_process import DataGenerator
from utils import *

import os

class Config(object):
    def __init__(self, **kwargs):
        self.sampling_rate = kwargs.get('sampling_rate', 16000)
        self.audio_duration = kwargs.get('audio_duration', 2)
        self.n_classes = kwargs.get('n_classes', 41)
        self.use_mfcc = kwargs.get('use_mfcc', False)
        self.n_mfcc = kwargs.get('n_mfcc', 20)
        self.use_folds = kwargs.get('use_folds', False)
        self.n_folds = kwargs.get('n_folds', 10)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.max_epochs = kwargs.get('max_epochs', 50)
        self.batch_size = kwargs.get('batch_size', 64)
        self.optimizer = kwargs.get('optimizer', 'adam')
        self.audio_pad_method = kwargs.get('audio_pad_method', 'constant')
        self.data_dir = kwargs.get('data_dir', '../data/')
        self.log_dir = kwargs.get('log_dir', '../logs/')
        self.tmp_dir = kwargs.get('tmp_dir', '../tmp/')
        self.model_name = kwargs.get('model_name', 'model_1')
        self.run_time = kwargs.get('run_time', 1)
        self.audio_length = self.sampling_rate * self.audio_duration
        if self.use_mfcc:
            self.dim = (self.n_mfcc, 1 + self.audio_length // 512, 1)
        else:
            self.dim = (self.audio_length, 1)

        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)
        if not os.path.exists(self.tmp_dir + self.model_name + '_%d/' % self.run_time):
            os.mkdir(self.tmp_dir + self.model_name + '_%d/' % self.run_time)


class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.model = self.__build_model()

    def __build_model(self):
        inp = Input(shape=(self.config.audio_length, 1))
        out = Dense(self.config.n_classes, activation=softmax)(inp)
        model = models.Model(inputs=inp, outputs=out)
        opt = optimizers.Adam(lr=self.config.learning_rate)
        model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        return model

    def fit(self, list_IDs, labels):
        train_data_dir = self.config.data_dir + 'audio_train/'
        if self.config.use_folds:
            history = []
            skf = StratifiedKFold(list_IDs, n_folds=self.config.n_folds)
            for i, (train_split, val_split) in enumerate(skf):
                train_list_IDs, val_list_IDs = list_IDs[train_split], list_IDs[val_split]
                train_labels, val_labels = labels[train_split], labels[val_split]
                checkpoint = ModelCheckpoint(self.config.tmp_dir + self.config.model_name
                                             + '_%d/best_%d.h5' % (self.config.run_time, i),
                                             monitor='val_loss',
                                             verbose=1,
                                             save_best_only=True)
                early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=10)
                tb = TensorBoard(log_dir=self.config.log_dir + self.config.model_name
                                         + '_%d/fold_%d' % (self.config.run_time, i),
                                 write_graph=True)
                callbacks_list = [checkpoint, early_stop, tb]

                print("Fold: ", i)
                train_generator = DataGenerator(self.config, train_data_dir, train_list_IDs, train_labels, batch_size=64,
                                                preprocessing_fn=audio_norm_min_max)
                val_generator = DataGenerator(self.config, train_data_dir, val_list_IDs, val_labels, batch_size=64,
                                              preprocessing_fn=audio_norm_min_max)
                res = self.model.fit_generator(train_generator,
                                                   callbacks=callbacks_list,
                                                   validation_data=val_generator,
                                                   epochs=self.config.max_epochs,
                                                   use_multiprocessing=True,
                                                   workers=6,
                                                   max_queue_size=20)
                history.append(res)
        else:
            train_list_IDs, val_list_IDs, train_labels, val_labels = train_test_split(list_IDs, labels, test_size=0.3)
            train_generator = DataGenerator(self.config, train_data_dir, train_list_IDs, train_labels, audio_norm_min_max)
            val_generator = DataGenerator(self.config, train_data_dir, val_list_IDs, val_labels, audio_norm_min_max)
            checkpoint = ModelCheckpoint(self.config.tmp_dir + self.config.model_name + '_%d/best.h5' % self.config.run_time,
                                         monitor='val_loss', verbose=1, save_best_only=True)
            early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=10)
            tb = TensorBoard(log_dir=self.config.log_dir + self.config.model_name + '_%d' % self.config.run_time, write_graph=True)
            callbacks_list = [checkpoint, early_stop, tb]
            history = self.model.fit_generator(train_generator,
                                               callbacks=callbacks_list,
                                               validation_data=val_generator,
                                               epochs=self.config.max_epochs,
                                               use_multiprocessing=True,
                                               workers=6,
                                               max_queue_size=20)
        return history

    def predict(self, list_IDs):
        test_data_dir = self.config.data_dir + 'audio_test/'
        test_generator = DataGenerator(self.config, test_data_dir, list_IDs, None, audio_norm_min_max)
        if self.config.use_folds:
            for i in range(self.config.n_folds):
                print('Fold: ', i)
                self.model.load_weights(self.config.tmp_dir + self.config.model_name
                                        + '_%d/best_%d.h5' % (self.config.run_time, i))
                predictions = self.model.predict_generator(test_generator,
                                                           use_multiprocessing=True,
                                                           workers=6,
                                                           max_queue_size=20,
                                                           verbose=1)
                np.save(self.config.tmp_dir + self.config.model_name + '_%d/pred_%d.npy' % (self.config.run_time, i),
                        predictions)
        else:
            self.model.load_weights(self.config.tmp_dir + self.config.model_name + '_%d/best.h5' % self.config.model_name)
            predictions = self.model.predict_generator(test_generator,
                                                       use_multiprocessing=True,
                                                       workers=6,
                                                       max_queue_size=20,
                                                       verbose=1)
            np.save(self.config.tmp_dir + self.config.model_name + '_%d/pred.npy' % self.config.run_time, predictions)

    def get_features(self, list_IDs, audio_path):
        audio_data_dir = self.config.data_dir + audio_path
        data_generator = DataGenerator(self.config, audio_data_dir, list_IDs, None, audio_norm_min_max)
        if audio_path.endswith('train/'):
            ss = 'train_features'
        else:
            ss = 'test_features'
        save_file = self.config.tmp_dir + self.config.model_name + '_%d/' % self.config.run_time
        if self.config.use_folds:
            for i in range(self.config.n_folds):
                print('Fold: ', i)
                self.model.load_weights(save_file + 'best_%d.h5' % i)
                feature_model = models.Model(inputs=self.model.input, outputs=self.model.layers[-3].output)
                extract_features = feature_model.predict_generator(data_generator,
                                                                   use_multiprocessing=True,
                                                                   workers=6,
                                                                   max_queue_size=20,
                                                                   verbose=1)

                np.save(save_file + ss + '_%d.npy' % i, extract_features)
        else:
            self.model.load_weights(save_file + 'best.h5')
            feature_model = models.Model(inputs=self.model.input, outputs=self.model.layers[-3].output)
            extract_features = feature_model.predict_generator(data_generator,
                                                               use_multiprocessing=True,
                                                               workers=6,
                                                               max_queue_size=20,
                                                               verbose=1)
            np.save(save_file + ss + '.npy', extract_features)


class Model1(BaseModel):
    # implement Conv1D
    def __init__(self, config):
        super(Model1, self).__init__(config)

    def __build_model(self):
        inp = Input(shape=(self.config.audio_length, 1))
        x = Convolution1D(16, 9, activation=relu, padding='same')(inp)
        x = Convolution1D(16, 9, activation=relu, padding='same')(x)
        x = MaxPool1D(16)(x)
        x = Dropout(0.25)(x)

        x = Convolution1D(64, 3, activation=relu, padding='same')(x)
        x = Convolution1D(64, 3, activation=relu, padding='same')(x)
        x = MaxPool1D(16)(x)
        x = Dropout(0.25)(x)

        x = Convolution1D(256, 3, activation=relu, padding='same')(x)
        x = Convolution1D(256, 3, activation=relu, padding='same')(x)
        x = MaxPool1D(16)(x)
        x = Flatten()(x)
        x = Dropout(0.25)(x)

        x = Dense(512, activation=relu)(x)
        x = Dropout(0.5)(x)
        out = Dense(self.config.n_classes, activation=softmax)(x)

        model = models.Model(inputs=inp, outputs=out)
        if self.config.optimizer == 'adam':
            opt = optimizers.Adam(lr=self.config.learning_rate)
        elif self.config.optimizer == 'rmsprop':
            opt = optimizers.RMSprop(lr=self.config.learning_rate)
        else:
            opt = optimizers.SGD(self.config.learning_rate)
        model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        return model


class Model2(BaseModel):
    # implement Conv2D
    def __init__(self, config):
        super(Model2, self).__init__(config)

    def __build_model(self):
        inp = Input(shape=(self.config.dim[0], self.config.dim[1], 1))
        x = BatchNormalization()(inp)

        x = Convolution2D(32, (4, 10), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPool2D()(x)
        x = Dropout(0.25)(x)

        x = Convolution2D(32, (4, 10), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPool2D()(x)
        x = Dropout(0.25)(x)

        x = Convolution2D(128, (4, 10), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPool2D()(x)
        x = Dropout(0.25)(x)

        x = Convolution2D(128, (4, 10), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPool2D()(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)
        x = Dense(256)(x)
        x = Dropout(0.5)(x)
        out = Dense(self.config.n_classes, activation=softmax)(x)

        model = models.Model(inputs=inp, outputs=out)
        if self.config.optimizer == 'adam':
            opt = optimizers.Adam(lr=self.config.learning_rate)
        elif self.config.optimizer == 'rmsprop':
            opt = optimizers.RMSprop(lr=self.config.learning_rate)
        else:
            opt = optimizers.SGD(self.config.learning_rate)
        model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        return model