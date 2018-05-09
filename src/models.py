# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Flatten
from keras.layers import Convolution1D, MaxPool1D, Convolution2D, MaxPool2D

from data_process import DataGenerator
from utils import *


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
        self.tmp_dir = kwargs.get('tmp_dir', '../tmp')
        self.model_name = kwargs.get('model_name', 'model_1')
        self.run_time = kwargs.get('run_time', 1)
        self.audio_length = self.sampling_rate * self.audio_duration
        if self.use_mfcc:
            self.dim = (self.n_mfcc, 1 + self.audio_length // 512, 1)
        else:
            self.dim = (self.audio_length, 1)


class Model1(object):
    # implement Conv1D
    def __init__(self, config):
        self.config = config
        self.model = self.__build_model()

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
        x = Flatten()(x)
        x = Dropout(0.25)(x)

        x = Dense(1024, activation=relu)(x)
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

    def fit(self, list_IDs, labels):
        train_list_IDs, val_list_IDs, train_labels, val_labels = train_test_split(list_IDs, labels, test_size=0.3)
        train_data_dir = self.config.data_dir + 'audio_train/'
        train_generator = DataGenerator(self.config, train_data_dir, train_list_IDs, train_labels, audio_norm_min_max)
        val_generator = DataGenerator(self.config, train_data_dir, val_list_IDs, val_labels, audio_norm_min_max)
        checkpoint = ModelCheckpoint(self.config.tmp_dir + 'model_1/' + 'best_%d.h5' % self.config.run_time,
                                     monitor='val_loss', verbose=1, save_best_only=True)
        early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5)
        tb = TensorBoard(log_dir=self.config.log_dir + 'model_1', write_graph=True)
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
        self.model.load_weights(self.config.tmp_dir + 'model_1/' + 'best_%d.h5' % self.config.run_time)
        predictions = self.model.predict_generator(test_generator,
                                                   use_multiprocessing=True,
                                                   workers=6,
                                                   max_queue_size=20,
                                                   verbose=1)
        return predictions

    def get_features(self, list_IDs, audio_path):
        audio_data_dir = self.config.data_dir + audio_path
        feature_model = models.Model(inputs=self.model.input, outputs=self.model.layers[-2].output)
        data_generator = DataGenerator(self.config, audio_data_dir, list_IDs, None, audio_norm_min_max)
        self.model.load_weights(self.config.tmp_dir + 'model_1/' + 'best_%d.h5' % self.config.run_time)
        extract_features = feature_model.predict_generator(data_generator,
                                                           user_multiprocessing=True,
                                                           workers=6,
                                                           max_queue_size=20,
                                                           verbose=1)
        return extract_features


class Model2(object):
    # implement Conv2D
    def __init__(self, config):
        self.config = config
        self.model = self.__build_model()

    def __build_model(self):
        inp = Input(shape=(self.config.dim[0], self.config.dim[1], 1))
        x = Convolution2D(32, (3, 3), padding='same')(inp)
        x = Convolution2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPool2D()(x)
        x = Dropout(0.25)(x)

        x = Convolution2D(64, (3, 3), padding='same')(x)
        x = Convolution2D(64, (3, 3), padding='same')(x)
        x = MaxPool2D()(x)
        x = Dropout(0.25)(x)

        x = Convolution2D(128, (3, 3), padding='same')(x)
        x = Convolution2D(128, (3, 3), padding='same')(x)
        x = MaxPool2D()(x)
        x = Dropout(0.25)(x)

        x = Convolution2D(256, (3, 3), padding='same')(x)
        x = Convolution2D(256, (3, 3), padding='same')(x)
        x = MaxPool2D()(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)
        x = Dense(1024)(x)
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

    def fit(self, list_IDs, labels):
        train_list_IDs, val_list_IDs, train_labels, val_labels = train_test_split(list_IDs, labels, test_size=0.3)
        train_data_dir = self.config.data_dir + 'audio_train/'
        train_generator = DataGenerator(self.config, train_data_dir, train_list_IDs, train_labels, audio_norm_min_max)
        val_generator = DataGenerator(self.config, train_data_dir, val_list_IDs, val_labels, audio_norm_min_max)
        checkpoint = ModelCheckpoint(self.config.tmp_dir + 'model_2/' 'best_%d.h5' % self.config.run_time,
                                     monitor='val_loss', verbose=1, save_best_only=True)
        early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5)
        tb = TensorBoard(log_dir=self.config.log_dir + 'model_2', write_graph=True)
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
        self.model.load_weights(self.config.tmp_dir + 'model_2/' + 'best_%d.h5' % self.config.run_time)
        predictions = self.model.predict_generator(test_generator,
                                                   use_multiprocessing=True,
                                                   workers=6,
                                                   max_queue_size=20,
                                                   verbose=1)
        return predictions

    def get_features(self, list_IDs, audio_path):
        audio_data_dir = self.config.data_dir + audio_path
        feature_model = models.Model(inputs=self.model.input, outputs=self.model.layers[-2].output)
        data_generator = DataGenerator(self.config, audio_data_dir, list_IDs, None, audio_norm_min_max)
        self.model.load_weights(self.config.tmp_dir + 'model_2/' + 'best_%d.h5' % self.config.run_time)
        extract_features = feature_model.predict_generator(data_generator,
                                                           user_multiprocessing=True,
                                                           workers=6,
                                                           max_queue_size=20,
                                                           verbose=1)
        return extract_features
