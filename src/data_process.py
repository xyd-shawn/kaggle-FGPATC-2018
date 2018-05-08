# -*- coding: utf-8 -*-


import librosa
import numpy as np
from keras.utils import Sequence, to_categorical


class DataGenerator(Sequence):
    def __init__(self, config, data_dir, list_IDs, labels=None, preprocessing_fn=lambda x: x):
        self.config = config
        self.data_dir = data_dir
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = self.config.batch_size
        self.dim = self.config.dim
        self.preprocessing_fn = preprocessing_fn
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        cur_list_IDs = [self.list_IDs[k] for k in indexes]
        return self.__data_generation(cur_list_IDs)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))

    def __data_generation(self, cur_list_IDs):
        cur_batch_size = len(cur_list_IDs)
        X = np.empty((cur_batch_size, *self.dim))
        config_audio_length = self.config.audio_length

        for i, ID in enumerate(cur_list_IDs):
            file_path = self.data_dir + ID

            # Read and Resample the audio
            data, _ = librosa.core.load(file_path, sr=self.config.sampling_rate, res_type='kaiser_fast')

            # Random offset / Cutting and Padding
            if len(data) != config_audio_length:
                max_offset = abs(len(data) - config_audio_length)
                offset = np.random.randint(max_offset)
                if len(data) > config_audio_length:
                    data = data[offset:(config_audio_length + offset)]
                else:
                    data = np.pad(data, (offset, config_audio_length - len(data) - offset), self.config.audio_pad_method)

            # Normalization + Other Preprocessing
            if self.config.use_mfcc:
                data = librosa.feature.mfcc(data, sr=self.config.sampling_rate,
                                            n_mfcc=self.config.n_mfcc)
                data = np.expand_dims(data, axis=-1)
            else:
                data = self.preprocessing_fn(data)[:, np.newaxis]
            X[i, :] = data

        if self.labels is not None:
            y = np.empty(cur_batch_size, dtype=int)
            for i, ID in enumerate(cur_list_IDs):
                y[i] = self.labels[ID]
            y = to_categorical(y, num_classes=self.config.n_classes)
            return X, y
        else:
            return X
