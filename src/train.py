# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from configs import *
from models import *


if __name__ == '__main__':
    train_file = pd.read_csv('../data/train.csv')
    test_file = pd.read_csv('../data/sample_submission.csv')

    label_names = list(train_file.label.unique())
    print('label_names')
    print(label_names)
    label_idx = {label: i for i, label in enumerate(label_names)}
    train_file.set_index('fname', inplace=True)
    test_file.set_index('fname', inplace=True)
    train_file['label_idx'] = train_file.label.apply(lambda x: label_idx[x])

    config = Config(**config1)
    model = Model1(config)
    _ = model.fit(train_file.index, train_file.label_idx)
    model.predict(test_file.index)
    model.get_features(train_file.index, 'audio_train/')
    model.get_features(test_file.index, 'audio_test/')

