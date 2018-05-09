# -*- coding: utf-8 -*-

import pandas as pd

from configs import *
from models import Config, ModelConv1D


if __name__ == '__main__':
    train_file = pd.read_csv('../data/train.csv')
    test_file = pd.read_csv('../data/sample_submission.csv')

    label_names = list(train_file.label.unique())
    print('label_names')
    print(label_names)
    label_idx = {label: i for i, label in enumerate(label_names)}
    train_file.set_index("fname", inplace=True)
    test_file.set_index("fname", inplace=True)
    train_file["label_idx"] = train_file.label.apply(lambda x: label_idx[x])

    config = Config(**config1)
    model_conv1d = ModelConv1D(config)
    res = model_conv1d.fit(train_file.index, train_file.label_idx)