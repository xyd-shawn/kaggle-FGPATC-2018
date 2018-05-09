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
    res = model.fit(train_file.index, train_file.label_idx)
    predictions = model.predict(test_file.index)
    top_3 = np.array(label_names)[np.argsort(-predictions, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(x)) for x in top_3]
    test_file = pd.read_csv('../data/sample_submission.csv')
    test_file['label'] = predicted_labels
    test_file[['fname', 'label']].to_csv('../submission/sub_180509_01.csv', index=False)