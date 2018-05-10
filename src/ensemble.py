# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from configs import *
from models import *

if __name__ == '__main__':
    config = Config(**config1)
    train_file = pd.read_csv('../data/train.csv')
    train_data = np.load(config.tmp_dir + config.model_name + '/train_feature_%d.npy' % config.run_time)
    test_data = np.load(config.tmp_dir + config.model_name + '/test_feature_%d.npy' % config.run_time)

    label_names = list(train_file.label.unique())
    label_idx = {label: i for i, label in enumerate(label_names)}
    train_file.set_index('fname', inplace=True)
    train_file['label_idx'] = train_file.label.apply(lambda x: label_idx[x])
    sample_weight = train_file['manually_verified'].values * 0.3 + 0.7

    train_set, val_set, train_label, val_label, train_weight, val_weight = train_test_split(train_data,
                                                                                            train_file.label_idx.values,
                                                                                            sample_weight,
                                                                                            test_size=0.3)
    print(train_set.shape)
    print(val_set.shape)
    print(train_label.shape)
    print(val_label.shape)

    ensemble_model = XGBClassifier(max_depth=8, n_estimators=100)
    ensemble_model.fit(train_set, sample_weight=train_weight)

    val_pred = ensemble_model.predict(val_set)
    print(accuracy_score(val_pred, val_label))

    test_prob = ensemble_model.predict_proba(test_data)
    top_3 = np.array(label_names)[np.argsort(-test_prob, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(x)) for x in top_3]
    test_file = pd.read_csv('../data/sample_submission.csv')
    test_file['label'] = predicted_labels
    test_file[['fname', 'label']].to_csv('../submission/sub_180510_01.csv', index=False)
