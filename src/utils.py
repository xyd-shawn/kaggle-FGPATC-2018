# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def audio_norm_min_max(data):
    # normalization: [-0.5, 0.5]
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data - min_data) / (max_data - min_data + 1e-6)
    return data - 0.5


def generate_submission_file(predictions, label_names, to_file_name):
    top_3 = np.array(label_names)[np.argsort(-predictions, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(x)) for x in top_3]
    test_file = pd.read_csv('../data/sample_submission.csv')
    test_file['label'] = predicted_labels
    test_file[['fname', 'label']].to_csv(to_file_name, index=False)
