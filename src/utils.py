# -*- coding: utf-8 -*-

import numpy as np


def audio_norm_min_max(data):
    # normalization: [-0.5, 0.5]
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data - min_data) / (max_data - min_data + 1e-6)
    return data - 0.5
