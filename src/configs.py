# -*- coding: utf-8 -*-

config1 = {
        'sampling_rate': 16000,
        'audio_duration': 2,
        'batch_size': 64,
        'n_classes': 41,
        'use_generator': True, 
        'use_mfcc': False,
        'n_mfcc': 20,
        'use_folds': False,
        'n_folds': 10,
        'learning_rate': 0.001,
        'max_epochs': 50,
        'optimizer': 'adam',
        'audio_pad_method': 'constant',
        'data_dir': '../data/',
        'log_dir': '../logs/',
        'tmp_dir': '../tmp/',
        'model_name': 'model_1',
        'run_time': 1
}

