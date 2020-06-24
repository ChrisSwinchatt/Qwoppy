#!/usr/bin/env python3

import os
import re
import sys

from PIL import Image

import numpy as np

if __name__ == '__main__':
    split = None
    try:
        split = float(sys.argv[1])
    except:
        print(
            'Usage: %s <train/test split>\nFor example, say `%s 0.8` to put 80%% of sentences into training set and 20%% into test set'
            % (sys.argv[0], sys.argv[0])
        )
        exit(1)

    path      = os.path.join('data', 'images', 'sentences')
    regex     = re.compile(r'(.*)_metres.png')
    sentences = list(map(lambda s: regex.sub(r'\1', s), os.listdir(path)))
    with open(os.path.join('data', 'train.csv'), 'w') as train:
        with open(os.path.join('data', 'test.csv'), 'w') as test:
            for i, sentence in enumerate(sentences):
                filename = sentence + '_metres.png'
                if np.random.uniform() > 0.8:
                    test.write(filename + '\n')
                else:
                    train.write('{},{}\n'.format(filename, sentence))
