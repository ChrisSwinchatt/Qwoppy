#!/usr/bin/env python3

'''
This file is part of Qwoppy.

Copyright (C) 2020 Chris Swinchatt

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
'''

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
