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

from   io import StringIO
import os 
import re
import torch
import unittest

from project.ocr import *

class TestOcr(unittest.TestCase):
    def test_to_sentence(self):
        indices1 = torch.randint(2, NUM_TOKENS, (10,)) # Don't include START or STOP
        sentence = to_sentence(indices1)
        indices2 = [TOKENS.index(c) for c in sentence]
        self.assertEqual(list(indices1), indices2)

    def test_to_indices(self):
        indices1 = torch.randint(2, NUM_TOKENS, (10,))
        sentence = [TOKENS[i.item()] for i in indices1]
        indices2 = to_indices(sentence)
        self.assertEqual(list(indices1), indices2)

    def test_to_one_hot(self):
        indices1 = torch.randint(2, NUM_TOKENS, (10,))
        sentence = [TOKENS[i.item()] for i in indices1]
        onehot   = to_one_hot(sentence)
        indices2 = [torch.argmax(oh).item() for oh in onehot]
        self.assertEqual(list(indices1), indices2)

    def test_multiple(self):
        sentence1 = '-3.14 metres'
        one_hot   = to_one_hot(sentence1)
        sentence2 = to_sentence(one_hot)
        self.assertEqual(sentence1, sentence2)

    def test_load_training_data(self):
        images, labels = load_training_data(basePath='../data')
        for X, y in zip(images, labels):
            self.assertTrue(hasattr(X, 'filename'))
            filename = os.path.basename(X.filename)
            y = to_sentence(torch.tensor(y)).replace(' ', '_')
            self.assertTrue(y.endswith('\n'))
            y = y[:-1]
            self.assertNotEqual(None,      re.match(r'-?[0-9]+(\.[0-9])?_metres\.png', filename))
            self.assertTrue(filename.startswith(y))
            self.assertEqual('L',          X.mode)
            self.assertEqual(IMAGE_HEIGHT, X.height)

    def test_preprocess_training_data(self):
        images, labels = load_training_data(basePath='../data')
        self.assertEqual(len(images), len(labels))
        ppImages, ppLabels = preprocess_training_data(images, labels, None)
        # ppImages has 3 dimensions - image width, number of samples and image height
        self.assertEqual(3,            len(ppImages.size()))
        self.assertEqual(IMAGE_WIDTH,  ppImages.size(0))
        self.assertEqual(IMAGE_HEIGHT, ppImages.size(2))
        # ppImages and ppLabels both have the same number of and sample count
        self.assertEqual(len(ppImages.size()), len(ppLabels.size()))
        self.assertEqual(ppImages.size(1),     ppLabels.size(1))
        # ppLabels has 3 dimensions -  output sequence length, number of samples and number of output classes
        self.assertEqual(3,               len(ppLabels.size()))
        self.assertEqual(SEQUENCE_LENGTH, ppLabels.size(0))
        self.assertEqual(NUM_TOKENS,      ppLabels.size(2))
        # Make sure image can be reassembled to match the originals and that the sentences and image filenames match
        for i, (image,width,height,filename) in enumerate(images):
            # Test that the image data can be reassembled into the original bytes
            for j in range(min(width, ppImages.size(0))):
                for k in range(min(height, ppImages.size(2))):
                    x1 = int(255*ppImages[j, i, k].item())
                    x2 = int(image[k*width + j])
                    self.assertEqual(x1, x2)
            # Test that the label matches the original filename when converted to a string
            y = to_sentence(ppLabels[:, i, :]).replace(' ', '_')
            self.assertTrue(y.endswith('\n'))
            y = y[:-1]
            self.assertTrue(os.path.basename(filename).startswith(y))

    def test_create_batches(self):
        images, labels = load_training_data(basePath='../data')
        self.assertEqual(len(images), len(labels))
        ppImages, ppLabels = preprocess_training_data(images, labels, None)
        for batch_x, batch_y in create_batches(ppImages, ppLabels, None, 2):
            for i in range(batch_x.size(1)):
                X = batch_x[:, i, :]
                y = batch_y[:, i, :]
                found  = False
                index  = 0
                s1, s2 = None, None
                for index in range(ppLabels.size(1)):
                    s1 = to_sentence(y)
                    s2 = to_sentence(ppLabels[:, index, :])
                    if s1 == s2:
                        found = True
                        break
                self.assertTrue(found)
                _, width, height, __ = images[index]
                for j in range(min(width, ppImages.size(0))):
                    for k in range(min(height, ppImages.size(2))):
                        self.assertEqual(ppImages[j, index, k], X[j, k])

if __name__ == '__main__':
    unittest.main()
