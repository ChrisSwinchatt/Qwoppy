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

import numpy as np

from PIL import Image

def gen_sentence(inputs):
    inputs += ['_', 'm', 'e', 't', 'r', 'e', 's']
    images = [None]*len(inputs)
    totalX = 0
    maxY   = 0
    for i in range(len(images)):
        name      = '{}.png'.format(inputs[i])
        images[i] = Image.open(os.path.join('data', 'images', 'tiles', name))
        x, y    = images[i].size
        totalX += x
        maxY    = y if y > maxY else maxY
    output  = Image.new('1', (totalX, maxY))
    xOffset = 0
    for image in images:
        output.paste(image, (xOffset, 0))
        x, y = image.size
        xOffset += x
    name = '{}.png'.format(''.join(inputs))
    output.save(os.path.join('data', 'images', 'sentences', name))
    return totalX, maxY

def update(x, y, minX, minY, maxX, maxY):
    if x < minX:
        minX = x
    if x > maxX:
        maxX = x
    if y < minY:
        minY = y
    if y > maxY:
        maxY = y
    return minX, minY, maxX, maxY

if __name__ == '__main__':
    minI, maxI = 0, 101
    minJ, maxJ = 1, 10
    maxCount   = 2*(maxI - minI) + 2*(maxI - minI)*(maxJ - minJ)
    minX, minY, maxX, maxY = np.inf, np.inf, -np.inf, -np.inf
    count = 0
    for i in range(minI, maxI):
        x, y = gen_sentence(list(str(i)))
        minX, minY, maxX, maxY = update(x, y, minX, minY, maxX, maxY)
        x, y = gen_sentence(list(str(-i)))
        minX, minY, maxX, maxY = update(x, y, minX, minY, maxX, maxY)
        count += 2
        for j in range(minJ, maxJ):
            x, y = gen_sentence(list(str(i) + '.' + str(j)))
            minX, minY, maxX, maxY = update(x, y, minX, minY, maxX, maxY)
            x, y = gen_sentence(list(str(-i) + '.' + str(j)))
            minX, minY, maxX, maxY = update(x, y, minX, minY, maxX, maxY)
            count += 2
            if count % 100 == 0:
                print('Generated {}/{} ({}%) images ranging from {}x{} to {}x{}'.format(count, maxCount, int(100*count/maxCount), minX, minY, maxX, maxY))
    print('Generated {}/{} ({}%) images ranging from {}x{} to {}x{}'.format(count, maxCount, int(100*count/maxCount), minX, minY, maxX, maxY))
