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

import logging
import time

import numpy as np

def try_parse(distance_text):
    try:
        return float(distance_text)
    except ValueError:
        return None

def get_distance(qwop, ocr):
    image         = qwop.get_image()
    distance_text = ocr(image.crop(qwop.distance_rect))
    distance      = try_parse(distance_text)
    if distance is None:
        distance = distance_text
    return image, distance

def take_action(qwop, agent, distance):
    sequence = agent.generate_action(distance)
    if not hasattr(sequence, '__len__'):
        sequence = [sequence]
    for action in sequence:
        if action == 0:
            qwop.sendSpace()
            logging.info('Sent space')
        elif action == 1:
            qwop.click()
            logging.info('Sent click')
        elif action == 2:
            qwop.sendQ()
            logging.info('Sent Q')
        elif action == 3:
            qwop.sendW()
            logging.info('Sent W')
        elif action == 4:
            qwop.sendO()
            logging.info('Sent O')
        elif action == 5:
            qwop.sendP()
            logging.info('Sent P')

def update_ui(ui, image, distance, frames, start):
    ui.set_distance(distance)
    ui.set_image(image)
    ui.update()
    seconds = time.perf_counter() - start
    ui.set_title('Frame {} ({} fps, ~{} s)'.format(
        frames,
         np.round(frames/seconds, 2), # FPS
         np.round(seconds/frames, 2)  # Average frame time
    ))

def mainloop(qwop, ocr, ui, agent):
    logging.info('Starting the game...')
    frames = 1
    start  = time.perf_counter()
    while ui.is_open:
        image, distance = get_distance(qwop, ocr)
        if isinstance(distance, float):
            take_action(qwop, agent, distance)
        else:
            logging.warning('Skipping frame - got unparseable distance {}'.format(distance))
        update_ui(ui, image, distance, frames, start)
        frames += 1


