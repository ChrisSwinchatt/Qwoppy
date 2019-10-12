import logging as log
import time

import numpy as np

from ..controller import SeleniumQwopController
from ..ocr        import PyTesseractOcrProvider
from ..ui         import TkUiProvider
from ..model      import Model

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

def take_action(qwop, nn, distance):
    sequence = nn.generate_action(distance)
    if not hasattr(sequence, '__len__'):
        sequence = [sequence]
    for action in sequence:
        if action == 0:
            qwop.sendSpace()
            log.info('Sent space')
        elif action == 1:
            qwop.click()
            log.info('Sent click')
        elif action == 2:
            qwop.sendQ()
            log.info('Sent Q')
        elif action == 3:
            qwop.sendW()
            log.info('Sent W')
        elif action == 4:
            qwop.sendO()
            log.info('Sent O')
        elif action == 5:
            qwop.sendP()
            log.info('Sent P')

def update_ui(ui, image, distance, frames, start):
    ui.set_distance(distance)
    ui.set_image(image)
    ui.update()
    seconds = time.process_time() - start
    ui.set_title('Frame {} ({} fps, ~{} s)'.format(
        frames,
         np.round(frames/seconds, 2), # FPS
         np.round(seconds/frames, 2)  # Average frame time
    ))

def mainloop(qwop, ocr, ui, nn):
    log.info('Starting the game...')
    frames = 1
    start  = time.process_time()
    while ui.is_open:
        image, distance = get_distance(qwop, ocr)
        if isinstance(distance, float):
            take_action(qwop, nn, distance)
        else:
            log.warning('Skipping frame - got unparseable distance {}'.format(distance))
        update_ui(ui, image, distance, frames, start)
        frames += 1
