import atexit
import logging as log
import time

import numpy as np

from ..controller import SeleniumQwopController
from ..ocr        import PyTesseractOcrProvider
from ..ui         import TkUiProvider
from ..model      import Model

def init_logging():
    log.basicConfig(level=log.INFO, format='[%(asctime)s %(levelname)s] %(message)s')
    log.info('Initializing...')

def init_ocr():
    ocr = PyTesseractOcrProvider()
    log.info('* OCR provider   [done]')
    return ocr

def init_ui(rect):
    ui = TkUiProvider()
    ui.set_position(rect['x'] + rect['width'], 0)
    ui.set_size(800, rect['height'])
    ui.update()
    log.info('* User interface [done]')
    return ui

def init_controller():
    qwop = SeleniumQwopController()
    rect = qwop.browser.get_window_rect()
    log.info('* Controller     [done]')
    return qwop, rect

def init_model():
    nn = Model()
    log.info('* Model          [done]')
    return nn

def init_cleanup(*args):
    def cleanup(things):
        log.info('Cleaning up...')
        for thing in things:
            del thing
        log.info('Goodbye!')
    atexit.register(lambda: cleanup(args))

def wait_for_init():
    return
    #log.info('Waiting for initialization completion...')
    #time.sleep(3)
