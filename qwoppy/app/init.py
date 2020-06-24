import atexit
import logging
import os
import time

import numpy as np

from ..controller import SeleniumQwopController
from ..ocr        import make_ocr
from ..ui         import TkUiProvider
from ..agent      import Agent

def init_logging():
    level = logging.INFO
    if os.environ.get('DEBUG') in (1,'1'):
        level = logging.DEBUG
    logging.basicConfig(level=level, format='[%(asctime)s %(levelname)s] %(message)s')
    logging.info('Initializing...')

def init_ocr(modelName):
    ocr = make_ocr(modelName)
    logging.info('* OCR provider   [done]')
    return ocr

def init_ui(rect):
    ui = TkUiProvider()
    ui.set_position(rect['x'] + rect['width'], 0)
    ui.set_size(800, rect['height'])
    ui.update()
    logging.info('* User interface [done]')
    return ui

def init_controller():
    qwop = SeleniumQwopController()
    rect = qwop.browser.get_window_rect()
    logging.info('* Controller     [done]')
    return qwop, rect

def init_agent():
    agent = Agent()
    logging.info('* Agent          [done]')
    return agent

def init_cleanup(*args):
    def cleanup(things):
        logging.info('Cleaning up...')
        for thing in things:
            del thing
        logging.info('Goodbye!')
    atexit.register(lambda: cleanup(args))

def wait_for_init():
    return
    #logging.info('Waiting for initialization completion...')
    #time.sleep(3)
