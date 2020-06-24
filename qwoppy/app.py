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

import argparse
import atexit
import logging
import os
import time

import numpy      as np
import torch.nn   as nn

import qwoppy.ml   as     ml
import qwoppy.ocr  as     ocr

from   .agent      import Agent
from   .controller import SeleniumQwopController
from   .ml         import get_device
from   .ocr        import train_ocr, test_ocr, make_ocr
from   .ui         import TkUiProvider

class Qwoppy:
    def __init__(self, modelName):
        device = get_device()
        logging.info('* ML provider    [done]')

        self.ocr = make_ocr(device, modelName)
        logging.info('* OCR provider   [done]')

        self.agent = Agent(device)
        logging.info('* Agent          [done]')

        self.qwop = SeleniumQwopController()
        self.rect = self.qwop.browser.get_window_rect()

        logging.info('* Controller     [done]')
        self.ui = TkUiProvider()
        self.ui.set_position(self.rect['x'] + self.rect['width'], 0)
        self.ui.update()
        logging.info('* User interface [done]')

        self.frames     = 1
        self.start_time = time.perf_counter()

    def try_parse(self, distance_text):
        try:
            return float(distance_text)
        except ValueError:
            return None

    def get_distance(self):
        image = self.qwop.get_image()
        distance, distance_text = self.ocr(image.crop(self.qwop.distance_rect))
        return image, distance, distance_text

    def take_action(self, distance):
        sequence = self.agent.generate_action(distance)
        if not hasattr(sequence, '__len__'):
            sequence = [sequence]
        for action in sequence:
            if action == 0:
                self.qwop.sendSpace()
                logging.info('Sent space')
            elif action == 1:
                self.qwop.click()
                logging.info('Sent click')
            elif action == 2:
                self.qwop.sendQ()
                logging.info('Sent Q')
            elif action == 3:
                self.qwop.sendW()
                logging.info('Sent W')
            elif action == 4:
                self.qwop.sendO()
                logging.info('Sent O')
            elif action == 5:
                self.qwop.sendP()
                logging.info('Sent P')

    def update_ui(self, image, distance_text):
        self.ui.set_distance(distance_text)
        self.ui.set_image(image)
        self.ui.update()
        seconds = time.perf_counter() - self.start_time
        self.ui.set_title('Frame {} ({} fps, ~{} s)'.format(
            self.frames,
            np.round(self.frames/seconds, 2), # FPS
            np.round(seconds/self.frames, 2)  # Average frame time
        ))

    def start(self):
        logging.info('Starting the game...')
        self.run()

    def stop(self):
        self.ui.close()

    def run(self):
        self.frames = 1
        self.start_time  = time.perf_counter()
        while self.ui.is_open:
            image, distance, distance_text = self.get_distance()
            self.take_action(distance)
            self.update_ui(image, distance_text)
            self.frames += 1

def setup_switches(args):
    logging.debug('Setting switches (args={{{}}})'.format(', '.join(map(lambda kv: '\'{}\'=\'{}\''.format(*kv), args.items()))))
    if args.get('cpu'):
        logging.debug('CPU forcing enabled')
        ml.Settings.FORCE_CPU = True
    elif args.get('cuda'):
        logging.debug('CUDA forcing enabled')
        ml.Settings.FORCE_CUDA = True
    if args.get('col'):
        logging.debug('Set column-major image order')
        ocr.Settings.COL_MAJOR = True
    elif args.get('row'):
        logging.debug('Set row-major image order')
        ocr.Settings.COL_MAJOR=False
    if args.get('e-gru'):
        logging.debug('Encoder RNN set to GRU')
        ocr.Settings.ENCODER_RNN = nn.GRU
    elif args.get('e-lstm'):
        logging.debug('Encoder RNN set to LSTM')
        ocr.Settings.ENCODER_RNN = nn.LSTM
    if args.get('d-gru'):
        logging.debug('Decoder RNN set to GRU')
        ocr.Settings.DECODER_RNN = nn.GRU
    elif args.get('d-lstm'):
        logging.debug('Decoder RNN set to LSTM')
        ocr.Settings.DECODER_RNN = nn.LSTM
    x = args.get('epochs')
    if x:
        logging.debug('Epoch count set to {}'.format(x))
        ocr.NUM_EPOCHS = x
    x = args.get('-bs')
    if x:
        logging.debug('Batch size set to {}'.format(x))
        ocr.BATCH_SIZE = x
    if args.get('metres'):
        logging.debug('Including metres in training set')
        ocr.LEARN_METRES = True

def run_qwoppy(args):
    setup_switches(args)
    qwoppy = Qwoppy(args['ocr'])
    qwoppy.start()

def train_ocr_wrapper(args):
    setup_switches(args)
    train_ocr(args.get('ocr'))

def test_ocr_wrapper(args):
    setup_switches(args)
    if not args.get('ocr'):
        raise ValueError('Expected model name')
    test_ocr(args['ocr'])

def parse_args():
    parser = argparse.ArgumentParser(prog='qwoppy', description='Run Qwoppy or train or test OCR')

    parser.add_argument('-metres', action='store_true', help='Include \'metres\' in test samples (default: off)')

    g = parser.add_mutually_exclusive_group()
    g.add_argument('-col',    action='store_true', help='Process images in column-major order (the default)')
    g.add_argument('-row',    action='store_true', help='Process images in row-major order')

    g = parser.add_mutually_exclusive_group()
    g.add_argument('-cuda',   action='store_true', help='Force CUDA (default: off; CUDA will be used if available)')
    g.add_argument('-cpu',    action='store_true', help='Force CPU (default: off)')

    sub = parser.add_subparsers(help='Subcommand to run')
    run = sub.add_parser('run', help='Run Qwoppy agent')
    run.add_argument('ocr', action='store', type=str, help='')
    run.set_defaults(func=run_qwoppy)

    train = sub.add_parser('train-ocr', help='Train OCR')
    train.add_argument('-epochs',  action='store', type=int,   help='Set epoch count (default: {})'.format(ocr.Settings.NUM_EPOCHS))
    train.add_argument('-bs',      action='store', type=int,   help='Set batch size (default: largest factor)')
    train.add_argument('-loss',    action='store', type=str,   help='Set loss function (default: {})'.format(ocr.Settings.LOSS))
    train.add_argument('-e-stack', action='store', type=int,   help='Number of RNNs to stack in encoder (default: {})'.format(ocr.Settings.ENCODER_STACK))
    train.add_argument('-e-size',  action='store', type=int,   help='Hidden size of encoder RNN (default: {})'.format(ocr.Settings.ENCODER_HIDDEN))
    train.add_argument('-e-opt',   action='store', type=str,   help='Set optimizer for encoder (default: {})'.format(ocr.Settings.ENCODER_OPT))
    train.add_argument('-e-lr',    action='store', type=float, help='Set learning rate for encoder (default: {})'.format(ocr.Settings.ENCODER_LR))
    train.add_argument('-d-stack', action='store', type=int,   help='Number of RNNs to stack in decoder (default: {})'.format(ocr.Settings.DECODER_STACK))
    train.add_argument('-d-size',  action='store', type=int,   help='Hidden size of decoder RNN (default: {})'.format(ocr.Settings.DECODER_HIDDEN))
    train.add_argument('-d-opt',   action='store', type=str,   help='Set optimizer for decoder (default: {})'.format(ocr.Settings.DECODER_OPT))
    train.add_argument('-d-lr',    action='store', type=float, help='Set learning rate for decoder (default: {})'.format(ocr.Settings.DECODER_LR))
    train.add_argument('ocr',      action='store', type=str,   help='Name of the OCR model (will be created if it doesn\'t exist). The default is to use the configuration options.', nargs='?')
    train.set_defaults(func=train_ocr_wrapper)

    g = train.add_mutually_exclusive_group()
    g.add_argument('-e-gru',   action='store_true', help='Use GRU as the RNN implementation for encoder (the default)')
    g.add_argument('-e-lstm',  action='store_true', help='Use LSTM as the RNN implementation for encoder')

    g = train.add_mutually_exclusive_group()
    g.add_argument('-d-gru',   action='store_true', help='Use GRU as the RNN implementation for decoder (the default)')
    g.add_argument('-d-lstm',  action='store_true', help='Use LSTM as the RNN implementation for decoder')

    test = sub.add_parser('test-ocr', help='Test OCR')
    test.add_argument('-bs', action='store', type=int, help='Set batch size')
    test.set_defaults(func=test_ocr_wrapper)
    test.add_argument('ocr', action='store', type=str, help='Name of the OCR model (must exist)')

    return parser.parse_args()

logging_is_initialised=False
def main():
    global logging_is_initialised
    if not logging_is_initialised:
        level = logging.INFO
        if os.environ.get('DEBUG', '0') == '1':
            level = logging.DEBUG
        logging.basicConfig(level=level, format='[%(asctime)s %(levelname)s] %(message)s')
        logging.info('Initialising...')
        logging_is_initialised = True
    args = parse_args()
    args.func(vars(args))
