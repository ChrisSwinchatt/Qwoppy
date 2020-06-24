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
import logging

import torch.nn as nn

from   .init      import init_logging, init_controller, init_ocr, init_ui, init_agent, init_cleanup, wait_for_init
from   .mainloop  import mainloop
from   ..ocr      import train_ocr, test_ocr
import qwoppy.ocr as     ocr

def setup_switches(args):
    logging.debug('Setting switches (args={{{}}})'.format(', '.join(map(lambda kv: '\'{}\'=\'{}\''.format(*kv), args.items()))))
    if args.get('cpu'):
        logging.debug('CPU forcing enabled')
        ocr.FORCE_CPU = True
    elif args.get('cuda'):
        logging.debug('CUDA enabled')
        ocr.USING_CUDA = True
    if args.get('col'):
        logging.debug('Set column-major image order')
        ocr.COL_MAJOR = True
    elif args.get('row'):
        logging.debug('Set row-major image order')
        ocr.COL_MAJOR=False
    if args.get('e-gru'):
        logging.debug('Encoder RNN set to GRU')
        ocr.ENCODER_RNN = nn.GRU
    elif args.get('e-lstm'):
        logging.debug('Encoder RNN set to LSTM')
        ocr.ENCODER_RNN = nn.LSTM
    if args.get('d-gru'):
        logging.debug('Decoder RNN set to GRU')
        ocr.DECODER_RNN = nn.GRU
    elif args.get('d-lstm'):
        logging.debug('Decoder RNN set to LSTM')
        ocr.DECODER_RNN = nn.LSTM
    x = args.get('epochs')
    if x:
        logging.debug('Epoch count set to {}'.format(x))
        ocr.NUM_EPOCHS = x
    x = args.get('-bs')
    if x:
        logging.debug('Batch size set to {}'.format(x))
        ocr.BATCH_SIZE = x
    if args.get('metres'):
        ocr.LEARN_METRES = True

def run_qwoppy(args):
    setup_switches(args)
    agent      = init_agent()
    ocr        = init_ocr(args['ocr'])
    qwop, rect = init_controller()
    ui         = init_ui(rect)
    init_cleanup(qwop, ocr, ui, agent)
    wait_for_init()
    mainloop(qwop, ocr, ui, agent)

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
    g.add_argument('-cuda',   action='store_true', help='Use CUDA if available (default: on)')
    g.add_argument('-cpu',    action='store_true', help='Use CPU even if CUDA is available (default: off)')

    sub = parser.add_subparsers(help='Subcommand to run')
    run = sub.add_parser('run', help='Run Qwoppy agent')
    run.add_argument('ocr', action='store', type=str, help='')
    run.set_defaults(func=run_qwoppy)

    train = sub.add_parser('train-ocr', help='Train OCR')
    train.add_argument('-epochs',  action='store', type=int,   help='Set epoch count')
    train.add_argument('-bs',      action='store', type=int,   help='Set batch size')
    train.add_argument('-loss',    action='store', type=str,   help='Set loss function (default: {})'.format(ocr.LOSS))
    train.add_argument('-e-stack', action='store', type=int,   help='Number of RNNs to stack in encoder (default: {})'.format(ocr.ENCODER_STACK))
    train.add_argument('-e-size',  action='store', type=int,   help='Hidden size of encoder RNN (default: {})'.format(ocr.ENCODER_HIDDEN))
    train.add_argument('-e-opt',   action='store', type=str,   help='Set optimizer for encoder (default: {})'.format(ocr.ENCODER_OPT))
    train.add_argument('-e-lr',    action='store', type=float, help='Set learning rate for encoder (default: {})'.format(ocr.ENCODER_LR))
    train.add_argument('-d-stack', action='store', type=int,   help='Number of RNNs to stack in decoder (default: {})'.format(ocr.DECODER_STACK))
    train.add_argument('-d-size',  action='store', type=int,   help='Hidden size of decoder RNN (default: {})'.format(ocr.DECODER_HIDDEN))
    train.add_argument('-d-opt',   action='store', type=str,   help='Set optimizer for decoder (default: {})'.format(ocr.DECODER_OPT))
    train.add_argument('-d-lr',    action='store', type=float, help='Set learning rate for decoder (default: {})'.format(ocr.DECODER_LR))
    train.add_argument('ocr',      action='store', type=str,   help='Name of the OCR model (will be created if it doesn\'t exist). The default is to use the configuration options.', nargs='?')
    train.set_defaults(func=train_ocr_wrapper)

    g = train.add_mutually_exclusive_group()
    g.add_argument('-e-gru',   action='store_true',      help='Use GRU as the RNN implementation for encoder (the default)')
    g.add_argument('-e-lstm',  action='store_true',      help='Use LSTM as the RNN implementation for encoder')

    g = train.add_mutually_exclusive_group()
    g.add_argument('-d-gru',   action='store_true',      help='Use GRU as the RNN implementation for decoder (the default)')
    g.add_argument('-d-lstm',  action='store_true',      help='Use LSTM as the RNN implementation for decoder')

    test = sub.add_parser('test-ocr', help='Test OCR')
    test.add_argument('-bs', action='store', type=int, help='Set batch size')
    test.set_defaults(func=test_ocr_wrapper)
    test.add_argument('ocr', action='store', type=str, help='Name of the OCR model (must exist)')

    return parser.parse_args()

def main():
    init_logging()
    args = parse_args()
    args.func(vars(args))
    exit(0)
