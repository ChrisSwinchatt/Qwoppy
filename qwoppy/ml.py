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

import torch

from .util import gigabytes

class Settings:
    FORCE_CPU  = False
    FORCE_CUDA = False
    USE_CUDNN  = False
    USING_CUDA = False

def print_cuda_runtime_info(device):
    logging.info(' * CUDA allocated    : {}'.format(gigabytes(torch.cuda.memory_allocated(device))))
    logging.info(' * CUDA cached       : {}'.format(gigabytes(torch.cuda.memory_cached(device))))

def print_device_info(device):
    logging.info('Device information:')
    logging.info(' * Force CPU         : {}'.format('yes' if Settings.FORCE_CPU  else 'no'))
    logging.info(' * Force CUDA        : {}'.format('yes' if Settings.FORCE_CUDA else 'no'))
    logging.info(' * cuDNN             : {}'.format('enabled' if Settings.USE_CUDNN else 'disabled'))
    logging.info(' * CUDA available    : {}'.format('yes' if torch.cuda.is_available() else 'no'))
    logging.info(' * CUDA initialized  : {}'.format('yes' if torch.cuda.is_initialized() else 'no'))
    logging.info(' * Selected device   : {} {}'.format('[' + str(device) + ']', torch.cuda.get_device_name(0) if USING_CUDA else ''))
    if USING_CUDA:
        print_cuda_runtime_info(device)

def get_device():
    global USING_CUDA
    if Settings.FORCE_CUDA and not torch.cuda.is_available():
        logging.error('CUDA is force-enabled but CUDA is not available')
        exit(1)
    USING_CUDA = not Settings.FORCE_CPU and torch.cuda.is_available()
    device = torch.device('cuda') if USING_CUDA else torch.device('cpu')
    torch.backends.cudnn.enabled = Settings.USE_CUDNN
    print_device_info(device)
    return device
