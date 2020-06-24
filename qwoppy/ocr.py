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

from   abc import ABC, abstractmethod
import logging
import os
import re
import time

import numpy            as np
import torch
import torch.autograd   as autograd
import torch.nn         as nn
import torch.optim      as optim
import torch.functional as F

from .filemgr import FileManager
from .ml      import get_device, print_device_info, print_cuda_runtime_info
from .util    import fuzzy_time, probability

class OcrProvider(ABC):
    @abstractmethod
    def __call__(self, image):
        pass

class Constants:
    TOKENS          = ['\t', '.', '-', ' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'e', 'm', 'r', 's', 't', '\n']
    SEQUENCE_LENGTH = 20
    NUM_TOKENS      = len(TOKENS)
    START           = TOKENS.index('\t')
    STOP            = TOKENS.index('\n')

assert Constants.START >= 0
assert Constants.STOP >= 0
assert Constants.START != Constants.STOP

class Settings:
    IMAGE_WIDTH    = 208
    IMAGE_HEIGHT   = 24
    LEARN_METRES   = True
    COL_MAJOR      = True
    NUM_EPOCHS     = 200
    BATCH_SIZE     = None
    ENCODER_RNN    = nn.GRU
    ENCODER_STACK  = 1
    ENCODER_HIDDEN = 256
    ENCODER_OPT    = optim.Adam
    ENCODER_LR     = 0.0001
    DECODER_RNN    = nn.GRU
    DECODER_STACK  = 1
    DECODER_HIDDEN = 256
    DECODER_OPT    = optim.Adam
    DECODER_LR     = 0.0001
    RAND_HIDDEN    = True
    LOSS           = nn.MSELoss

def model_name_from_options():
    return 'epochs={},bs={},img={}x{}-{},metres={},loss={},hidden={},e={}x{}{},e.opt={},e.lr={},d={}x{}{},d.opt={},d.lr={}'.format(
        Settings.NUM_EPOCHS,
        Settings.BATCH_SIZE,
        Settings.IMAGE_WIDTH,
        Settings.IMAGE_HEIGHT,
        'col-maj' if Settings.COL_MAJOR else 'row-maj',
        'yes'     if Settings.LEARN_METRES else 'no',
        Settings.LOSS.__name__,
        'rand' if Settings.RAND_HIDDEN else 'zero',
        Settings.ENCODER_STACK,
        Settings.ENCODER_HIDDEN,
        Settings.ENCODER_RNN.__name__,
        Settings.ENCODER_OPT.__name__,
        Settings.ENCODER_LR,
        Settings.DECODER_STACK,
        Settings.DECODER_HIDDEN,
        Settings.DECODER_RNN.__name__,
        Settings.DECODER_OPT.__name__,
        Settings.DECODER_LR
    )

def summarise_ocr_options():
    logging.info('OCR options:')
    logging.info(' * General options:')
    logging.info(' `- Epochs       : {}'.format(Settings.NUM_EPOCHS))
    logging.info(' `- Batch size   : {}'.format(Settings.BATCH_SIZE))
    logging.info(' `- Image width  : {}'.format(Settings.IMAGE_WIDTH))
    logging.info(' `- Image height : {}'.format(Settings.IMAGE_HEIGHT))
    logging.info(' `- Learn metres : {}'.format('yes' if Settings.LEARN_METRES else 'no'))
    logging.info(' `- Image order  : {}'.format('column-major' if Settings.COL_MAJOR else 'row-major'))
    logging.info(' `- Hidden state : {}'.format('random' if Settings.RAND_HIDDEN else 'zeros'))
    logging.info(' `- Loss fn      : {}'.format(Settings.LOSS.__name__))

    logging.info(' * Encoder options: ')
    logging.info(' `- RNN          : {}'.format(Settings.ENCODER_RNN.__name__))
    logging.info(' `- Stack size   : {}'.format(Settings.ENCODER_STACK))
    logging.info(' `- Hidden size  : {}'.format(Settings.ENCODER_HIDDEN))
    logging.info(' `- Optimizer    : {}'.format(Settings.ENCODER_OPT.__name__))
    logging.info(' `- Learn rate   : {}'.format(Settings.ENCODER_LR))

    logging.info(' * Decoder options: ')
    logging.info(' `- RNN          : {}'.format(Settings.DECODER_RNN.__name__))
    logging.info(' `- Stack size   : {}'.format(Settings.DECODER_STACK))
    logging.info(' `- Hidden size  : {}'.format(Settings.DECODER_HIDDEN))
    logging.info(' `- Optimizer    : {}'.format(Settings.DECODER_OPT.__name__))
    logging.info(' `- Learn rate   : {}'.format(Settings.DECODER_LR))

def image_to_tensor(image, device=None, width=None, height=None, out='torch'):
    if width is None and height is None:
        # Assuming image is PIL.Image.
        width, height = image.size
        image         = image.convert('L').tobytes()
    X = None
    if Settings.COL_MAJOR:
        X = np.zeros((Settings.IMAGE_WIDTH, Settings.IMAGE_HEIGHT))
        i = 0
        for i in range(min(width, Settings.IMAGE_WIDTH)):
            for j in range(min(height, Settings.IMAGE_HEIGHT)):
                X[i, j] = image[j*width + i]/255.0
    else:
        X = np.zeros((Settings.IMAGE_HEIGHT, Settings.IMAGE_WIDTH))
        for j in range(min(height, Settings.IMAGE_HEIGHT)):
            for i in range(min(width, Settings.IMAGE_WIDTH)):
                X[j, i] = image[j*width + i]/255.0
    if out == 'torch':
        X = torch.tensor(X, device=device)
    return X

def to_sentence(X):
    if isinstance(X, torch.Tensor) and len(X.size()) > 1:
        X = torch.argmax(X, dim=-1)
    elif isinstance(X, np.ndarray) and len(X.shape) > 1:
        X = np.argmax(X, axis=-1)
    sentence = ''
    for i in X:
        token = Constants.TOKENS[i]
        sentence += token
        if i == Constants.STOP:
            break
    return sentence

def to_indices(sentence, pad=False):
    indices = []
    for c in sentence:
        if c not in Constants.TOKENS:
            continue
        indices.append(Constants.TOKENS.index(c))
        if c == Constants.TOKENS[Constants.STOP]:
            break
    if pad:
        while len(indices) < Constants.SEQUENCE_LENGTH:
            indices.append(Constants.STOP)
    return indices

def ind_to_one_hot(indices, device=None, tensor=True):
    X = None
    if tensor:
        X = torch.zeros((len(indices),Constants.NUM_TOKENS), device=device)
    else:
        X = np.zeros((len(indices),Constants.NUM_TOKENS))
    for i, j in enumerate(indices):
        X[i][j] = 1
    return X

def to_one_hot(sentence, pad=False, device=None, tensor=True):
    indices = to_indices(sentence, pad=pad)
    return ind_to_one_hot(indices, device, tensor)

class TorchOcrProvider(OcrProvider):
    def __init__(self, device, model):
        if model is None:
            self.model = self.model(device).to(device)
        else:
            self.model = model.to(device)
        self.device = device

    def __call__(self, image, raw=False):
        X = image_to_tensor(image, device=self.device).unsqueeze(1).float()
        X = self.model(X)
        return X, to_sentence(X)

    class model(nn.Module):
        def __init__(self, device):
            super().__init__()
            self.device  = device
            self.encoder = self.Encoder(device).to(device)
            self.decoder = self.Decoder(device).to(device)

        def forward(self, X, y_target=None, teacher_forcing_ratio=0.5):
            batch_size = X.size(1)
            y          = torch.zeros((Constants.SEQUENCE_LENGTH, batch_size, Constants.NUM_TOKENS), device=self.device)
            X, h   = self.encoder(X)
            X          = ind_to_one_hot([[Constants.START]]*batch_size, device=self.device, tensor=True)
            for i in range(Constants.SEQUENCE_LENGTH):
                yi, h      = self.decoder(X, h)
                y[i, :, :] = yi
                if y_target is not None and probability(teacher_forcing_ratio):
                    X = y_target[i, :, :]
                else:
                    X = yi
            return y

        class Encoder(nn.Module):
            def __init__(self, device):
                super().__init__()
                self.device = device
                if Settings.COL_MAJOR:
                    self.inputs = Settings.IMAGE_HEIGHT
                else:
                    self.inputs = Settings.IMAGE_WIDTH
                self.hidden = Settings.ENCODER_HIDDEN
                self.stack  = Settings.ENCODER_STACK
                self.rnn    = Settings.ENCODER_RNN(self.inputs, self.hidden, self.stack).to(device)

            def forward(self, X, h=None):
                if not h:
                    shape = (self.stack, X.size(1), self.hidden)
                    fn    = torch.zeros
                    if Settings.RAND_HIDDEN:
                        fn = torch.randn
                    if Settings.ENCODER_RNN == nn.LSTM:
                        h = (fn(shape, device=self.device), fn(shape, device=self.device))
                    else:
                        h = fn(shape, device=self.device)
                return self.rnn(X, h)

        class Decoder(nn.Module):
            def __init__(self, device):
                super().__init__()
                self.device  = device
                self.hidden  = Settings.DECODER_HIDDEN
                self.stack   = Settings.DECODER_STACK
                self.rnn     = Settings.DECODER_RNN(Constants.NUM_TOKENS, self.hidden, self.stack).to(device)
                self.lin     = nn.Linear(self.hidden, Constants.NUM_TOKENS).to(device)
                self.softmax = nn.Softmax(-1).to(device)

            def forward(self, X, h):
                X, h = self.rnn(X.unsqueeze(0), h)
                X    = self.lin(X.squeeze(0))
                X    = self.softmax(X)
                return X, h



def print_epoch_summary(epoch, count, epoch_loss, prev_loss, epoch_acc, mean_time, epoch_time, total_time, device):
    logging.info('Epoch {} summary:'.format(epoch + 1))
    logging.info(' * Average loss      : {} ({}{})'.format(epoch_loss, '+' if epoch_loss > prev_loss else '', epoch_loss - prev_loss))
    logging.info(' * Average accuracy  : {}%'.format(int(100*epoch_acc)))
    logging.info(' * Total elapsed     : {}'.format(fuzzy_time(total_time)))
    logging.info(' * Epoch time        : {}'.format(fuzzy_time(epoch_time)))
    logging.info(' * Avg. sample time  : {}'.format(fuzzy_time(mean_time)))
    logging.info(' * Approx. time left : {}'.format(fuzzy_time((Settings.NUM_EPOCHS - epoch)*epoch_time)))
    print_cuda_runtime_info(device)

def load_training_data_with_file(file, basePath=None):
    logging.info('Load training data...')
    start  = time.perf_counter()
    images = []
    labels = []
    for line in file:
        s, r = line.split(',')
        s = FileManager.openSentence(s, basePath=basePath)
        r = str(r)[:-1]
        if Settings.LEARN_METRES:
            r += ' metres\n'
        r = to_one_hot(r, pad=True, tensor=False)
        images.append(s)
        labels.append(r)
    logging.info('Loaded {} samples in {}'.format(len(images), fuzzy_time(time.perf_counter() - start)))
    return images, labels

def load_training_data(name='train.csv', basePath=None):
    with FileManager.openOcrCsv(name, basePath=basePath) as file:
        return load_training_data_with_file(file, basePath=basePath)

def preprocess_training_data(images, labels, device, in_place=True):
    if not in_place:
        images = list(images)
        labels = list(labels)
    start    = time.perf_counter()
    count    = len(labels)
    img_dim0 = Settings.IMAGE_WIDTH
    img_dim2 = Settings.IMAGE_HEIGHT
    if not Settings.COL_MAJOR:
        img_dim0 = Settings.IMAGE_HEIGHT
        img_dim2 = Settings.IMAGE_WIDTH
    ppImages = np.zeros((img_dim0,                  count, img_dim2),   dtype=np.float32)
    ppLabels = np.zeros((Constants.SEQUENCE_LENGTH, count, Constants.NUM_TOKENS), dtype=np.float32)
    for i in range(count):
        image  = images[i]
        width  = None
        height = None
        if isinstance(image, tuple):
            image, width, height, _ = image
        else:
            width, height = image.size
            filename      = image.filename if hasattr(image, 'filename') else '<no filename>'
            image         = image.tobytes()
            images[i]     = (image, width, height, filename)
        ppImages[:, i, :] = image_to_tensor(image, width=width, height=height, out='numpy')
        for j in range(Constants.SEQUENCE_LENGTH):
            for k in range(Constants.NUM_TOKENS):
                ppLabels[j, i, k] = labels[i][j][k]
    ppImages = torch.tensor(ppImages, device=device)
    ppLabels = torch.tensor(ppLabels, device=device)
    logging.info('Preprocessed training set in {}'.format(fuzzy_time(time.perf_counter() - start)))
    return ppImages, ppLabels

def create_batches(images, labels, device, batch_size, num_batches, shuffle=True):
    indices = np.arange(images.size(1))
    if shuffle:
        np.random.shuffle(indices)
    indices    = iter(indices)
    for _ in range(num_batches):
        indices_for_batch = torch.tensor([next(indices) for _ in range(batch_size)], dtype=torch.int64, device=device)
        images_for_batch  = images.index_select(1, indices_for_batch)
        labels_for_batch  = labels.index_select(1, indices_for_batch)
        yield images_for_batch, labels_for_batch

def test_batches(batches, device, ocr, loss_fn, batch_start, batch_end, batch_size, num_batches, batch_losses, batch_accuracies, batch_times):
    count = 0
    for X, y_target in batches:
        if hasattr(batch_start, '__call__'):
            batch_start()

        batch_start_time = time.perf_counter()

        y_pred = ocr.model(X)

        loss = loss_fn(y_pred, y_target)
        if hasattr(batch_end, '__call__'):
            batch_end(loss)

        batch_losses[count]     = loss.item()
        batch_times[count]      = time.perf_counter() - batch_start_time
        batch_accuracies[count] = torch.sum((torch.argmax(y_pred, -1) == torch.argmax(y_target, -1)).float())/(y_pred.size(0)*y_pred.size(1))
        count += 1

        logging.info(
            '\t{}\t{}%\t\t{:22} \'{}\''.format(
                count,
                int(100*count/num_batches),
                '\'' + to_sentence(y_target[:, -1, :]).replace('\n', '\\n') + '\'',
                to_sentence(y_pred[:, -1, :]).replace('\n', '\\n')
            )
        )

def train_batches(batches, device, ocr, encoder_optimizer, decoder_optimizer, loss_fn, batch_size, num_batches, batch_losses, batch_accuracies, batch_times):
    def batch_start():
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
    def batch_end(loss):
        loss.backward(retain_graph=True)
        encoder_optimizer.step()
        decoder_optimizer.step()
    test_batches(batches, device, ocr, loss_fn, batch_start, batch_end, batch_size, num_batches, batch_losses, batch_accuracies, batch_times)

def choose_batching_parameters(N):
    '''
    Select the factors of N with the largest mean.
    '''
    factors = set()
    for x in range(2, N):
        if N%x == 0:
            y = N//x
            factors.add((min(x, y), max(x, y)))
    if not factors:
        logging.warn('Couldn\'t find any factors of {}, using 1 batch'.format(N))
        return 1, N
    return max(factors, key=lambda x: (x[0] + x[1])//2)

class TrainTestHarness:
    def __init__(self, dataset, modelName):
        if os.environ.get('DEBUG') == '1':
            logging.info('Starting in debug mode')
            autograd.set_detect_anomaly(True)
        summarise_ocr_options()
        self.modelName = modelName
        if not self.modelName:
            self.modelName = model_name_from_options()
        self.device              = get_device()
        self.images, self.labels = load_training_data(name='.'.join((dataset, 'csv')))
        self.images, self.labels = preprocess_training_data(self.images, self.labels, self.device)
        model                    = FileManager.loadOcrModelIfExists(name=self.modelName)
        self.loadedModel         = model is not None
        self.ocr                 = TorchOcrProvider(self.device, model=model)
        self.model               = self.ocr.model
        self.loss_fn             = Settings.LOSS()
        self.encoder_optimizer   = Settings.ENCODER_OPT(self.model.encoder.parameters(), lr=Settings.ENCODER_LR)
        self.decoder_optimizer   = Settings.DECODER_OPT(self.model.decoder.parameters(), lr=Settings.DECODER_LR)
        if Settings.BATCH_SIZE is None:
            self.num_batches, self.batch_size = choose_batching_parameters(self.images.size(1))
        else:
            self.batch_size  = Settings.BATCH_SIZE
            self.num_batches = self.images.size(1)//Settings.BATCH_SIZE
        logging.info('Using {} batches of {} samples'.format(self.num_batches, self.batch_size))
        self.batch_losses     = np.empty(self.num_batches, dtype=float)
        self.batch_accuracies = np.empty(self.num_batches, dtype=float)
        self.batch_times      = np.empty(self.num_batches, dtype=float)

    def train(self):
        if self.loadedModel:
            logging.info('Continue training...')
        else:
            logging.info('Start training...')
        start_time = time.perf_counter()
        prev_loss  = 0
        try:
            for epoch in range(Settings.NUM_EPOCHS):
                logging.info(
                    '========================================[ Epoch {}/{} ]========================================'.format(
                        epoch + 1,
                        Settings.NUM_EPOCHS
                    )
                )
                logging.info('\tSamples\tProgress\tExpected\t\tActual')
                epoch_start_time = time.perf_counter()

                batches = list(create_batches(self.images, self.labels, self.device, self.batch_size, self.num_batches, shuffle=True))
                train_batches(
                    batches,
                    self.device,
                    self.ocr,
                    self.encoder_optimizer,
                    self.decoder_optimizer,
                    self.loss_fn,
                    self.batch_size,
                    self.num_batches,
                    self.batch_losses,
                    self.batch_accuracies,
                    self.batch_times
                )

                epoch_loss = np.mean(self.batch_losses)
                epoch_acc  = np.mean(self.batch_accuracies)
                mean_time  = np.mean(self.batch_times)
                epoch_time = time.perf_counter() - epoch_start_time
                total_time = time.perf_counter() - start_time
                print_epoch_summary(epoch, self.num_batches, epoch_loss, prev_loss, epoch_acc, mean_time, epoch_time, total_time, self.device)
                prev_loss  = epoch_loss
                logging.info('Training completed')
        except KeyboardInterrupt:
            logging.info('Training stopped by user')
        except SystemExit:
            logging.info('Training stopped by exit signal')
        except Exception:
            logging.exception('Training stopped by exception')
            exit(1)
        finally:
            FileManager.saveOcrModel(self.model, name=self.modelName)
            logging.info('Fin.')
            exit(0)

    def test(self):
        if self.model is None:
            logging.error('Model is not loaded')
            exit(1)
        start_time = time.perf_counter()
        prev_loss  = 0
        batches = list(create_batches(self.images, self.labels, self.device, self.batch_size, self.num_batches, shuffle=True))
        test_batches(
            batches,
            self.device,
            self.ocr,
            self.loss_fn,
            None,
            None,
            self.batch_size,
            self.num_batches,
            self.batch_losses,
            self.batch_accuracies,
            self.batch_times
        )
        epoch_loss = np.mean(self.batch_losses)
        epoch_acc  = np.mean(self.batch_accuracies)
        mean_time  = np.mean(self.batch_times)
        total_time = time.perf_counter() - start_time
        print_epoch_summary(0, self.num_batches, epoch_loss, prev_loss, epoch_acc, mean_time, total_time, total_time, self.device)
        prev_loss  = epoch_loss
        logging.info('Fin.')
        exit(0)

def make_ocr(device, modelName):
    model  = FileManager.loadOcrModel(name=modelName)
    return TorchOcrProvider(device, model)

def train_ocr(modelName='model'):
    TrainTestHarness('train', modelName).train()

def test_ocr(modelName):
    TrainTestHarness('test', modelName).test()
