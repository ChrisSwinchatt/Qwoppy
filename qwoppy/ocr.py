from   abc import ABC, abstractmethod
import logging
import re
import time

import torch
import torch.autograd   as autograd
import torch.nn         as nn
import torch.optim      as optim
import torch.functional as F

from .filemgr import *
from .util    import *

class OcrProvider(ABC):
    @abstractmethod
    def __call__(self, image):
        pass

TOKENS          = ['\t', '.', '-', ' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'e', 'm', 'r', 's', 't', '\n']
SEQUENCE_LENGTH = 20
NUM_TOKENS      = len(TOKENS)
START           = TOKENS.index('\t')
STOP            = TOKENS.index('\n')
IMAGE_WIDTH     = 208
IMAGE_HEIGHT    = 24
LEARN_METRES    = False
COL_MAJOR       = True

FORCE_CPU  = False
NUM_EPOCHS = 5000
BATCH_SIZE = 83
USING_CUDA = True

ENCODER_RNN = nn.GRU
DECODER_RNN = nn.GRU

ENCODER_STACK  = 1
ENCODER_HIDDEN = 256
ENCODER_OPT    = optim.Adam
ENCODER_LR     = 0.0001

DECODER_STACK  = 1
DECODER_HIDDEN = 256
DECODER_OPT    = optim.Adam
DECODER_LR     = 0.0001

RAND_HIDDEN = True

LOSS = nn.MSELoss

assert START >= 0
assert STOP >= 0
assert START != STOP

def model_name_from_options():
    return 'epochs={},bs={},img={}x{}-{},metres={},loss={},hidden={},e={}x{}{},e.opt={},e.lr={},d={}x{}{},d.opt={},d.lr={}'.format(
        NUM_EPOCHS,
        BATCH_SIZE,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        'col-maj' if COL_MAJOR else 'row-maj',
        'yes'     if LEARN_METRES else 'no',
        LOSS.__name__,
        'rand' if RAND_HIDDEN else 'zero',
        ENCODER_STACK,
        ENCODER_HIDDEN,
        ENCODER_RNN.__name__,
        ENCODER_OPT.__name__,
        ENCODER_LR,
        DECODER_STACK,
        DECODER_HIDDEN,
        DECODER_RNN.__name__,
        DECODER_OPT.__name__,
        DECODER_LR
    )

def summarise_ocr_options():
    logging.info('OCR options:')
    logging.info(' * General options:')
    logging.info(' `- Epochs       : {}'.format(NUM_EPOCHS))
    logging.info(' `- Batch size   : {}'.format(BATCH_SIZE))
    logging.info(' `- Image width  : {}'.format(IMAGE_WIDTH))
    logging.info(' `- Image height : {}'.format(IMAGE_HEIGHT))
    logging.info(' `- Learn metres : {}'.format('yes' if LEARN_METRES else 'no'))
    logging.info(' `- Image order  : {}'.format('column-major' if COL_MAJOR else 'row-major'))
    logging.info(' `- Force CPU    : {}'.format('yes' if FORCE_CPU  else 'no'))
    logging.info(' `- Use CUDA     : {}'.format('yes' if USING_CUDA else 'no'))
    logging.info(' `- Hidden state : {}'.format('random' if RAND_HIDDEN else 'zeros'))
    logging.info(' `- Loss fn      : {}'.format(LOSS.__name__))

    logging.info(' * Encoder options: ')
    logging.info(' `- RNN          : {}'.format(ENCODER_RNN.__name__))
    logging.info(' `- Stack size   : {}'.format(ENCODER_STACK))
    logging.info(' `- Hidden size  : {}'.format(ENCODER_HIDDEN))
    logging.info(' `- Optimizer    : {}'.format(ENCODER_OPT.__name__))
    logging.info(' `- Learn rate   : {}'.format(ENCODER_LR))

    logging.info(' * Decoder options: ')
    logging.info(' `- RNN          : {}'.format(DECODER_RNN.__name__))
    logging.info(' `- Stack size   : {}'.format(DECODER_STACK))
    logging.info(' `- Hidden size  : {}'.format(DECODER_HIDDEN))
    logging.info(' `- Optimizer    : {}'.format(DECODER_OPT.__name__))
    logging.info(' `- Learn rate   : {}'.format(DECODER_LR))

def image_to_tensor(image, device=None):
    w, h  = image.size
    image = image.tobytes()
    X     = [[] for i in range(IMAGE_WIDTH)]
    for i in range(min(w, IMAGE_WIDTH)):
        X[i] = [0]*h
        for j in range(h):
            b       = image[j*w + i]
            X[i][j] = float(b)
    while i < IMAGE_WIDTH:
        X[i] = [0]*h
        i += 1
    return torch.tensor(X, device=device)

def to_sentence(X):
    if isinstance(X, torch.Tensor) and len(X.size()) > 1:
        X = torch.argmax(X, dim=-1)
    elif isinstance(X, np.ndarray) and len(X.shape) > 1:
        X = np.argmax(X, axis=-1)
    sentence = ''
    for i in X:
        token = TOKENS[i]
        sentence += token
        if i == STOP:
            break
    return sentence

def to_indices(sentence, pad=False):
    indices = []
    for c in sentence:
        if c not in TOKENS:
            continue
        indices.append(TOKENS.index(c))
        if c == TOKENS[STOP]:
            break
    if pad:
        while len(indices) < SEQUENCE_LENGTH:
            indices.append(STOP)
    return indices

def ind_to_one_hot(indices, device=None, tensor=True):
    X = None
    if tensor:
        X = torch.zeros((len(indices),NUM_TOKENS), device=device)
    else:
        X = np.zeros((len(indices),NUM_TOKENS))
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

    def __call__(self, image):
        X = image_to_tensor(image, device=self.device)
        X = self.model(X)
        return to_sentence(X)

    class model(nn.Module):
        def __init__(self, device):
            super().__init__()
            self.device  = device
            self.encoder = self.Encoder(device).to(device)
            self.decoder = self.Decoder(device).to(device)

        def forward(self, X, y_target=None, teacher_forcing_ratio=0.5):
            batch_size = X.size(1)
            y          = torch.zeros((SEQUENCE_LENGTH, batch_size, NUM_TOKENS), device=self.device)
            X, h   = self.encoder(X)
            X          = ind_to_one_hot([[START]]*batch_size, device=self.device, tensor=True)
            for i in range(SEQUENCE_LENGTH):
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
                if COL_MAJOR:
                    self.inputs = IMAGE_HEIGHT
                else:
                    self.inputs = IMAGE_WIDTH
                self.hidden = ENCODER_HIDDEN
                self.stack  = ENCODER_STACK
                self.rnn    = ENCODER_RNN(self.inputs, self.hidden, self.stack).to(device)

            def forward(self, X, h=None):
                if not h:
                    shape = (self.stack, X.size(1), self.hidden)
                    fn    = torch.zeros
                    if RAND_HIDDEN:
                        fn = torch.randn
                    if ENCODER_RNN == nn.LSTM:
                        h = (fn(shape, device=self.device), fn(shape, device=self.device))
                    else:
                        h = fn(shape, device=self.device)
                return self.rnn(X, h)

        class Decoder(nn.Module):
            def __init__(self, device):
                super().__init__()
                self.device  = device
                self.hidden  = DECODER_HIDDEN
                self.stack   = DECODER_STACK
                self.rnn     = DECODER_RNN(NUM_TOKENS, self.hidden, self.stack).to(device)
                self.lin     = nn.Linear(self.hidden, NUM_TOKENS).to(device)
                self.softmax = nn.Softmax(-1).to(device)

            def forward(self, X, h):
                X, h = self.rnn(X.unsqueeze(0), h)
                X    = self.lin(X.squeeze(0))
                X    = self.softmax(X)
                return X, h

def print_cuda_runtime_info(device):
    if USING_CUDA:
        logging.info(' * CUDA allocated    : {}'.format(gigabytes(torch.cuda.memory_allocated(device))))
        logging.info(' * CUDA cached       : {}'.format(gigabytes(torch.cuda.memory_cached(device))))

def print_device_info(device):
    logging.info('Device information:')
    logging.info(' * CUDA available    : {}'.format('yes' if torch.cuda.is_available() else 'no'))
    logging.info(' * CUDA initialized  : {}'.format('yes' if torch.cuda.is_initialized() else 'no'))
    logging.info(' * Using CUDA        : {}'.format('yes' if USING_CUDA else 'no'))
    logging.info(' * CPU forced        : {}'.format('yes' if FORCE_CPU else 'no'))
    logging.info(' * Selected device   : {} {}'.format('[' + str(device) + ']', torch.cuda.get_device_name(0) if USING_CUDA else ''))
    print_cuda_runtime_info(device)

def print_epoch_summary(epoch, count, epoch_loss, prev_loss, epoch_acc, mean_time, epoch_time, total_time, device):
    logging.info('Epoch {} summary:'.format(epoch + 1))
    logging.info(' * Average loss      : {} ({}{})'.format(epoch_loss, '+' if epoch_loss > prev_loss else '', epoch_loss - prev_loss))
    logging.info(' * Average accuracy  : {}%'.format(int(100*epoch_acc)))
    logging.info(' * Total elapsed     : {}'.format(fuzzy_time(total_time)))
    logging.info(' * Epoch time        : {}'.format(fuzzy_time(epoch_time)))
    logging.info(' * Avg. sample time  : {}'.format(fuzzy_time(mean_time)))
    logging.info(' * Approx. time left : {}'.format(fuzzy_time((NUM_EPOCHS - epoch)*epoch_time)))
    print_cuda_runtime_info(device)

def get_device():
    global USING_CUDA
    USING_CUDA = not FORCE_CPU and torch.cuda.is_available()
    device = torch.device('cuda') if USING_CUDA else torch.device('cpu')
    print_device_info(device)
    return device

def load_training_data_with_file(file, basePath=None):
    logging.info('Load training data...')
    start  = time.perf_counter()
    images = []
    labels = []
    for line in file:
        s, r = line.split(',')
        s = FileManager.openSentence(s, basePath=basePath)
        r = str(r)[:-1]
        if LEARN_METRES:
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
    img_dim0 = IMAGE_WIDTH
    img_dim2 = IMAGE_HEIGHT
    if not COL_MAJOR:
        img_dim0 = IMAGE_HEIGHT
        img_dim2 = IMAGE_WIDTH
    ppImages = np.zeros((img_dim0,        count, img_dim2), dtype=np.float32)
    ppLabels = np.zeros((SEQUENCE_LENGTH, count, NUM_TOKENS),   dtype=np.float32)
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
        if COL_MAJOR:
            for j in range(min(width, IMAGE_WIDTH)):
                for k in range(min(height, IMAGE_HEIGHT)):
                    ppImages[j, i, k] = image[k*width + j]/255.0
        else:
            for j in range(min(height, IMAGE_HEIGHT)):
                for k in range(min(width, IMAGE_WIDTH)):
                    ppImages[j, i, k] = image[j*width + k]/255.0
        for j in range(SEQUENCE_LENGTH):
            for k in range(NUM_TOKENS):
                ppLabels[j, i, k] = labels[i][j][k]
    ppImages = torch.tensor(ppImages, device=device)
    ppLabels = torch.tensor(ppLabels, device=device)
    logging.info('Preprocessed training set in {}'.format(fuzzy_time(time.perf_counter() - start)))
    return ppImages, ppLabels

def create_batches(images, labels, device, num_batches, shuffle=True):
    indices = np.arange(images.size(1))
    if shuffle:
        np.random.shuffle(indices)
    indices    = iter(indices)
    batch_size = images.size(1)//num_batches
    for _ in range(num_batches):
        indices_for_batch = torch.tensor([next(indices) for _ in range(batch_size)], dtype=torch.int64, device=device)
        images_for_batch  = images.index_select(1, indices_for_batch)
        labels_for_batch  = labels.index_select(1, indices_for_batch)
        yield images_for_batch, labels_for_batch

def accuracy(y_pred, y_target):
    count   = y_pred.size(0)
    correct = float(sum([1 if all(y_pred[i] == y_target[i]) else 0 for i in range(count)]))
    return correct/count

def train_batches(batches, device, ocr, encoder_optimizer, decoder_optimizer, loss_fn, num_batches, batch_losses, batch_accuracies, batch_times):
    count = 0
    for X, y_target in batches:
        batch_start_time = time.perf_counter()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        y_pred = ocr.model(X)

        loss = loss_fn(y_pred, y_target)
        loss.backward(retain_graph=True)
        encoder_optimizer.step()
        decoder_optimizer.step()

        y_pred_n   = torch.argmax(y_pred,   dim=-1)
        y_target_n = torch.argmax(y_target, dim=-1)
        acc        = accuracy(y_pred_n, y_target_n)

        batch_losses[count]     = loss.item()
        batch_times[count]      = time.perf_counter() - batch_start_time
        batch_accuracies[count] = acc
        count += 1

        logging.info(
            '\t{}\t{}%\t\t{:22} \'{}\''.format(
                count,
                int(100*count/num_batches),
                '\'' + to_sentence(y_target[:, -1, :]).replace('\n', '\\n') + '\'',
                to_sentence(y_pred[:, -1, :]).replace('\n', '\\n')
            )
        )

class TrainTestHarness:
    def __init__(self, mode, modelName):
        if os.environ.get('DEBUG') == '1':
            logging.info('Starting in debug mode')
            autograd.set_detect_anomaly(True)
        summarise_ocr_options()
        self.modelName           = modelName
        if not self.modelName:
            self.modelName = model_name_from_options()
        self.device              = get_device()
        self.images, self.labels = load_training_data(name='.'.join((mode, 'csv')))
        self.images, self.labels = preprocess_training_data(self.images, self.labels, self.device)
        self.ocr                 = TorchOcrProvider(self.device, model=FileManager.loadOcrModelIfExists(name=self.modelName))
        self.model               = self.ocr.model
        self.loss_fn             = LOSS()
        self.encoder_optimizer   = ENCODER_OPT(self.model.encoder.parameters(), lr=ENCODER_LR)
        self.decoder_optimizer   = DECODER_OPT(self.model.decoder.parameters(), lr=DECODER_LR)
        self.num_batches         = self.images.size(1)//BATCH_SIZE
        self.batch_losses        = np.empty(self.num_batches, dtype=float)
        self.batch_accuracies    = np.empty(self.num_batches, dtype=float)
        self.batch_times         = np.empty(self.num_batches, dtype=float)

    def train(self):
        if self.model is None:
            logging.info('Start training...')
        else:
            logging.info('Continue training...')
            autograd.set_detect_anomaly(True)
        start_time = time.perf_counter()
        prev_loss  = 0
        try:
            for epoch in range(NUM_EPOCHS):
                logging.info('========================================[ Epoch {} ]========================================'.format(epoch + 1))
                logging.info('\tSamples\tProgress\tExpected\t\tActual')
                epoch_start_time = time.perf_counter()

                batches = list(create_batches(self.images, self.labels, self.device, self.num_batches, shuffle=True))
                train_batches(
                    batches,
                    self.device,
                    self.ocr,
                    self.encoder_optimizer,
                    self.decoder_optimizer,
                    self.loss_fn,
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
        except (KeyboardInterrupt, SystemExit):
            logging.info('Training stopped early')
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
        batches = list(create_batches(self.images, self.labels, self.device, self.num_batches, shuffle=True))
        train_batches(
            batches,
            self.device,
            self.ocr,
            self.encoder_optimizer,
            self.decoder_optimizer,
            self.loss_fn,
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

def make_ocr(modelName):
    device = get_device()
    model  = FileManager.loadOcrModel(name=modelName)
    return TorchOcrProvider(device, model)

def train_ocr(modelName='model'):
    TrainTestHarness('train', modelName).train()

def test_ocr(modelName):
    TrainTestHarness('test', modelName).test()
