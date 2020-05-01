from   abc import ABC, abstractmethod
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
LEARN_METRES    = True

FORCE_CPU  = False
NUM_EPOCHS = 1000
BATCH_SIZE = 83
USING_CUDA = True

ENCODER_RNN = nn.GRU
DECODER_RNN = nn.GRU

assert START >= 0
assert STOP >= 0
assert START != STOP

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
            self.model = self.Model(device).to(device)
        else:
            self.model = model.to(device)
        self.device = device

    def __call__(self, image):
        X = image_to_tensor(image, device=self.device)
        X = self.model(X)
        return to_sentence(X)

    class Model(nn.Module):
        def __init__(self, device, hidden=256):
            super().__init__()
            self.hidden  = hidden
            self.device  = device
            self.encoder = self.Encoder(device, IMAGE_HEIGHT, hidden).to(device)
            self.decoder = self.Decoder(device, hidden).to(device)

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
            def __init__(self, device, inputs, hidden):
                super().__init__()
                self.device = device
                self.rnn    = ENCODER_RNN(inputs, hidden).to(device)
                self.hidden = hidden

            def forward(self, X, h=None):
                if not h:
                    if ENCODER_RNN == nn.LSTM:
                        h = (
                            torch.randn((1, X.size(1), self.hidden), device=self.device),
                            torch.randn((1, X.size(1), self.hidden), device=self.device)
                        )
                    else:
                        h = torch.randn((1, X.size(1), self.hidden), device=self.device)
                return self.rnn(X, h)

        class Decoder(nn.Module):
            def __init__(self, device, hidden):
                super().__init__()
                self.device  = device
                self.hidden  = hidden
                self.rnn     = DECODER_RNN(NUM_TOKENS, hidden).to(device)
                self.lin     = nn.Linear(hidden, NUM_TOKENS).to(device)
                self.softmax = nn.Softmax(-1).to(device)

            def forward(self, X, h):
                X, h = self.rnn(X.unsqueeze(0), h)
                X    = self.lin(X.squeeze(0))
                X    = self.softmax(X)
                return X, h

def print_cuda_runtime_info(device):
    if USING_CUDA:
        print(' * CUDA allocated    :', gigabytes(torch.cuda.memory_allocated(device)))
        print(' * CUDA cached       :', gigabytes(torch.cuda.memory_cached(device)))

def print_device_info(device):
    print('Device information:')
    print(' * CUDA available    :', 'yes' if torch.cuda.is_available() else 'no')
    print(' * CUDA initialized  :', 'yes' if torch.cuda.is_initialized() else 'no')
    print(' * Using CUDA        :', 'yes' if USING_CUDA else 'no')
    print(' * CPU forced        :', 'yes' if FORCE_CPU else 'no')
    print(' * Selected device   :', '[' + str(device) + ']', torch.cuda.get_device_name(0) if USING_CUDA else '')
    print_cuda_runtime_info(device)

def print_epoch_summary(epoch, count, epoch_loss, prev_loss, epoch_acc, mean_time, epoch_time, total_time, device):
    print('Epoch {} summary:'.format(epoch + 1))
    print(' * Average loss      : {} ({}{})'.format(epoch_loss, '+' if epoch_loss > prev_loss else '', epoch_loss - prev_loss))
    print(' * Average accuracy  : {} %'.format(int(100*epoch_acc)))
    print(' * Total elapsed     :', fuzzy_time(total_time))
    print(' * Epoch time        :', fuzzy_time(epoch_time))
    print(' * Avg. sample time  :', fuzzy_time(mean_time))
    print(' * Approx. time left :', fuzzy_time((NUM_EPOCHS - epoch)*epoch_time))
    print_cuda_runtime_info(device)

def get_device():
    global USING_CUDA
    USING_CUDA = not FORCE_CPU and torch.cuda.is_available()
    device = torch.device('cuda') if USING_CUDA else torch.device('cpu')
    print_device_info(device)
    return device

def load_training_data_with_file(file, basePath=None):
    print('Load training data...')
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
    print('Loaded {} samples in {}'.format(len(images), fuzzy_time(time.perf_counter() - start)))
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
    ppImages = np.zeros((IMAGE_WIDTH,     count, IMAGE_HEIGHT), dtype=np.float32)
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
        for j in range(min(width, IMAGE_WIDTH)):
            for k in range(min(height, IMAGE_HEIGHT)):
                ppImages[j, i, k] = image[k*width + j]/255.0
        for j in range(SEQUENCE_LENGTH):
            for k in range(NUM_TOKENS):
                ppLabels[j, i, k] = labels[i][j][k]
    ppImages = torch.tensor(ppImages, device=device)
    ppLabels = torch.tensor(ppLabels, device=device)
    print('Preprocessed training set in {}'.format(fuzzy_time(time.perf_counter() - start)))
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

        y_pred_n   = torch.argmax(y_pred, dim=-1)
        y_target_n = torch.argmax(y_target, dim=-1)
        acc        = accuracy(y_pred_n, y_target_n)

        batch_losses[count]     = loss.item()
        batch_times[count]      = time.perf_counter() - batch_start_time
        batch_accuracies[count] = acc
        count += 1

        print(
            '\t{}\t{}%\t\t{:22} \'{}\''.format(
                count,
                int(100*count/num_batches),
                '\'' + to_sentence(y_target[:, -1, :]).replace('\n', '\\n') + '\'',
                to_sentence(y_pred[:, -1, :]).replace('\n', '\\n')
            )
        )

def train_epochs(images, labels, device, model=None):
    if model is None:
        print('Start training...')
    else:
        print('Continue training...')
    if os.environ.get('DEBUG') == '1':
        print('Starting in debug mode')
        autograd.set_detect_anomaly(True)
    ocr               = TorchOcrProvider(device, model=model)
    model             = ocr.model
    loss_fn           = nn.MSELoss()
    encoder_optimizer = optim.Adam(ocr.model.encoder.parameters(), lr=0.0001)
    decoder_optimizer = optim.Adam(ocr.model.decoder.parameters(), lr=0.0001)
    start_time        = time.perf_counter()
    prev_loss         = 0
    num_batches       = images.size(1)//BATCH_SIZE
    batch_losses      = [0]*num_batches
    batch_accuracies  = [0]*num_batches
    batch_times       = [0]*num_batches
    try:
        for epoch in range(NUM_EPOCHS):
            print('========================================[ Epoch {} ]========================================'.format(epoch + 1))
            print('\tSamples\tProgress\tExpected\t\tActual')
            epoch_start_time = time.perf_counter()

            batches = list(create_batches(images, labels, device, num_batches, shuffle=True))
            train_batches(
                batches,
                device,
                ocr,
                encoder_optimizer,
                decoder_optimizer,
                loss_fn,
                num_batches,
                batch_losses,
                batch_accuracies,
                batch_times
            )

            epoch_loss = np.mean(batch_losses)
            epoch_acc  = np.mean(batch_accuracies)
            mean_time  = np.mean(batch_times)
            epoch_time = time.perf_counter() - epoch_start_time
            total_time = time.perf_counter() - start_time
            print_epoch_summary(epoch, num_batches, epoch_loss, prev_loss, epoch_acc, mean_time, epoch_time, total_time, device)
            prev_loss  = epoch_loss
    except (KeyboardInterrupt, SystemExit):
        print('Training stopped early')
    finally:
        return ocr.model

def train_ocr():
    modelName = 'model'
    if len(sys.argv) > 1:
        modelName = sys.argv[1]
    model = None
    device         = get_device()
    images, labels = load_training_data()
    images, labels = preprocess_training_data(images, labels, device)
    model          = FileManager.loadOcrModelIfExists(name=modelName)
    model          = train_epochs(images, labels, device, model=model)
    FileManager.saveOcrModel(model)
    print('Fin.')
    exit(0)


def test_ocr():
    if len(sys.argv) < 2:
        print('Usage: {} <model name>'.format(sys.argv[1]), file=sys.stderr)
        exit(1)
    modelName         = sys.argv[1]
    device            = get_device()
    images, labels    = load_training_data('test.csv')
    images, labels    = preprocess_training_data(images, labels, device)
    model             = FileManager.loadOcrModel(name=modelName)
    ocr               = TorchOcrProvider(device, model=model)
    num_batches       = len(images)
    batches           = list(create_batches(images, labels, device, num_batches, shuffle=True))
    loss_fn           = nn.MSELoss()
    encoder_optimizer = optim.Adam(ocr.model.encoder.parameters(), lr=0.0001)
    decoder_optimizer = optim.Adam(ocr.model.decoder.parameters(), lr=0.0001)
    start_time        = time.perf_counter()
    num_batches       = images.size(1)//BATCH_SIZE
    batch_losses      = [0]*num_batches
    batch_accuracies  = [0]*num_batches
    batch_times       = [0]*num_batches

    train_batches(
        batches,
        device,
        ocr,
        encoder_optimizer,
        decoder_optimizer,
        loss_fn,
        num_batches,
        batch_losses,
        batch_accuracies,
        batch_times
    )

    epoch_loss = np.mean(batch_losses)
    epoch_acc  = np.mean(batch_accuracies)
    mean_time  = np.mean(batch_times)
    epoch_time = time.perf_counter() - start_time
    total_time = time.perf_counter() - start_time
    print_epoch_summary(0, num_batches, epoch_loss, 0, epoch_acc, mean_time, epoch_time, total_time, device)

