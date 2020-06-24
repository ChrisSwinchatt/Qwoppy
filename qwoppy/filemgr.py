import os
import sys

import numpy as np

from PIL import Image

import torch

BASE_PATH = os.path.join(os.getcwd(), os.path.dirname(sys.argv[0]), 'data')

class FileManager:
    @classmethod
    def openOcrCsv(cls, name, basePath=None):
        return open(cls._join(name, basePath=basePath))
    
    @classmethod
    def openSentence(cls, name, basePath=None):
        i = Image.open(cls._join('images', 'sentences', name, basePath=basePath), )
        filename = i.filename
        i = i.convert('L')
        if not hasattr(i, 'filename'):
            setattr(i, 'filename', filename)
        return i

    @classmethod
    def saveOcrModel(cls, model, name='model', basePath=None):
        torch.save(model, cls._join('models', name + '.pkl', basePath=basePath))

    @classmethod
    def loadOcrModelIfExists(cls, name='model', basePath=None):
        p = cls._join('models', name + '.pkl', basePath=basePath)
        if os.path.exists(p):
            return torch.load(p)

    @classmethod
    def loadOcrModel(cls, name='model', basePath=None):
        return torch.load(cls._join(basePath, 'models', name + '.pkl', basePath=basePath))

    @classmethod
    def _join(cls, *args, basePath=None):
        if basePath is None:
            basePath = BASE_PATH
        return os.path.join(basePath, 'ocr', *args)
