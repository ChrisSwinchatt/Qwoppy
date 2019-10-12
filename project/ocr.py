from abc import ABC, abstractmethod

import pytesseract

class OcrProvider(ABC):
    @abstractmethod
    def __call__(self, image):
        pass

class PyTesseractOcrProvider(OcrProvider):
    def __call__(self, image):
        return pytesseract.image_to_string(image)
