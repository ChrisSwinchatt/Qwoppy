from   abc import ABC, abstractmethod
import re

import pytesseract

DISTANCE_REGEX = re.compile(r'(-?[0-9]+(\.[0-9]+)?) metres')

class OcrProvider(ABC):
    @abstractmethod
    def __call__(self, image):
        pass

class PyTesseractOcrProvider(OcrProvider):
    def __call__(self, image):
        text  = pytesseract.image_to_string(image)
        match = DISTANCE_REGEX.match(text)
        if match:
            return float(match.group(0).split(' ')[0])
        return text
