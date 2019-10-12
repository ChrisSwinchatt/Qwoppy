import time

from project import QwopController, OcrProvider, PyTesseractOcrProvider

if __name__ == '__main__':
    print('Loading QWOP')
    qwop = QwopController(PyTesseractOcrProvider())
    time.sleep(3)
    qwop.click()
    while True:
        print(qwop.get_distance())
