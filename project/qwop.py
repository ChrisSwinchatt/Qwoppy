import logging                        as     log
import PIL.ImageGrab                  as     ImageGrab
from   selenium                       import webdriver
from   selenium.webdriver.common.keys import Keys
from   selenium.webdriver.common.by   import By

OFFSET_X = 200
OFFSET_Y = 95
WIDTH    = 290
HEIGHT   = 40

class QwopController:
    def __init__(self, ocr):
        log.debug('Opening browser')
        self.browser = webdriver.Firefox()
        log.debug('Loading QWOP')
        self.browser.get('http://foddy.net/Athletics.html')
        self.canvas = self.browser.find_element_by_id('gameContent')
        self.ocr    = ocr

    def __del__(self):
        self.browser.close()

    @property
    def location(self):
        return self.canvas.location['x'], self.canvas.location['y']

    @property
    def size(self):
        return self.canvas.size['width'], self.canvas.size['height']

    @property
    def distance_rect(self):
        x, y = self.location
        x += OFFSET_X
        y += OFFSET_Y
        return x, y, x + WIDTH, y + HEIGHT

    def get_distance(self):
        rect  = self.distance_rect
        image = ImageGrab.grab(rect)
        text  = self.ocr(image)
        image.show()
        return float(text.split(' ')[0])

    def click(self):
        self.canvas.click()
    
    def q(self):
        self.canvas.key_down('q')
        self.canvas.key_up('q')
    
    def w(self):
        self.canvas.key_down('w')
        self.canvas.key_up('w')
    
    def o(self):
        self.canvas.key_down('o')
        self.canvas.key_up('o')
    
    def p(self):
        self.canvas.key_down('p')
        self.canvas.key_up('p')

    def space(self):
        self.canvas.key_down(' ')
        self.canvas.key_up(' ')
