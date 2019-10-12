from   abc                            import ABC, abstractmethod, abstractproperty
import logging                        as     log
from   PIL                            import ImageGrab
from   selenium                       import webdriver
from   selenium.webdriver.common.keys import Keys
from   selenium.webdriver.common.by   import By

class QwopController(ABC):
    @abstractproperty
    def location(self):
        raise NotImplementedError

    @abstractproperty
    def size(self):
        raise NotImplementedError

    @abstractproperty
    def distance_rect(self):
        raise NotImplementedError

    @abstractmethod
    def get_image(self):
        raise NotImplementedError

    @abstractmethod
    def click(self):
        raise NotImplementedError
    
    @abstractmethod
    def sendQ(self):
        raise NotImplementedError
    
    @abstractmethod
    def sendW(self):
        raise NotImplementedError
    
    @abstractmethod
    def sendO(self):
        raise NotImplementedError
    
    @abstractmethod
    def sendP(self):
        raise NotImplementedError

    @abstractmethod
    def sendSpace(self):
        raise NotImplementedError


BROWSER_WIDTH  = 1024
BROWSER_HEIGHT = 768
QWOP_URL       = 'http://foddy.net/Athletics.html'
QWOP_OFFSET_X  = 200
QWOP_OFFSET_Y  = 90
QWOP_WIDTH     = 250
QWOP_HEIGHT    = 40

class SeleniumQwopController(QwopController):
    def __init__(self):
        import time
        log.info('Opening browser')
        self.browser = webdriver.Firefox()
        self.browser.set_window_position(0, 0)
        self.browser.set_window_size(BROWSER_WIDTH, BROWSER_HEIGHT)
        log.info('Loading QWOP')
        self.browser.get(QWOP_URL)
        time.sleep(1)
        self.canvas = self.browser.find_element_by_id('window1')

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
        return QWOP_OFFSET_X, QWOP_OFFSET_Y, QWOP_OFFSET_X + QWOP_WIDTH, QWOP_OFFSET_Y + QWOP_HEIGHT

    def get_image(self):
        x, y       = self.location
        _, _, w, h = self.distance_rect
        return ImageGrab.grab((x, y, x + w + 1, y + h + 1))

    def click(self):
        self.canvas.click()
    
    def sendQ(self):
        webdriver.ActionChains(self.browser).send_keys_to_element(self.canvas, 'q').perform()
    
    def sendW(self):
        webdriver.ActionChains(self.browser).send_keys_to_element(self.canvas, 'w').perform()
    
    def sendO(self):
        webdriver.ActionChains(self.browser).send_keys_to_element(self.canvas, 'o').perform()
    
    def sendP(self):
        webdriver.ActionChains(self.browser).send_keys_to_element(self.canvas, 'p').perform()

    def sendSpace(self):
        webdriver.ActionChains(self.browser).send_keys_to_element(self.canvas, ' ').perform()
