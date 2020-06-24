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

from   abc     import ABC, abstractmethod, abstractproperty
from   PIL     import ImageTk
import tkinter as     tk

class UiProvider(ABC):
    @abstractmethod
    def set_title(self, title):
        raise NotImplementedError

    @abstractmethod
    def set_position(self, x, y):
        raise NotImplementedError

    @abstractmethod
    def set_distance(self, distance):
        raise NotImplementedError
        
    @abstractmethod
    def set_image(self, image):
        raise NotImplementedError

    @abstractmethod
    def update(self):
        raise NotImplementedError

    @abstractproperty
    def is_open(self):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

class TkUiProvider(UiProvider):
    def __init__(self, title_base='QWOP'):
        self.root = tk.Tk()
        self.root.wm_title(title_base)
        self.title_base = title_base
        
        self._currentScoreLabel = tk.Label(self.root)
        self._currentScoreLabel.pack(side=tk.TOP, fill=tk.X, expand=1)
        
        self._canvas = tk.Canvas(self.root)
        self._canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self._imageTk = None
        self._is_open = True
    
    def __del__(self):
        self._is_open = False

    def set_title(self, title):
        self.root.wm_title('{} - {}'.format(self.title_base, title))

    def set_position(self, x, y):
        self.root.wm_geometry('+{}+{}'.format(x, y))
    
    def set_size(self, w, h):
        self.root.wm_geometry('{}x{}'.format(w, h))

    def set_distance(self, distance):
        self._currentScoreLabel['text'] = 'Distance: {}'.format(distance)

    def set_image(self, image):
        self._imageTk = ImageTk.PhotoImage(image)
        x = abs((self._canvas.winfo_width()  - self._imageTk.width()))//2
        y = abs((self._canvas.winfo_height() - self._imageTk.height()))//2
        self._canvas.create_image((x, y), image=self._imageTk)

    def update(self):
        self.root.update()

    def close(self):
        self.root.close()

    @property
    def is_open(self):
        return self._is_open and self.root.state() == 'normal' and self.root.winfo_exists()
