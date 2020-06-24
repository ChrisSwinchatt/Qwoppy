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

import numpy as np

def probability(x):
    return np.random.uniform() < x

def fuzzy_time(sec):
    lines = []
    if sec >= 3600:
        h   = int(sec/3600)
        lines.append('{} hour{}'.format(h, '' if h == 1 else 's'))
        sec -= h*3600
    if sec >= 60:
        m = int(sec/60)
        lines.append('{} minute{}'.format(m, '' if m == 1 else 's'))
        sec -= m*60
    if sec >= 1:
        sec = int(sec)
        lines.append('{} second{}'.format(sec, '' if sec == 1 else 's'))
    elif not lines:
        lines.append('{} seconds'.format(round(sec, 2)))
    if len(lines) == 1:
        return lines[0]
    return ' and '.join((', '.join(lines[:-1]), lines[-1]))

def gigabytes(mem):
    return str(round(mem/1024**3,1)) + ' GB'
