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
