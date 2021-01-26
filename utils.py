# helper functions
import math
import numpy as np
import logging

def print_divider():
    print('-------')


def get_commit_hash():
    return ''


def get_filename_no_ext(full_name):
    '''Returns a filename stripped of its extension, without any slash
    Return full_name itself if there is no dot
    '''
    pos = full_name.rfind('.')
    if pos < 0:
        return full_name
    return full_name[:pos]


def print_size_in_mb(ar):
    print('itemsize : %s' % ar.itemsize)
    print('size : %s' % ar.size)
    print('total : %s mb' % (ar.size * ar.itemsize/1024**2))


def size_in_mb(s):
    return s/1024**2


def get_normalized_radius(i, j, middle):
    return math.sqrt(math.pow(i-middle-1/2, 2) + math.pow(-j+middle+1/2, 2))/middle


def get_arctan_ratio(x, y, middle):
    return np.divide(-y+middle+1/2, x-middle-1/2)


def get_arctan_offset(x, y, middle):
    new_x = x-middle-1/2
    new_y = -y+middle+1/2
    if(new_x >= 0 and new_y >= 0):
        return 0
    if(new_x < 0 and new_y >= 0):
        return np.pi
    if(new_x < 0 and new_y < 0):
        return np.pi
    return 2*np.pi
