from os.path import join as pjoin
from os.path import dirname as getdir
from os.path import basename as getbase
from os.path import splitext
from tqdm.auto import tqdm
import math, random
import numpy as np
import time
from glob import glob
import csv
import torch
from typing import List
import os

def restore_pic(x):
    """
    A function to restore image data from a normalized and rearranged format back to its original format.
    Parameters:
        x (numpy array) : The input image data. Shape is (batch x C x H x W) and values are in range [0,1]
    Returns:
        numpy array : The restored image. Shape is (H x W x C) and values are in range [0,255]
    """
    x = x * 255
    x = x.clamp_(0, 255)
    x = jnp.round(x)
    x = x.astype(jnp.uint8)
    x = jnp.transpose(x, (0, 2, 3, 1))
    return x


def slice_idxlist(start, end, array_length, need_slice_remain=True):
    """
    Returns two lists of indices for the main slice and the remaining slice.
    The main slice contains indices from start to end, wrapping around the array if necessary.
    The remaining slice contains all other indices.

    Args:
    start (int): Start index of the main slice.
    end (int): End index of the main slice.
    array_length (int): Length of the array to be sliced.

    Returns:
    tuple: A tuple containing two lists of indices (main_slice_indices, remaining_slice_indices).
    """

    main_slice_indices = [i % array_length for i in range(start, end)]
    remaining_slice_indices = [i % array_length for i in range(end, start + array_length)]

    return main_slice_indices, remaining_slice_indices


def write_dict_to_csv(data: dict, file_path):
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)

def become_deterministic(seed=0):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False # train speed is slower after enabling this opts.

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True)


def dict_add(dictionary: dict, key, value, acc='list'):
    """
    This function allows the addition of a value to an existing key in a dictionary or initialises a new key with a list or set.
    
    :param dictionary: dict. The main dictionary where the key-value pair will be added.
    :param key: The key which maps to the value.
    :param value: The value to be added for the associated key.
    :param acc: 'list' or 'set'. Determines whether a list or set should be initialized if the key isn't already present in dictionary.
    :return: None. The function modifies the dictionary in-place.
    :raises AssertionError: If acc parameter isn't 'list' or 'set'.
    """
    if key not in dictionary.keys():
        if acc=='list':
            dictionary[key] = []
        elif acc=='set':
            dictionary[key] = set()
        else:
            assert False, 'acc must either be "list" or "set"'
    dictionary[key].append(value)
    
class measure_time:
    """
    This class is used as a context manager to measure the execution time of a block of code.

    When used in a `with` statement, it will print the time elapsed since entering the block upon exiting the block.
    """
    def __init__(self):
        pass
    
    def __enter__(self):
        """
        Start the timer as the context is entered.
        """
        self.start_time = time.time()
        
    def __exit__(self, type, value, traceback):
        """
        Stop the timer as the context is exited and print the time elapsed.
        """
        print('time elapsed', time.strftime("%H:%M:%S", time.gmtime(time.time() - self.start_time)))


def get_str_after_substring(text:str, substring:str):
    """
    This function gets a substring and the text after it from the original text string.
    
    :param text: str. The original text string from which a substring is to be identified.
    :param substring: str. The substring to be identified in the original text.
    :return: str/None. The substring and the text after it in the original text string, or None if the substring is not in the original text.
    """
    index = text.find(substring)
    if index >= 0:
        next_char = text[index + len(substring):]
        return substring + next_char
    else:
        return None

def fn_comb(kwargs: List):
    def comb(X):
        for fn in kwargs:
            X = fn(X)
        return X
    return comb



class printc:
    """colorful print, but now I want colorul logging to show the message"""
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    END = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    BLACK = "\033[1;30m"
    RED = "\033[1;31m"
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[1;34m"
    PURPLE = "\033[1;35m"
    CYAN = "\033[1;36m"
    WHITE = "\033[1;37m"

    @staticmethod
    def blue(*text):
        printc.uni(printc.BLUE, text)
    @staticmethod
    def green(*text):
        printc.uni(printc.GREEN, text)
    @staticmethod
    def yellow(*text):
        printc.uni(printc.YELLOW, text)
    @staticmethod
    def red(*text):
        printc.uni(printc.RED, text)
    @staticmethod
    def uni(color, text:tuple):
        print(color + ' '.join([str(x) for x in text]) + printc.END)

class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


# utils performed by np
def accuracy(y_pred, y_test):
    """
    This function calculates the accuracy of mean prediction of Gaussian Process
    :param y_pred: np.ndarray. Prediction of Gaussian Process.
    :param y_test: np.ndarray. Ground truth label.
    :return: a float for accuracy.
    """
    return np.mean(np.argmax(y_pred, axis=-1) == np.argmax(y_test, axis=-1))

# Data Loading
def _partial_flatten_and_normalize(x):
    """Flatten all but the first dimension of an `np.ndarray`."""
    x = np.reshape(x, (x.shape[0], -1))
    return (x - np.mean(x)) / np.std(x)

def _flatten(x):
    """Flatten all but the first dimension of an `np.ndarray`."""
    return np.reshape(x, (x.shape[0], -1))/255

def _normalize(x):
    """Flatten all but the first dimension of an `np.ndarray`."""
    return x / 255


def _one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)