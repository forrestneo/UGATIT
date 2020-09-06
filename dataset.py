#encoding=utf-8
from multiprocessing import cpu_count
import numpy as np
from PIL import Image
import paddle
import os
import random
import os.path
import matplotlib.pyplot as plt
from work.hapi.vision.transform import transforms
import paddle.fluid as fluid

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir):
    images = []
    extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                item = (path, 0)
                images.append(item)

    return images



def custom_reader(root, transform=None):
    '''
    自定义reader
    '''
    def reader():
        samples = make_dataset(root)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root))
        for idx in range(len(samples)):
            path, target = samples[idx]
            img = np.array(Image.open(path).convert('RGB')).astype('float32')
            img = img/255.0             # 0 ~ 1
            img = (img - 0.5) / 0.5     # -1 ~ 1

            yield img #, target
    return reader




