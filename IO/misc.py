import torch

from skimage.transform import resize as skimage_resize


class Resize(object):
    """Rescale a numpy array to a given size.

    Args:
        output_size (tuple or int): Desired output size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (tuple))
        self.output_size = output_size

    def __call__(self, img):
        img = skimage_resize(img, self.output_size, mode='constant')
        return img


class ToFloatTensor(object):
    """Transforms the numpy array without adding a new dimension

    """

    def __call__(self, img):
        return torch.FloatTensor(img)
