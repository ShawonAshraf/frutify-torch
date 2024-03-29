# source : https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#transforms
import torch
from skimage import io, transform
import numpy as np


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, item):
        image, label, file_name = item["image"], item["label"], item["file_name"]

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w))

        return {
            "image": image,
            "label": label,
            "file_name": file_name
        }


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, item):
        image, label, file_name = item["image"], item["label"], item["file_name"]

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        return {
            "image": image,
            "label": label,
            "file_name": file_name
        }


class RotateImage(object):
    """
        Rotates image by a specified angle
        angle is a float value

        -90, 90, 180, -180 can be possible values

        270 may not preserve the shape of the image
    """

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, item):
        image, label, file_name = item["image"], item["label"], item["file_name"]

        image = transform.rotate(image, angle=self.angle)

        return {
            "image": image,
            "label": label,
            "file_name": file_name
        }


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, item):
        image, label, file_name = item["image"], item["label"], item["file_name"]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        return {
            "image": torch.from_numpy(image).float(),
            "label": torch.tensor(label),
            "file_name": file_name
        }
