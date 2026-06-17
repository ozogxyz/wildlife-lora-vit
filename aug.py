"""cv2 image augmentations operating on dict samples ({"image": np.ndarray, ...}).

The interesting bits over torchvision: color-temperature warping via LUT splines
and gamma-aware brightness. Lifted from the MIPT spoof thesis, trimmed.
"""

import random

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import UnivariateSpline
from torchvision.transforms import Compose as TorchvisionCompose
from torchvision.transforms import Lambda
from torchvision.transforms import functional as TVF


def adjust_brightness(src, brightness_factor):
    """0 -> black, 1 -> identity, 2 -> 2x brighter."""
    return np.array(TVF.adjust_brightness(Image.fromarray(src), brightness_factor))


def adjust_contrast(src, contrast_factor):
    """0 -> solid gray, 1 -> identity, 2 -> 2x contrast."""
    return np.array(TVF.adjust_contrast(Image.fromarray(src), contrast_factor))


def adjust_hue(img, hue_factor):
    """Cyclic shift of the H channel in HSV. hue_factor in [-0.5, 0.5]."""
    return np.array(TVF.adjust_hue(Image.fromarray(img), hue_factor))


def adjust_gamma(img, gamma, gain=1):
    """Power-law transform: out = 255 * gain * (in/255)**gamma. gamma < 1 lifts shadows."""
    if gamma < 0:
        raise ValueError("Gamma should be a non-negative real number")
    return np.array(TVF.adjust_gamma(Image.fromarray(img), gamma, gain))


def adjust_saturation(img, factor=0.0):
    """-1 -> grayscale, 0 -> identity, 1 -> oversaturated."""
    return np.array(TVF.adjust_saturation(Image.fromarray(img), factor))


def create_LUT_8UC1(x, y):
    spl = UnivariateSpline(x, y)
    return spl(range(256))


def adjust_temperature(img, factor=1):
    """Warp color temperature of a BGR image by reshaping the B/R histograms.

    factor in [-1, 1]: +1 == warm, -1 == cold. Source:
    http://www.askaswiss.com/2016/02/how-to-manipulate-color-temperature-opencv-python.html
    """
    if not (-1 <= factor <= 1):
        raise ValueError(f"{factor} is not in [-1, 1].")
    if factor == 0:
        return img

    inp = [0, 64, 128, 192, 256]
    dest_warm = [0, 70, 140, 210, 256]
    dest_cold = [0, 30, 80, 120, 192]

    abs_factor = abs(factor)
    delta = np.array(dest_warm) - np.array(inp)
    dest_warm = (np.array(inp) + np.array(delta) * abs_factor).tolist()
    delta = np.array(dest_cold) - np.array(inp)
    dest_cold = (np.array(inp) + np.array(delta) * abs_factor).tolist()

    incr_ch_lut = create_LUT_8UC1(inp, dest_warm)
    decr_ch_lut = create_LUT_8UC1(inp, dest_cold)

    if factor > 0:
        incr_ch_lut, decr_ch_lut = decr_ch_lut, incr_ch_lut

    # warm -> boost R, drop B
    img[:, :, 2] = cv2.LUT(img[:, :, 2], incr_ch_lut).astype(np.uint8)
    img[:, :, 0] = cv2.LUT(img[:, :, 0], decr_ch_lut).astype(np.uint8)

    # warm -> boost saturation (and vice versa)
    hsv_lut = incr_ch_lut if factor < 0 else decr_ch_lut
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 1] = cv2.LUT(img_hsv[:, :, 1], hsv_lut).astype(np.uint8)
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


class RandomHorizontalFlip:
    """Flip the tagged images left-right with probability p."""

    def __init__(self, p=0.5, tags=("image",)):
        self.p = p
        self.tags = tags

    def __call__(self, sample_dict):
        to_flip = random.random() < self.p
        for tag in set(self.tags).intersection(sample_dict.keys()):
            if to_flip:
                sample_dict[tag] = np.fliplr(sample_dict[tag].copy())
        return sample_dict

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


class ColorJitterCV:
    """Randomly jitter brightness / contrast / saturation / hue / gamma / color temp.

    Each arg is a scalar s (range becomes symmetric around the identity) or an
    explicit [lo, hi] pair. Factors are clipped to sane bounds: brightness &
    contrast (0, inf), hue (-0.5, 0.5), gamma (0.5, 1.5), temp (-1, 1).
    p is the probability the whole jitter is applied.
    """

    def __init__(
        self,
        brightness=0,
        contrast=0,
        saturation=0,
        hue=0,
        gamma=0,
        temp=0,
        p=0.75,
        tags=("image",),
    ):
        self.brightness = (
            brightness
            if self.check_param(brightness)
            else [max(0, 1 - brightness), 1 + brightness]
        )
        self.contrast = (
            contrast
            if self.check_param(contrast)
            else [max(0, 1 - contrast), 1 + contrast]
        )
        self.saturation = (
            saturation if self.check_param(saturation) else [-saturation, saturation]
        )
        self.hue = hue if self.check_param(hue) else [-hue, hue]
        self.gamma = gamma if self.check_param(gamma) else [1 - gamma, 1 + gamma]
        self.temp = temp if self.check_param(temp) else [-temp, temp]

        self.brightness = np.clip(self.brightness, 0, None)
        self.contrast = np.clip(self.contrast, 0, None)
        self.hue = np.clip(self.hue, -0.5, 0.5)
        self.gamma = np.clip(self.gamma, 0.5, 1.5)
        self.temp = np.clip(self.temp, -1, 1)

        self.p = p
        self.tags = tags

    @staticmethod
    def check_param(param):
        return hasattr(param, "__len__") and len(param) == 2

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness},"
            f"contrast={self.contrast},"
            f"saturation={self.saturation},"
            f"hue={self.hue},"
            f"gamma={self.gamma},"
            f"temp={self.temp})"
        )

    @staticmethod
    def get_params(brightness, contrast, saturation, hue, gamma, temp):
        """Build a randomized, shuffled Compose of the active sub-transforms."""
        transforms = []

        gamma_factor = 1
        if not np.allclose(gamma, 1):
            gain_factor = 1
            gamma_factor = np.clip(random.uniform(gamma[0], gamma[1]), 0.5, 1.5)
            transforms.append(
                Lambda(lambda img: adjust_gamma(np.array(img), gamma_factor, gain_factor))
            )
        if not np.allclose(brightness, 1):
            if gamma_factor < 1 and brightness[1] > 1:
                brightness_factor = random.uniform(1, brightness[1])
            elif gamma_factor > 1 and brightness[0] < 1:
                brightness_factor = random.uniform(brightness[0], 1)
            elif gamma_factor == 1:
                brightness_factor = random.uniform(brightness[0], brightness[1])
            else:
                brightness_factor = 1
            transforms.append(
                Lambda(lambda img: adjust_brightness(img, brightness_factor))
            )
        if not np.allclose(contrast, 1):
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: adjust_contrast(img, contrast_factor)))
        if not np.allclose(saturation, 0):
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(
                Lambda(lambda img: adjust_saturation(img, saturation_factor))
            )
        if not np.allclose(temp, 0):
            temp_factor = random.uniform(temp[0], temp[1])
            transforms.append(Lambda(lambda img: adjust_temperature(img, temp_factor)))
        if not np.allclose(hue, 1):
            hue_factor = float(np.clip(random.uniform(hue[0], hue[1]), -0.5, 0.5))
            transforms.append(Lambda(lambda img: adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        return TorchvisionCompose(transforms)

    def __call__(self, sample_dict):
        transform_func = self.get_params(
            self.brightness,
            self.contrast,
            self.saturation,
            self.hue,
            self.gamma,
            self.temp,
        )
        if np.random.random() < self.p:
            for tag in self.tags:
                sample_dict[tag] = transform_func(sample_dict[tag].squeeze())
        return sample_dict


class RandomGaussianBlur:
    """Gaussian blur with sigma drawn from `radius`, applied with probability p."""

    def __init__(self, radius=(1, 3), tags=("image",), p=0.5):
        self.p = p
        self.tags = tags
        assert (
            type(radius) is tuple and len(radius) == 2
        ) or (type(radius) is int and radius > 1), "Wrong input: radius"
        self.radius = (1, radius) if isinstance(radius, float) else radius

    def get_params(self):
        # NOTE: deviates from the thesis, which had `radius[1] - radius[1]` (always 0),
        # pinning sigma at radius[0]. Here sigma is drawn across the full [lo, hi] range.
        return {"sigmaX": self.radius[0] + random.random() * (self.radius[1] - self.radius[0])}

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, radius={self.radius}"

    def __call__(self, sample_dict):
        params = self.get_params()
        # NOTE: deviates from the thesis, which fired on `> self.p` (inverted; benign
        # only at p=0.5). Here blur fires with probability p, as the name implies.
        if random.random() < self.p:
            for tag in self.tags:
                sample_dict[tag] = cv2.GaussianBlur(sample_dict[tag].copy(), None, **params)
        return sample_dict
