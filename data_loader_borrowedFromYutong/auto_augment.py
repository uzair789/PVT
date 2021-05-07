from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random
import torch
import torchvision.transforms.functional as F
from torch import Tensor


import PIL
from PIL import Image

PIL.Image.MAX_IMAGE_PIXELS = 200000000

import os
import cv2

import mmcv
# import torchvision

class ImageNetPolicy(object):
    """ Randomly choose one of the best 24 Sub-policies on ImageNet.
        Example:
        >>> policy = ImageNetPolicy()
        >>> transformed = policy(image)
        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     ImageNetPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),

            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"


class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.
        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)
        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class SVHNPolicy(object):
    """ Randomly choose one of the best 25 Sub-policies on SVHN.
        Example:
        >>> policy = SVHNPolicy()
        >>> transformed = policy(image)
        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     SVHNPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.9, "shearX", 4, 0.2, "invert", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.7, "invert", 5, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.6, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 3, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "equalize", 1, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.8, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.4, "invert", 5, fillcolor),
            SubPolicy(0.9, "shearY", 5, 0.2, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 6, 0.8, "autocontrast", 1, fillcolor),
            SubPolicy(0.6, "equalize", 3, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.3, "solarize", 3, fillcolor),
            SubPolicy(0.8, "shearY", 8, 0.7, "invert", 4, fillcolor),
            SubPolicy(0.9, "equalize", 5, 0.6, "translateY", 6, fillcolor),
            SubPolicy(0.9, "invert", 4, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.3, "contrast", 3, 0.8, "rotate", 4, fillcolor),

            SubPolicy(0.8, "invert", 5, 0.0, "translateY", 2, fillcolor),
            SubPolicy(0.7, "shearY", 6, 0.4, "solarize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 0.8, "rotate", 4, fillcolor),
            SubPolicy(0.3, "shearY", 7, 0.9, "translateX", 3, fillcolor),
            SubPolicy(0.1, "shearX", 6, 0.6, "invert", 5, fillcolor),

            SubPolicy(0.7, "solarize", 2, 0.6, "translateY", 7, fillcolor),
            SubPolicy(0.8, "shearY", 4, 0.8, "invert", 8, fillcolor),
            SubPolicy(0.7, "shearX", 9, 0.8, "translateY", 3, fillcolor),
            SubPolicy(0.8, "shearY", 5, 0.7, "autocontrast", 3, fillcolor),
            SubPolicy(0.7, "shearX", 2, 0.1, "invert", 5, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment SVHN Policy"


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
        return img


class Cutout(object):
    def __init__(self, length=16):
        self.length = length

    def __call__(self, img):
        img = np.array(img)

        mask_val = img.mean()

        top = np.random.randint(0 - self.length//2, img.shape[0] - self.length)
        left = np.random.randint(0 - self.length//2, img.shape[1] - self.length)
        bottom = top + self.length
        right = left + self.length

        top = 0 if top < 0 else top
        left = 0 if left < 0 else top

        img[top:bottom, left:right, :] = mask_val

        img = Image.fromarray(img)

        return img

class RandomResize(object):
    def __init__(self, scale=(0.5, 1), prob=0.7):
        self.scale = scale
        self.prob = prob

    def __call__(self, img):
        original_size = img.size

        if np.random.rand() < self.prob:
            scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
            img = img.resize((int(original_size[0] * scale), int(original_size[1] * scale)))
            img = img.resize(original_size)

        return img

class HueNormalize(torch.nn.Module):
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def get_im_hue(self, img):
        hue = img.new_zeros(img.shape[1], img.shape[2])

        hue[img[2] == img.max(0)[0]] = 4.0 + ((img[0] - img[1]) / (img.max(0)[0] - img.min(0)[0] + 1e-7))[
            img[2] == img.max(0)[0]]
        hue[img[1] == img.max(0)[0]] = 2.0 + ((img[2] - img[0]) / (img.max(0)[0] - img.min(0)[0] + 1e-7))[
            img[1] == img.max(0)[0]]
        hue[img[0] == img.max(0)[0]] = (0.0 + ((img[1] - img[2]) / (img.max(0)[0] - img.min(0)[0] + 1e-7))[
            img[0] == img.max(0)[0]]) % 6

        hue[img.min(0)[0] == img.max(0)[0]] = 0.0
        hue = hue / 6
        return torch.cat([img, hue.unsqueeze(0)], dim=0)

    def forward(self, tensor: Tensor) -> Tensor:
        tensor = self.get_im_hue(tensor)

        return F.normalize(tensor, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class SquarePad():
    def __init__(self, fill=None):
        if fill is None:
            fill = np.random.randint(0, 256)
        self.fill = fill

    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, self.fill, 'constant')


def rotatexy(h, w, f=1, rotation=(0, 0, 0)):
    r_x, r_y, r_z = rotation
    # convert degree angles to rad
    theta_rx = np.deg2rad(r_x)
    theta_ry = np.deg2rad(r_y)
    theta_rz = np.deg2rad(r_z)
    # set the image from cartesian to projective dimension
    H_M = np.array([[1, 0, -w / 2],
                    [0, 1, -h / 2],
                    [0, 0, 1],
                    [0, 0, 1]])
    # set the image projective to carrtesian dimension
    Hp_M = np.array([[f, 0, w / 2, 0],
                     [0, f, h / 2, 0],
                     [0, 0, 1, 0]])
    # calculate cos and sin of angles
    sin_rx, cos_rx = np.sin(theta_rx), np.cos(theta_rx)
    sin_ry, cos_ry = np.sin(theta_ry), np.cos(theta_ry)
    sin_rz, cos_rz = np.sin(theta_rz), np.cos(theta_rz)
    R_Mx = np.array([[1, 0, 0, 0],
                     [0, cos_rx, -sin_rx, 0],
                     [0, sin_rx, cos_rx, 0],
                     [0, 0, 0, 1]])
    R_My = np.array([[cos_ry, 0, -sin_ry, 0],
                     [0, 1, 0, 0],
                     [sin_ry, 0, cos_ry, 0],
                     [0, 0, 0, 1]])
    R_Mz = np.array([[cos_rz, -sin_rz, 0, 0],
                     [sin_rz, cos_rz, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    R_M = np.dot(np.dot(R_Mx, R_My), R_Mz)
    # compute the full transform matrix
    return np.dot(Hp_M, np.dot(np.dot(R_M, np.eye(4, 4)), H_M))


class fangyi_sc_transform(object):

    def __init__(self):
        self.warp_list = [i for i in range(1, 16)]
        self.view_list = [i for i in range(5)]
        self.angle_list = [float(i) / 1000. for i in range(0, 110, 3)]
        self.final_size = (1000, 256)

    def rescale(self, im, scale):
        im, scale_factor = mmcv.imrescale(im, scale, return_scale=True)
        return im

    def detection(self, image, threshold=1):
        try:
            # h, w, _ = image.shape
            # x1, x2, y1, y2 = 0, h - 1, 0, w - 1
            # while image[x1].max() == image[x1].min():
            #     x1 += 1
            # while image[x2 - 1].max() == image[x2 - 1].min():
            #     x2 -= 1
            # while image[:, y1].max() == image[:, y1].min():
            #     y1 += 1
            # while image[:, y2 - 1].max() == image[:, y2 - 1].min():
            #     y2 -= 1
            img = image.max(2)

            residual_x = img.max(1) - img.min(1)
            non_zero = np.where(residual_x > threshold)[0]
            x1, x2 = non_zero.min(), non_zero.max()

            residual_y = img.max(0) - img.min(0)
            non_zero = np.where(residual_y > threshold)[0]
            y1, y2 = non_zero.min(), non_zero.max()
            im = image[x1:x2, y1:y2, :]
        except:
            im = image
        return im

    def warp(self, im, ind):
        h, w, _ = im.shape
        rect = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        # idendity
        dst1 = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        # sym left short
        dst2 = np.array([[0, h / 3], [w, 0], [w, h], [0, h * 2 / 3]], dtype=np.float32)
        dst3 = np.array([[0, h / 4], [w, 0], [w, h], [0, h * 3 / 4]], dtype=np.float32)
        dst4 = np.array([[0, h / 5], [w, 0], [w, h], [0, h * 4 / 5]], dtype=np.float32)
        # sym right short
        dst5 = np.array([[0, 0], [w, h / 3], [w, h * 2 / 3], [0, h]], dtype=np.float32)
        dst6 = np.array([[0, 0], [w, h / 4], [w, h * 3 / 4], [0, h]], dtype=np.float32)
        dst7 = np.array([[0, 0], [w, h / 5], [w, h * 4 / 5], [0, h]], dtype=np.float32)
        # keep upper left short
        dst8 = np.array([[0, 0], [w, 0], [w, h], [0, h * 2 / 3]], dtype=np.float32)
        dst9 = np.array([[0, 0], [w, 0], [w, h], [0, h * 3 / 4]], dtype=np.float32)
        dst10 = np.array([[0, 0], [w, 0], [w, h], [0, h * 4 / 5]], dtype=np.float32)
        # keep upper right short
        dst11 = np.array([[0, 0], [w, 0], [w, h * 2 / 3], [0, h]], dtype=np.float32)
        dst12 = np.array([[0, 0], [w, 0], [w, h * 3 / 4], [0, h]], dtype=np.float32)
        dst13 = np.array([[0, 0], [w, 0], [w, h * 4 / 5], [0, h]], dtype=np.float32)
        # top down view
        dst14 = np.array([[0, 0], [w, 0], [w * 3 / 4, h], [w / 4, h]], dtype=np.float32)
        dst15 = np.array([[0, 0], [w, 0], [w * 4 / 5, h], [w / 5, h]], dtype=np.float32)
        dst16 = np.array([[0, 0], [w, 0], [w * 5 / 6, h], [w / 6, h]], dtype=np.float32)

        dst_bank = [dst1, dst2, dst3, dst4, dst5, dst6, dst7, dst8, dst9, dst10, dst11, dst12, dst13, dst14, dst15,
                    dst16]
        dst = dst_bank[ind]

        T = cv2.getPerspectiveTransform(rect, dst)
        im = cv2.warpPerspective(im, T, (w, h), borderValue=(255, 255, 255))
        return im

    def rotatexy(self, im, angle, ind):
        h, w, _ = im.shape
        degree = angle
        M1 = rotatexy(h=h, w=0, f=1, rotation=(0, degree, 0))  # rotate alone y axis around (h/2, 0)
        M2 = rotatexy(h=0, w=0, f=1, rotation=(0, degree, 0))  # rotate along y axis around (0, 0)
        M3 = rotatexy(h=h, w=2 * w, f=1, rotation=(0, -degree, 0))  # rotate along y axis around (h/2, w)
        M4 = rotatexy(h=0, w=2 * w, f=1, rotation=(0, -degree, 0))  # rotate along y axis around (0, w)
        M5 = rotatexy(h=0, w=w, f=1, rotation=(degree, 0, 0))  # rotate along x axis around (0, w/2)
        M_bank = [M1, M2, M3, M4, M5]
        M = M_bank[ind]
        im = cv2.warpPerspective(im, M, (w, h), borderValue=(255, 255, 255))
        return im

    def __call__(self, image):

        ind = np.random.randint(len(self.warp_list) + len(self.view_list) + 1, size=1)[0]
        angle = np.random.randint(len(self.angle_list), size=1)[0]
        im = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        im = self.rescale(im, scale=(800, 800))  # initial rescale for convenience. Will rescale again
        im = self.detection(im)  # original

        if ind == 0:
            keep = self.rescale(im, scale=self.final_size)
        elif 0 < ind <= len(self.warp_list):
            i = ind - 1
            keep = self.rescale(self.detection(self.warp(im, ind=i)), scale=self.final_size)
        elif len(self.warp_list) < ind:
            i = ind - 1 - len(self.warp_list)
            angle = self.angle_list[angle]
            keep = self.rescale(self.detection(self.rotatexy(im, angle=angle, ind=i)), scale=self.final_size)
        else:
            raise ValueError

        img = cv2.cvtColor(keep, cv2.COLOR_BGR2RGB)

        return Image.fromarray(img)
