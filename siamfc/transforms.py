from __future__ import absolute_import, division

import cv2
import numpy as np
import numbers
import torch

import ops  # from . import ops
import backbones

__all__ = ['SiamFCTransforms']


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class RandomStretch(object):

    def __init__(self, max_stretch=0.05):
        self.max_stretch = max_stretch

    def __call__(self, img):
        interp = np.random.choice([
            # 几种差值
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_NEAREST,
            cv2.INTER_LANCZOS4])
        # 函数原型： numpy.random.uniform(low, high, size)
        # 功能：从一个均匀分布[low, high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
        # low: 采样下界，float类型，默认值为0；
        # high: 采样上界，float类型，默认值为1；
        # size: 输出样本数目，为int或元组(tuple)
        # 类型，例如，size = (m, n, k), 则输出
        # m * n * k个样本，缺省时输出1个值。
        # 返回值：ndarray类型，其形状和参数size中描述一致。
        scale = 1.0 + np.random.uniform(
            -self.max_stretch, self.max_stretch)
        out_size = (
            round(img.shape[1] * scale),
            round(img.shape[0] * scale))
        # cv2.resize(src, (dst_w, dst_h), interpolation)  # src是输入图，dst是目标图，w是宽,h是高，interpolation是差值方法，差值方法共有五种
        return cv2.resize(img, out_size, interpolation=interp)


class CenterCrop(object):

    def __init__(self, size):
        # isinstance()函数来判断一个对象是否是一个已知的类型，类似type()。如果要判断两个类型是否相同推荐使用 isinstance()。
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        h, w = img.shape[:2]
        tw, th = self.size
        i = round((h - th) / 2.)
        j = round((w - tw) / 2.)

        npad = max(0, -i, -j)
        if npad > 0:
            avg_color = np.mean(img, axis=(0, 1))
            # cv2.copyMakeBorder()用来给图片添加边框，它有下面几个参数：
            # src：要处理的原图
            # top, bottom, left, right：上下左右要扩展的像素数
            # borderType：边框类型，这个就是需要关注的填充方式，详情请参考：BorderTypes
            # 其中默认方式和固定值方式最常用，我们详细说明一下：
            # 固定值填充（cv2.BORDER_CONSTANT),顾名思义，这种方式就是边框都填充成一个固定值，
            img = cv2.copyMakeBorder(
                img, npad, npad, npad, npad,
                cv2.BORDER_CONSTANT, value=avg_color)
            i += npad
            j += npad

        return img[i:i + th, j:j + tw]


class RandomCrop(object):

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        h, w = img.shape[:2]
        tw, th = self.size
        i = np.random.randint(0, h - th + 1)
        j = np.random.randint(0, w - tw + 1)
        return img[i:i + th, j:j + tw]


class ToTensor(object):

    def __call__(self, img):
        return torch.from_numpy(img).float().permute((2, 0, 1))


# 主要是对输入的ground_truth的z, x, bbox_z, bbox_x进行一系列变换，构成孪生网络的输入
class SiamFCTransforms(object):

    def __init__(self, exemplar_sz=127, instance_sz=255, context=0.5):
        self.exemplar_sz = exemplar_sz
        self.instance_sz = instance_sz
        self.context = context

        self.transforms_z = Compose([
            RandomStretch(),
            CenterCrop(instance_sz - 8),
            RandomCrop(instance_sz - 2 * 8),
            CenterCrop(exemplar_sz),
            ToTensor()])
        self.transforms_x = Compose([
            RandomStretch(),
            CenterCrop(instance_sz - 8),
            RandomCrop(instance_sz - 2 * 8),
            ToTensor()])

    def __call__(self, z, x, box_z, box_x):
        z = self._crop(z, box_z, self.instance_sz)
        x = self._crop(x, box_x, self.instance_sz)
        z = self.transforms_z(z)  # z.size = (3, 127, 127)
        x = self.transforms_x(x)  # x.size = (3, 239, 239)
        return z, x

    def _crop(self, img, box, out_size):
        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        center, target_sz = box[:2], box[2:]

        context = self.context * np.sum(target_sz)
        size = np.sqrt(np.prod(target_sz + context))
        size *= out_size / self.exemplar_sz

        avg_color = np.mean(img, axis=(0, 1), dtype=float)
        interp = np.random.choice([
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_NEAREST,
            cv2.INTER_LANCZOS4])
        patch = ops.crop_and_resize(
            img, center, size, out_size,
            border_value=avg_color, interp=interp)
        return patch
