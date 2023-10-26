from __future__ import absolute_import

import os

import torch
from got10k.datasets import *

from siamfc import TrackerSiamFC

# if __name__ == '__main__':
#     root_dir = os.path.expanduser('/home/wenjunzhou/PycharmProjects/siamfc-master/data/GOT-10K')
#     seqs = GOT10k(root_dir, subset='train', return_meta=True)
#     net_path = '/home/wenjunzhou/PycharmProjects/siamfc-master/不shuffle的VGG pretrained 第三层/siamfc_alexnet_e50.pth'
#     tracker = TrackerSiamFC(net_path=net_path)
#     # tracker = TrackerSiamFC()
#     tracker.train_over(seqs)
if __name__ == '__main__':
    root_dir = os.path.expanduser('/home/wenjunzhou/PycharmProjects/siamfc-master/data/GOT-10K')
    seqs = GOT10k(root_dir, subset='train', return_meta=True)
    net_path = '/home/wenjunzhou/PycharmProjects/siamfc-master/alex关键信息提取最后一层+vgg16  第49个epoch/siamfc_alexnet_e49.pth'
    tracker = TrackerSiamFC(net_path)
    tracker.train_over(seqs)
