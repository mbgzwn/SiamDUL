# from __future__ import absolute_import
#
# import os
# import glob
# import numpy as np
# import sys
# sys.path.append("/home/liang/Downloads/siamfc-pytorch-master")
# import siamfc.siamfc
# from siamfc import TrackerSiamFC
#
#
# if __name__ == '__main__':
#     seq_dir = os.path.expanduser('/home/liang/Downloads/data/OTB100/Crossing/')
#     img_files = sorted(glob.glob(seq_dir + 'img/*.jpg')) # '/home/liang/Downloads/siamfc-pytorch-master/data/OTB/Crossing/img/0001.jpg'
#     #print(img_files)
#     anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt',delimiter=',')  # 存ground数据，四个一组，n×4
#     # net_path = '/home/liang/Downloads/siamfc-pytorch-master/不shuffle的VGG pretrained/siamfc_alexnet_e50.pth'
#     net_path = '/home/liang/Downloads/siamfc-pytorch-master/不shuffle的VGG pretrained/siamfc_alexnet_e50.pth'
#     tracker = TrackerSiamFC(net_path=net_path)
#     tracker.track(img_files, anno[0], visualize=True)


from __future__ import absolute_import

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import glob
import numpy as np
from siamfc import TrackerSiamFC

if __name__ == '__main__':
    seq_dir = os.path.expanduser('/home/wenjunzhou/PycharmProjects/siamfc-master/OTB/Basketball/Basketball')
    img_files = sorted(glob.glob(seq_dir + '/img/*.jpg'))
    # anno = np.loadtxt(seq_dir + 'groundtruth.txt')
    anno = np.loadtxt(seq_dir + '/groundtruth_rect.txt', delimiter=',')

    net_path = '/home/wenjunzhou/PycharmProjects/siamfc-master/基准pretrained/siamfc_alexnet_e46.pth'
    tracker = TrackerSiamFC(net_path=net_path)
    s = tracker.track(img_files, anno[0], visualize=True)
    print(s)
