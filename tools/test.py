from __future__ import absolute_import

import os

from got10k.experiments import *

from siamfc import TrackerSiamFC

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.add_dll_directory("G:\\InstallSoftware\\Anaconda\\envs\\python_env\\Library\\bin\\geos_c.dll")

# if __name__ == '__main__':
#     net_path = '../不shuffle的VGG pretrained/siamfc_alexnet_e50.pth'
#     tracker = TrackerSiamFC(net_path=net_path)
#

#     root_dir = os.path.expanduser('/home/liang/Downloads/data/OTB100')
#     results = '/home/liang/Downloads/siamfc-pytorch-master/results'
#     report = '/home/liang/Downloads/siamfc-pytorch-master/report'
#     e = ExperimentOTB(root_dir, version='tb100', result_dir=results, report_dir=report)
#     e.run(tracker, visualize=True)
#     e.report([tracker.name])
if __name__ == '__main__':
    net_path = '/home/wenjunzhou/PycharmProjects/siamfc-master/alex多尺度特征最后一层+vgg16 第50个epoch/siamfc_alexnet_e50.pth'
    tracker = TrackerSiamFC(net_path=net_path)
    root_dir = os.path.expanduser('/home/wenjunzhou/PycharmProjects/siamfc-master/data/OTB')
    # root_dir = os.path.expanduser('/home/wenjunzhou/PycharmProjects/siamfc-master/data/GOT-10K')
    e = ExperimentOTB(root_dir, version='tb100')
    e.run(tracker, visualize=True)
    e.report([tracker.name])
    # experiment = ExperimentGOT10k(
    #     root_dir=root_dir,  # GOT-10k's root directory
    #     subset='test',  # 'train' | 'val' | 'test'
    #     result_dir='results',  # where to store tracking results
    #     report_dir='reports'  # where to store evaluation reports
    # )
    # experiment.run(tracker, visualize=True)
    #
    # # report tracking performance
    # experiment.report([tracker.name])
