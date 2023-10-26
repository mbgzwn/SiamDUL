# from __future__ import absolute_import, division, print_function
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import numpy as np
# import time
# import cv2
# import sys
# import torchvision.models.vgg
#
# # 将模块路径加到当前模块扫描的路径里
# from torch.utils import model_zoo
#
# sys.path.append("/home/wenjunzhou/PycharmProjects/siamfc-master/siamfc")
# import os
# from collections import namedtuple
# from torch.optim.lr_scheduler import ExponentialLR
# from torch.utils.data import DataLoader
# from got10k.trackers import Tracker
# import ops
# from backbones import VGG16
# from backbones import Net3
# import backbones
# from backbones import Net2
# from backbones import AlexNetV1  # from .backbones import AlexNetV1
# from heads import SiamFC  # from .heads import DASiamRPN
# from losses import GHMCLoss
# from losses import FocalLoss
# from losses import BalancedLoss  # from .losses import BalancedLoss
# from datasets import Pair  # from .datasets import Pair
# from transforms import SiamFCTransforms  # from .transforms import SiamFCTransforms
#
# __all__ = ['TrackerSiamFC']  # 直接进入TrackerSiamFC
#
#
# class Net(nn.Module):  # 继承nn.Module
#
#     def __init__(self, backbone, backbone1, head):
#         super(Net, self).__init__()
#         self.head = head
#         self.backbone1 = backbone1
#         self.backbone = backbone
#         self.conv = nn.Sequential(nn.Conv2d(2, 1, 1))
#         # self.att = att
#
#     def forward(self, z, x):
#         z1 = self.backbone(z)
#         # z1, z2 = self.backbone(z)
#         x1 = self.backbone(x)
#         # x1, x2 = self.backbone(x)
#         z2 = self.backbone1(z)
#         x2 = self.backbone1(x)
#         out1 = self.head(z1, x1)
#         out2 = self.head(z2, x2)
#         out = torch.cat([out1, out2], dim=1)
#         out = self.conv(out)
#         return out
#
#
# class TrackerSiamFC(Tracker):  # 继承got10k 也就是TrackerSiamFC可以调用Tracker里的类
#
#     def __init__(self, net_path=None, **kwargs):  # 定义self的属性
#         super(TrackerSiamFC, self).__init__('SiamFC', True)  # 继承父类Tracker的属性name和is_deterministic
#         # self.name = DASiamRPN
#         # self.is_deterministic = True
#         self.cfg = self.parse_args(**kwargs)
#
#         # setup GPU device if available
#         self.cuda = torch.cuda.is_available()
#         self.device = torch.device('cuda:0' if self.cuda else 'cpu')
#
#         # setup model  先新建一个model，然后加载预训练好的权重，就不用再训练一次了。所以创建的model和传入的pretrained weights一定是要对应的
#         self.net = Net(
#             # backbone2=backbones.VGG163(),
#             backbone=backbones.vgg(),
#             # backbone=backbones.resnet34(),
#             backbone1=backbones.AlexNetV(),
#             head=SiamFC(self.cfg.out_scale),
#             # head2=SiamFC(self.cfg.out_scale, False),
#         )
#         ops.init_weights(self.net),
#
#         # load checkpoint if provided 检查是否传入了pretrained weights，有的话就读进去，没有的话，就没有了
#         if net_path is not None:  # net_path -> siamfc_alexmet_e50.pth
#             self.net.load_state_dict(torch.load(
#                 net_path,
#                 map_location=lambda storage, load_state_dictc: storage), strict=False)  # # Load all tensors onto the CPU, using a function
#         self.net = self.net.to(self.device)  # siamfc_alexmet_e50.pth 先用cpu读然后再转到gpu
#
#         # pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
#         # model_dict = {}
#         # state_dict = model.state_dict()
#         # for k, v in pretrain_dict.items()
#         #     if k in state_dict:
#         #         model_dict[k] = v
#         # state_dict.update(model_dict)
#         # model.load_state_dict(state_dict)
#
#         # setup criterion
#         # self.criterion = FocalLoss()
#         self.criterion = BalancedLoss()
#
#         # setup optimizerBCE
#         self.optimizer = optim.SGD(  # 梯度下降法 带momentum的
#             self.net.parameters(),
#             lr=self.cfg.initial_lr,  # 学习率
#             weight_decay=self.cfg.weight_decay,  # 设置权值衰减，目的是防止过拟合
#             momentum=self.cfg.momentum)
#
#         # setup lr scheduler 这块不懂，还没看
#         # gamma = np.power(  # np.power(a,b) 数组的幂计算 a的b次方
#         #     self.cfg.ultimate_lr / self.cfg.initial_lr,  # gamma = le-3^0.02
#         #     1.0 / self.cfg.epoch_num)
#         gamma = np.power(  # np.power(a,b) 数组的幂计算 a的b次方
#             self.cfg.ultimate_lr / self.cfg.initial_lr,  # gamma = le-3^0.02
#             1.0 / self.cfg.epoch_num)
#         self.lr_scheduler = ExponentialLR(self.optimizer, 0.87)
#         # m = 1;
#         # if m >= 65:
#         #     self.lr_scheduler = ExponentialLR(self.optimizer, 0.87)
#         #     m = m + 1
#         # elif m >= 75:
#         #     self.lr_scheduler = ExponentialLR(self.optimizer, 0.7)
#         #     m = m + 1
#         # elif m >= 85:
#         #     self.lr_scheduler = ExponentialLR(self.optimizer, 0.6)
#         #     m = m + 1
#         # else:
#         #     self.lr_scheduler = ExponentialLR(self.optimizer, 0.5)
#
#     def parse_args(self, **kwargs):
#         # default parameters
#         cfg = {
#             # basic parameters
#             'out_scale': 0.001,
#             'exemplar_sz': 127,  # 模板固定为127*127
#             'instance_sz': 255,  # current frame固定为255*255
#             'context': 0.5,
#             # inference parameters
#             'scale_num': 3,
#             'scale_step': 1.0375,
#             # 'scale_step': 1.0482,
#             # 'scale_lr': 0.65,
#             'scale_lr': 0.59,
#             'scale_penalty': 0.9745,
#             'window_influence': 0.176,
#             'response_sz': 17,  # 相应尺寸是固定的17×17
#             'response_up': 16,  # 16×17=272 把croped patch还原成orignal image
#             'total_stride': 8,
#             # train parameters
#             'epoch_num': 65,
#             'batch_size': 8,
#             'num_workers': 16,
#             'initial_lr': 1e-2,
#             'ultimate_lr': 1e-5,
#             'weight_decay': 5e-4,
#             'momentum': 0.9,
#             'r_pos': 16,  # radius for positive pairs 论文2.2小节
#             'r_neg': 0}  # radius for negative pairs
#
#         for key, val in kwargs.items():  # 如果传了字段名称进来，就对字段名称的数值进行更新
#             if key in cfg:
#                 cfg.update({key: val})
#         return namedtuple('Config', cfg.keys())(
#             **cfg)  # 返回具名元组，可以用index访问也可以用字段名称访问。如访问‘out_scale’可以是Config(0)也可以是Config['out_scale']
#
#     @torch.no_grad()  # 下面的定义类，不用计算梯度，也不做反向传播（speed up）
#     def init(self, img, box):
#         # set to evaluation mode
#         self.net.eval()  # 把BN和Dropout固定，不再取平均或调整，直接用训练好的参数值
#
#         # convert box to 0-indexed and center based [y, x, h, w]
#         box = np.array([  # from[ltx,lty,w,h] -> [cy,cx,h,w]
#             box[1] - 1 + (box[3] - 1) / 2,  # ok but why minus 1??
#             box[0] - 1 + (box[2] - 1) / 2,
#             box[3], box[2]], dtype=np.float32)
#         self.center, self.target_sz = box[:2], box[2:]
#
#         # create hanning window
#         self.upscale_sz = self.cfg.response_up * self.cfg.response_sz  # 上采样到272*272
#         self.hann_window = np.outer(  # 外积，np.outer([m],[n])则生产m行n列的数组,行元素为m[i]*n的每一个元素
#             np.hanning(self.upscale_sz),  # 高维数组会自动flatten成1维然后进行计算 最后的形式是分块数组组成的数组
#             np.hanning(self.upscale_sz))
#         self.hann_window /= self.hann_window.sum()  # 归一化
#
#         # search scale factors 生成尺度池，尺度因子在一定范围内均匀递增
#         self.scale_factors = self.cfg.scale_step ** np.linspace(  # np.linspace(start,stop,num)生成均匀间隔的num个数值序列
#             -(self.cfg.scale_num // 2),  # **等价于np.pow() 幂运算
#             self.cfg.scale_num // 2,
#             self.cfg.scale_num)  # np.linspace[-1 1 3] 所以是self.scale_factors = [1.0375^(-1), 1.0375^0, 1.0375^1] -> [
#         # 0.9638 1 1.0375 ]
#
#         # exemplar and search sizes
#         context = self.cfg.context * np.sum(self.target_sz)  # context就是padding出来的区域 0.5(w+h)
#         self.z_sz = np.sqrt(np.prod(self.target_sz + context))  # squrt[(w+0.5(w+h))*(h+0.5(w+h))]
#         self.x_sz = self.z_sz * \
#                     self.cfg.instance_sz / self.cfg.exemplar_sz  # 输入x应该固定为255，模板z固定为127，所以输入的模板z乘上一个缩放比例变255，但模板z
#         # 因为padding了，所以最后得到的x尺度是255附近
#
#         # exemplar image
#         self.avg_color = np.mean(img, axis=(0, 1))  # 算像素平均，用来padding的
#         z = ops.crop_and_resize(  # resize到固定的127或255
#             img, self.center, self.z_sz,
#             out_size=self.cfg.exemplar_sz,
#             border_value=self.avg_color)
#
#         # exemplar features
#         z = torch.from_numpy(z).to(
#             # 换成gpu能处理的tensor 但是为什么要做permute？ 解决： [w,h,c] -> [c,w,h]  !!!因为got10k的数据类型是ncwh，所以用这个
#             self.device).permute(2, 0, 1).unsqueeze(0).float()  # unsqueeze() ，则维度变[mini-batch,c,w,h] NCWH格式，数据类型float
#         # self.kernel1, self.kernel2 = self.net.backbone(z)  # crop的z经过backbone（卷积神经网络）输出z的feature，这里backbone用的是AlexNet_V1
#         # self.kernel = self.net.backbone(z)  # crop的z经过backbone（卷积神经网络）输出z的feature，这里backbone用的是AlexNet_V1
#         self.kernel1 = self.net.backbone(z)  # crop的z经过backbone（卷积神经网络）输出z的feature，这里backbone用的是AlexNet_V1
#         self.kernel2 = self.net.backbone1(z)  # crop的z经过backbone（卷积神经网络）输出z的feature，这里backbone用的是AlexNet_V1
#         print("for breakpoint")
#
#     #  self.kernel就用第一帧初始化了，后面不会变，用与和当前帧做conv2d卷积得到响应。到这里初始化就完成了，主要是设定模板z。即Siamese的上半部分
#
#     @torch.no_grad()  # 这里是做跟踪的时候新的帧进来则做处理，直接跟踪，所以不再做梯度，并且冻结网络中的batchnum和dropout等参数
#     def update(self, img):
#         # set to evaluation mode
#         self.net.eval()
#
#         # search images
#         x = [ops.crop_and_resize(
#             img, self.center, self.x_sz * f,
#             out_size=self.cfg.instance_sz,
#             border_value=self.avg_color) for f in self.scale_factors]  # 这里输出的x是三种尺度的patch，是三个数组
#         # s_sc1 = torch.from_numpy(x[0]).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
#         # s_sc2 = torch.from_numpy(x[1]).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
#         # s_sc3 = torch.from_numpy(x[2]).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
#
#         x = np.stack(x, axis=0)  # 把三个数组堆叠为一个数组，此时shape=(3,255,255,3) 第一个3是三张patch对应的3维，第二个3是rgb的3维，255 255是每张x的h w
#         x = torch.from_numpy(x).to(
#             self.device).permute(0, 3, 1, 2).float()  # [patch_num, channels, h, w] 即patch_num CHW
#
#         # responses 这里的responses是包含3种尺度变换后的响应，后面通过最大响应来确定用哪张response，即哪个尺度
#         # a,b,c,d = s_sc1.size()
#         # x = self.net.backbone(x)  # 得到feature，此时x=[3*22*22*128]
#         x1 = self.net.backbone(x)  # 得到feature，此时x=[3*22*22*128]
#         x2 = self.net.backbone1(x)  # 得到feature，此时x=[3*22*22*128]
#         # x1, x2 = self.net.backbone(x)  # 得到feature，此时x=[3*22*22*128]
#
#         # self.kernel, x= self.net.feature_enhance(self.kernel,x)
#         # x1 = self.net.feature_enhance(self.kernel, x[1])
#         # x2 = self.net.feature_enhance(self.kernel, x[2])
#
#         # responses = self.net.head(self.kernel, x)  # 此时x=[3*17*17*128] head就是SiameseNet，计算当前帧与模板z的相关响应（卷积响应），模板z则是前面init初始化好的self.kernel
#         responses1 = self.net.head(self.kernel1, x1)  # 此时x=[3*17*17*128] head就是SiameseNet，计算当前帧与模板z的相关响应（卷积响应），模板z则是前面init初始化好的self.kernel
#         responses2 = self.net.head(self.kernel2, x2)  # 此时x=[3*17*17*128] head就是SiameseNet，计算当前帧与模板z的相关响应（卷积响应），模板z则是前面init初始化好的self.kernel
#         responses = self.net.conv(torch.cat([responses1, responses2], dim=1))  # 此时x=[3*17*17*128] head就是SiameseNet，计算当前帧与模板z的相关响应（卷积响应），模板z则是前面init初始化好的self.kernel
#         # responses = 0.3 * self.net.head(self.kernel1, x1) + 0.7 * self.net.head(self.kernel2, x2)  # 此时x=[3*17*17*128] head就是SiameseNet，计算当前帧与模板z的相关响应（卷积响应），模板z则是前面init初始化好的self.kernel
#         responses = responses.squeeze(1).cpu().numpy()  # 删掉1维（channels） 此时responses.shape=[3,17,17]
#
#         # upsample responses and penalize scale changes
#         responses = np.stack([cv2.resize(  # 把3*17*17 resize到 3*272*272 (后面要映射回到原image)
#             u, (self.upscale_sz, self.upscale_sz),  # 做上采样，用的是三次样条插值
#             interpolation=cv2.INTER_CUBIC)
#             for u in responses])
#         responses[
#         :self.cfg.scale_num // 2] *= self.cfg.scale_penalty  # responses包含三张尺度的response，对第一张响应和第三张响应做惩罚，而原尺寸响应不变  为什么？
#         responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty  #
#         # peak scale
#         scale_id = np.argmax(np.amax(responses, axis=(1, 2)))  # 分别找三张图的最大响应，然后再从这三个响应中找最大响应，尺度因子则用这张最大响应所对应的
#         #                                                  (1,2)对于图的wh，即找每一张图的，0对应第几张
#
#         # peak location
#         response = responses[scale_id]  # 选最大响应对应的那张
#         # s = response.max()
#         # print(s)
#         response -= response.min()  # 都减去最小响应值 一种平滑处理吧
#         response /= response.sum() + 1e-16  # 归一化 为什么要加1e-16？
#         response = (1 - self.cfg.window_influence) * response + \
#                    self.cfg.window_influence * self.hann_window  # 边界效应的处理
#         loc = np.unravel_index(response.argmax(),
#                                response.shape)  # 中心点 croped的patch (y,x) 以patch为标尺的center，后面还要映射回原image
#
#         # locate target center 这里的计算不太看得懂（懂啦！！） 就是逆回去把center映射到原图片 因为图片已经经过了上采样 crop等操作 [138 134]->[2.5 -1.5]->[1.25 -0.75]->[0.45269844 -0.27161906]->[110.433945 61.620953]
#         disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2  # 倒数第一步是upsampling，所以先映射回未upsampling时的center
#         disp_in_instance = disp_in_response * \
#                            self.cfg.total_stride / self.cfg.response_up  # 映射回backbond之前的center backbond多层卷积所有的stride为total_stride，所以要乘回去变回之前的尺寸
#         #                                                        再映射回没做resize之前的patch
#         disp_in_image = disp_in_instance * self.x_sz * \
#                         self.scale_factors[
#                             scale_id] / self.cfg.instance_sz  # 从patch映射回没做padding之前的center，即在原image的center
#         # 这里的disp_in_xx都是一个相对的差值，最后用原center加上这个差值就得到映射回来的center了
#         self.center += disp_in_image
#
#         # update target size 就是稀疏更新那个式子 （1-尺度学习率）+尺度学习率×尺度因子
#         scale = (1 - self.cfg.scale_lr) * 1.0 + \
#                 self.cfg.scale_lr * self.scale_factors[scale_id]
#         self.target_sz *= scale  # 按照尺度因子更新尺度
#         self.z_sz *= scale  # 按照尺度因子更新尺度
#         self.x_sz *= scale  # 按照尺度因子更新尺度
#
#         # return 1-indexed and left-top based bounding box
#         box = np.array([  # box还原回：顶角点 w h 因为要和计算跟踪性能，数据集提供的数据是[顶角点 w h]的格式
#             self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
#             self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
#             self.target_sz[1], self.target_sz[0]])
#
#         return box
#
#     # 读frame->crop&resize(with scale)->features->do conv with z->response->choose best scale resopnse-> updata paras
#     # -> return boxes -> visualize
#     def track(self, img_files, box, visualize=False):
#         frame_num = len(img_files)
#         boxes = np.zeros((frame_num, 4))
#         boxes[0] = box
#         times = np.zeros(frame_num)
#
#         for f, img_file in enumerate(img_files):
#             img = ops.read_image(img_file)
#
#             begin = time.time()
#             if f == 0:
#                 self.init(img, box)  # 第一帧时做初始化，生成模板z（self.kernel)
#             else:
#                 boxes[f, :] = self.update(img)  # 计算每一帧img返回的box，box=[tlx,tly,w,h] 存到boxes里，用来做可视化
#                 # self.init(img, boxes[f])
#             times[f] = time.time() - begin
#
#             if visualize:
#                 ops.show_image(img, boxes[f, :])
#
#         return boxes, times
#
#     # 具体在这个函数里面实现了训练和反向传播
#     def train_step(self, batch, backward=True):
#         # set network mode
#         self.net.train(backward)
#
#         # parse batch data
#         z = batch[0].to(self.device, non_blocking=self.cuda)  # z.shape: torch.Size([8,3,127,127])   *NCWH格式
#         x = batch[1].to(self.device, non_blocking=self.cuda)  # x.shape: torch.Size([8,3,239,239])
#
#         with torch.set_grad_enabled(backward):
#             # inference
#             responses = self.net(z, x)  # responses.shape: torch.Size([8, 1, 15, 15])
#
#             # calculate loss
#             labels = self._create_labels(responses.size())  # 要把labels resize成responses的size，因为后面计算loss的时候是element-wise的
#             loss = self.criterion(responses, labels)
#
#             if backward:
#                 # back propagation
#                 self.optimizer.zero_grad()
#                 if hasattr(torch.cuda, 'empty_cache'):
#                     torch.cuda.empty_cache()
#
#                 loss.backward()
#                 if hasattr(torch.cuda, 'empty_cache'):
#                     torch.cuda.empty_cache()
#                 self.optimizer.step()
#
#         return loss.item()
#
#     @torch.enable_grad()
#     def train_over(self, seqs, val_seqs=None,
#                    save_dir='VGG pretrained 46个epoch'):
#         # set to train mode 设置为train模式，参数可迭代
#         self.net.train()
#
#         # create save_dir folder
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#
#         # setup dataset
#         transforms = SiamFCTransforms(
#             exemplar_sz=self.cfg.exemplar_sz,
#             instance_sz=self.cfg.instance_sz,
#             context=self.cfg.context)
#         # 这里的dataset经过Pair是返回一对（z,x），且已经经过裁剪成为合格的输入
#         dataset = Pair(
#             seqs=seqs,
#             transforms=transforms)
#
#         # setup dataloader
#         dataloader = DataLoader(
#             dataset,
#             batch_size=self.cfg.batch_size,  # batch_size = 8
#             shuffle=True,  # batch取到的可以是与其它batch重复的，这样会存在一个问题，有的data可能从头到尾都没有被抽到
#             num_workers=self.cfg.num_workers,  # 加载数据的次数，按cfg设为32，则分32次加载（是不是就是batch？？）
#             pin_memory=self.cuda,  # copy Tensors into CUDA
#             drop_last=True)  # 最后一个batch drop掉
#
#         # loop over epochs
#         for epoch in range(self.cfg.epoch_num):
#             # update lr at each epoch
#             self.lr_scheduler.step(epoch=epoch)
#
#             # loop over dataloader
#             # print("let's see what's in batch")
#             # print(batch)
#             for it, batch in enumerate(
#                     dataloader):  # enumerate枚举dataloader中的元素，返回值包括index和datas 则it对应index，batch对应dataloader
#                 # Epoch: 49[it+1: 914 / len(dataloader): 1166] Loss: 0.19454
#                 loss = self.train_step(batch, backward=True)
#                 print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
#                     epoch + 1, it + 1, len(dataloader), loss))
#                 sys.stdout.flush()
#             # save checkpoint
#             if not os.path.exists(save_dir):
#                 os.makedirs(save_dir)
#             net_path = os.path.join(
#                 save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
#             torch.save(self.net.state_dict(), net_path)
#
#     def _create_labels(self, size):  # train step里用于生成当前response的label然后与gt的label算loss，然后backward
#         # skip if same sized labels already created
#         if hasattr(self,
#                    'labels') and self.labels.size() == size:  # hasattr找属性，如果self里有labels这个属性则返回真; self里的labels属性size和给定的相同，则直接返回labels
#             return self.labels  # 这里的size主要是维度是否一致，后面的目的就是扩展维度使一致
#
#         def logistic_labels(x, y, r_pos, r_neg):
#             dist = np.abs(x) + np.abs(y)  # block distance
#             labels = np.where(dist <= r_pos,  # np.where 判断0/1,相应返回0/1
#                               np.ones_like(x),  # 满足dist<=r_neg则返回np.ones_like(x) 反之返回后面的输出
#                               np.where(dist < r_neg,
#                                        np.ones_like(x) * 0.5,
#                                        np.zeros_like(x)))
#             return labels  # 在设定的positive范围内像素点值设为1, 在positive外又未到negative则设为0.5,其余设为0
#
#         # distances along x- and y-axis
#         n, c, h, w = size
#         x = np.arange(w) - (w - 1) / 2  # 生成一个0-w的固定步长排列，stride=1 ->> 算中心点（下同）
#         y = np.arange(h) - (h - 1) / 2
#         x, y = np.meshgrid(x, y)  # 生成网格点坐标矩阵 but why????
#
#         # create logistic labels 论文2.2小节
#         r_pos = self.cfg.r_pos / self.cfg.total_stride  # radius of pos 换算到original image上的pos半径
#         r_neg = self.cfg.r_neg / self.cfg.total_stride  # r_neg 0
#         labels = logistic_labels(x, y, r_pos, r_neg)  # 返回一张图，仅有中心部分位置是0
#
#         # repeat to size
#         labels = labels.reshape((1, 1, h, w))
#         labels = np.tile(labels, (n, c, 1, 1))  # labels的四个维度分别按size扩展，就和前面判断labels的size是否相同对得上了
#         # n个batch，每个batch c个通道 每个通道是1×1数组（就是一个点嘛）
#         # convert to tensors
#         self.labels = torch.from_numpy(labels).to(self.device).float()  # numyp to tensor
#
#         return self.labels

# from __future__ import absolute_import, division, print_function
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import numpy as np
# import time
# import cv2
# import sys
# import torchvision.models.vgg
#
# # 将模块路径加到当前模块扫描的路径里
# from torch.utils import model_zoo
#
# sys.path.append("/home/wenjunzhou/PycharmProjects/siamfc-master/siamfc")
# import os
# from collections import namedtuple
# from torch.optim.lr_scheduler import ExponentialLR
# from torch.utils.data import DataLoader
# from got10k.trackers import Tracker
# import ops
# from backbones import VGG16
# from backbones import Net3
# import backbones
# from backbones import Net2
# from backbones import AlexNetV1  # from .backbones import AlexNetV1
# from heads import SiamFC  # from .heads import DASiamRPN
# from losses import GHMCLoss
# from losses import FocalLoss
# from losses import BalancedLoss  # from .losses import BalancedLoss
# from datasets import Pair  # from .datasets import Pair
# from transforms import SiamFCTransforms  # from .transforms import SiamFCTransforms
#
# __all__ = ['TrackerSiamFC']  # 直接进入TrackerSiamFC
#
#
# class Net(nn.Module):  # 继承nn.Module
#
#     def __init__(self, backbone, backbone1, head):
#         super(Net, self).__init__()
#         self.head = head
#         self.backbone1 = backbone1
#         self.backbone = backbone
#         self.conv = nn.Sequential(nn.Conv2d(512, 256, 1))
#         self.maxpool = nn.Sequential(nn.Conv2d(256, 256, 2, 1, 0))
#         # self.att = att
#
#     def forward(self, z, x):
#         z1 = self.backbone(z)
#         # z1, z2 = self.backbone(z)
#         x1 = self.backbone(x)
#         # x1, x2 = self.backbone(x)
#         z2 = self.backbone1(z)
#         x2 = self.backbone1(x)
#         z2 = self.maxpool(z2)
#         x2 = self.maxpool(x2)
#         z = torch.cat([z1, z2], dim=1)
#         x = torch.cat([x1, x2], dim=1)
#         z = self.conv(z)
#         x = self.conv(x)
#         out = self.head(z,x)
#         return out
#
#
# class TrackerSiamFC(Tracker):  # 继承got10k 也就是TrackerSiamFC可以调用Tracker里的类
#
#     def __init__(self, net_path=None, **kwargs):  # 定义self的属性
#         super(TrackerSiamFC, self).__init__('SiamFC', True)  # 继承父类Tracker的属性name和is_deterministic
#         # self.name = DASiamRPN
#         # self.is_deterministic = True
#         self.cfg = self.parse_args(**kwargs)
#
#         # setup GPU device if available
#         self.cuda = torch.cuda.is_available()
#         self.device = torch.device('cuda:0' if self.cuda else 'cpu')
#
#         # setup model  先新建一个model，然后加载预训练好的权重，就不用再训练一次了。所以创建的model和传入的pretrained weights一定是要对应的
#         self.net = Net(
#             # backbone2=backbones.VGG163(),
#             backbone=backbones.vgg(),
#             # backbone=backbones.resnet34(),
#             backbone1=backbones.AlexNetV(),
#             head=SiamFC(self.cfg.out_scale),
#             # head2=SiamFC(self.cfg.out_scale, False),
#         )
#         ops.init_weights(self.net),
#
#         # load checkpoint if provided 检查是否传入了pretrained weights，有的话就读进去，没有的话，就没有了
#         if net_path is not None:  # net_path -> siamfc_alexmet_e50.pth
#             self.net.load_state_dict(torch.load(
#                 net_path,
#                 map_location=lambda storage, load_state_dictc: storage), strict=False)  # # Load all tensors onto the CPU, using a function
#         self.net = self.net.to(self.device)  # siamfc_alexmet_e50.pth 先用cpu读然后再转到gpu
#
#         # pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
#         # model_dict = {}
#         # state_dict = model.state_dict()
#         # for k, v in pretrain_dict.items()
#         #     if k in state_dict:
#         #         model_dict[k] = v
#         # state_dict.update(model_dict)
#         # model.load_state_dict(state_dict)
#
#         # setup criterion
#         # self.criterion = FocalLoss()
#         self.criterion = BalancedLoss()
#
#         # setup optimizerBCE
#         self.optimizer = optim.SGD(  # 梯度下降法 带momentum的
#             self.net.parameters(),
#             lr=self.cfg.initial_lr,  # 学习率
#             weight_decay=self.cfg.weight_decay,  # 设置权值衰减，目的是防止过拟合
#             momentum=self.cfg.momentum)
#
#         # setup lr scheduler 这块不懂，还没看
#         # gamma = np.power(  # np.power(a,b) 数组的幂计算 a的b次方
#         #     self.cfg.ultimate_lr / self.cfg.initial_lr,  # gamma = le-3^0.02
#         #     1.0 / self.cfg.epoch_num)
#         gamma = np.power(  # np.power(a,b) 数组的幂计算 a的b次方
#             self.cfg.ultimate_lr / self.cfg.initial_lr,  # gamma = le-3^0.02
#             1.0 / self.cfg.epoch_num)
#         self.lr_scheduler = ExponentialLR(self.optimizer, 0.87)
#         # m = 1;
#         # if m >= 65:
#         #     self.lr_scheduler = ExponentialLR(self.optimizer, 0.87)
#         #     m = m + 1
#         # elif m >= 75:
#         #     self.lr_scheduler = ExponentialLR(self.optimizer, 0.7)
#         #     m = m + 1
#         # elif m >= 85:
#         #     self.lr_scheduler = ExponentialLR(self.optimizer, 0.6)
#         #     m = m + 1
#         # else:
#         #     self.lr_scheduler = ExponentialLR(self.optimizer, 0.5)
#
#     def parse_args(self, **kwargs):
#         # default parameters
#         cfg = {
#             # basic parameters
#             'out_scale': 0.001,
#             'exemplar_sz': 127,  # 模板固定为127*127
#             'instance_sz': 255,  # current frame固定为255*255
#             'context': 0.5,
#             # inference parameters
#             'scale_num': 3,
#             'scale_step': 1.0375,
#             # 'scale_step': 1.0482,
#             # 'scale_lr': 0.65,
#             'scale_lr': 0.87,
#             'scale_penalty': 0.9745,
#             'window_influence': 0.176,
#             'response_sz': 17,  # 相应尺寸是固定的17×17
#             'response_up': 16,  # 16×17=272 把croped patch还原成orignal image
#             'total_stride': 8,
#             # train parameters
#             'epoch_num': 65,
#             'batch_size': 8,
#             'num_workers': 16,
#             'initial_lr': 1e-2,
#             'ultimate_lr': 1e-5,
#             'weight_decay': 5e-4,
#             'momentum': 0.9,
#             'r_pos': 16,  # radius for positive pairs 论文2.2小节
#             'r_neg': 0}  # radius for negative pairs
#
#         for key, val in kwargs.items():  # 如果传了字段名称进来，就对字段名称的数值进行更新
#             if key in cfg:
#                 cfg.update({key: val})
#         return namedtuple('Config', cfg.keys())(
#             **cfg)  # 返回具名元组，可以用index访问也可以用字段名称访问。如访问‘out_scale’可以是Config(0)也可以是Config['out_scale']
#
#     @torch.no_grad()  # 下面的定义类，不用计算梯度，也不做反向传播（speed up）
#     def init(self, img, box):
#         # set to evaluation mode
#         self.net.eval()  # 把BN和Dropout固定，不再取平均或调整，直接用训练好的参数值
#
#         # convert box to 0-indexed and center based [y, x, h, w]
#         box = np.array([  # from[ltx,lty,w,h] -> [cy,cx,h,w]
#             box[1] - 1 + (box[3] - 1) / 2,  # ok but why minus 1??
#             box[0] - 1 + (box[2] - 1) / 2,
#             box[3], box[2]], dtype=np.float32)
#         self.center, self.target_sz = box[:2], box[2:]
#
#         # create hanning window
#         self.upscale_sz = self.cfg.response_up * self.cfg.response_sz  # 上采样到272*272
#         self.hann_window = np.outer(  # 外积，np.outer([m],[n])则生产m行n列的数组,行元素为m[i]*n的每一个元素
#             np.hanning(self.upscale_sz),  # 高维数组会自动flatten成1维然后进行计算 最后的形式是分块数组组成的数组
#             np.hanning(self.upscale_sz))
#         self.hann_window /= self.hann_window.sum()  # 归一化
#
#         # search scale factors 生成尺度池，尺度因子在一定范围内均匀递增
#         self.scale_factors = self.cfg.scale_step ** np.linspace(  # np.linspace(start,stop,num)生成均匀间隔的num个数值序列
#             -(self.cfg.scale_num // 2),  # **等价于np.pow() 幂运算
#             self.cfg.scale_num // 2,
#             self.cfg.scale_num)  # np.linspace[-1 1 3] 所以是self.scale_factors = [1.0375^(-1), 1.0375^0, 1.0375^1] -> [
#         # 0.9638 1 1.0375 ]
#
#         # exemplar and search sizes
#         context = self.cfg.context * np.sum(self.target_sz)  # context就是padding出来的区域 0.5(w+h)
#         self.z_sz = np.sqrt(np.prod(self.target_sz + context))  # squrt[(w+0.5(w+h))*(h+0.5(w+h))]
#         self.x_sz = self.z_sz * \
#                     self.cfg.instance_sz / self.cfg.exemplar_sz  # 输入x应该固定为255，模板z固定为127，所以输入的模板z乘上一个缩放比例变255，但模板z
#         # 因为padding了，所以最后得到的x尺度是255附近
#
#         # exemplar image
#         self.avg_color = np.mean(img, axis=(0, 1))  # 算像素平均，用来padding的
#         z = ops.crop_and_resize(  # resize到固定的127或255
#             img, self.center, self.z_sz,
#             out_size=self.cfg.exemplar_sz,
#             border_value=self.avg_color)
#
#         # exemplar features
#         z = torch.from_numpy(z).to(
#             # 换成gpu能处理的tensor 但是为什么要做permute？ 解决： [w,h,c] -> [c,w,h]  !!!因为got10k的数据类型是ncwh，所以用这个
#             self.device).permute(2, 0, 1).unsqueeze(0).float()  # unsqueeze() ，则维度变[mini-batch,c,w,h] NCWH格式，数据类型float
#         # self.kernel1, self.kernel2 = self.net.backbone(z)  # crop的z经过backbone（卷积神经网络）输出z的feature，这里backbone用的是AlexNet_V1
#         # self.kernel = self.net.backbone(z)  # crop的z经过backbone（卷积神经网络）输出z的feature，这里backbone用的是AlexNet_V1
#         self.kernel1 = self.net.backbone(z)  # crop的z经过backbone（卷积神经网络）输出z的feature，这里backbone用的是AlexNet_V1
#         self.kernel2 = self.net.backbone1(z)  # crop的z经过backbone（卷积神经网络）输出z的feature，这里backbone用的是AlexNet_V1
#         self.kernel2 = self.net.maxpool(self.kernel2)
#         self.kernel = self.net.conv(torch.cat([self.kernel1, self.kernel2], dim=1))
#         print("for breakpoint")
#
#     #  self.kernel就用第一帧初始化了，后面不会变，用与和当前帧做conv2d卷积得到响应。到这里初始化就完成了，主要是设定模板z。即Siamese的上半部分
#
#     @torch.no_grad()  # 这里是做跟踪的时候新的帧进来则做处理，直接跟踪，所以不再做梯度，并且冻结网络中的batchnum和dropout等参数
#     def update(self, img):
#         # set to evaluation mode
#         self.net.eval()
#
#         # search images
#         x = [ops.crop_and_resize(
#             img, self.center, self.x_sz * f,
#             out_size=self.cfg.instance_sz,
#             border_value=self.avg_color) for f in self.scale_factors]  # 这里输出的x是三种尺度的patch，是三个数组
#         # s_sc1 = torch.from_numpy(x[0]).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
#         # s_sc2 = torch.from_numpy(x[1]).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
#         # s_sc3 = torch.from_numpy(x[2]).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
#
#         x = np.stack(x, axis=0)  # 把三个数组堆叠为一个数组，此时shape=(3,255,255,3) 第一个3是三张patch对应的3维，第二个3是rgb的3维，255 255是每张x的h w
#         x = torch.from_numpy(x).to(
#             self.device).permute(0, 3, 1, 2).float()  # [patch_num, channels, h, w] 即patch_num CHW
#
#         # responses 这里的responses是包含3种尺度变换后的响应，后面通过最大响应来确定用哪张response，即哪个尺度
#         # a,b,c,d = s_sc1.size()
#         # x = self.net.backbone(x)  # 得到feature，此时x=[3*22*22*128]
#         x1 = self.net.backbone(x)  # 得到feature，此时x=[3*22*22*128]
#         x2 = self.net.backbone1(x)  # 得到feature，此时x=[3*22*22*128]
#         # x1, x2 = self.net.backbone(x)  # 得到feature，此时x=[3*22*22*128]
#         x2 = self.net.maxpool(x2)
#         x = self.net.conv(torch.cat([x1, x2], dim=1))
#         # self.kernel, x= self.net.feature_enhance(self.kernel,x)
#         # x1 = self.net.feature_enhance(self.kernel, x[1])
#         # x2 = self.net.feature_enhance(self.kernel, x[2])
#
#         responses = self.net.head(self.kernel, x)  # 此时x=[3*17*17*128] head就是SiameseNet，计算当前帧与模板z的相关响应（卷积响应），模板z则是前面init初始化好的self.kernel
#         responses = responses.squeeze(1).cpu().numpy()  # 删掉1维（channels） 此时responses.shape=[3,17,17]
#
#         # upsample responses and penalize scale changes
#         responses = np.stack([cv2.resize(  # 把3*17*17 resize到 3*272*272 (后面要映射回到原image)
#             u, (self.upscale_sz, self.upscale_sz),  # 做上采样，用的是三次样条插值
#             interpolation=cv2.INTER_CUBIC)
#             for u in responses])
#         responses[
#         :self.cfg.scale_num // 2] *= self.cfg.scale_penalty  # responses包含三张尺度的response，对第一张响应和第三张响应做惩罚，而原尺寸响应不变  为什么？
#         responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty  #
#         # peak scale
#         scale_id = np.argmax(np.amax(responses, axis=(1, 2)))  # 分别找三张图的最大响应，然后再从这三个响应中找最大响应，尺度因子则用这张最大响应所对应的
#         #                                                  (1,2)对于图的wh，即找每一张图的，0对应第几张
#
#         # peak location
#         response = responses[scale_id]  # 选最大响应对应的那张
#         # s = response.max()
#         # print(s)
#         response -= response.min()  # 都减去最小响应值 一种平滑处理吧
#         response /= response.sum() + 1e-16  # 归一化 为什么要加1e-16？
#         response = (1 - self.cfg.window_influence) * response + \
#                    self.cfg.window_influence * self.hann_window  # 边界效应的处理
#         loc = np.unravel_index(response.argmax(),
#                                response.shape)  # 中心点 croped的patch (y,x) 以patch为标尺的center，后面还要映射回原image
#
#         # locate target center 这里的计算不太看得懂（懂啦！！） 就是逆回去把center映射到原图片 因为图片已经经过了上采样 crop等操作 [138 134]->[2.5 -1.5]->[1.25 -0.75]->[0.45269844 -0.27161906]->[110.433945 61.620953]
#         disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2  # 倒数第一步是upsampling，所以先映射回未upsampling时的center
#         disp_in_instance = disp_in_response * \
#                            self.cfg.total_stride / self.cfg.response_up  # 映射回backbond之前的center backbond多层卷积所有的stride为total_stride，所以要乘回去变回之前的尺寸
#         #                                                        再映射回没做resize之前的patch
#         disp_in_image = disp_in_instance * self.x_sz * \
#                         self.scale_factors[
#                             scale_id] / self.cfg.instance_sz  # 从patch映射回没做padding之前的center，即在原image的center
#         # 这里的disp_in_xx都是一个相对的差值，最后用原center加上这个差值就得到映射回来的center了
#         self.center += disp_in_image
#
#         # update target size 就是稀疏更新那个式子 （1-尺度学习率）+尺度学习率×尺度因子
#         scale = (1 - self.cfg.scale_lr) * 1.0 + \
#                 self.cfg.scale_lr * self.scale_factors[scale_id]
#         self.target_sz *= scale  # 按照尺度因子更新尺度
#         self.z_sz *= scale  # 按照尺度因子更新尺度
#         self.x_sz *= scale  # 按照尺度因子更新尺度
#
#         # return 1-indexed and left-top based bounding box
#         box = np.array([  # box还原回：顶角点 w h 因为要和计算跟踪性能，数据集提供的数据是[顶角点 w h]的格式
#             self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
#             self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
#             self.target_sz[1], self.target_sz[0]])
#
#         return box
#
#     # 读frame->crop&resize(with scale)->features->do conv with z->response->choose best scale resopnse-> updata paras
#     # -> return boxes -> visualize
#     def track(self, img_files, box, visualize=False):
#         frame_num = len(img_files)
#         boxes = np.zeros((frame_num, 4))
#         boxes[0] = box
#         times = np.zeros(frame_num)
#
#         for f, img_file in enumerate(img_files):
#             img = ops.read_image(img_file)
#
#             begin = time.time()
#             if f == 0:
#                 self.init(img, box)  # 第一帧时做初始化，生成模板z（self.kernel)
#             else:
#                 boxes[f, :] = self.update(img)  # 计算每一帧img返回的box，box=[tlx,tly,w,h] 存到boxes里，用来做可视化
#                 # self.init(img, boxes[f])
#             times[f] = time.time() - begin
#
#             if visualize:
#                 ops.show_image(img, boxes[f, :])
#
#         return boxes, times
#
#     # 具体在这个函数里面实现了训练和反向传播
#     def train_step(self, batch, backward=True):
#         # set network mode
#         self.net.train(backward)
#
#         # parse batch data
#         z = batch[0].to(self.device, non_blocking=self.cuda)  # z.shape: torch.Size([8,3,127,127])   *NCWH格式
#         x = batch[1].to(self.device, non_blocking=self.cuda)  # x.shape: torch.Size([8,3,239,239])
#
#         with torch.set_grad_enabled(backward):
#             # inference
#             responses = self.net(z, x)  # responses.shape: torch.Size([8, 1, 15, 15])
#
#             # calculate loss
#             labels = self._create_labels(responses.size())  # 要把labels resize成responses的size，因为后面计算loss的时候是element-wise的
#             loss = self.criterion(responses, labels)
#
#             if backward:
#                 # back propagation
#                 self.optimizer.zero_grad()
#                 if hasattr(torch.cuda, 'empty_cache'):
#                     torch.cuda.empty_cache()
#
#                 loss.backward()
#                 if hasattr(torch.cuda, 'empty_cache'):
#                     torch.cuda.empty_cache()
#                 self.optimizer.step()
#
#         return loss.item()
#
#     @torch.enable_grad()
#     def train_over(self, seqs, val_seqs=None,
#                    save_dir='VGG pretrained 46个epoch'):
#         # set to train mode 设置为train模式，参数可迭代
#         self.net.train()
#
#         # create save_dir folder
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#
#         # setup dataset
#         transforms = SiamFCTransforms(
#             exemplar_sz=self.cfg.exemplar_sz,
#             instance_sz=self.cfg.instance_sz,
#             context=self.cfg.context)
#         # 这里的dataset经过Pair是返回一对（z,x），且已经经过裁剪成为合格的输入
#         dataset = Pair(
#             seqs=seqs,
#             transforms=transforms)
#
#         # setup dataloader
#         dataloader = DataLoader(
#             dataset,
#             batch_size=self.cfg.batch_size,  # batch_size = 8
#             shuffle=True,  # batch取到的可以是与其它batch重复的，这样会存在一个问题，有的data可能从头到尾都没有被抽到
#             num_workers=self.cfg.num_workers,  # 加载数据的次数，按cfg设为32，则分32次加载（是不是就是batch？？）
#             pin_memory=self.cuda,  # copy Tensors into CUDA
#             drop_last=True)  # 最后一个batch drop掉
#
#         # loop over epochs
#         for epoch in range(self.cfg.epoch_num):
#             # update lr at each epoch
#             self.lr_scheduler.step(epoch=epoch)
#
#             # loop over dataloader
#             # print("let's see what's in batch")
#             # print(batch)
#             for it, batch in enumerate(
#                     dataloader):  # enumerate枚举dataloader中的元素，返回值包括index和datas 则it对应index，batch对应dataloader
#                 # Epoch: 49[it+1: 914 / len(dataloader): 1166] Loss: 0.19454
#                 loss = self.train_step(batch, backward=True)
#                 print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
#                     epoch + 1, it + 1, len(dataloader), loss))
#                 sys.stdout.flush()
#             # save checkpoint
#             if not os.path.exists(save_dir):
#                 os.makedirs(save_dir)
#             net_path = os.path.join(
#                 save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
#             torch.save(self.net.state_dict(), net_path)
#
#     def _create_labels(self, size):  # train step里用于生成当前response的label然后与gt的label算loss，然后backward
#         # skip if same sized labels already created
#         if hasattr(self,
#                    'labels') and self.labels.size() == size:  # hasattr找属性，如果self里有labels这个属性则返回真; self里的labels属性size和给定的相同，则直接返回labels
#             return self.labels  # 这里的size主要是维度是否一致，后面的目的就是扩展维度使一致
#
#         def logistic_labels(x, y, r_pos, r_neg):
#             dist = np.abs(x) + np.abs(y)  # block distance
#             labels = np.where(dist <= r_pos,  # np.where 判断0/1,相应返回0/1
#                               np.ones_like(x),  # 满足dist<=r_neg则返回np.ones_like(x) 反之返回后面的输出
#                               np.where(dist < r_neg,
#                                        np.ones_like(x) * 0.5,
#                                        np.zeros_like(x)))
#             return labels  # 在设定的positive范围内像素点值设为1, 在positive外又未到negative则设为0.5,其余设为0
#
#         # distances along x- and y-axis
#         n, c, h, w = size
#         x = np.arange(w) - (w - 1) / 2  # 生成一个0-w的固定步长排列，stride=1 ->> 算中心点（下同）
#         y = np.arange(h) - (h - 1) / 2
#         x, y = np.meshgrid(x, y)  # 生成网格点坐标矩阵 but why????
#
#         # create logistic labels 论文2.2小节
#         r_pos = self.cfg.r_pos / self.cfg.total_stride  # radius of pos 换算到original image上的pos半径
#         r_neg = self.cfg.r_neg / self.cfg.total_stride  # r_neg 0
#         labels = logistic_labels(x, y, r_pos, r_neg)  # 返回一张图，仅有中心部分位置是0
#
#         # repeat to size
#         labels = labels.reshape((1, 1, h, w))
#         labels = np.tile(labels, (n, c, 1, 1))  # labels的四个维度分别按size扩展，就和前面判断labels的size是否相同对得上了
#         # n个batch，每个batch c个通道 每个通道是1×1数组（就是一个点嘛）
#         # convert to tensors
#         self.labels = torch.from_numpy(labels).to(self.device).float()  # numyp to tensor
#
#         return self.labels



from __future__ import absolute_import, division, print_function
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import cv2
import sys
import torchvision.models.vgg

# 将模块路径加到当前模块扫描的路径里
from torch.utils import model_zoo

sys.path.append("/home/wenjunzhou/PycharmProjects/siamfc-master/siamfc")
import os
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from got10k.trackers import Tracker
import ops
from backbones import VGG16
from backbones import Net3
import backbones
from backbones import Net2
from backbones import AlexNetV1  # from .backbones import AlexNetV1
from heads import SiamFC  # from .heads import DASiamRPN
from losses import GHMCLoss
from losses import FocalLoss
from losses import BalancedLoss  # from .losses import BalancedLoss
from datasets import Pair  # from .datasets import Pair
from transforms import SiamFCTransforms  # from .transforms import SiamFCTransforms

__all__ = ['TrackerSiamFC']  # 直接进入TrackerSiamFC


class Net(nn.Module):  # 继承nn.Module

    def __init__(self, backbone, backbone1, head):
        super(Net, self).__init__()
        self.head = head
        self.backbone1 = backbone1
        self.backbone = backbone
        self.conv = nn.Sequential(nn.Conv2d(512, 256, 1))
        self.maxpool = nn.Sequential(nn.Conv2d(256, 256, 2, 1, 0))
        self.conv1 = nn.Sequential(nn.Conv2d(3, 1, 1))
        # self.att = att

    def forward(self, z, x):
        z1 = self.backbone(z)
        # z1, z2 = self.backbone(z)
        x1 = self.backbone(x)
        # x1, x2 = self.backbone(x)
        z2 = self.backbone1(z)
        x2 = self.backbone1(x)
        z3 = self.maxpool(z2)
        x3 = self.maxpool(x2)
        out1 = self.head(z1, x1)
        out2 = self.head(z2, x2)
        z3 = torch.cat([z1, z3], dim=1)
        x3 = torch.cat([x1, x3], dim=1)
        z3 = self.conv(z3)
        x3 = self.conv(x3)
        out3 = self.head(z3,x3)
        out = self.conv1(torch.cat([out1, out2, out3], dim=1))
        return out


class TrackerSiamFC(Tracker):  # 继承got10k 也就是TrackerSiamFC可以调用Tracker里的类

    def __init__(self, net_path=None, **kwargs):  # 定义self的属性
        super(TrackerSiamFC, self).__init__('SiamFC', True)  # 继承父类Tracker的属性name和is_deterministic
        # self.name = DASiamRPN
        # self.is_deterministic = True
        self.cfg = self.parse_args(**kwargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model  先新建一个model，然后加载预训练好的权重，就不用再训练一次了。所以创建的model和传入的pretrained weights一定是要对应的
        self.net = Net(
            # backbone2=backbones.VGG163(),
            backbone=backbones.vgg(),
            # backbone=backbones.resnet34(),
            backbone1=backbones.AlexNetV(),
            head=SiamFC(self.cfg.out_scale),
            # head2=SiamFC(self.cfg.out_scale, False),
        )
        ops.init_weights(self.net),

        # load checkpoint if provided 检查是否传入了pretrained weights，有的话就读进去，没有的话，就没有了
        if net_path is not None:  # net_path -> siamfc_alexmet_e50.pth
            self.net.load_state_dict(torch.load(
                net_path,
                map_location=lambda storage, load_state_dictc: storage), strict=False)  # # Load all tensors onto the CPU, using a function
        self.net = self.net.to(self.device)  # siamfc_alexmet_e50.pth 先用cpu读然后再转到gpu

        # pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        # model_dict = {}
        # state_dict = model.state_dict()
        # for k, v in pretrain_dict.items()
        #     if k in state_dict:
        #         model_dict[k] = v
        # state_dict.update(model_dict)
        # model.load_state_dict(state_dict)

        # setup criterion
        # self.criterion = FocalLoss()
        self.criterion = BalancedLoss()

        # setup optimizerBCE
        self.optimizer = optim.SGD(  # 梯度下降法 带momentum的
            self.net.parameters(),
            lr=self.cfg.initial_lr,  # 学习率
            weight_decay=self.cfg.weight_decay,  # 设置权值衰减，目的是防止过拟合
            momentum=self.cfg.momentum)

        # setup lr scheduler 这块不懂，还没看
        # gamma = np.power(  # np.power(a,b) 数组的幂计算 a的b次方
        #     self.cfg.ultimate_lr / self.cfg.initial_lr,  # gamma = le-3^0.02
        #     1.0 / self.cfg.epoch_num)
        gamma = np.power(  # np.power(a,b) 数组的幂计算 a的b次方
            self.cfg.ultimate_lr / self.cfg.initial_lr,  # gamma = le-3^0.02
            1.0 / self.cfg.epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer, 0.87)
        # m = 1;
        # if m >= 65:
        #     self.lr_scheduler = ExponentialLR(self.optimizer, 0.87)
        #     m = m + 1
        # elif m >= 75:
        #     self.lr_scheduler = ExponentialLR(self.optimizer, 0.7)
        #     m = m + 1
        # elif m >= 85:
        #     self.lr_scheduler = ExponentialLR(self.optimizer, 0.6)
        #     m = m + 1
        # else:
        #     self.lr_scheduler = ExponentialLR(self.optimizer, 0.5)

    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 127,  # 模板固定为127*127
            'instance_sz': 255,  # current frame固定为255*255
            'context': 0.5,
            # inference parameters
            'scale_num': 3,
            'scale_step': 1.0375,
            # 'scale_step': 1.0482,
            # 'scale_lr': 0.65,
            'scale_lr': 0.87,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,  # 相应尺寸是固定的17×17
            'response_up': 16,  # 16×17=272 把croped patch还原成orignal image
            'total_stride': 8,
            # train parameters
            'epoch_num': 50,
            'batch_size': 8,
            'num_workers': 16,
            'initial_lr': 1e-2,
            'ultimate_lr': 1e-5,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,  # radius for positive pairs 论文2.2小节
            'r_neg': 0}  # radius for negative pairs

        for key, val in kwargs.items():  # 如果传了字段名称进来，就对字段名称的数值进行更新
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(
            **cfg)  # 返回具名元组，可以用index访问也可以用字段名称访问。如访问‘out_scale’可以是Config(0)也可以是Config['out_scale']

    @torch.no_grad()  # 下面的定义类，不用计算梯度，也不做反向传播（speed up）
    def init(self, img, box):
        # set to evaluation mode
        self.net.eval()  # 把BN和Dropout固定，不再取平均或调整，直接用训练好的参数值

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([  # from[ltx,lty,w,h] -> [cy,cx,h,w]
            box[1] - 1 + (box[3] - 1) / 2,  # ok but why minus 1??
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz  # 上采样到272*272
        self.hann_window = np.outer(  # 外积，np.outer([m],[n])则生产m行n列的数组,行元素为m[i]*n的每一个元素
            np.hanning(self.upscale_sz),  # 高维数组会自动flatten成1维然后进行计算 最后的形式是分块数组组成的数组
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()  # 归一化

        # search scale factors 生成尺度池，尺度因子在一定范围内均匀递增
        self.scale_factors = self.cfg.scale_step ** np.linspace(  # np.linspace(start,stop,num)生成均匀间隔的num个数值序列
            -(self.cfg.scale_num // 2),  # **等价于np.pow() 幂运算
            self.cfg.scale_num // 2,
            self.cfg.scale_num)  # np.linspace[-1 1 3] 所以是self.scale_factors = [1.0375^(-1), 1.0375^0, 1.0375^1] -> [
        # 0.9638 1 1.0375 ]

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)  # context就是padding出来的区域 0.5(w+h)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))  # squrt[(w+0.5(w+h))*(h+0.5(w+h))]
        self.x_sz = self.z_sz * \
                    self.cfg.instance_sz / self.cfg.exemplar_sz  # 输入x应该固定为255，模板z固定为127，所以输入的模板z乘上一个缩放比例变255，但模板z
        # 因为padding了，所以最后得到的x尺度是255附近

        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))  # 算像素平均，用来padding的
        z = ops.crop_and_resize(  # resize到固定的127或255
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)

        # exemplar features
        z = torch.from_numpy(z).to(
            # 换成gpu能处理的tensor 但是为什么要做permute？ 解决： [w,h,c] -> [c,w,h]  !!!因为got10k的数据类型是ncwh，所以用这个
            self.device).permute(2, 0, 1).unsqueeze(0).float()  # unsqueeze() ，则维度变[mini-batch,c,w,h] NCWH格式，数据类型float
        # self.kernel1, self.kernel2 = self.net.backbone(z)  # crop的z经过backbone（卷积神经网络）输出z的feature，这里backbone用的是AlexNet_V1
        # self.kernel = self.net.backbone(z)  # crop的z经过backbone（卷积神经网络）输出z的feature，这里backbone用的是AlexNet_V1
        self.kernel1 = self.net.backbone(z)  # crop的z经过backbone（卷积神经网络）输出z的feature，这里backbone用的是AlexNet_V1
        self.kernel2 = self.net.backbone1(z)  # crop的z经过backbone（卷积神经网络）输出z的feature，这里backbone用的是AlexNet_V1
        self.kernel3 = self.net.maxpool(self.kernel2)
        self.kernel3 = self.net.conv(torch.cat([self.kernel1, self.kernel3], dim=1))
        print("for breakpoint")

    #  self.kernel就用第一帧初始化了，后面不会变，用与和当前帧做conv2d卷积得到响应。到这里初始化就完成了，主要是设定模板z。即Siamese的上半部分

    @torch.no_grad()  # 这里是做跟踪的时候新的帧进来则做处理，直接跟踪，所以不再做梯度，并且冻结网络中的batchnum和dropout等参数
    def update(self, img):
        # set to evaluation mode
        self.net.eval()

        # search images
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]  # 这里输出的x是三种尺度的patch，是三个数组
        # s_sc1 = torch.from_numpy(x[0]).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
        # s_sc2 = torch.from_numpy(x[1]).to(self.device).permute(2, 0, 1).unsqueeze(0).float()
        # s_sc3 = torch.from_numpy(x[2]).to(self.device).permute(2, 0, 1).unsqueeze(0).float()

        x = np.stack(x, axis=0)  # 把三个数组堆叠为一个数组，此时shape=(3,255,255,3) 第一个3是三张patch对应的3维，第二个3是rgb的3维，255 255是每张x的h w
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()  # [patch_num, channels, h, w] 即patch_num CHW

        # responses 这里的responses是包含3种尺度变换后的响应，后面通过最大响应来确定用哪张response，即哪个尺度
        # a,b,c,d = s_sc1.size()
        # x = self.net.backbone(x)  # 得到feature，此时x=[3*22*22*128]
        x1 = self.net.backbone(x)  # 得到feature，此时x=[3*22*22*128]
        x2 = self.net.backbone1(x)  # 得到feature，此时x=[3*22*22*128]
        # x1, x2 = self.net.backbone(x)  # 得到feature，此时x=[3*22*22*128]
        x3 = self.net.maxpool(x2)
        x3 = self.net.conv(torch.cat([x1, x3], dim=1))
        # self.kernel, x= self.net.feature_enhance(self.kernel,x)
        # x1 = self.net.feature_enhance(self.kernel, x[1])
        # x2 = self.net.feature_enhance(self.kernel, x[2])

        responses1 = self.net.head(self.kernel1, x1)
        responses2 = self.net.head(self.kernel2, x2)
        responses3 = self.net.head(self.kernel3, x3)
        responses = self.net.conv1(torch.cat([responses1, responses2, responses3], dim=1))

        # responses = self.net.head(self.kernel, x)  # 此时x=[3*17*17*128] head就是SiameseNet，计算当前帧与模板z的相关响应（卷积响应），模板z则是前面init初始化好的self.kernel
        responses = responses.squeeze(1).cpu().numpy()  # 删掉1维（channels） 此时responses.shape=[3,17,17]

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(  # 把3*17*17 resize到 3*272*272 (后面要映射回到原image)
            u, (self.upscale_sz, self.upscale_sz),  # 做上采样，用的是三次样条插值
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[
        :self.cfg.scale_num // 2] *= self.cfg.scale_penalty  # responses包含三张尺度的response，对第一张响应和第三张响应做惩罚，而原尺寸响应不变  为什么？
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty  #
        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))  # 分别找三张图的最大响应，然后再从这三个响应中找最大响应，尺度因子则用这张最大响应所对应的
        #                                                  (1,2)对于图的wh，即找每一张图的，0对应第几张

        # peak location
        response = responses[scale_id]  # 选最大响应对应的那张
        # s = response.max()
        # print(s)
        response -= response.min()  # 都减去最小响应值 一种平滑处理吧
        response /= response.sum() + 1e-16  # 归一化 为什么要加1e-16？
        response = (1 - self.cfg.window_influence) * response + \
                   self.cfg.window_influence * self.hann_window  # 边界效应的处理
        loc = np.unravel_index(response.argmax(),
                               response.shape)  # 中心点 croped的patch (y,x) 以patch为标尺的center，后面还要映射回原image

        # locate target center 这里的计算不太看得懂（懂啦！！） 就是逆回去把center映射到原图片 因为图片已经经过了上采样 crop等操作 [138 134]->[2.5 -1.5]->[1.25 -0.75]->[0.45269844 -0.27161906]->[110.433945 61.620953]
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2  # 倒数第一步是upsampling，所以先映射回未upsampling时的center
        disp_in_instance = disp_in_response * \
                           self.cfg.total_stride / self.cfg.response_up  # 映射回backbond之前的center backbond多层卷积所有的stride为total_stride，所以要乘回去变回之前的尺寸
        #                                                        再映射回没做resize之前的patch
        disp_in_image = disp_in_instance * self.x_sz * \
                        self.scale_factors[
                            scale_id] / self.cfg.instance_sz  # 从patch映射回没做padding之前的center，即在原image的center
        # 这里的disp_in_xx都是一个相对的差值，最后用原center加上这个差值就得到映射回来的center了
        self.center += disp_in_image

        # update target size 就是稀疏更新那个式子 （1-尺度学习率）+尺度学习率×尺度因子
        scale = (1 - self.cfg.scale_lr) * 1.0 + \
                self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale  # 按照尺度因子更新尺度
        self.z_sz *= scale  # 按照尺度因子更新尺度
        self.x_sz *= scale  # 按照尺度因子更新尺度

        # return 1-indexed and left-top based bounding box
        box = np.array([  # box还原回：顶角点 w h 因为要和计算跟踪性能，数据集提供的数据是[顶角点 w h]的格式
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box

    # 读frame->crop&resize(with scale)->features->do conv with z->response->choose best scale resopnse-> updata paras
    # -> return boxes -> visualize
    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            img = ops.read_image(img_file)

            begin = time.time()
            if f == 0:
                self.init(img, box)  # 第一帧时做初始化，生成模板z（self.kernel)
            else:
                boxes[f, :] = self.update(img)  # 计算每一帧img返回的box，box=[tlx,tly,w,h] 存到boxes里，用来做可视化
                # self.init(img, boxes[f])
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, boxes[f, :])

        return boxes, times

    # 具体在这个函数里面实现了训练和反向传播
    def train_step(self, batch, backward=True):
        # set network mode
        self.net.train(backward)

        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)  # z.shape: torch.Size([8,3,127,127])   *NCWH格式
        x = batch[1].to(self.device, non_blocking=self.cuda)  # x.shape: torch.Size([8,3,239,239])

        with torch.set_grad_enabled(backward):
            # inference
            responses = self.net(z, x)  # responses.shape: torch.Size([8, 1, 15, 15])

            # calculate loss
            labels = self._create_labels(responses.size())  # 要把labels resize成responses的size，因为后面计算loss的时候是element-wise的
            loss = self.criterion(responses, labels)

            if backward:
                # back propagation
                self.optimizer.zero_grad()
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()

                loss.backward()
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                self.optimizer.step()

        return loss.item()

    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None,
                   save_dir='VGG pretrained 46个epoch'):
        # set to train mode 设置为train模式，参数可迭代
        self.net.train()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        # 这里的dataset经过Pair是返回一对（z,x），且已经经过裁剪成为合格的输入
        dataset = Pair(
            seqs=seqs,
            transforms=transforms)

        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,  # batch_size = 8
            shuffle=True,  # batch取到的可以是与其它batch重复的，这样会存在一个问题，有的data可能从头到尾都没有被抽到
            num_workers=self.cfg.num_workers,  # 加载数据的次数，按cfg设为32，则分32次加载（是不是就是batch？？）
            pin_memory=self.cuda,  # copy Tensors into CUDA
            drop_last=True)  # 最后一个batch drop掉

        # loop over epochs
        for epoch in range(self.cfg.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)

            # loop over dataloader
            # print("let's see what's in batch")
            # print(batch)
            for it, batch in enumerate(
                    dataloader):  # enumerate枚举dataloader中的元素，返回值包括index和datas 则it对应index，batch对应dataloader
                # Epoch: 49[it+1: 914 / len(dataloader): 1166] Loss: 0.19454
                loss = self.train_step(batch, backward=True)
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                sys.stdout.flush()
            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)

    def _create_labels(self, size):  # train step里用于生成当前response的label然后与gt的label算loss，然后backward
        # skip if same sized labels already created
        if hasattr(self,
                   'labels') and self.labels.size() == size:  # hasattr找属性，如果self里有labels这个属性则返回真; self里的labels属性size和给定的相同，则直接返回labels
            return self.labels  # 这里的size主要是维度是否一致，后面的目的就是扩展维度使一致

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,  # np.where 判断0/1,相应返回0/1
                              np.ones_like(x),  # 满足dist<=r_neg则返回np.ones_like(x) 反之返回后面的输出
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels  # 在设定的positive范围内像素点值设为1, 在positive外又未到negative则设为0.5,其余设为0

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2  # 生成一个0-w的固定步长排列，stride=1 ->> 算中心点（下同）
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)  # 生成网格点坐标矩阵 but why????

        # create logistic labels 论文2.2小节
        r_pos = self.cfg.r_pos / self.cfg.total_stride  # radius of pos 换算到original image上的pos半径
        r_neg = self.cfg.r_neg / self.cfg.total_stride  # r_neg 0
        labels = logistic_labels(x, y, r_pos, r_neg)  # 返回一张图，仅有中心部分位置是0

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))  # labels的四个维度分别按size扩展，就和前面判断labels的size是否相同对得上了
        # n个batch，每个batch c个通道 每个通道是1×1数组（就是一个点嘛）
        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()  # numyp to tensor

        return self.labels