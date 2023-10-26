from __future__ import absolute_import, division

import torch.nn as nn
import cv2
import numpy as np


def init_weights(model, gain=1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def read_image(img_file, cvt_code=cv2.COLOR_BGR2RGB):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    return img


def show_image(img, boxes=None, box_fmt='ltwh', colors=None,
               thickness=3, fig_n=1, delay=1, visualize=True,
               cvt_code=cv2.COLOR_RGB2BGR):
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)

    # resize img if necessary
    max_size = 960
    if max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        out_size = (
            int(img.shape[1] * scale),
            int(img.shape[0] * scale))
        img = cv2.resize(img, out_size)
        if boxes is not None:
            boxes = np.array(boxes, dtype=np.float32) * scale

    if boxes is not None:
        assert box_fmt in ['ltwh', 'ltrb']
        boxes = np.array(boxes, dtype=np.int32)
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        if box_fmt == 'ltrb':
            boxes[:, 2:] -= boxes[:, :2]

        # clip bounding boxes
        bound = np.array(img.shape[1::-1])[None, :]
        boxes[:, :2] = np.clip(boxes[:, :2], 0, bound)
        boxes[:, 2:] = np.clip(boxes[:, 2:], 0, bound - boxes[:, :2])

        if colors is None:
            colors = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0),
                (0, 0, 128),
                (0, 128, 0),
                (128, 0, 0),
                (0, 128, 128),
                (128, 0, 128),
                (128, 128, 0)]
        colors = np.array(colors, dtype=np.int32)
        if colors.ndim == 1:
            colors = np.expand_dims(colors, axis=0)

        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            pt1 = (box[0], box[1])
            pt2 = (box[0] + box[2], box[1] + box[3])
            img = cv2.rectangle(img, pt1, pt2, color.tolist(), thickness)

    if visualize:
        winname = 'window_{}'.format(fig_n)
        cv2.imshow(winname, img)
        cv2.waitKey(delay)

    return img


def crop_and_resize(img, center, size, out_size,  # size是x_sz
                    border_type=cv2.BORDER_CONSTANT,  # 添加边界框像素（常数），即padding的数值为常数（这里用的是average color）
                    border_value=(0, 0, 0),  # 初始化时用的是padding 0,传參用的是·average color
                    interp=cv2.INTER_LINEAR):  # 线性插值
    # convert box to corners (0-indexed)
    size = round(size)
    # numpy.concatenate((a1, a2,…), axis=0)函数，能够一次完成多个数组的拼接。其中a1, a2,…是数组类型的参数,axis=0表示按列拼接,axis默认为0
    corners = np.concatenate((  # 左上和右下（角点）
        np.round(center - (size - 1) / 2),
        np.round(center - (size - 1) / 2) + size))
    corners = np.round(corners).astype(int)  # corners = [y_min, x_min, y_max, x_max]
    # pad image if necessary
    pads = np.concatenate((  # 其中一组 [-62 -14 -82 -250]
        -corners[:2], corners[2:] - img.shape[:2]))  # 按照search size和center画出来的search area在整张image边界里时，就不用做padding，如果有在边界外的就得给原image做padding，这样才能在原image里把search area完整裁剪出来一块patch
    npad = max(0, int(pads.max()))
    if npad > 0:
        img = cv2.copyMakeBorder(
            img, npad, npad, npad, npad,  # search area超出原image边界时，给原image padding一圈，以最大那个超出区域为准
            border_type, value=border_value)  # padding的时候用的是color_average

    # crop image patch
    corners = (corners + npad).astype(int)  # 则新conner变为原connor+padding num，向左上、右下展开（底下作了示意图）
    patch = img[corners[0]:corners[2], corners[1]:corners[3]]  # 从原image中取出以conner为边界的像素点，作为截取的patch

    # resize to out_size
    patch = cv2.resize(patch, (out_size, out_size),
                       # 把patch resize成exemplar_sz或instance_sz 即127或255 初始化时处理的是模板z用127,跟踪时处理的是当前帧x用的是255
                       interpolation=interp)  # 用线性插值做resize

    return patch

#    ------------------------
#    -                      -
#    -                      -
#    -  original image      -
#    -                      -
#    -                      -
#    -                      -
#    -                      -
#    -                      -
#    ------------------------

#### 假如search area在original image边界里时就不用padding
#    ------------------------
#    -                      -
#    -    ++++++++++        -
#    -    +        +        -
#    -    + search +        -
#    -    + aera   +        -
#    -    ++++++++++        -
#    -                      -
#    -                      -
#    ------------------------

#### 假如searcg area超出original image边界，做padding，且以超出边界中最大那个长度来padding
#### 则新conner变为原connor+padding num，向左上、右下展开

####                      padding to
####                    ---------------->                 
####                    (左2右2上2下2 总4)

#  +++++++++++++++++                                    *+++++++++++++++**************
#  +               +                                    *+ -           +             *
#  +               +                                    *+ -           +             *
#  + --------------+---------                           *+ ------------+-----------  *
#  + -             +        -                           *+ -           +          -  *
#  + -             +        -                           *+ -           +          -  *
#  +++++++++++++++++        -                           *+++++++++++++++          -  *
#    -                      -                           *  -                      -  *
#    -                      -                           *  -                      -  *
#    -                      -                           *  -                      -  *
#    -                      -                           *  -                      -  *
#    -                      -                           *  -                      -  *
#    ------------------------                           *  ------------------------  *                          -
#                                                       *                            *
#                                                       *                            *
#                                                       ******************************  
#
