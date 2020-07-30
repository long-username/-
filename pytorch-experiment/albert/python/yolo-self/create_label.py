import numpy as np



def computeIOU(predict, truth):
    '''
        计算 IOU ground truth 与 predict
        输入格式:
        predict:
        [
            [ # 对应一个 grund truth
                [], # B 个预测框
                []
            ],
            [
                [],
                []
            ],
            ...
        ]
        ground truth:
        [
            [], #grund truth
            [],
            ...
        ]
        输出格式
        [
            [x,x], B 个 IOU 值
            []
        ]
    '''
    truth = np.expand_dims(truth, axis=1)
    max = np.maximum(truth, predict)
    min = np.minimum(truth, predict)
    max[..., 2:4] = min[..., 2:4]
    res = max[..., 2:4] - max[..., 0:2]
    mask = np.where(res < 0)
    res[[mask[0]], [mask[1]], mask[2]] = 0
    IOU = res[..., 0] * res[..., 1]

    truth_area = truth[..., 2:4] - truth[..., 0:2]
    truth_area = truth_area[..., 0] * truth_area[..., 1]

    prd_area = predict[..., 2:4] - predict[..., 0:2]
    prd_area = prd_area[..., 0] * prd_area[..., 1]
    # print(prd_area)  # 出现第一个 bug prd_area 可能是是负数

    total_IOU = IOU / (truth_area + prd_area - IOU)
    # print(total_IOU)
    return total_IOU


def box_normlization(ground_truth, position, image_size, grid_size):
    '''
        目的是将[y, x, yy, xx] --> [y-offset, x-offset, 
        height-normlization, width-normlization]

        输入格式 ground_truth:
        [ # 一幅图片
            [], #一个 ground_truth 坐标 [y, x, yy, xx]
            [],
            ...
        ]
        position: #对应 ground truth 位置
        [
            [s, s],
            [],
            ...
        ]
    '''
    label = ground_truth.copy()
    # print(label.shape, position.shape)

    label[..., 2:4] -= label[..., 0:2]
    label[..., 0:2] += label[..., 2:4]/2
    image_size = np.ones_like(position) * image_size
    grid_size = np.ones_like(position) * grid_size
    w = np.concatenate((grid_size, image_size), axis=-1)
    label /= w                     
    label[..., 0:2] -= position
    # print(label)
    return label


def normlization2comm(predict, position, image_size, grid_size):
    '''
        作用是将[y_normalization, x_normlization,
         h_offset, w_offset]] --> [y, x, yy, xx](左上, 右下)
        
        输入格式 predict:
        [
            [ #代表某一个 ground truth
                [] #代表 B 个 bbox
                []
            ],
            ...
        ]
        postion:
        [
            [], #一个 ground truth 的位置信息
            [],
            ...
        ]
    '''
    position = np.expand_dims(position, axis=1)
    image_size = np.ones_like(position) * image_size
    grid_size = np.ones_like(position) * grid_size
    w = np.concatenate((grid_size, image_size), axis=-1)
    predict[..., 0:2] += position
    # print(predict)  
    predict *= w
    # print(predict)
    predict[..., 0:2] = predict[..., 0:2] - predict[..., 2:4] / 2
    predict[..., 2:4] = predict[..., 0:2] + predict[..., 2:4]
    # print(predict)
    return predict


def ground_truth2label(ground_truth_norm, IOU_):
    '''
        流程:
            寻找 响应 bunding_box 然后给出 label
    '''

    n, B = IOU_.shape
    label = -np.ones((n, B * 5))
    index = np.ones((n, 5))
    index *= np.arange(0, n).reshape((n, -1))  # 达到[[0, 0, 0, 0, 0][1, 1, 1..]]
    IOU = np.max(IOU_, axis=-1)

    #假设 IOU 相同 那么 不会寻找下一个 都会在一个争抢
    resposed_bbox = np.argmax(
        IOU_, axis=1).reshape((n, 1)) * 5 * np.ones((n, 5)) 
    resposed_bbox += np.arange(0, 5).reshape((1, -1))

    label[index.astype(np.int64), resposed_bbox.astype(np.int64)] = np.concatenate(
        (ground_truth_norm, IOU.reshape((n, -1))), axis=-1)
    # print(label)

    return label, resposed_bbox


def create_label(images, S, C, B, predict):
'''
    label n*S*S*(B*5+20) predict n*S*S*(B*5+20) tensor
    具体做法:
    1.找到 ground truth 的中心点
    2.找到预测 tensor 的中心点中的两个 bbox
    3.转换 bbox -> 正常坐标
    4.计算 IOU 找到 responsed_bbox i.e. 最大的 IOU
    5.把 ground truth 转化成 归一化格式 当做 label 赋值到 ground truth 中心点 bbox
    6.返回 label
'''

    size = 224
    per_grid = size / S
    n = len(images)

    labels = np.ones((n, S, S, (B * 5 + C)))
    labels = -labels
    mask = []

    for i in range(n):  #开始循环每一张图片
        ground_truth = images[i][1]
        centre = (ground_truth[:, 2:4] + ground_truth[:, 0:2]) / 2
        position = centre // per_grid #- 1  #计算错误, 应该是整除

        index = np.arange(0, B * 5)
        index = index.reshape((2, 5))
        index = index[:, 0:4]  #应该是 0:4
        # index = index.reshape((-1, ))

        # 取出对应bbox 的相应坐标 希望得到[[5],[5]]
        s1, s2 = position.T.astype(np.int32)
        pre_bbox = predict[i, s1, s2][..., index]  #这个地方比较难解决
        pre_bbox = normlization2comm(pre_bbox, position, size, per_grid)
        # print(pre_bbox, ground_truth)
        IOU = computeIOU(pre_bbox, ground_truth)
        # print(IOU)

        # 目前需要调试, 应选出最大的 IOU 加在 Ground_truth 后面
        ground_truth, resposed_bbox = ground_truth2label(
            box_normlization(ground_truth, position, size, per_grid), IOU)
        
        labels[i, s1, s2, 0:B * 5] = ground_truth
        mask.append([position, resposed_bbox])

    return mask, labels


if __name__ == '__main__':

    b = ([
        np.ones((224, 224, 3)),
        np.array([[13, 14, 56, 77], [23, 24, 76, 97]], dtype=np.float64),
        np.array([1])
    ], )
    a = np.ones((1, 7, 7, 30))  #为什么 n 是2
    _, label = create_label(b, 7, 20, 2, a)
    # print(_, '\n', label[0, 1:2, 1:2, 0:10])

    _, label = create_label(b, 7, 20, 2, label)
    print(_, '\n', label[0, 1:2, 1:2, 0:10])
