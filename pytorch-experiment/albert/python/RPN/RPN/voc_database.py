from chainercv.datasets import VOCBboxDataset, voc_bbox_label_names
from cv2 import imwrite, rectangle
import numpy as np

train_dataset = VOCBboxDataset(year='2007', split='train')
val_dataset = VOCBboxDataset(year='2007', split='val')
trainval_dataset = VOCBboxDataset(year='2007', split='trainval')
test_dataset = VOCBboxDataset(year='2007', split='test')


def drawRect(img, istorch=True):

    i = np.array(img[0])
    if istorch:
        i = np.transpose(i, (1, 2, 0))
    for pos in img[1]:
        [y, x, yy, xx] = pos
        rectangle(i, (x, y), (xx, yy), (255, 0, 0))
    i = i[..., [2, 1, 0]]

    return i


def get_label(predict, turth):

    predict = np.expand_dims(predict, axis=0)
    turth = np.expand_dims(turth, axis=1)
    max = np.maximum(turth, predict)
    min = np.minimum(turth, predict)

    max[..., 2:4] = min[..., 2:4]
    res = max[..., 2:4] - max[..., 0:2]

    mask = np.where(res < 0)
    res[[mask[0]], [mask[1]], [mask[2]]] = 0
    IOU = res[..., 0] * res[..., 1]
    # print(IOU)
    turth_area = turth[..., 2:4] - turth[..., 0:2]
    turth_area = turth_area[..., 0] * turth_area[..., 1]
    # print(turth_area)
    # prd_area = predict[..., 2:4] - predict[..., 0:2]
    # prd_area = prd_area[..., 0] * prd_area[..., 1]
    # # print(prd_area)
    total_IOU = IOU / (turth_area)# + prd_area - IOU)
    return total_IOU


if __name__ == '__main__':

    print(train_dataset.shape)
