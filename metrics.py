import torch
import numpy as np

# 阈值函数，根据指定阈值将张量进行二值化
def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)  # 如果指定了阈值，返回大于阈值的部分
    else:
        return x  # 否则返回原始张量

# 将输入x和y转换为张量，并应用Sigmoid函数
def _list_tensor(x, y):
    m = torch.nn.Sigmoid()
    if type(x) is list:
        x = torch.tensor(np.array(x))  # 如果x是列表，转换为numpy数组再转换为张量
        y = torch.tensor(np.array(y))  # 同样处理y
    if x.min() < 0:
        x = m(x)  # 如果x中有负值，应用Sigmoid函数
    return x, y  # 返回处理后的x和y

# 计算IoU (Intersection over Union) 指标
def iou(pr, gt, eps=1e-7, threshold=0.5):
    pr_, gt_ = _list_tensor(pr, gt)  # 将预测和真实值转换为张量
    pr_ = _threshold(pr_, threshold=threshold)  # 应用阈值
    gt_ = _threshold(gt_, threshold=threshold)  # 应用阈值
    intersection = torch.sum(gt_ * pr_, dim=[1, 2, 3])  # 计算交集
    union = torch.sum(gt_, dim=[1, 2, 3]) + torch.sum(pr_, dim=[1, 2, 3]) - intersection  # 计算并集
    return ((intersection + eps) / (union + eps)).cpu().numpy()  # 计算IoU并返回numpy数组

# 计算Dice系数
def dice(pr, gt, eps=1e-7, threshold=0.5):
    pr_, gt_ = _list_tensor(pr, gt)  # 将预测和真实值转换为张量
    pr_ = _threshold(pr_, threshold=threshold)  # 应用阈值
    gt_ = _threshold(gt_, threshold=threshold)  # 应用阈值
    intersection = torch.sum(gt_ * pr_, dim=[1, 2, 3])  # 计算交集
    union = torch.sum(gt_, dim=[1, 2, 3]) + torch.sum(pr_, dim=[1, 2, 3])  # 计算联合
    return ((2. * intersection + eps) / (union + eps)).cpu().numpy()  # 计算Dice系数并返回numpy数组

# 计算分割指标
def SegMetrics(pred, label, metrics):
    metric_list = []  # 存储指标结果的列表
    if isinstance(metrics, str):
        metrics = [metrics, ]  # 如果metrics是字符串，转换为包含该字符串的列表
    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):
            continue  # 如果metric不是字符串，跳过
        elif metric == 'iou':
            metric_list.append(np.mean(iou(pred, label)))  # 计算IoU并添加到列表
        elif metric == 'dice':
            metric_list.append(np.mean(dice(pred, label)))  # 计算Dice系数并添加到列表
        else:
            raise ValueError('metric %s not recognized' % metric)  # 如果metric未识别，抛出错误
    if pred is not None:
        metric = np.array(metric_list)  # 将结果列表转换为numpy数组
    else:
        raise ValueError('metric mistakes in calculations')  # 如果预测为空，抛出错误
    return metric  # 返回计算结果
