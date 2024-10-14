from albumentations.pytorch import ToTensorV2  # 导入将图像转换为张量的模块
import cv2  # 导入OpenCV用于图像处理
import albumentations as A  # 导入图像增强库
import torch  # 导入PyTorch用于深度学习
import numpy as np  # 导入NumPy用于数组操作
from torch.nn import functional as F  # 导入PyTorch的功能性模块
from skimage.measure import label, regionprops  # 导入用于图像分析的模块
import torch.nn as nn  # 导入PyTorch的神经网络模块
import logging  # 导入日志模块
import os  # 导入操作系统接口模块


# 定义日志记录器函数
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}  # 定义日志等级字典
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"  # 定义日志格式
    )
    logger = logging.getLogger(name)  # 获取日志记录器
    logger.setLevel(level_dict[verbosity])  # 设置日志等级

    os.makedirs(os.path.dirname(filename), exist_ok=True)  # 创建日志文件目录

    fh = logging.FileHandler(filename, "w")  # 定义文件处理器
    fh.setFormatter(formatter)  # 设置文件处理器格式
    logger.addHandler(fh)  # 将文件处理器添加到日志记录器

    sh = logging.StreamHandler()  # 定义流处理器
    sh.setFormatter(formatter)  # 设置流处理器格式
    logger.addHandler(sh)  # 将流处理器添加到日志记录器

    return logger  # 返回日志记录器


# 定义保存掩码函数
def save_masks(preds, save_path, mask_name):
    preds = torch.sigmoid(preds)  # 对预测结果应用Sigmoid函数
    preds[preds > 0.5] = int(1)  # 大于0.5的值设置为1
    preds[preds <= 0.5] = int(0)  # 小于等于0.5的值设置为0

    mask = preds.squeeze().cpu().numpy()  # 将张量转换为NumPy数组
    mask = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)  # 将掩码转换为灰度图像

    os.makedirs(save_path, exist_ok=True)  # 创建保存路径
    mask_path = os.path.join(save_path, f"{mask_name}")  # 构建掩码保存路径
    cv2.imwrite(mask_path, np.uint8(mask))  # 保存掩码图像



# 定义Focal Loss类
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):  # 初始化函数，设置gamma和alpha参数
        super(FocalLoss, self).__init__()  # 调用父类构造函数
        self.gamma = gamma  # 设置gamma值
        self.alpha = alpha  # 设置alpha值

    def forward(self, pred, mask):  # 前向传播函数
        if isinstance(pred, list):  # 如果预测结果是列表
            loss = 0.0  # 初始化损失
            num_pos = torch.sum(mask)  # 计算正类样本数
            num_neg = mask.numel() - num_pos  # 计算负类样本数
            for pr in pred:  # 遍历预测结果
                assert pr.shape == mask.shape, "pred and mask should have the same shape."  # 断言预测结果和掩码形状相同
                p = torch.sigmoid(pr)  # 对预测结果应用Sigmoid函数
                w_pos = (1 - p) ** self.gamma  # 计算正类权重
                w_neg = p ** self.gamma  # 计算负类权重

                loss_pos = -self.alpha * mask * w_pos * torch.log(p + 1e-12)  # 计算正类损失
                loss_neg = -(1 - self.alpha) * (1 - mask) * w_neg * torch.log(1 - p + 1e-12)  # 计算负类损失
                loss += (torch.sum(loss_pos) + torch.sum(loss_neg)) / (num_pos + num_neg + 1e-12)  # 累加损失
            return loss / 4  # 返回平均损失

        else:  # 如果预测结果不是列表
            assert pred.shape == mask.shape, "pred and mask should have the same shape."  # 断言预测结果和掩码形状相同
            p = torch.sigmoid(pred)  # 对预测结果应用Sigmoid函数
            num_pos = torch.sum(mask)  # 计算正类样本数
            num_neg = mask.numel() - num_pos  # 计算负类样本数
            w_pos = (1 - p) ** self.gamma  # 计算正类权重
            w_neg = p ** self.gamma  # 计算负类权重

            loss_pos = -self.alpha * mask * w_pos * torch.log(p + 1e-12)  # 计算正类损失
            loss_neg = -(1 - self.alpha) * (1 - mask) * w_neg * torch.log(1 - p + 1e-12)  # 计算负类损失

            loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (num_pos + num_neg + 1e-12)  # 计算总损失
            return loss  # 返回损失


# 定义Focal Loss类
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):  # 初始化函数，设置平滑参数
        super(DiceLoss, self).__init__()  # 调用父类构造函数
        self.smooth = smooth  # 设置平滑参数

    def forward(self, pred, mask):  # 前向传播函数
        if isinstance(pred, list):  # 如果预测结果是列表
            loss = 0.0  # 初始化损失
            for pr in pred:  # 遍历预测结果
                assert pr.shape == mask.shape, "pred and mask should have the same shape."  # 断言预测结果和掩码形状相同
                p = torch.sigmoid(pr)  # 对预测结果应用Sigmoid函数
                intersection = torch.sum(p * mask)  # 计算交集
                union = torch.sum(p) + torch.sum(mask)  # 计算并集
                dice_loss = (2.0 * intersection + self.smooth) / (union + self.smooth)  # 计算Dice系数
                loss += (1 - dice_loss)  # 累加损失
            return loss / 4  # 返回平均损失
        else:  # 如果预测结果不是列表
            assert pred.shape == mask.shape, "pred and mask should have the same shape."  # 断言预测结果和掩码形状相同
            p = torch.sigmoid(pred)  # 对预测结果应用Sigmoid函数
            intersection = torch.sum(p * mask)  # 计算交集
            union = torch.sum(p) + torch.sum(mask)  # 计算并集
            dice_loss = (2.0 * intersection + self.smooth) / (union + self.smooth)  # 计算Dice系数
            return 1 - dice_loss  # 返回Dice损失


class FocalDiceloss(nn.Module):  # 定义Focal Dice Loss类
    def __init__(self, weight=0.5):  # 初始化函数，设置权重
        super(FocalDiceloss, self).__init__()  # 调用父类构造函数
        self.weight = weight  # 设置权重值
        self.focal_loss = FocalLoss()  # 实例化Focal Loss
        self.dice_loss = DiceLoss()  # 实例化Dice Loss

    def forward(self, pred, mask):  # 前向传播函数
        focal_loss = self.focal_loss(pred, mask)  # 计算Focal Loss
        dice_loss = self.dice_loss(pred, mask)  # 计算Dice Loss
        loss = self.weight * focal_loss + dice_loss * (1 - self.weight)  # 计算加权总损失
        return loss  # 返回损失


def tversky_loss(pred, gt):  # 定义Tversky损失函数
    num = gt.size(0)  # 获取批次大小
    beta = 0.75  # 设置β值

    gt = gt.view(num, -1)  # 将掩码展开为向量
    pred = pred.view(num, -1)  # 将预测值展开为向量

    tp = (gt * pred).sum(1)  # 计算真正例
    fp = ((1 - gt) * pred).sum(1)  # 计算假正例
    fn = (gt * (1 - pred)).sum(1)  # 计算假负例

    tversky = 1. - (tp / (tp + beta * fn + (1 - beta) * fp)).sum() / num  # 计算Tversky损失

    return tversky  # 返回Tversky损失


class BoundaryLoss(nn.Module):  # 定义Boundary Loss类
    def __init__(self, theta0=3, theta=3):  # 初始化函数，设置θ值
        super().__init__()  # 调用父类构造函数
        self.theta0 = theta0  # 设置θ0值
        self.theta = theta  # 设置θ值

    def forward(self, pred, gt):  # 前向传播函数
        if isinstance(pred, list):  # 如果预测结果是列表
            score = 0.0  # 初始化得分
            for pr in pred:  # 遍历预测结果
                assert pr.shape == gt.shape, "pred and mask should have the same shape."  # 断言预测结果和掩码形状相同
                pr = torch.sigmoid(pr)  # 对预测结果应用Sigmoid函数
                gt_b = F.max_pool2d(1 - gt.float(), kernel_size=self.theta0, stride=1,
                                    padding=(self.theta0 - 1) // 2)  # 计算掩码边界
                gt_b -= (1 - gt)  # 调整掩码边界
                pred_b = F.max_pool2d(1 - pr, kernel_size=self.theta0, stride=1,
                                      padding=(self.theta0 - 1) // 2)  # 计算预测边界
                pred_b -= (1 - pr)  # 调整预测边界
                score += tversky_loss(pred_b, gt_b)  # 计算Tversky损失并累加
            return score / 4  # 返回平均得分

        else:  # 如果预测结果不是列表
            pred = torch.sigmoid(pred)  # 对预测结果应用Sigmoid函数
            gt_b = F.max_pool2d(1 - gt.float(), kernel_size=self.theta0, stride=1,
                                padding=(self.theta0 - 1) // 2)  # 计算掩码边界
            gt_b -= (1 - gt)  # 调整掩码边界
            pred_b = F.max_pool2d(1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)  # 计算预测边界
            pred_b -= (1 - pred)  # 调整预测边界
            score = tversky_loss(pred_b, gt_b)  # 计算Tversky损失
            return score  # 返回得分


class Boundary_Diceloss(nn.Module):  # 定义Boundary Dice Loss类
    def __init__(self):  # 初始化函数
        super(Boundary_Diceloss, self).__init__()  # 调用父类构造函数
        self.boundaryloss = BoundaryLoss()  # 实例化Boundary Loss
        self.diceLoss = DiceLoss()  # 实例化Dice Loss

    def forward(self, logits, targets):  # 前向传播函数
        alpha = 0.6  # 设置权重系数
        diceLoss = self.diceLoss(logits, targets)  # 计算Dice Loss
        boundaryloss = self.boundaryloss(logits, targets)  # 计算Boundary Loss
        score = diceLoss * alpha + (1 - alpha) * boundaryloss  # 计算加权总得分
        return score  # 返回得分
