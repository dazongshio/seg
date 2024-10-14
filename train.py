from network import SegModel  # 从自定义的network模块中导入分割模型SegModel
import torch  # 导入PyTorch库
import argparse  # 导入argparse模块，用于处理命令行参数
import os  # 导入os模块，用于与操作系统进行交互
from torch import optim  # 从PyTorch的优化器模块中导入optim
from torch.utils.data import DataLoader  # 从PyTorch的数据加载模块中导入DataLoader
from DataLoader import DatasetLoader  # 从自定义的数据加载模块中导入DatasetLoader
from utils import FocalDiceloss, DiceLoss, get_logger  # 从自定义的utils模块中导入损失函数和日志记录器
from metrics import SegMetrics  # 从自定义的metrics模块中导入分割评估指标
import time  # 导入time模块，用于计时
from tqdm import tqdm  # 导入tqdm模块，用于显示进度条
import numpy as np  # 导入numpy库，用于数值计算
from datetime import datetime  # 导入datetime模块，用于处理日期和时间
from torch.nn import functional as F  # 从PyTorch的神经网络模块中导入函数式API
import matplotlib.pyplot as plt  # 导入matplotlib库，用于绘图
from torchvision import transforms
import numpy as np

from model.UNet import Unet, resnet34_unet  # 从自定义的unet模块中导入UNet模型
from model.unetpp import NestedUNet
from model.attention_unet import AttU_Net
from model.cenet import CE_Net_
from model.r2unet import R2U_Net
from model.segnet import SegNet
from model.FAT_Net import FAT_Net

from model.networks.vit_seg_modeling import VisionTransformer as trans_ViT_seg
from model.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

from model.swin_unet.vision_transformer import SwinUnet as swin_ViT_seg
from model.swin_unet.config import get_config

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def parse_args():
    # 解析命令行参数的函数
    parser = argparse.ArgumentParser()  # 创建ArgumentParser对象
    parser.add_argument("--work_dir", type=str, default="workdir828", help="工作目录")  # 添加工作目录参数
    parser.add_argument("--run_name", type=str, default="unet", help="运行模型名称")  # 添加运行名称参数
    parser.add_argument("--epochs", type=int, default=150, help="训练的epoch数")  # 添加训练的epoch参数
    parser.add_argument("--batch_size", type=int, default=32, help="训练批次大小")  # 添加训练批次大小参数
    parser.add_argument("--image_size", type=int, default=256, help="输入图像的尺寸")  # 添加输入图像尺寸参数
    parser.add_argument("--data_path", type=str, default="dataset/isic2018", help="数据集路径")  # 添加数据集路径参数
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="评估指标")  # 添加评估指标参数
    parser.add_argument('--device', type=str, default='cuda:1', help="运行设备")  # 添加运行设备参数
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")  # 添加学习率参数
    parser.add_argument("--resume", type=str, default=None, help="加载已有模型")  # 添加模型恢复路径参数
    parser.add_argument('--lr_scheduler', type=bool, default=True, help='是否使用学习率调度器')  # 添加学习率调度器开关
    parser.add_argument('--cfg', type=str,
                        default='./model/swin_unet/swin_tiny_patch4_window7_224_lite.yaml',
                        metavar="FILE", help='path to config file', )
    parser.add_argument('--deepsupervision', default=0)
    parser.add_argument('--augment', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--run_all', type=bool, default=False)
    args = parser.parse_args()  # 解析命令行参数
    return args  # 返回解析结果


def getModel(args):
    if args.run_name == 'unet':  # 8GB
        model = Unet(3, 1).to(args.device)
    if args.run_name == 'resnet34_unet':  # 4G
        model = resnet34_unet(1, pretrained=True).to(args.device)
    if args.run_name == 'unet++':  # 22.5GB
        args.deepsupervision = False
        model = NestedUNet(args, 3, 1).to(args.device)
    if args.run_name == 'attention_unet':  # 21.5GB
        model = AttU_Net(3, 1).to(args.device)
    if args.run_name == 'segnet':  # mem out batch_size 4 --23GB
        args.batch_size = 4
        model = SegNet(3, 1).to(args.device)
    if args.run_name == 'r2unet':  # men out batch  16--22.5G
        args.batch_size = 16
        model = R2U_Net(3, 1).to(args.device)
    if args.run_name == 'cenet':  # 4GB
        model = CE_Net_().to(args.device)
    if args.run_name == 'fat_net':  # 10GB
        args.image_size = 224
        model = FAT_Net(3, 1).to(args.device)
    if args.run_name == 'mamba_sam':
        args.batch_size = 12
        args.image_size = 192
        model = SegModel(out_chans=1024, freeze_encoder=True, input_size=args.image_size, deep_supervision=True).to(
            args.device)  # 初始化SegModel并加载到指定设备
    if args.run_name == 'trans_unet':
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = 1
        config_vit.n_skip = 3
        model = trans_ViT_seg(config_vit, img_size=args.image_size, num_classes=config_vit.n_classes).to(args.device)
        model.load_from(weights=np.load(config_vit.pretrained_path))
    if args.run_name == 'swin_unet':
        config = get_config(args)
        args.image_size = 224
        model = swin_ViT_seg(config, img_size=args.image_size, num_classes=args.num_classes).to(args.device)
        model.load_from(config)

    return model


def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion):
    # 训练一个epoch的函数
    train_loader = tqdm(train_loader)  # 包装train_loader以显示进度条
    train_losses = []  # 存储每个batch的训练损失
    train_iter_metrics = [0] * len(args.metrics)  # 初始化评估指标
    model.train()  # 设置模型为训练模式
    for batch, batched_input in enumerate(train_loader):
        image, label = batched_input["image"].float().to(args.device), batched_input["label"].to(
            args.device)  # 将图像和标签加载到指定设备

        masks = model(image)  # 前向传播，获取模型输出
        loss = criterion(masks, label)  # 计算损失
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数
        optimizer.zero_grad()  # 清空梯度

        train_losses.append(loss.item())  # 记录当前batch的损失

        gpu_info = {}
        gpu_info['gpu_name'] = args.device  # 获取设备信息
        train_loader.set_postfix(train_loss=loss.item(), gpu_info=gpu_info)  # 在进度条上显示训练损失和设备信息
        if isinstance(masks, list):
            masks = masks[-1]  # 如果masks是列表，取最后一个元素
        train_batch_metrics = SegMetrics(masks, label, args.metrics)  # 计算当前batch的评估指标
        train_iter_metrics = [train_iter_metrics[i] + train_batch_metrics[i] for i in
                              range(len(args.metrics))]  # 累加评估指标

    return train_losses, train_iter_metrics  # 返回训练损失和评估指标


def test_one_epoch(args, model, test_loader, criterion):
    # 测试一个epoch的函数
    test_loader = tqdm(test_loader)  # 包装test_loader以显示进度条
    test_losses = []  # 存储每个batch的测试损失
    test_iter_metrics = [0] * len(args.metrics)  # 初始化评估指标
    model.eval()  # 设置模型为评估模式
    for batch, batched_input in enumerate(test_loader):
        image, label = batched_input["image"].float().to(args.device), batched_input["label"].to(
            args.device)  # 将图像和标签加载到指定设备

        with torch.no_grad():  # 禁用梯度计算
            masks = model(image)  # 前向传播，获取模型输出
            loss = criterion(masks, label)  # 计算损失

        test_losses.append(loss.item())  # 记录当前batch的损失
        if isinstance(masks, list):
            masks = masks[-1]  # 如果masks是列表，取最后一个元素
        test_batch_metrics = SegMetrics(masks, label, args.metrics)  # 计算当前batch的评估指标
        test_batch_metrics = [float('{:.4f}'.format(metric)) for metric in test_batch_metrics]  # 格式化指标值
        for j in range(len(args.metrics)):
            test_iter_metrics[j] += test_batch_metrics[j]  # 累加评估指标
    return test_losses, test_iter_metrics  # 返回测试损失和评估指标


def plot_combined_results(train_loss, test_loss, train_dices, test_dices, train_ious, test_ious, save_path, run_name,
                          current_time, data_path):
    # 计算每个测试数据中的最佳值
    best_test_loss = min(test_loss)
    best_test_dice = max(test_dices)
    best_test_iou = max(test_ious)

    # 生成x轴刻度
    x_ticks = np.arange(len(train_loss))

    # 创建一个3x1的子图
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))

    # 绘制损失曲线
    axs[0].plot(x_ticks, train_loss, label='Train Loss')
    axs[0].plot(x_ticks, test_loss, label=f'Test Loss (Best: {best_test_loss:.4f})')
    axs[0].set_title(f'Loss (Best Test: {best_test_loss:.4f}) batch_size:{args.batch_size} lr:{args.lr}', fontsize=20)
    axs[0].set_xlabel('Epoch', fontsize=20)
    axs[0].set_ylabel('Loss', fontsize=20)
    axs[0].legend(fontsize=20)

    # 绘制Dice曲线
    axs[1].plot(x_ticks, train_dices, label='Train Dice')
    axs[1].plot(x_ticks, test_dices, label=f'Test Dice (Best: {best_test_dice:.4f})')
    axs[1].set_title(f'Dice (Best Test: {best_test_dice:.4f}) batch_size:{args.batch_size} lr:{args.lr}', fontsize=20)
    axs[1].set_xlabel('Epoch', fontsize=20)
    axs[1].set_ylabel('Dice', fontsize=20)
    axs[1].legend(fontsize=20)

    # 绘制IoU曲线
    axs[2].plot(x_ticks, train_ious, label='Train IoU')
    axs[2].plot(x_ticks, test_ious, label=f'Test IoU (Best: {best_test_iou:.4f})')
    axs[2].set_title(f'IoU (Best Test: {best_test_iou:.4f}) batch_size:{args.batch_size} lr:{args.lr}', fontsize=20)
    axs[2].set_xlabel('Epoch', fontsize=20)
    axs[2].set_ylabel('IoU', fontsize=20)
    axs[2].legend(fontsize=20)

    # 自动调整布局以防止重叠
    plt.tight_layout()

    # 获取数据集名称
    data_path = data_path.split('/')[-1]

    # 使用更新的文件名保存图像
    filename = f'{run_name}_{data_path}_{current_time}.png'
    plt.savefig(os.path.join(save_path, filename))

    # 关闭图表以释放内存
    plt.close()


def main(args):
    # 主函数，负责训练和测试过程
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    model = getModel(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # 初始化Adam优化器
    criterion = FocalDiceloss()  # 使用FocalDice损失函数
    # criterion = DiceLoss()  # 可以选择使用Dice损失函数（此行代码已注释掉）

    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.5)  # 设置学习率调度器
        print('*******Use MultiStepLR')

    if args.resume is not None:
        with open(args.resume, "rb") as f:
            checkpoint = torch.load(f)  # 加载模型检查点
            model.load_state_dict(checkpoint['model'])  # 加载模型权重
            optimizer.load_state_dict(checkpoint['optimizer'].state_dict())  # 加载优化器状态
            print(f"*******load {args.resume}")

    train_dataset = DatasetLoader(args.data_path, image_size=args.image_size, mode='train',
                                  requires_name=False, augment_multiplier=args.augment)  # 加载训练数据集
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)  # 创建训练数据加载器
    print('*******Train data:', len(train_dataset))  # 打印训练数据集的大小

    test_dataset = DatasetLoader(args.data_path, image_size=args.image_size, mode='test',
                                 requires_name=False)  # 加载测试数据集
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)  # 创建测试数据加载器
    print('*******Test data:', len(test_dataset))  # 打印测试数据集的大小

    loggers = get_logger(os.path.join(args.work_dir, "logs",
                                      f"{args.run_name}_{datetime.now().strftime('%Y%m%d-%H%M.log')}"))  # 创建日志记录器

    best_iou = 0  # 初始化最佳IoU
    l = len(train_loader)  # 获取训练数据加载器的长度
    train_loss, train_dices, train_ious = [], [], []  # 初始化训练损失和Dice系数的列表
    test_loss, test_dices, test_ious = [], [], []  # 初始化测试损失和Dice系数的列表

    for epoch in range(0, args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}')
        # 循环训练每个epoch
        train_metrics = {}  # 初始化训练评估指标
        start = time.time()  # 记录当前时间
        os.makedirs(os.path.join(f"{args.work_dir}/models", args.run_name), exist_ok=True)  # 创建保存模型的目录
        train_losses, train_iter_metrics = train_one_epoch(args, model, optimizer, train_loader, epoch,
                                                           criterion)  # 训练一个epoch
        test_losses, test_iter_metrics = test_one_epoch(args, model, test_loader, criterion)  # 测试一个epoch

        if args.lr_scheduler is not None:
            scheduler.step()  # 更新学习率

        train_iter_metrics = [metric / l for metric in train_iter_metrics]  # 计算每个评估指标的平均值
        train_metrics = {args.metrics[i]: '{:.4f}'.format(train_iter_metrics[i]) for i in
                         range(len(train_iter_metrics))}  # 格式化训练评估指标

        test_iter_metrics = [metric / len(test_loader) for metric in test_iter_metrics]  # 计算每个评估指标的平均值
        test_metrics = {args.metrics[i]: '{:.4f}'.format(test_iter_metrics[i]) for i in
                        range(len(test_iter_metrics))}  # 格式化测试评估指标

        avgtrain_loss = np.mean(train_losses)  # 计算平均训练损失
        avgtest_loss = np.mean(test_losses)  # 计算平均测试损失
        lr = scheduler.get_last_lr()[0] if args.lr_scheduler is not None else args.lr  # 获取当前学习率
        loggers.info(f"epoch: {epoch + 1}, lr: {lr}, Train loss: {avgtrain_loss:.4f}, metrics: {train_metrics}  \n "
                     f"Test loss: {avgtest_loss:.4f}, metrics: {test_metrics}")  # 记录训练和测试的日志信息

        train_loss.append(avgtrain_loss)  # 记录训练损失
        train_dices.append(float(train_metrics['dice']))  # 记录训练的Dice系数
        train_ious.append(float(train_metrics['iou']))  # 记录训练的Dice系数

        test_loss.append(avgtest_loss)  # 记录测试损失
        test_dices.append(float(test_metrics['dice']))  # 记录测试的Dice系数
        test_ious.append(float(test_metrics['iou']))  # 记录测试的Dice系数

        if float(test_metrics['iou']) > best_iou:
            best_iou = float(test_metrics['iou'])  # 更新最佳IoU
            save_path = os.path.join(args.work_dir, "models", args.run_name,
                                     f"batch_size_{args.batch_size}_{current_time}.pth")  # 定义模型保存路径
            state = {'model': model.float().state_dict(), 'optimizer': optimizer}  # 创建保存的模型状态
            torch.save(state, save_path)  # 保存模型

        end = time.time()  # 记录结束时间
        print("Run epoch time: %.2fs" % (end - start))  # 打印本次epoch运行时间

        # 在训练循环的最后调用这个新函数
        plot_combined_results(train_loss, test_loss, train_dices, test_dices,
                              train_ious, test_ious, args.work_dir, args.run_name, current_time, args.data_path)


if __name__ == '__main__':
    total_start_time = time.time()  # 记录总运行开始时间

    args = parse_args()  # 解析命令行参数
    if args.run_all == True:

        model_names = [
            # 'unet', 'resnet34_unet', 'cenet', 'fat_net',
            # 'trans_unet', 'swin_unet',
            'mamba_sam',
            'unet++', 'attention_unet', 'r2unet'
        ]
        for model_name in model_names:
            print(f'Running model: {model_name}')
            args.run_name = model_name  # 动态设置模型名称
            args.image_size = 256
            args.batch_size = 32
            main(args)
    else:
        main(args)
    total_end_time = time.time()  # 记录总运行结束时间
    total_elapsed_time = total_end_time - total_start_time
    print(f"All models have been run in {total_elapsed_time:.2f} seconds.")
