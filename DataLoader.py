import os
from torch.utils.data import Dataset
import cv2
import torch
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


# 定义训练数据的变换
def train_transforms(img_size):
    transforms = [
        A.Resize(int(img_size), int(img_size), interpolation=cv2.INTER_NEAREST),  # 调整图像大小
        A.Transpose(p=0.5),  # 以50%的概率随机转置图像
        A.HorizontalFlip(p=0.25),  # 以50%的概率随机水平翻转图像
        A.VerticalFlip(p=0.25),  # 以50%的概率随机垂直翻转图像
        A.ShiftScaleRotate(p=0.25),  # 以50%的概率随机平移、缩放和旋转图像
        A.RandomRotate90(p=0.25),  # 以50%的概率随机旋转图像90度
        ToTensorV2(p=1.0)  # 将图像和掩码转换为PyTorch张量
    ]
    return A.Compose(transforms, p=1.)


# 定义测试数据的变换
def test_transforms(img_size):
    transforms = [
        A.Resize(int(img_size), int(img_size), interpolation=cv2.INTER_NEAREST),  # 调整图像大小
        ToTensorV2(p=1.0)  # 将图像和掩码转换为PyTorch张量
    ]
    return A.Compose(transforms, p=1.)


# 自定义数据集加载器
class DatasetLoader(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, augment_multiplier=1):
        self.image_size = image_size
        self.requires_name = requires_name
        self.pixel_mean = [123.675, 116.28, 103.53]  # 用于归一化的像素均值
        self.pixel_std = [58.395, 57.12, 57.375]  # 用于归一化的像素标准差
        self.data_dir = data_dir
        self.augment_multiplier = augment_multiplier  # 数据增强倍数

        self.mode = mode
        if mode == 'train':
            self.image_path = os.path.join(data_dir, f'{mode}/images')
            self.image_list = os.listdir(self.image_path)
            self.image_list = self.image_list * self.augment_multiplier  # 扩展列表
            self.transforms = train_transforms(image_size)  # 使用训练变换
        else:
            self.image_path = os.path.join(data_dir, f'{mode}/images')
            self.image_list = os.listdir(self.image_path)
            self.transforms = test_transforms(image_size)  # 使用测试变换

    def __getitem__(self, index):
        image_input = {}
        try:
            image_path = os.path.join(self.image_path, self.image_list[index % len(os.listdir(self.image_path))])
            mask_path = image_path.replace('images', 'masks')
            mask_path = mask_path.split('.jpg')[0] + '_segmentation.png'
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = (image - self.pixel_mean) / self.pixel_std  # 归一化图像

            mask = cv2.imread(mask_path, 0)
            h, w = mask.shape
            if mask.max() == 255:
                mask = mask / 255  # 如果需要，归一化掩码
            ori_mask = torch.tensor(mask).unsqueeze(0)  # 将掩码转换为张量并添加通道维度
        except:
            print(f'check data: {os.path.join(self.image_path, self.image_list[index % len(os.listdir(self.image_path))])}')
            image = None
            mask = None

        augments = self.transforms(image=image, mask=mask)
        image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)  # 应用变换

        image_input["image"] = image_tensor
        image_input["label"] = mask_tensor.unsqueeze(0)  # 为掩码添加通道维度

        image_name = mask_path.split('/')[-1].split('\\')[-1]
        if self.requires_name and self.mode == 'test':
            image_input["name"] = image_name
            image_input["original_size"] = (h, w)
            image_input["ori_label"] = ori_mask
            return image_input
        else:
            return image_input

    def __len__(self):
        return len(self.image_list)


if __name__ == "__main__":
    # 设置数据增强倍数，倍数为2代表每张图像产生2个变换版本
    train_dataset = DatasetLoader("dataset/isic2018", image_size=256, mode='train', requires_name=True, augment_multiplier=2)
    print("Dataset:", len(train_dataset))
    # train_batch_sampler = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # for i, batched_image in enumerate(tqdm(train_batch_sampler)):
    #     print(batched_image["image"].shape, batched_image["label"].shape)
