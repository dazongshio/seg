import os
import shutil
import random

# 定义路径
base_dir = "isic2017"
new_base_dir = "new_isic2017"

# 创建新的数据集目录结构
new_train_images_dir = os.path.join(new_base_dir, "train", "images")
new_train_masks_dir = os.path.join(new_base_dir, "train", "masks")
new_test_images_dir = os.path.join(new_base_dir, "test", "images")
new_test_masks_dir = os.path.join(new_base_dir, "test", "masks")

os.makedirs(new_train_images_dir, exist_ok=True)
os.makedirs(new_train_masks_dir, exist_ok=True)
os.makedirs(new_test_images_dir, exist_ok=True)
os.makedirs(new_test_masks_dir, exist_ok=True)

# 收集所有的images和masks文件路径
all_images = []
all_masks = []

for folder in ["train", "test"]:
    images_dir = os.path.join(base_dir, folder, "images")
    masks_dir = os.path.join(base_dir, folder, "masks")

    images_files = sorted(os.listdir(images_dir))
    masks_files = sorted(os.listdir(masks_dir))

    for img_file, mask_file in zip(images_files, masks_files):
        all_images.append(os.path.join(images_dir, img_file))
        all_masks.append(os.path.join(masks_dir, mask_file))

# 确保图片和mask的数量一致
assert len(all_images) == len(all_masks), "Images and masks count mismatch!"

# 将数据混合并打乱顺序
data = list(zip(all_images, all_masks))
random.shuffle(data)

# 按照7:3的比例分割数据集
split_idx = int(0.7 * len(data))
train_data = data[:split_idx]
test_data = data[split_idx:]

# 复制数据到新的目录结构
for img_path, mask_path in train_data:
    shutil.copy(img_path, os.path.join(new_train_images_dir, os.path.basename(img_path)))
    shutil.copy(mask_path, os.path.join(new_train_masks_dir, os.path.basename(mask_path)))

for img_path, mask_path in test_data:
    shutil.copy(img_path, os.path.join(new_test_images_dir, os.path.basename(img_path)))
    shutil.copy(mask_path, os.path.join(new_test_masks_dir, os.path.basename(mask_path)))

print("数据集重新划分完成并存储在 'new_isic2017' 文件夹下。")
