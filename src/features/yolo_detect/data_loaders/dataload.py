import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import albumentations as A

class YOLODataset(Dataset):
    def __init__(self, 
                 data_dir: str,
                 img_size: int = 640,
                 augment: bool = True):
        """
        初始化YOLO格式数据集
        :param data_dir: 数据目录路径（包含images和labels文件夹）
        :param img_size: 训练图像尺寸
        :param augment: 是否启用数据增强
        """
        self.img_files = list(Path(data_dir).glob("images/*.jpg"))
        self.label_files = [f.parent.parent/"labels"/f.stem.with_suffix(".txt") 
                           for f in self.img_files]
        self.img_size = img_size
        self.augment = augment
        print(img_size)
        # 定义数据增强管道
        # 更新数据增强配置
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(p=0.2),
            A.RandomResizedCrop(
                size=(img_size, img_size),  # 修改为元组格式
                scale=(0.5, 1.0),
                ratio=(0.75, 1.33),
                interpolation=1
            ),
        ], bbox_params=A.BboxParams(format='yolo'))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # 加载图像和标签
        img = cv2.imread(str(self.img_files[index]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        labels = []
        with open(self.label_files[index]) as f:
            for line in f:
                class_id, x, y, w, h = map(float, line.strip().split())
                labels.append([class_id, x, y, w, h])
        
        # 数据增强
        if self.augment:
            transformed = self.transform(image=img, bboxes=labels)
            img = transformed['image']
            labels = transformed['bboxes']
        
        # 转换为Tensor
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        labels = torch.tensor(labels)
        
        return img, labels

def create_dataloader(data_dir, batch_size=16,num_workers=4, **kwargs):
    """
    创建数据加载器
    :param data_dir: 数据集根目录
    :param batch_size: 批处理大小
    :return: DataLoader实例
    """
    dataset = YOLODataset(data_dir, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: tuple(zip(*batch))  # 保持图像和标签分离
    )

if __name__ == "__main__":
    data_dir = "E:/github/Integration-procedure/src/features/yolo_detect/datasets/yolo/train"  # 替换为你的数据集路径
    train_loader = create_dataloader(data_dir, batch_size=4, augment=True)
    #val_loader = create_dataloader(data_dir, batch_size=4, augment=False)