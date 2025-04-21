import os
import random
import shutil

def split_yolo_dataset(dataset_root, split_ratio=0.8):
    """
    拆分YOLO格式数据集
    :param dataset_root: 数据集根目录（包含images和labels目录）
    :param split_ratio: 训练集比例（默认0.8）
    """
    # 创建输出目录
    os.makedirs(os.path.join(dataset_root, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_root, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(dataset_root, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_root, 'val', 'labels'), exist_ok=True)

    # 获取所有图片文件
    image_dir = os.path.join(dataset_root, 'images')
    images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # 随机打乱顺序
    random.shuffle(images)
    split_idx = int(len(images) * split_ratio)
    train_files = images[:split_idx]
    val_files = images[split_idx:]

    # 复制训练集文件
    for file in train_files:
        # 复制图片
        src_img = os.path.join(image_dir, file)
        dst_img = os.path.join(dataset_root, 'train', 'images', file)
        shutil.copy2(src_img, dst_img)
        
        # 复制对应标签
        label_file = os.path.splitext(file)[0] + '.txt'
        src_label = os.path.join(dataset_root, 'labels', label_file)
        dst_label = os.path.join(dataset_root, 'train', 'labels', label_file)
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)

    # 复制验证集文件（同上逻辑）
    for file in val_files:
        src_img = os.path.join(image_dir, file)
        dst_img = os.path.join(dataset_root, 'val', 'images', file)
        shutil.copy2(src_img, dst_img)
        
        label_file = os.path.splitext(file)[0] + '.txt'
        src_label = os.path.join(dataset_root, 'labels', label_file)
        dst_label = os.path.join(dataset_root, 'val', 'labels', label_file)
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)

# 使用示例
if __name__ == "__main__":
    split_yolo_dataset("E:/github/autolabel/datasets/input/yolo3_new", split_ratio=0.9)