import torch
from data_loaders.dataload import create_dataloader
from models.model_load import load_yolo_model
import yaml

# 加载配置文件
with open('./configs/standard_config.yaml') as f:
    config = yaml.safe_load(f)

# 初始化数据加载器
train_loader = create_dataloader(
    data_dir=config['train'],
    img_size=640,
    augment=True,
    batch_size=32
)

val_loader = create_dataloader(
    data_dir=config['val'],
    img_size=640,
    augment=False,
    batch_size=16
)

# 初始化模型
model = load_yolo_model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练循环示例
def train():
    model.train()
    for epoch in range(100):
        for batch_idx, (images, targets) in enumerate(train_loader):
            # 将数据移至GPU
            images = [img.to('cuda') for img in images]
            targets = [tgt.to('cuda') for tgt in targets]
            
            # 前向传播
            losses = model(images, targets)
            
            # 反向传播
            optimizer.zero_grad()
            losses.sum().backward()
            optimizer.step()
            
            # 每50个批次打印日志
            if batch_idx % 50 == 0:
                print(f'Epoch: {epoch} | Batch: {batch_idx} | Loss: {losses.sum().item():.2f}')

if __name__ == '__main__':
    train()