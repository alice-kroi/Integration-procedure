import torch
from data_loaders.dataload import create_dataloader
from models.model_load import load_yolo_model
import yaml
from pathlib import Path
import time
from torch.utils.tensorboard import SummaryWriter

# 加载配置文件
config_path = Path(__file__).parent / 'configs' / 'standard_config.yaml'
with open(config_path,'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
# 初始化数据加载器
train_loader = create_dataloader(
    data_dir=config['paths']['dataset']['train'],  # 更新路径访问方式
    img_size=config['model']['input_size'][0],     # 使用配置中的输入尺寸
    augment=True,
    batch_size=config['training']['batch_size'],
    num_workers=config['hardware']['workers']      # 新增工作线程数参数
)

val_loader = create_dataloader(
    data_dir=config['paths']['dataset']['val'],    # 同理更新验证集路径
    img_size=config['model']['input_size'][0],
    augment=False,
    batch_size=config['validation']['batch_size'],
    num_workers=config['hardware']['workers']
)

# 初始化模型（适配新配置）
model = load_yolo_model(model_name=config['model']['name']).to(device)
optimizer = getattr(torch.optim, config['training']['optimizer']['type'])(
    model.parameters(), 
    lr=config['training']['optimizer']['lr']
)

def train():
    # 初始化日志系统
    writer = SummaryWriter() if config['logging']['tensorboard'] else None
    best_loss = float('inf')
    
    for epoch in range(config['training']['epochs']):
        # 添加学习率调度
        lr = config['training']['optimizer']['lr'] * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # 训练阶段
        model.train()
        for batch_idx, (images, targets) in enumerate(train_loader):
            # 自动设备转移
            images = [img.to(device) for img in images]
            targets = [tgt.to(device) for tgt in targets]
            
            # 混合精度训练
            with torch.cuda.amp.autocast():
                losses = model(images, targets)
            
            # 梯度累积
            loss = losses.sum() / 2  # 假设梯度累积步长为2
            loss.backward()
            
            if (batch_idx + 1) % 2 == 0:  # 每2个批次更新一次
                optimizer.step()
                optimizer.zero_grad()
            
            # 日志记录增强
            if batch_idx % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item():.2f} | LR: {current_lr:.5f}')
                if writer:
                    writer.add_scalar('Train/Loss', loss.item(), epoch*len(train_loader)+batch_idx)
    
    # 添加模型保存逻辑
    if config['logging']['checkpoint']['save_best'] and loss < best_loss:
        torch.save(model.state_dict(), config['paths']['weights_dir'] / f'best_{config["model"]["name"]}.pt')
        best_loss = loss


if __name__ == '__main__':
    train()