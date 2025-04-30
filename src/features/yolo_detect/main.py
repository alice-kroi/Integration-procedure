import torch
from data_loaders.dataload import create_dataloader
from models.model_load import load_yolo_model
import yaml
from pathlib import Path
import time
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from ultralytics import YOLO

# ... 保留原有导入 ...
from typing import Dict, Any, Tuple

class YOLOTrainer:
    """模块化训练器"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._get_device()
        self.model = self._init_model().to(self.device)
        self.optimizer = self._init_optimizer()
        self.train_loader, self.val_loader = self._init_dataloaders()
        self.scaler = torch.cuda.amp.GradScaler()
        self.writer = SummaryWriter() if config['logging']['tensorboard'] else None
    
    def _get_device(self) -> torch.device:
        """获取训练设备"""
        return torch.device(self.config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    
    def _init_model(self) -> torch.nn.Module:
        """初始化YOLO模型"""
        model = load_yolo_model(model_name=self.config['model']['name'], force_download=False)
        
        if self.config['model']['pretrained']:
            model.load_state_dict(torch.load(self.config['paths']['pretrained_weights']))
        return model
    
    def _init_optimizer(self) -> torch.optim.Optimizer:
        """初始化优化器"""
        return getattr(torch.optim, self.config['training']['optimizer']['type'])(
            self.model.parameters(), 
            lr=self.config['training']['optimizer']['lr']
        )
    
    def _init_dataloaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """初始化数据加载器"""
        train_loader = create_dataloader(
            data_dir=self.config['paths']['dataset']['train'],
            img_size=self.config['model']['input_size'][0],
            augment=True,
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['hardware']['workers']
        )
        val_loader = create_dataloader(
            data_dir=self.config['paths']['dataset']['val'],
            img_size=self.config['model']['input_size'][0],
            augment=False,
            batch_size=self.config['validation']['batch_size'],
            num_workers=self.config['hardware']['workers']
        )
        return train_loader, val_loader
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存训练状态"""
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        filename = f"yolo_{self.config['model']['name']}_epoch{epoch}.pth"
        torch.save(state, self.config['paths']['weights_dir'] / filename)
        if is_best:
            torch.save(state, self.config['paths']['weights_dir'] / f"best_{filename}")
    def train_epoch(self, epoch: int) -> float:
        """执行单个训练epoch"""
        self.model.train(self.config['training']['train_mode'])
        total_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            # 数据转移到设备
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # 混合精度训练
            with torch.cuda.amp.autocast():
                losses = self.model(images, targets)
                loss = losses.mean()
            
            # 梯度累积
            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % 2 == 0:  # 每2个批次更新一次
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            # 记录日志
            total_loss += loss.item()
            if batch_idx % 50 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                log_str = f"Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item():.4f} | LR: {lr:.5f}"
                print(log_str)
                if self.writer:
                    self.writer.add_scalar('Train/Loss', loss.item(), epoch*len(self.train_loader)+batch_idx)
        
        return total_loss / len(self.train_loader)
    
    def validate_epoch(self, epoch: int) -> float:
        """执行验证epoch"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                with torch.cuda.amp.autocast():
                    losses = self.model(images, targets)
                    loss = losses.mean()
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        print(f"Validation Epoch: {epoch} | Loss: {avg_loss:.4f}")
        if self.writer:
            self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        
        return avg_loss
    
def main():
    """主执行函数"""
    config_path = Path(__file__).parent / 'configs' / 'standard_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        config = OmegaConf.create(config)
        resolved_config = OmegaConf.resolve(config)  # 正确解析插值
          # 转换为普通字典
    print(config)
    # 新增配置验证

    trainer = YOLOTrainer(config)
    best_loss = float('inf')
    
    trainer.model.train(data='../configs/standard_yolo.yaml')
    '''
    for epoch in range(config['training']['epochs']):
        # 训练阶段
        train_loss = trainer.train_epoch(epoch)
        # 验证阶段
        val_loss = trainer.validate_epoch(epoch)
        
        # 保存最佳模型
        if val_loss < best_loss:
            trainer._save_checkpoint(epoch, is_best=True)
            best_loss = val_loss
            '''
    return trainer
if __name__ == '__main__':
    main()