import yaml
from pathlib import Path
from dataload import DataLoader

def load_from_config(config_path: str) -> DataLoader:
    """
    通过YAML配置文件加载数据集
    示例用法：
    >>> loader = load_from_config("configs/data_config.yaml")
    >>> dataset = loader.load()
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 处理路径配置
    input_path = Path(config['input_path']).expanduser().resolve()
    
    # 创建DataLoader实例
    loader = DataLoader(input_path)
    
    # 覆盖默认配置
    if 'img_extensions' in config:
        loader.img_extensions = tuple(config['img_extensions'])
    
    if 'ann_formats' in config:
        # 转换列表为集合类型
        for fmt in config['ann_formats'].values():
            if 'structure' in fmt:
                fmt['structure'] = set(fmt['structure'])
        loader.ann_formats = config['ann_formats']
    
    return loader