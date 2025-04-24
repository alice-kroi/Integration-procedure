import sys
from pathlib import Path
from data_loaders.dataload import load_from_config
# 添加目录到对应路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from data_loaders.dataload import DataLoader

# 方式1：使用独立函数
loader = load_from_config("configs/data_config.yaml")

# 方式2：使用类方法
loader = DataLoader.from_config("configs/data_config.yaml")

# 后续使用方式不变
dataset = loader.load()
loader.export(dataset, "output", "YOLO")