import torch
import os
from pathlib import Path

# 模型保存目录（可修改）
MODEL_DIR = Path("./models/yolo")

def load_yolo_model(model_name='yolov5s', force_download=False):
    """
    智能加载YOLO模型
    :param model_name: 模型名称
    :param force_download: 强制重新下载模型
    :return: 加载完成的YOLO模型
    """
    # 创建模型保存目录
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 模型路径检查
    model_path = MODEL_DIR / f"{model_name}.pt"
    
    # 本地模型存在且不需要强制下载
    if model_path.exists() and not force_download:
        print(f"使用本地模型: {model_path}")
        model = torch.hub.load('ultralytics/yolov5', 
                             'custom', 
                             path=model_path,
                             source='local')
    else:
        print(f"下载预训练模型: {model_name}")
        model = torch.hub.load('ultralytics/yolov5', 
                             model_name, 
                             pretrained=True,
                             verbose=False)
        # 保存模型到本地
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存至: {model_path}")
    
    return model.eval()

def run_inference(model, input_data):
    """
    执行模型推理
    :param model: 已加载的YOLO模型
    :param input_data: 输入数据 (PIL.Image / np.ndarray / 文件路径)
    :return: 检测结果对象
    """
    return model(input_data)