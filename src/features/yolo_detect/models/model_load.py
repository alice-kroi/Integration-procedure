import torch
import os
from pathlib import Path

# 模型保存目录（可修改）
MODEL_DIR = Path(__file__).parent / "yolo"

def load_yolo_model(model_name='yolov5s', force_download=False):
    """
    智能加载YOLO模型
    :param model_name: 模型名称
    :param force_download: 强制重新下载模型
    :return: 加载完成的YOLO模型
    """
    # 创建模型保存目录
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"模型保存目录: {MODEL_DIR}")
    # 模型路径检查
    model_path = MODEL_DIR / f"{model_name}.pt"
    
    # 本地模型存在且不需要强制下载
    if model_path.exists() and not force_download:
        print(f"直接加载本地模型: {model_path}")
        from ultralytics import YOLO
        model = YOLO(model_path)
    else:
        print(f"下载预训练模型: {model_name}")
        from ultralytics import YOLO
        model = YOLO(f'{model_name}.pt')  # Load official weight
        model.save(model_path)  # Save in ultralytics format
    
    return model

def run_inference(model, input_data):
    """
    执行模型推理
    :param model: 已加载的YOLO模型
    :param input_data: 输入数据 (PIL.Image / np.ndarray / 文件路径)
    :return: 检测结果对象
    """
    return model(input_data)

# ... 保留原有代码 ...

if __name__ == "__main__":
    # 测试模型加载功能
    print("="*50 + "\n测试模型加载功能")
    test_model = load_yolo_model(model_name='yolov5s', force_download=False)
    print(f"\n模型架构类型: {type(test_model)}")
    print(f"输入尺寸要求: {test_model.img_size}")
    print(f"类别数量: {test_model.model[-1].nc}")

    # 测试推理功能
    print("\n" + "="*50 + "\n测试推理功能")
    
    # 测试用例1: 使用随机张量输入
    dummy_input = torch.randn(1, 3, 640, 640)
    results = run_inference(test_model, dummy_input)
    print("\n[随机张量输入结果]")
    print(f"检测到对象数量: {len(results.xyxy[0])}")
    print(f"示例检测框坐标: {results.xyxy[0][0] if len(results.xyxy[0]) > 0 else '无'}")

    # 测试用例2: 使用图片路径输入
    try:
        from PIL import Image
        import numpy as np
        
        # 生成测试图像（白色图片加随机噪声）
        test_img = Image.fromarray((np.random.rand(640,640,3)*255).astype('uint8'))
        results = run_inference(test_model, test_img)
        
        print("\n[生成图片输入结果]")
        print(f"检测到对象数量: {len(results.xyxy[0])}")
        print(f"置信度统计: {results.pandas().xyxy[0].confidence.describe()[['mean', 'std']]}")
        
    except ImportError:
        print("\n[警告] 缺少PIL/numpy依赖，部分测试用例无法运行")

    # 测试模型保存功能验证
    print("\n" + "="*50 + "\n验证模型保存功能")
    saved_path = MODEL_DIR / "yolov5s.pt"
    print(f"模型文件存在: {saved_path.exists()}")
    print(f"文件大小: {saved_path.stat().st_size/(1024*1024):.2f} MB" if saved_path.exists() else "")