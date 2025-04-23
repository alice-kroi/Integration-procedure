from PIL import Image, ImageDraw
import numpy as np

def plot_results(results, conf_threshold=0.5):
    """
    可视化YOLO检测结果
    :param results: YOLO检测结果对象
    :param conf_threshold: 置信度阈值 (默认0.5)
    :return: 带标注的PIL图像
    """
    # 复制原始图像并转换为PIL格式
    img = Image.fromarray(results.render()[0].astype('uint8'))
    draw = ImageDraw.Draw(img)
    
    # 过滤低置信度检测结果
    detections = [det for det in results.xyxy[0] if det[4] >= conf_threshold]
    
    # 绘制每个检测结果
    for *xyxy, conf, cls in detections:
        label = f"{results.names[int(cls)]} {conf:.2f}"
        
        # 绘制边界框
        draw.rectangle(xyxy, outline="red", width=3)
        
        # 计算文本位置
        text_position = (xyxy[0], xyxy[1] - 15 if xyxy[1] > 15 else xyxy[1])
        
        # 绘制标签背景
        text_bbox = draw.textbbox(text_position, label)
        draw.rectangle(text_bbox, fill="red")
        
        # 绘制文本
        draw.text(text_position, text=label, fill="white")
    
    return img