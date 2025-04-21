import json
import cv2
import yaml
from pathlib import Path
from collections import defaultdict
import shutil
def yolo2labelme(yolo_dir, output_dir, classes_file):
    """
    Convert YOLO format dataset to Labelme format
    Args:
        yolo_dir: YOLO dataset directory (contains images/ and labels/)
        output_dir: Output directory for Labelme dataset
        data_yaml: Path to data.yaml containing class names
    """
    # Load class names from data.yaml
    with open(classes_file) as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]  # Read from txt

    
    # Create output directories
    output_dir = Path(output_dir)
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
    images_dir = Path(yolo_dir) / 'images'  # 移动到循环前
    labels_dir = Path(yolo_dir) / 'labels'  # 移动到循环前
    for img_path in images_dir.glob('*.*'):
        # 新增图片复制操作
        dst_img_path = output_dir / 'images' / img_path.name
        shutil.copy(img_path, dst_img_path)  # 复制原图到输出目录

        # 跳过非图片文件
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
            
        # Read image dimensions
        img = cv2.imread(str(img_path))
        height, width = img.shape[:2]
        
        # Prepare Labelme JSON structure
        json_data = {
            "version": "5.1.1",
            "flags": {},
            "shapes": [],
            "imagePath": f"images/{img_path.name}",
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width
        }
        
        # Read corresponding label file
        label_path = labels_dir / f'{img_path.stem}.txt'
        if label_path.exists():
            with open(label_path) as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                        
                    # Convert YOLO format to absolute coordinates
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    w = float(parts[3]) * width
                    h = float(parts[4]) * height
                    
                    # Calculate rectangle coordinates
                    xmin = x_center - w/2
                    ymin = y_center - h/2
                    xmax = x_center + w/2
                    ymax = y_center + h/2
                    
                    # Add to shapes
                    json_data['shapes'].append({
                        "label": class_names[class_id],
                        "points": [
                            [xmin, ymin],
                            [xmax, ymax]
                        ],
                        "group_id": None,
                        "shape_type": "rectangle",
                        "flags": {}
                    })
        
        # Save JSON file
        json_path = output_dir / f'{img_path.stem}.json'
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
def add_bboxes_to_labelme(json_path, bbox_list, label_name):
    """向Labelme JSON文件添加矩形标注
    
    Args:
        json_path: Labelme JSON文件路径
        bbox_list: 矩形列表，每个元素为(x, y, w, h)
        label_name: 标注类别名称
    """
    # 读取现有标注文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 添加新标注
    for x, y, w, h in bbox_list:
        data['shapes'].append({
            "label": label_name,
            "points": [[x, y], [x + w, y + h]],  # 左上和右下坐标
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        })
    
    # 保存更新后的文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def merge_labelme_annotations(parent_dir, output_dir):
    """
    合并子文件夹中的Labelme标注文件
    参数:
        parent_dir: 包含多个子文件夹的父目录
        output_dir: 合并后的输出目录
    """
    parent_path = Path(parent_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 收集所有JSON文件 {文件名: [(标签名, 文件路径), ...]}
    file_dict = defaultdict(list)
    
    # 遍历子文件夹
    for child_dir in parent_path.iterdir():
        if child_dir.is_dir():
            label_name = child_dir.name
            # 记录每个JSON文件及其对应的标签（去除标签后缀）
            for json_file in child_dir.glob('*.json'):
                # 新增：从文件名中去除标签后缀
                base_filename = json_file.name.replace(f'_{label_name}', '')
                file_dict[base_filename].append((label_name, json_file))
    
    # 处理每个要合并的JSON文件
    for filename, sources in file_dict.items():
        merged_data = None
        output_file = output_path / filename
        
        # 合并所有来源的标注
        for label, filepath in sources:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 初始化合并文件结构
            if not merged_data:
                merged_data = {
                    "version": data["version"],
                    "flags": data.get("flags", {}),
                    "shapes": [],
                    "imagePath": data["imagePath"],
                    "imageData": data.get("imageData"),
                    "imageHeight": data["imageHeight"],
                    "imageWidth": data["imageWidth"]
                }
            
            # 重设标签并添加标注
            for shape in data["shapes"]:
                new_shape = shape.copy()
                new_shape["label"] = label  # 使用文件夹名作为新标签
                merged_data["shapes"].append(new_shape)
        
        # 写入合并后的文件
        if merged_data:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    # Example usage
    yolo2labelme(
        yolo_dir='datasets/input/yolo_data',
        output_dir='datasets/output/labelme_data',
        classes_file='datasets/input/yolo_data/classes.txt'
    )