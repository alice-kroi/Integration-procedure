import json
import os
import shutil
from pathlib import Path
from collections import OrderedDict

def parse_labelme(json_path):
    """解析单个Labelme文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {
        'image_path': Path(json_path).parent / data['imagePath'],
        'width': data['imageWidth'],
        'height': data['imageHeight'],
        'shapes': data['shapes']
    }
def build_class_map(labelme_dir):
    """自动构建类别映射表"""
    classes = OrderedDict()
    # 修改为递归查找所有子目录中的json文件
    for json_path in Path(labelme_dir).rglob('*.json'):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for shape in data['shapes']:
                classes[shape['label']] = None
    return {cls: idx for idx, cls in enumerate(classes)}


def convert_shapes(shapes, class_map):
    """转换标注形状为YOLO格式"""
    print(shapes)
    for shape in shapes:
        label = shape['label']
        points = shape['points']
        
        # 添加类别缺失处理
        if label not in class_map:
            raise ValueError(f"Label '{label}' not found in class mapping. "
                           f"Available labels: {list(class_map.keys())}")
        
        # 获取边界框
        if shape['shape_type'] == 'rectangle':
            x1, y1 = points[0]
            x2, y2 = points[1]
        else:
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x1, x2 = min(x_coords), max(x_coords)
            y1, y2 = min(y_coords), max(y_coords)
        
        yield (class_map[label], x1, y1, x2, y2)

def save_yolo_annotation(image_info, class_map, output_dir):
    """保存YOLO格式标注文件"""
    # 生成目标路径
    image_stem = image_info['image_path'].stem
    txt_path = Path(output_dir) / 'labels' / f'{image_stem}.txt'
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 转换坐标系
    yolo_lines = []
    for class_id, x1, y1, x2, y2 in convert_shapes(image_info['shapes'], class_map):
        # 计算归一化坐标
        x_center = ((x1 + x2) / 2) / image_info['width']
        y_center = ((y1 + y2) / 2) / image_info['height']
        width = (x2 - x1) / image_info['width']
        height = (y2 - y1) / image_info['height']
        
        # 限制坐标范围
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    # 保存标注文件
    with open(txt_path, 'w') as f:
        f.write('\n'.join(yolo_lines))
    
    # 复制图像文件
    img_dest = Path(output_dir) / 'images' / image_info['image_path'].name
    img_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(image_info['image_path'], img_dest)
    
    return len(yolo_lines)

def build_class_map(labelme_dir):
    """自动构建类别映射表"""
    classes = OrderedDict()
    # 确保使用递归搜索 ↓↓↓
    for json_path in Path(labelme_dir).rglob('*.json'):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for shape in data['shapes']:
                classes[shape['label']] = None
    return {cls: idx for idx, cls in enumerate(classes)}

def convert_labelme_to_yolo(labelme_dir, output_dir, class_map=None):
    """主转换函数"""
    # 创建类别映射
    auto_class_map = build_class_map(labelme_dir) if class_map is None else {}
    final_class_map = class_map or auto_class_map
    
    # 保存类别文件
    if class_map is None:
        with open(Path(output_dir) / 'classes.txt', 'w') as f:
            f.write('\n'.join(final_class_map.keys()))
    
    # 处理所有json文件
    total_annotations = 0
    for json_path in Path(labelme_dir).rglob('*.json'):
        image_info = parse_labelme(json_path)
        count = save_yolo_annotation(image_info, final_class_map, output_dir)
        total_annotations += count
    
    print(f'转换完成！共处理 {len(list(Path(labelme_dir).glob("*.json")))} 个文件，'
          f'生成 {total_annotations} 个标注')
def convert_labelme_to_yolo2(labelme_dir, output_dir, class_map=None):
    """主转换函数"""
    # 创建类别映射
    auto_class_map = build_class_map(labelme_dir) if class_map is None else {}
    final_class_map = class_map or auto_class_map
    
    # 保存类别文件
    if class_map is None:
        with open(Path(output_dir) / 'classes.txt', 'w') as f:
            f.write('\n'.join(final_class_map.keys()))
    
    # 处理所有json文件（直接处理所有子目录中的文件）
    total_annotations = 0
    for json_path in Path(labelme_dir).rglob('*.json'):
        image_info = parse_labelme(json_path)
        count = save_yolo_annotation(image_info, final_class_map, output_dir)
        total_annotations += count

    print(f'转换完成！共处理 {len(list(Path(labelme_dir).rglob("*.json")))} 个文件，'
          f'生成 {total_annotations} 个标注')
def save_yolo_annotation2(image_info, class_map, output_dir):
    """保存YOLO格式标注文件"""
    # 生成目标路径
    image_stem = image_info['image_path'].stem
    txt_path = Path(output_dir) / 'labels' / f'{image_stem}.txt'
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 转换坐标系
    yolo_lines = []
    for class_id, x1, y1, x2, y2 in convert_shapes(image_info['shapes'], class_map):
        # 计算归一化坐标
        x_center = ((x1 + x2) / 2) / image_info['width']
        y_center = ((y1 + y2) / 2) / image_info['height']
        width = (x2 - x1) / image_info['width']
        height = (y2 - y1) / image_info['height']
        
        # 限制坐标范围
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    # 保存标注文件
    with open(txt_path, 'w') as f:
        f.write('\n'.join(yolo_lines))
    
    # 复制图像文件（强制覆盖）
    img_dest = Path(output_dir) / 'images' / image_info['image_path'].name
    img_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_info['image_path'], img_dest)  # 使用copy2保留元数据
    
    
    return len(yolo_lines)
def validate_and_fix_json_files(root_dir):
    """检查并修复缺失的imageWidth/imageHeight字段"""
    from PIL import Image
    
    for json_path in Path(root_dir).rglob('*.json'):
        with open(json_path, 'r+', encoding='utf-8') as f:
            try:
                data = json.load(f)
                needs_fix = False
                
                # 检查并修复宽度
                if 'imageWidth' not in data:
                    if 'width' in data:  # 优先使用现有width字段
                        data['imageWidth'] = data['width']
                    else:  # 读取实际图片尺寸
                        img_path = Path(json_path).parent / data['imagePath']
                        with Image.open(img_path) as img:
                            data['imageWidth'] = img.width
                    needs_fix = True
                
                # 检查并修复高度（逻辑同上）
                if 'imageHeight' not in data:
                    if 'height' in data:
                        data['imageHeight'] = data['height']
                    else:
                        img_path = Path(json_path).parent / data['imagePath']
                        with Image.open(img_path) as img:
                            data['imageHeight'] = img.height
                    needs_fix = True
                
                # 回写修复后的内容
                if needs_fix:
                    f.seek(0)
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    f.truncate()
                    print(f'已修复文件：{json_path}')
                    
            except Exception as e:
                print(f'修复 {json_path} 失败：{str(e)}')

if __name__ == '__main__':
    import argparse

    validate_and_fix_json_files('datasets/input/mark2_data')
    parser = argparse.ArgumentParser(description='Labelme转YOLO格式转换工具')
    parser.add_argument('-i', '--input', default='datasets/input/mark2_data/floorplan', 
                       help='Labelme数据目录（默认：datasets/input/test_data）')
    parser.add_argument('-o', '--output', default='datasets/input/floorplan',
                       help='YOLO输出目录（默认：datasets/input/yolo_data）')
    parser.add_argument('-c', '--classes',default='datasets/input/floorplan/classes.txt', help='可选类别映射文件')
    
    args = parser.parse_args()
    
    # 加载自定义类别映射
    custom_map = {}
    if args.classes:
        with open(args.classes) as f:
            custom_map = {line.strip(): idx for idx, line in enumerate(f)}
    
    convert_labelme_to_yolo2(
        labelme_dir=args.input,
        output_dir=args.output,
        class_map=custom_map if custom_map else None
    )

