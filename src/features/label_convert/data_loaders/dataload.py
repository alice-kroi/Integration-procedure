import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Union
import numpy as np
import yaml
class DataLoader:
    """通用标注数据加载器，支持自动识别格式
    
    Attributes:
        input_path (Path): 输入文件夹路径
        img_extensions (tuple): 支持的图像格式扩展名
        ann_formats (dict): 支持的标注格式特征
    """
    
    def __init__(self, input_path: Union[str, Path]):
        self.input_path = Path(input_path)
        self.img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        self.ann_formats = {
            'COCO': {'required': ['annotations.json'], 'structure': {'images', 'annotations'}},
            'YOLO': {'required': ['labels'], 'extensions': ('.txt',)},
            'VOC': {'required': ['Annotations'], 'extensions': ('.xml',)},
            'Labelme': {'extensions': ('.json',), 'pattern': r'^.*\.json$'}, 
            'Custom': {'extensions': ('.json', '.txt', '.xml')}
        }
        
    def detect_format(self) -> str:
        """自动检测标注格式"""
        if (self.input_path / 'annotations.json').exists():
            return 'COCO'
        if (self.input_path / 'labels').is_dir():
            return 'YOLO'
        if (self.input_path / 'Annotations').is_dir():
            return 'VOC'
        json_files = list(self.input_path.glob('*.json'))
        if json_files:
            # 随机采样3个文件验证格式
            sample_files = [f for f in json_files[:3] if f.is_file()]
            if all(self._is_labelme_file(f) for f in sample_files):
                return 'Labelme'
        # 检查文件扩展名匹配
        all_files = [f.suffix.lower() for f in self.input_path.iterdir() if f.is_file()]
        for fmt, specs in self.ann_formats.items():
            if any(ext in specs.get('extensions', ()) for ext in all_files):
                return fmt
        raise ValueError("无法识别的标注格式")

    def load(self) -> Dict:
        """加载并返回统一格式的数据集"""
        '''返回数据格式
                {
            "info": {"format": "原始格式"},
            "categories": {"类别名": 统一ID},
            "images": [
                {
                    "file_name": "绝对路径",
                    "width": int,
                    "height": int,
                    "annotations": [
                        {
                            "bbox": [四舍五入后坐标],
                            "category_id": 统一ID,
                            "category_name": "统一名称",
                            "original_info": {原始标注数据}
                        }
                    ]
                }
            ]
        }
        '''
        fmt = self.detect_format()
        dataset = {
            'info': {'format': fmt},  # 添加数据格式标识
            'images': [],
            'categories': {}         # 新增统一类别字典
        }
        
        # 原始数据加载
        raw_data = {
            'COCO': self._load_coco,
            'YOLO': self._load_yolo,
            'VOC': self._load_voc,
            'Labelme': self._load_labelme,
            'Custom': self._load_custom
        }[fmt]()
        
        # 统一转换逻辑
        category_index = 1  # 统一类别ID从1开始
        for img in raw_data['images']:
            # 转换图像信息
            unified_img = {
                'file_name': str(self.input_path / img['file_name']),  # 统一为绝对路径
                'width': img['width'],
                'height': img['height'],
                'annotations': []
            }
            
            # 转换标注信息
            for ann in img['annotations']:
                # 统一类别处理
                raw_id = str(ann.get('category_id', '0')).strip().lower()
                category_name = ann.get('category_name') or f'class_{raw_id}'
                
                if category_name not in dataset['categories']:
                    dataset['categories'][category_name] = category_index
                    category_index += 1
                    
                unified_ann = {
                    'bbox': [round(float(x), 4) for x in ann['bbox']],  # 统一精度
                    'category_id': dataset['categories'][category_name],
                    'category_name': category_name,
                    'original_info': ann  # 保留原始信息
                }
                unified_img['annotations'].append(unified_ann)
            
            dataset['images'].append(unified_img)
        
        return dataset
    
    def _load_coco(self) -> Dict:
        """加载COCO格式数据"""
        with open(self.input_path / 'annotations.json') as f:
            data = json.load(f)
        dataset = {'images': []}
        for img in data['images']:
            dataset['images'].append({
                'id': img['id'],
                'file_name': img['file_name'],
                'width': img['width'],
                'height': img['height'],
                'annotations': [{
                    'bbox': ann['bbox'],
                    'category_id': ann['category_id']
                } for ann in data['annotations'] if ann['image_id'] == img['id']]
            })
        return dataset
    
    def _load_yolo(self) -> Dict:
        """加载YOLO格式数据"""
        # 实现YOLO格式解析逻辑
        dataset = {'images': []}
        labels_dir = self.input_path / 'labels'
        images_dir = self.input_path / 'images'
        
        # 获取类别映射表
        class_map = {}
        classes_file = self.input_path / 'classes.txt'
        if classes_file.exists():
            with open(classes_file, 'r') as f:
                for idx, line in enumerate(f):
                    class_map[idx] = line.strip()

        # 遍历所有标注文件
        for label_file in labels_dir.glob('*.txt'):
            # 匹配对应的图像文件
            img_stem = label_file.stem
            img_path = None
            for ext in self.img_extensions:
                candidate = images_dir / f"{img_stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            
            if not img_path or not img_path.exists():
                continue

            # 读取图像尺寸
            try:
                from PIL import Image
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                continue

            # 解析标注内容
            annotations = []
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                    except ValueError:
                        continue
                    
                    # 转换归一化坐标为绝对坐标
                    abs_x = x_center * img_width
                    abs_y = y_center * img_height
                    abs_w = width * img_width
                    abs_h = height * img_height
                    
                    # 转换为COCO格式的[x_min, y_min, width, height]
                    x_min = abs_x - (abs_w / 2)
                    y_min = abs_y - (abs_h / 2)
                    
                    annotations.append({
                        'category_id': class_id,
                        'category_name': class_map.get(class_id, f'class_{class_id}'),
                        'bbox': [x_min, y_min, abs_w, abs_h]
                    })

            
            dataset['images'].append({
                'file_name': str(img_path.relative_to(self.input_path)),  # 改为相对路径
                'width': img_width,
                'height': img_height,
                'annotations': annotations
            })    
        
        return dataset
    
    def _load_voc(self) -> Dict:
        """加载VOC格式数据"""
        # 实现VOC格式解析逻辑
        dataset = {'images': []}
        annotations_dir = self.input_path / 'Annotations'
        images_dir = self.input_path / 'JPEGImages'

        # 遍历所有标注文件
        for xml_file in annotations_dir.glob('*.xml'):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # 解析图像基本信息
                filename = root.find('filename').text
                size = root.find('size')
                img_width = int(size.find('width').text)
                img_height = int(size.find('height').text)
                
                # 查找图像文件
                img_path = images_dir / filename
                if not img_path.exists():
                    img_path = self.input_path / filename
                    if not img_path.exists():
                        continue

                # 解析标注对象
                annotations = []
                for obj in root.iter('object'):
                    name = obj.find('name').text
                    bndbox = obj.find('bndbox')
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)
                    
                    # 转换为COCO格式
                    width = xmax - xmin
                    height = ymax - ymin
                    annotations.append({
                        'category_id': name,
                        'bbox': [xmin, ymin, width, height]
                    })


                dataset['images'].append({
                    'file_name': str(img_path.relative_to(self.input_path)),  # 改为相对路径
                    'width': img_width,
                    'height': img_height,
                    'annotations': annotations
                })
                
            except Exception as e:
                continue
        
        return dataset
    def _load_labelme(self) -> Dict:
        """加载Labelme格式数据"""
        dataset = {'images': []}
        for json_file in self.input_path.glob('*.json'):
            if not self._is_labelme_file(json_file):
                continue
                
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 构建图像路径并验证存在性
            img_path = self.input_path / data['imagePath']
            if not img_path.exists():
                continue

            # 转换标注格式
            annotations = []
            for shape in data['shapes']:
                if shape['shape_type'] == 'rectangle':
                    points = shape['points']
                    x_min = min(p[0] for p in points)
                    y_min = min(p[1] for p in points)
                    x_max = max(p[0] for p in points)
                    y_max = max(p[1] for p in points)
                    annotations.append({
                        'category_id': shape['label'],
                        'bbox': [x_min, y_min, x_max-x_min, y_max-y_min]
                    })

            dataset['images'].append({
                'file_name': str(img_path.name),
                'width': data['imageWidth'],
                'height': data['imageHeight'],
                'annotations': annotations
            })
        
        return dataset
    def _load_custom(self) -> Dict:
        """加载自定义格式数据"""
        # 实现自定义格式解析逻辑
        pass
    def _is_labelme_file(self, file_path: Path) -> bool:
        """验证是否为Labelme格式文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return all(key in data for key in ['version', 'shapes', 'imagePath'])
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False
    def export(self, dataset: Dict, output_dir: Union[str, Path], target_format: str):
        """将统一格式数据导出为指定格式
        
        Args:
            dataset: load()方法返回的统一格式数据
            output_dir: 输出目录路径
            target_format: 目标格式 (COCO/YOLO/VOC/Labelme)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 根据目标格式选择导出器
        exporters = {
            'COCO': self._export_coco,
            'YOLO': self._export_yolo,
            'VOC': self._export_voc,
            'Labelme': self._export_labelme
        }
        
        if target_format not in exporters:
            raise ValueError(f"不支持的导出格式: {target_format}，可用格式: {list(exporters.keys())}")
        
        # 执行导出并保留原始信息
        exporters[target_format](dataset, output_dir)
    def _export_coco(self, dataset: Dict, output_dir: Path):
        """导出为COCO格式"""
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": v, "name": k} for k, v in dataset['categories'].items()]
        }
        
        ann_id = 1
        for img in dataset['images']:
            # 添加图像信息
            coco_img = {
                "id": len(coco_data["images"]) + 1,
                "file_name": Path(img['file_name']).name,
                "width": img['width'],
                "height": img['height']
            }
            coco_data["images"].append(coco_img)
            
            # 添加标注信息
            for ann in img['annotations']:
                coco_ann = {
                    "id": ann_id,
                    "image_id": coco_img['id'],
                    "category_id": ann['category_id'],
                    "bbox": ann['original_info']['bbox']
                }
                coco_data["annotations"].append(coco_ann)
                ann_id += 1
        
        # 保存annotations.json
        with open(output_dir / "annotations.json", 'w') as f:
            json.dump(coco_data, f)
    def _export_yolo(self, dataset: Dict, output_dir: Path):
        """导出为YOLO格式"""
        # 创建目录结构
        (output_dir / "images").mkdir(exist_ok=True)
        labels_dir = output_dir / "labels"
        labels_dir.mkdir(exist_ok=True)
        
        # 生成类别映射文件
        with open(output_dir / "classes.txt", 'w') as f:
            for name in dataset['categories']:
                f.write(f"{name}\n")
        
        for img in dataset['images']:
            # 转换标注信息
            txt_lines = []
            for ann in img['annotations']:
                # 转换为YOLO格式（归一化中心坐标）
                x_center = (ann['bbox'][0] + ann['bbox'][2]/2) / img['width']
                y_center = (ann['bbox'][1] + ann['bbox'][3]/2) / img['height']
                width = ann['bbox'][2] / img['width']
                height = ann['bbox'][3] / img['height']
                
                txt_lines.append(f"{ann['category_id']-1} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # 写入labels文件
            label_path = labels_dir / f"{Path(img['file_name']).stem}.txt"
            with open(label_path, 'w') as f:
                f.write("\n".join(txt_lines))
    def _export_voc(self, dataset: Dict, output_dir: Path):
        """导出为VOC格式"""
        annotations_dir = output_dir / "Annotations"
        annotations_dir.mkdir(exist_ok=True)
        jpeg_dir = output_dir / "JPEGImages"
        jpeg_dir.mkdir(exist_ok=True)

        for img in dataset['images']:
            # 创建XML结构
            root = ET.Element("annotation")
            ET.SubElement(root, "filename").text = Path(img['file_name']).name
            size = ET.SubElement(root, "size")
            ET.SubElement(size, "width").text = str(img['width'])
            ET.SubElement(size, "height").text = str(img['height'])
            
            # 添加标注对象
            for ann in img['annotations']:
                obj = ET.SubElement(root, "object")
                ET.SubElement(obj, "name").text = ann['category_name']
                bndbox = ET.SubElement(obj, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(int(ann['bbox'][0]))
                ET.SubElement(bndbox, "ymin").text = str(int(ann['bbox'][1]))
                ET.SubElement(bndbox, "xmax").text = str(int(ann['bbox'][0] + ann['bbox'][2]))
                ET.SubElement(bndbox, "ymax").text = str(int(ann['bbox'][1] + ann['bbox'][3]))
            
            # 保存XML文件
            tree = ET.ElementTree(root)
            xml_path = annotations_dir / f"{Path(img['file_name']).stem}.xml"
            tree.write(xml_path, encoding='utf-8')
    def _export_labelme(self, dataset: Dict, output_dir: Path):
        """导出为Labelme格式"""
        for img in dataset['images']:
            labelme_data = {
                "version": "4.5.6",
                "flags": {},
                "shapes": [],
                "imagePath": Path(img['file_name']).name,
                "imageData": None,
                "imageHeight": img['height'],
                "imageWidth": img['width']
            }
            
            # 转换标注信息
            for ann in img['annotations']:
                shape = {
                    "label": ann['category_name'],
                    "shape_type": "rectangle",
                    "points": [
                        [ann['bbox'][0], ann['bbox'][1]],
                        [ann['bbox'][0] + ann['bbox'][2], 
                        ann['bbox'][1] + ann['bbox'][3]]
                    ]
                }
                labelme_data["shapes"].append(shape)
            
            # 保存JSON文件
            json_path = output_dir / f"{Path(img['file_name']).stem}.json"
            with open(json_path, 'w') as f:
                json.dump(labelme_data, f, indent=2)
    @classmethod
    def from_config(cls, config_path: str):
        """类方法实现配置加载"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        input_path = Path(config['input_path']).expanduser().resolve()
        instance = cls(input_path)
        
        if 'img_extensions' in config:
            instance.img_extensions = tuple(config['img_extensions'])
        
        if 'ann_formats' in config:
            for fmt in config['ann_formats'].values():
                if 'structure' in fmt:
                    fmt['structure'] = set(fmt['structure'])
            instance.ann_formats = config['ann_formats']
        
        return instance
# 使用示例
if __name__ == "__main__":
    loader = DataLoader(r"E:\github\Integration-procedure\src\features\label_convert\test_data\yolo")
    dataset = loader.load()
    print(f"成功加载 {len(dataset['images'])} 张图像")
    # 导出为指定格式（支持任意选择）
    #loader.export(dataset, "output/yolo", "YOLO")  # 导出为YOLO格式
    loader.export(dataset, r"E:\github\Integration-procedure\src\features\label_convert\test_data\voc", "VOC")    # 导出为VOC格式