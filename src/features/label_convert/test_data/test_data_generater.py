import json
import xml.etree.ElementTree as ET
from enum import Enum
import random
from pathlib import Path
import shutil

class DatasetFormat(Enum):
    YOLO = "yolo"
    VOC = "voc"
    COCO = "coco"
    LABELME = "labelme"

class AnnotationGenerator:
    def __init__(self, class_names=None, format=DatasetFormat.YOLO):
        # 自动生成默认类别（可自定义数量）
        self.class_names = class_names or [f"class_{i}" for i in range(3)]
        self.format = format
        self.coco_data = {
            "images": [], 
            "annotations": [],
            "categories": [{"id": i, "name": name} for i, name in enumerate(self.class_names)]
        } if format == DatasetFormat.COCO else None
        print(f"初始化标注生成器 | 格式: {format.value} | 类别数: {len(self.class_names)}")  # 添加初始化提示
    def _generate_bbox(self):
        """生成随机边界框（返回绝对坐标和归一化坐标）"""
        x = random.randint(50, 600)
        y = random.randint(50, 400)
        w = random.randint(50, 150)
        h = random.randint(50, 150)
        return (x, y, w, h), (x/640, y/480, w/640, h/480)

    def generate(self, img_path, dest_dir, img_size=(640, 480)):
        """根据指定格式生成标注"""
        file_stem = Path(img_path).stem
        class_id = random.randint(0, len(self.class_names)-1)
        abs_bbox, norm_bbox = self._generate_bbox()
        print(f"\n处理图像: {img_path.name}")
        print(f"│─ 随机类别: {self.class_names[class_id]}({class_id})")
        print(f"└─ 绝对坐标: {abs_bbox} | 归一化坐标: {tuple(f'{x:.4f}' for x in norm_bbox)}")

        if self.format == DatasetFormat.YOLO:
            label_path = dest_dir / "labels" / f"{file_stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            with open(label_path, "a") as f:
                f.write(f"{class_id} {' '.join(f'{v:.6f}' for v in norm_bbox)}\n")
            print(f"生成 YOLO 标签: {label_path.relative_to(dest_dir)}")
        elif self.format == DatasetFormat.VOC:
            annotation = ET.Element("annotation")
            ET.SubElement(annotation, "filename").text = img_path.name
            size = ET.SubElement(annotation, "size")
            ET.SubElement(size, "width").text = str(img_size[0])
            ET.SubElement(size, "height").text = str(img_size[1])
            
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = self.class_names[class_id]
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(abs_bbox[0])
            ET.SubElement(bndbox, "ymin").text = str(abs_bbox[1])
            ET.SubElement(bndbox, "xmax").text = str(abs_bbox[0] + abs_bbox[2])
            ET.SubElement(bndbox, "ymax").text = str(abs_bbox[1] + abs_bbox[3])
            
            label_path = dest_dir / "Annotations" / f"{file_stem}.xml"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            ET.ElementTree(annotation).write(label_path)
            print(f"生成 VOC 标注: {label_path.relative_to(dest_dir)}")
        elif self.format == DatasetFormat.LABELME:
            label_data = {
                "version": "5.0.1",
                "flags": {},
                "shapes": [{
                    "label": self.class_names[class_id],
                    "points": [
                        [abs_bbox[0], abs_bbox[1]],
                        [abs_bbox[0]+abs_bbox[2], abs_bbox[1]+abs_bbox[3]]
                    ],
                    "shape_type": "rectangle"
                }]
            }
            label_path = dest_dir / f"{file_stem}.json"
            with open(label_path, "w") as f:
                json.dump(label_data, f, indent=2)
            print(f"生成 Labelme JSON: {label_path.relative_to(dest_dir)}")
        elif self.format == DatasetFormat.COCO:
            img_id = len(self.coco_data["images"]) + 1
            self.coco_data["images"].append({
                "id": img_id,
                "file_name": img_path.name,
                "width": img_size[0],
                "height": img_size[1]
            })
            
            self.coco_data["annotations"].append({
                "id": img_id,
                "image_id": img_id,
                "category_id": class_id,
                "bbox": list(abs_bbox),
                "area": abs_bbox[2] * abs_bbox[3],
                "iscrowd": 0
            })
            print(f"缓存 COCO 标注 (图像ID: {img_id})")
def process_dataset(src_dir, dest_dir, class_names, format=DatasetFormat.YOLO):
    """处理数据集并生成指定格式"""
    generator = AnnotationGenerator(class_names, format)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*40}\n开始转换数据集\n格式: {format.value}\n源目录: {src_dir}\n目标目录: {dest_dir}")
    print(f"类别列表: {class_names}")
    for img_path in Path(src_dir).glob("*"):
        print(img_path)
        if img_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
            # 复制图片到目标目录
            dest_img = dest_dir / ("JPEGImages" if format==DatasetFormat.VOC else "images") / img_path.name
            dest_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_path, dest_img)
            
            # 生成对应格式的标注
            generator.generate(img_path, dest_dir)

    print(f"\n复制图像: {img_path.name} => {dest_img.relative_to(dest_dir)}")
    # COCO格式需要单独保存
    if format == DatasetFormat.COCO:
        with open(dest_dir / "annotations.json", "w") as f:
            json.dump(generator.coco_data, f, indent=2)
        print(f"\n保存 COCO 总标注文件: {dest_dir}/annotations.json (含 {len(generator.coco_data['images'])} 张图像)")
        
if __name__ == "__main__":
    src_dir = r"E:\github\Integration-procedure\src\features\label_convert\test_data\images"  # 原始图片目录
    dest_dir = r"E:\github\Integration-procedure\src\features\label_convert\test_data\yolo"  # 目标目录
    class_names = ["class_1", "class_2", "class_3"]  # 自定义类别
    process_dataset(src_dir, dest_dir, ["cat", "dog"], DatasetFormat.YOLO)