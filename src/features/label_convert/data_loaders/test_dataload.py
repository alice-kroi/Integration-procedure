# test_dataloader.py
import unittest
import tempfile
import shutil
import os
import json
from pathlib import Path
from dataload import DataLoader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # 创建临时目录
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        
    def tearDown(self):
        # 清理临时目录
        shutil.rmtree(self.test_dir)
    
    def create_coco_dataset(self):
        """创建COCO格式测试数据"""
        # 创建images目录
        images_dir = self.test_path / "images"
        images_dir.mkdir()
        (images_dir / "img1.jpg").touch()
        (images_dir / "img2.jpg").touch()
        
        # 创建annotations.json
        coco_data = {
            "images": [
                {"id": 1, "file_name": "images/img1.jpg", "width": 640, "height": 480},
                {"id": 2, "file_name": "images/img2.jpg", "width": 800, "height": 600}
            ],
            "annotations": [
                {"image_id": 1, "bbox": [100, 100, 200, 200], "category_id": 1},
                {"image_id": 2, "bbox": [150, 150, 300, 300], "category_id": 2}
            ]
        }
        with open(self.test_path / "annotations.json", "w") as f:
            json.dump(coco_data, f)
    
    def create_yolo_dataset(self):
        """创建YOLO格式测试数据"""
        # 创建images目录
        images_dir = self.test_path / "images"
        images_dir.mkdir()
        (images_dir / "img1.jpg").touch()
        (images_dir / "img2.jpg").touch()
        
        # 创建labels目录和标注文件
        labels_dir = self.test_path / "labels"
        labels_dir.mkdir()
        with open(labels_dir / "img1.txt", "w") as f:
            f.write("0 0.1 0.1 0.2 0.2\n")
        with open(labels_dir / "img2.txt", "w") as f:
            f.write("1 0.2 0.2 0.3 0.3\n")
    
    def create_voc_dataset(self):
        """创建VOC格式测试数据"""
        # 创建JPEGImages目录
        jpeg_dir = self.test_path / "JPEGImages"
        jpeg_dir.mkdir()
        (jpeg_dir / "img1.jpg").touch()
        (jpeg_dir / "img2.jpg").touch()
        
        # 创建Annotations目录和XML文件
        ann_dir = self.test_path / "Annotations"
        ann_dir.mkdir()
        
        # 创建第一个XML文件
        xml1 = f"""<annotation>
            <size><width>640</width><height>480</height></size>
            <object><name>cat</name><bndbox><xmin>100</xmin><ymin>100</ymin><xmax>200</xmax><ymax>200</ymax></bndbox></object>
        </annotation>"""
        with open(ann_dir / "img1.xml", "w") as f:
            f.write(xml1)
            
        # 创建第二个XML文件
        xml2 = f"""<annotation>
            <size><width>800</width><height>600</height></size>
            <object><name>dog</name><bndbox><xmin>150</xmin><ymin>150</ymin><xmax>300</xmax><ymax>300</ymax></bndbox></object>
        </annotation>"""
        with open(ann_dir / "img2.xml", "w") as f:
            f.write(xml2)
    
    def test_load_coco_format(self):
        """测试加载COCO格式数据"""
        self.create_coco_dataset()
        loader = DataLoader(self.test_dir)
        dataset = loader.load()
        
        # 验证基本信息
        self.assertEqual(dataset['info']['format'], 'COCO')
        self.assertEqual(len(dataset['images']), 2)
        self.assertEqual(len(dataset['categories']), 2)
        
        # 验证图像信息
        img1 = dataset['images'][0]
        self.assertTrue(str(self.test_path) in img1['file_name'])
        self.assertEqual(img1['width'], 640)
        self.assertEqual(img1['height'], 480)
        self.assertEqual(len(img1['annotations']), 1)
        
        # 验证标注信息
        ann1 = img1['annotations'][0]
        self.assertEqual(len(ann1['bbox']), 4)
        self.assertEqual(ann1['category_name'], 'class_1')
        self.assertIn('original_info', ann1)
    
    def test_load_yolo_format(self):
        """测试加载YOLO格式数据"""
        self.create_yolo_dataset()
        loader = DataLoader(self.test_dir)
        dataset = loader.load()
        
        # 验证基本信息
        self.assertEqual(dataset['info']['format'], 'YOLO')
        self.assertEqual(len(dataset['images']), 2)
        self.assertEqual(len(dataset['categories']), 2)
        
        # 验证图像信息
        img1 = dataset['images'][0]
        self.assertTrue(str(self.test_path) in img1['file_name'])
        self.assertEqual(len(img1['annotations']), 1)
        
        # 验证标注信息
        ann1 = img1['annotations'][0]
        self.assertEqual(len(ann1['bbox']), 4)
        self.assertEqual(ann1['category_name'], 'class_0')
    
    def test_load_voc_format(self):
        """测试加载VOC格式数据"""
        self.create_voc_dataset()
        loader = DataLoader(self.test_dir)
        dataset = loader.load()
        
        # 验证基本信息
        self.assertEqual(dataset['info']['format'], 'VOC')
        self.assertEqual(len(dataset['images']), 2)
        self.assertEqual(len(dataset['categories']), 2)
        
        # 验证图像信息
        img1 = dataset['images'][0]
        self.assertTrue(str(self.test_path) in img1['file_name'])
        self.assertEqual(img1['width'], 640)
        self.assertEqual(img1['height'], 480)
        self.assertEqual(len(img1['annotations']), 1)
        
        # 验证标注信息
        ann1 = img1['annotations'][0]
        self.assertEqual(ann1['category_name'], 'cat')
        self.assertEqual(ann1['bbox'], [100.0, 100.0, 200.0, 200.0])
    
    def test_load_empty_directory(self):
        """测试加载空目录"""
        loader = DataLoader(self.test_dir)
        dataset = loader.load()
        
        self.assertEqual(dataset['info']['format'], 'Custom')
        self.assertEqual(len(dataset['images']), 0)
        self.assertEqual(len(dataset['categories']), 0)
    
    def test_load_invalid_directory(self):
        """测试加载不存在的目录"""
        with self.assertRaises(FileNotFoundError):
            loader = DataLoader("/nonexistent/path")
            loader.load()

if __name__ == '__main__':
    unittest.main()