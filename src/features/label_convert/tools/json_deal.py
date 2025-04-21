import json
import os
import base64
import shutil
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) 
from scripts import fix
import cv2
import yolo2labelme
def remove_image_date(file_path):
    """
    删除指定JSON文件中的imageDate字段
    :param file_path: JSON文件路径
    """
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 删除目标字段
    data.pop('imageDate', None)
    
    # 写回修改后的内容
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# 使用示例
# remove_image_date('c:/Users/Administrator/Desktop/标完改/combined_000004.json')


def split_labelme_json(source_json, output_dir):
    """
    将Labelme JSON文件按标签拆分为多个文件
    :param source_json: 源JSON文件路径
    :param output_dir: 输出目录路径
    """
    with open(source_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建按标签分类的字典
    label_dict = {}
    for shape in data['shapes']:
        label = shape['label']
        if label not in label_dict:
            label_dict[label] = []
        label_dict[label].append(shape)
    
    # 生成基础文件名
    base_name = os.path.splitext(os.path.basename(source_json))[0]
    
    # 为每个标签创建新JSON
    for label, shapes in label_dict.items():
        new_data = {
            "version": data["version"],
            "flags": data["flags"],
            "shapes": shapes,
            "imagePath": data["imagePath"]
        }
        
        output_path = os.path.join(output_dir, f"{base_name}_{label}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)

# 使用示例
# split_labelme_json('input.json', 'output_directory')
# ... 保留已有函数 ...

def process_folder(input_dir, output_root):
    """
    处理整个文件夹的Labelme JSON文件
    :param input_dir: 包含原始JSON的文件夹路径
    :param output_root: 输出根目录路径
    """
    # 创建输出根目录
    os.makedirs(output_root, exist_ok=True)
    
    # 遍历所有JSON文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(input_dir, filename)
            
            # 为每个文件创建专属输出目录
            base_name = os.path.splitext(filename)[0]
            output_dir = os.path.join(output_root, base_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # 执行拆分操作
            split_labelme_json(file_path, output_dir)

# 使用示例
# process_folder('c:/输入文件夹', 'c:/输出根目录')
def add_image_data(json_path, image_dir=None):
    """
    为Labelme JSON添加imageData字段
    :param json_path: JSON文件路径
    :param image_dir: 图片目录（默认为JSON文件所在目录）
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取图片路径
    if not image_dir:
        image_dir = os.path.dirname(json_path)
    image_path = os.path.join(image_dir, data['imagePath'])
    
    # 读取图片并编码
    with open(image_path, 'rb') as image_file:
        data['imageData'] = base64.b64encode(image_file.read()).decode('utf-8')
    
    # 写回JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def process_all_in_one(input_dir, output_root):
    """
    一站式处理文件夹中的所有Labelme JSON文件
    执行顺序：删除字段 -> 添加图片数据 -> 拆分标签
    :param input_dir: 输入文件夹路径
    :param output_root: 输出根目录路径
    """
    # 创建临时工作目录
    temp_dir = os.path.join(output_root, "_temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 第一步：复制原文件到临时目录
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            src = os.path.join(input_dir, filename)
            dst = os.path.join(temp_dir, filename)
            shutil.copy2(src, dst)
    
    # 第二步：在临时目录执行字段操作
    for filename in os.listdir(temp_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(temp_dir, filename)
            remove_image_date(file_path)
            add_image_data(file_path)
    
    # 第三步：处理拆分操作
    process_folder(temp_dir, output_root)
    
    # 清理临时目录
    shutil.rmtree(temp_dir)
def batch_add_image_data(root_dir):
    """
    递归遍历所有文件夹，为JSON文件添加imageData
    :param root_dir: 要遍历的根目录
    """
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                try:
                    add_image_data(json_path)
                    print(f"成功处理：{json_path}")
                except Exception as e:
                    print(f"处理失败 {json_path}: {str(e)}")
# ... 保留已有函数 ...

def reorganize_by_labels(root_dir):
    """
    将按图片分类的目录结构转换为按标签分类
    原始结构：
    root_dir/
        ├─pic1/
        │   ├─pic1_door.json
        │   └─pic1_window.json
        └─pic2/
            ├─pic2_door.json
            └─pic2_wall.json
    
    目标结构：
    root_dir/
        ├─door/
        │   ├─pic1_door.json
        │   └─pic2_door.json
        └─window/
            ├─pic1_window.json
            └─pic2_window.json
    """
    # 第一遍遍历：收集所有标签
    labels = set()
    for pic_dir in os.listdir(root_dir):
        pic_path = os.path.join(root_dir, pic_dir)
        if os.path.isdir(pic_path):
            for f in os.listdir(pic_path):
                if f.endswith('.json') and '_' in f:
                    label = f.split('_')[-1].split('.')[0]
                    labels.add(label)
    
    # 创建所有标签目录
    for label in labels:
        os.makedirs(os.path.join(root_dir, label), exist_ok=True)
    
    # 第二遍遍历：处理文件
    for pic_dir in os.listdir(root_dir):
        pic_path = os.path.join(root_dir, pic_dir)
        if os.path.isdir(pic_path) and not pic_dir in labels:
            for f in os.listdir(pic_path):
                if f.endswith('.json') and '_' in f:
                    # 解析标签
                    label = f.split('_')[-1].split('.')[0]
                    
                    # 原始文件路径
                    src_json = os.path.join(pic_path, f)
                    dest_dir = os.path.join(root_dir, label)
                    
                    # 读取原JSON数据
                    with open(src_json, 'r', encoding='utf-8') as json_file:
                        data = json.load(json_file)
                    
                    # 处理图片文件
                    old_img_name = data['imagePath']
                    new_img_name = f"{os.path.splitext(f)[0]}{os.path.splitext(old_img_name)[1]}"
                    src_img = os.path.join(pic_path, old_img_name)
                    dest_img = os.path.join(dest_dir, new_img_name)
                    
                    # 复制并重命名图片
                    if os.path.exists(src_img):
                        shutil.copy2(src_img, dest_img)
                        # 更新JSON中的图片路径
                        data['imagePath'] = new_img_name
                    
                    # 写回修改后的JSON
                    with open(src_json, 'w', encoding='utf-8') as json_file:
                        json.dump(data, json_file, indent=2, ensure_ascii=False)
                    
                    # 移动JSON文件到标签目录
                    shutil.move(src_json, os.path.join(dest_dir, f))

# 使用示例
# reorganize_by_labels('c:/Users/Administrator/Desktop/标完改')
def batch_rename_media(root_dir, class_list):
    """
    为指定分类的媒体文件添加分类标签后缀
    现在支持图片文件在分类目录外的场景
    """
    for class_name in class_list:
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.exists(class_dir):
            continue

        # 处理每个JSON文件
        for filename in os.listdir(class_dir):
            if filename.endswith('.json'):
                json_path = os.path.join(class_dir, filename)
                
                # 读取JSON数据获取原图路径
                with open(json_path, 'r+', encoding='utf-8') as f:
                    data = json.load(f)
                    old_img_name = data['imagePath']
                    old_img_path = os.path.join(root_dir, old_img_name)  # 根目录下的原图
                    
                    # 生成新文件名（保持原扩展名）
                    base_name = os.path.splitext(old_img_name)[0]
                    ext = os.path.splitext(old_img_name)[1]
                    new_img_name = f"{base_name}_{class_name}{ext}"
                    
                    # 复制并重命名图片到分类目录
                    if os.path.exists(old_img_path):
                        shutil.copy2(
                            old_img_path,
                            os.path.join(class_dir, new_img_name)
                        )
                        
                        # 更新JSON中的图片路径
                        data['imagePath'] = new_img_name
                        f.seek(0)
                        json.dump(data, f, indent=2, ensure_ascii=False)
                        f.truncate()

# 使用示例
# batch_rename_media('c:/标完改', ['window', 'wall', 'floorplan', 'door'])
def process_labelme_dataset(root_dir):
    """
    全自动处理Labelme数据集
    1. 收集所有标注种类
    2. 创建分类目录
    3. 拆分标注文件
    4. 复制并重命名图片
    """
    # 第一阶段：收集所有标签
    labels = set()
    
    # 遍历所有子目录
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.json'):
                json_path = os.path.join(dirpath, filename)
                with open(json_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        for shape in data.get('shapes', []):
                            labels.add(shape['label'])
                    except Exception as e:
                        print(f"解析失败 {json_path}: {str(e)}")
    
    # 创建所有标签目录
    for label in labels:
        os.makedirs(os.path.join(root_dir, label), exist_ok=True)
    
    # 第二阶段：处理文件
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.json'):
                json_path = os.path.join(dirpath, filename)
                base_name = os.path.splitext(filename)[0]
                
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 获取原图信息
                img_name = data['imagePath']
                img_ext = os.path.splitext(img_name)[1]
                src_img = os.path.join(dirpath, img_name)

                # 按标签拆分
                label_dict = {}
                for shape in data['shapes']:
                    label = shape['label']
                    if label not in label_dict:
                        label_dict[label] = []
                    label_dict[label].append(shape)
                
                # 为每个标签创建新文件
                for label, shapes in label_dict.items():
                    dest_dir = os.path.join(root_dir, label)

                    


                    # 生成新文件名，避免标签名重复添加
                    if base_name.endswith(f"_{label}"):
                        new_base = base_name
                    else:
                        new_base = f"{base_name}_{label}"
                    new_json = f"{new_base}.json"
                    new_img = f"{new_base}{img_ext}"
                    # 复制图片文件
                    dst_img = os.path.join(dest_dir, new_img)
                    # 添加检查逻辑，确保源文件和目标文件路径不同
                    if os.path.exists(src_img) and os.path.normpath(src_img) != os.path.normpath(dst_img):
                        shutil.copy2(src_img, dst_img)
                    # 创建新JSON数据
                    
                    
                    new_data = {
                        "version": data["version"],
                        "flags": data.get("flags", {}),
                        "shapes": shapes,
                        "imagePath": new_img,# 更新图片路径
                        "imageData": data["imageData"], 
                        "imageHeight": img.shape[0],
                        "imageWidth": img.shape[1]
                    }
                    
                    # 保存JSON文件
                    with open(os.path.join(dest_dir, new_json), 'w', encoding='utf-8') as f:
                        json.dump(new_data, f, indent=2, ensure_ascii=False)
                    
                    
# ... 其他函数 ...
def batch_rename_sequence(parent_dir):
    """
    批量重命名子目录中的文件为序号_标签格式
    格式示例：1_window.json / 1_window.jpg
    :param parent_dir: 父目录路径，包含多个分类子文件夹
    """
    # 遍历每个分类子目录
    for subdir in os.listdir(parent_dir):
        subdir_path = os.path.join(parent_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        
        # 获取目录下所有JSON文件并排序
        json_files = [f for f in os.listdir(subdir_path) if f.endswith('.json')]
        json_files.sort()
        
        # 为每个文件生成序号
        for idx, json_file in enumerate(json_files, 1):
            json_path = os.path.join(subdir_path, json_file)
            
            # 从原文件名提取标签（最后一段下划线后的内容）
            base_name = os.path.splitext(json_file)[0]
            label = base_name.split('_')[-1] if '_' in base_name else 'unknown'
            
            # 生成新文件名
            new_base = f"{idx}_{label}"
            new_json = f"{new_base}.json"
            new_json_path = os.path.join(subdir_path, new_json)
            
            # 读取并更新JSON内容
            with open(json_path, 'r+', encoding='utf-8') as f:
                data = json.load(f)
                # 获取原图扩展名并生成新图片名
                img_ext = os.path.splitext(data['imagePath'])[1]
                new_img = f"{new_base}{img_ext}"
                # 重命名图片文件
                old_img_path = os.path.join(subdir_path, data['imagePath'])
                new_img_path = os.path.join(subdir_path, new_img)
                if os.path.exists(old_img_path):
                    os.rename(old_img_path, new_img_path)
                # 更新JSON内容
                data['imagePath'] = new_img
                f.seek(0)
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.truncate()
            
            # 重命名JSON文件
            os.rename(json_path, new_json_path)

def create_labelme_dataset(image_dir):
    """
    创建Labelme标注文件夹结构
    Args:
        image_dir: 包含原始图片的目录（支持子目录）
    """
    # 创建输出目录
    output_dir = image_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 递归遍历所有图片文件
    for root, _, files in os.walk(image_dir):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, filename)
                
                # 读取图片尺寸（添加异常处理）
                img = cv2.imread(img_path)
                if img is None:
                    print(f"警告：无法读取图片文件 {img_path}，已跳过")
                    continue
                # 读取图片数据
                with open(img_path, 'rb') as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                
                # 生成JSON数据结构
                json_data = {
                    "version": "5.1.1",
                    "flags": {},
                    "shapes": [],
                    "imagePath": os.path.relpath(img_path, output_dir),  # 相对路径
                    "imageData": image_data,
                    "imageHeight": img.shape[0],
                    "imageWidth": img.shape[1]
                }
                
                # 保存JSON文件（保持原文件名）
                json_filename = os.path.splitext(filename)[0] + '.json'
                json_path = os.path.join(output_dir, json_filename)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)

# 使用示例
# create_labelme_dataset('C:/Users/Administrator/Desktop/标完改')

def merge_labelme_dataset(root_dir):
    """
    将按标签分类的Labelme标注合并为原始图片对应的单个JSON文件
    :param root_dir: 根目录路径（包含按标签分类的子目录）
    """
    # 收集所有基名对应的标注信息
    base_dict = {}
    
    # 遍历所有标签目录
    for label_dir in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label_dir)
        if not os.path.isdir(label_path):
            continue
            
        # 处理每个JSON文件
        for json_file in os.listdir(label_path):
            if json_file.endswith('.json'):
                json_path = os.path.join(label_path, json_file)
                
                # 解析基名和标签（示例：pic1_door.json -> base="pic1", label="door"）
                parts = json_file.rsplit('_', 1)
                if len(parts) != 2:
                    continue
                base_name, label = parts[0], parts[1].split('.')[0]
                
                # 读取JSON数据
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 初始化基名记录
                if base_name not in base_dict:
                    base_dict[base_name] = {
                        'shapes': [],
                        'image_info': {
                            'imagePath': f"{base_name}{os.path.splitext(data['imagePath'])[1]}",
                            'version': data.get('version', '5.1.1'),
                            'flags': data.get('flags', {})
                        }
                    }
                
                # 合并标注形状（添加标签信息）
                for shape in data['shapes']:
                    # 如果shape已存在则跳过（根据坐标和标签判断）
                    if shape not in base_dict[base_name]['shapes']:
                        base_dict[base_name]['shapes'].append(shape)
                
                # 移动关联图片到根目录
                src_img = os.path.join(label_path, data['imagePath'])
                dst_img = os.path.join(root_dir, base_dict[base_name]['image_info']['imagePath'])
                if os.path.exists(src_img) and not os.path.exists(dst_img):
                    shutil.move(src_img, dst_img)
    
    # 生成合并后的JSON文件
    for base_name, data in base_dict.items():
        merged_json = os.path.join(root_dir, f"{base_name}.json")
        merged_data = {
            **data['image_info'],
            "shapes": data['shapes'],
            "imageData": None  # 建议后续使用 add_image_data 添加
        }
        
        with open(merged_json, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)

def get_image_paths(folder_path):
    """获取指定文件夹下所有图片文件路径（包含子目录）
    Args:
        folder_path: 目标文件夹路径
    Returns:
        list: 包含所有图片绝对路径的列表
    """
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    img_paths = []
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in img_exts:
                img_paths.append(str(file_path.resolve()))
    
    return img_paths

if __name__ == '__main__':
    # 使用示例（输入输出目录不同）
    '''
    process_all_in_one(
        'E:/github/autolabel/datasets/input/test_data',
        'E:/github/autolabel/datasets/input/test_data'
    )
    
    batch_add_image_data('E:/github/autolabel/datasets/input/test_data')
    reorganize_by_labels('E:/github/autolabel/datasets/input/test_data')
    batch_rename_media('E:/github/autolabel/datasets/input/test_data', ['window', 'wall', 'floorplan', 'door'])'''
    #process_labelme_dataset('E:/github/autolabel/datasets/input/test_data')

    #batch_rename_sequence('C:/Users/Administrator/Desktop/标完改')
    create_labelme_dataset('E:/github/autolabel/datasets/output/labelme_data/images')
    img_list=get_image_paths('E:/github/autolabel/datasets/output/labelme_data/images')
    print(img_list)
    for img_path in img_list:
        img=cv2.imread(img_path)
        print(img_path)
        result, windows, doors= fix.annotate_color_blocks(img)
        json_path = os.path.splitext(img_path)[0] + '.json'
        print(json_path)
        yolo2labelme.add_bboxes_to_labelme(json_path, windows, 'window')
        yolo2labelme.add_bboxes_to_labelme(json_path, doors, 'door')
        result_img, base_points, opt_points = fix.smart_point_sampling(img)
        right_half = result[:, result.shape[1]//2:]
        hsv = cv2.cvtColor(right_half, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0,0,220), (179,15,255))
        
        # 绘制严格在白色区域内的十字线
        result,walls = fix.draw_white_rectangles(result_img, opt_points, white_mask)
        yolo2labelme.add_bboxes_to_labelme(json_path, walls, 'wall')
        yolo2labelme.add_bboxes_to_labelme(json_path, walls, 'floorplan')
    #batch_rename_media('E:/github/autolabel/datasets/output/labelme_data/images', ['window', 'wall', 'floorplan', 'door'])
    process_labelme_dataset('E:/github/autolabel/datasets/output/labelme_data/images')
    
