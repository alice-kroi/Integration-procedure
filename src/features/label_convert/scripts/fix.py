import os
import cv2
import numpy as np
import sys
from pathlib import Path
from torchvision import transforms
from sklearn.cluster import DBSCAN
from collections import defaultdict
sys.path.append(str(Path(__file__).parent.parent)) 
from data_loaders.data_loader import CustomDataset,create_dataloader

def preprocess_white_fill(img):
    """白色区域预填充处理"""
    # 增强的白色检测
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 230])
    upper_white = np.array([179, 10, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # 三级形态学处理流程
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    
    # 1. 填充孔洞和小间隙
    filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_fill, iterations=5)
    # 2. 连接邻近区域
    connected = cv2.dilate(filled, kernel_smooth, iterations=3)
    # 3. 平滑边缘
    smoothed = cv2.erode(connected, kernel_smooth, iterations=1)
    
    # 应用填充结果
    result = img.copy()
    result[smoothed == 255] = (255, 255, 255)
    return result
def dilate_white_regions(img):
    """白色区域定向膨胀"""
    # 获取当前白色区域掩膜
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0,0,220]), np.array([179,15,255]))
    
    # 创建方向性核（水平+垂直增强）
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2))
    dilated = cv2.dilate(mask, kernel, iterations=2)
    
    # 应用膨胀结果
    result = img.copy()
    result[dilated == 255] = (255,255,255)
    return result
def process_repires(input_dir, output_dir):
    # 创建数据集实例（移除尺寸调整）
    transform = transforms.Compose([
        transforms.ToTensor()  # 仅转换到[0,1]范围，保持原始尺寸
    ])
    dataset = CustomDataset(root_dir=input_dir, transform=transform)
    
    # 处理所有样本
    for idx in range(len(dataset)):
        _, _, right_half = dataset[idx]
        
        # 转换张量为OpenCV格式（保持原始尺寸）
        img = right_half.numpy().transpose(1, 2, 0)
        img = (img * 255).astype(np.uint8)
        repaired = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        repaired=dilate_white_regions(repaired)
        reparied=preprocess_white_fill(repaired)
        #repaired = reshape_color_blocks(repaired)
        #repaired = reshape_white_areas(repaired)


        cv2.imwrite(os.path.join(output_dir, f"right_half_{idx:04d}.jpg"), repaired)



def repair_edges(image):
    """使用颜色量化+形态学操作修复图像边缘"""
    # 颜色量化（新增）
    pixels = image.reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, palette = cv2.kmeans(pixels.astype(np.float32), 
                                   K=4,  # 根据实际颜色数量调整
                                   bestLabels=None,
                                   criteria=criteria,
                                   attempts=10,
                                   flags=cv2.KMEANS_RANDOM_CENTERS)
    quantized = palette[labels.flatten()].reshape(image.shape)
    
    # 边缘增强（新增）
    gray = cv2.cvtColor(quantized.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
    sharpened = cv2.addWeighted(gray, 1.5, laplacian, -0.5, 0)
    
    # 形态学操作闭合边缘（新增）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    morphed = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel)
    
    # 查找轮廓时使用修正后的二值图像（修改）
    _, thresh = cv2.threshold(morphed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 绘制修正后的轮廓
    result = quantized.copy()
    for cnt in contours:
        epsilon = 0.005 * cv2.arcLength(cnt, True)  # 减小多边形逼近系数
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(result, [approx], -1, (0,0,0), 1)
        
    return result
def reshape_color_blocks(img):
    # 转换到BGR颜色空间
    bgr = img.copy()
    
    # 精确颜色阈值定义 (BGR格式)
    pure_red_mask = (bgr[:,:,0] <= 5) & (bgr[:,:,1] <= 5) & (bgr[:,:,2] >= 250)
    pure_green_mask = (bgr[:,:,0] <= 5) & (bgr[:,:,1] >= 250) & (bgr[:,:,2] <= 5)

    # 转换掩膜为uint8格式
    red_mask = pure_red_mask.astype(np.uint8) * 255
    green_mask = pure_green_mask.astype(np.uint8) * 255

    # 形态学处理参数
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    
    # 处理红色区域
    red_processed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_contours, _ = cv2.findContours(red_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 处理绿色区域
    green_processed = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    green_contours, _ = cv2.findContours(green_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 合并颜色区域
    color_mask = cv2.bitwise_or(red_processed, green_processed)
    # 查找轮廓并绘制矩形
    result = img.copy()
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # 获取直立矩形区域
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 创建精确掩膜
        mask = np.zeros_like(color_mask)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        
        # 计算掩膜区域平均颜色
        mean_color = cv2.mean(img, mask=mask)[:3]
        
        # 根据平均颜色确定边框
        border_color = (0, 0, 255) if mean_color[2] > mean_color[1] else (0, 255, 0)
        
        # 绘制直立矩形
        cv2.rectangle(result, (x, y), (x+w, y+h), border_color, 2)

    return result
def reshape_white_areas(img):
    """专用白色区域矩形修正"""
    # 增强的白色检测（包含更广的亮度范围）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 220])  # 降低亮度阈值
    upper_white = np.array([179, 15, 255])  # 提高饱和度容忍
    
    # 生成精确掩膜
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # 分阶段形态学处理
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    kernel_merge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    merged = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_merge)
    
    # 多层级轮廓处理
    result = img.copy()
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            # 使用非旋转矩形避免角度干扰
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 创建临时掩膜检查实际覆盖范围
            temp_mask = np.zeros_like(mask)
            cv2.rectangle(temp_mask, (x, y), (x+w, y+h), 255, -1)
            actual_coverage = cv2.bitwise_and(temp_mask, mask)
            
            # 仅当实际覆盖率达到80%以上才进行填充
            if cv2.countNonZero(actual_coverage)/(w*h) > 0.8:
                cv2.rectangle(result, (x, y), (x+w, y+h), (255,255,255), -1)
    
    return result
def fill_color_areas(img_path, output_path):
    """填充红绿区域并保存结果"""
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 红色检测范围（HSV空间）
    red_mask = cv2.inRange(hsv, np.array([0, 150, 50]), np.array([10, 255, 255])) | \
               cv2.inRange(hsv, np.array([170, 150, 50]), np.array([180, 255, 255]))
               
    # 绿色检测范围（HSV空间）  
    green_mask = cv2.inRange(hsv, np.array([36, 50, 50]), np.array([89, 255, 255]))
    
    # 形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    red_filled = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    green_filled = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    # 创建纯色填充层
    red_layer = np.zeros_like(img)
    red_layer[red_filled == 255] = (0, 0, 255)  # BGR红色
    
    green_layer = np.zeros_like(img)
    green_layer[green_filled == 255] = (0, 255, 0)  # BGR绿色

    # 合成结果
    result = img.copy()
    result = cv2.addWeighted(result, 0.7, red_layer, 0.3, 0)
    result = cv2.addWeighted(result, 0.7, green_layer, 0.3, 0)
    
    cv2.imwrite(output_path, result)
    return result

def detect_color_bars(img, color='red'):
    """识别指定颜色的细长条状区域"""
    # 增强型颜色检测（HSV空间）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 根据颜色设置阈值
    if color == 'red':
        lower = np.array([0, 150, 150])
        upper = np.array([10, 255, 255])
        lower2 = np.array([170, 150, 150])
        upper2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower, upper) | cv2.inRange(hsv, lower2, upper2)
    else:  # green
        lower = np.array([40, 150, 50])
        upper = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    
    # 方向自适应的形态学处理
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (10,1))  # 水平核
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1,10))  # 垂直核
    processed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_h)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel_v)
    
    return processed

def annotate_color_blocks(img):
    """返回红色（窗）和绿色（门）的标注框位置列表"""
    result = img.copy()
    windows = []  # 存储红色窗的位置 (x, y, w, h)
    doors = []     # 存储绿色门的位置 (x, y, w, h)
    mid_x = img.shape[1] // 2  # 新增：获取图片水平中线
    
    # 处理红色（窗）
    processed_red = detect_color_bars(img, 'red')
    red_contours, _ = cv2.findContours(processed_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in red_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 新增右半区判断：x + w > mid_x
        if (w/h > 1.5 or h/w > 1.5) and min(w,h) <= 10 and (x + w) > mid_x:
            windows.append((x, y, w, h))
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 0), 2)
    
    # 处理绿色（门）
    processed_green = detect_color_bars(img, 'green')
    green_contours, _ = cv2.findContours(processed_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in green_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 新增右半区判断：x + w > mid_x
        if (w/h > 1.5 or h/w > 1.5) and min(w,h) <= 10 and (x + w) > mid_x:
            doors.append((x, y, w, h))
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 0), 2)
    
    return result, windows, doors
def detect_other_areas(img):
    """检测并标记非基础色区域的直角多边形"""
    # 创建排除掩膜（红、绿、白、黑）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 排除颜色阈值定义
    red_mask = cv2.inRange(hsv, (0,150,50), (10,255,255)) | cv2.inRange(hsv, (170,150,50), (180,255,255))
    green_mask = cv2.inRange(hsv, (40,50,50), (80,255,255))
    white_mask = cv2.inRange(hsv, (0,0,220), (179,15,255))
    black_mask = cv2.inRange(hsv, (0,0,0), (180,255,30))
    
    # 合并排除区域并取反
    combined_mask = red_mask | green_mask | white_mask | black_mask
    other_mask = cv2.bitwise_not(combined_mask)
    
    # 修改形态学处理为保持区域独立
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    processed = cv2.morphologyEx(other_mask, cv2.MORPH_OPEN, kernel)  # 改为开运算去除噪点
    
    # 强化区域分离
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result = img.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue

        # 获取初始多边形逼近
        epsilon = 0.011 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # 强制直角化处理
        rect_points = []
        for i in range(len(approx)):
            # 连接相邻顶点生成线段
            x1, y1 = approx[i-1][0]
            x2, y2 = approx[i][0]
            
            # 计算线段走向
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            
            # 强制水平或垂直
            if dx > dy:  # 水平优先
                rect_points.append((x1, y1))
                rect_points.append((x2, y1))  # 保持Y坐标一致
            else:  # 垂直优先
                rect_points.append((x1, y1))
                rect_points.append((x1, y2))  # 保持X坐标一致

        # 去除重复点并重构多边形
        rect_points = list(dict.fromkeys(rect_points))
        if len(rect_points) >= 4:
            rect_approx = np.array(rect_points, dtype=np.int32).reshape(-1,1,2)
            cv2.drawContours(result, [rect_approx], -1, (255, 0, 255), 2)
    
    # 添加返回语句
    return result

def detect_areas(img):
    """检测并标记非基础色区域的直角多边形"""
    # 创建排除掩膜（红、绿、白、黑）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 排除颜色阈值定义
    red_mask = cv2.inRange(hsv, (0,150,50), (10,255,255)) | cv2.inRange(hsv, (170,150,50), (180,255,255))
    green_mask = cv2.inRange(hsv, (40,50,50), (80,255,255))
    white_mask = cv2.inRange(hsv, (0,0,220), (179,15,255))
    black_mask = cv2.inRange(hsv, (0,0,0), (180,255,30))
    
    # 合并排除区域并取反
    combined_mask = red_mask | green_mask | white_mask | black_mask
    other_mask = cv2.bitwise_not(combined_mask)
    
    # 修改形态学处理为保持区域独立
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    processed = cv2.morphologyEx(other_mask, cv2.MORPH_OPEN, kernel)  # 改为开运算去除噪点
    
    # 强化区域分离
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result = img.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue

        # 获取初始多边形逼近
        epsilon = 0.011 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # 强制直角化处理
        rect_points = []
        for i in range(len(approx)):
            # 连接相邻顶点生成线段
            x1, y1 = approx[i-1][0]
            x2, y2 = approx[i][0]
            
            # 计算线段走向
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            
            # 强制水平或垂直
            if dx > dy:  # 水平优先
                rect_points.append((x1, y1))
                rect_points.append((x2, y1))  # 保持Y坐标一致
            else:  # 垂直优先
                rect_points.append((x1, y1))
                rect_points.append((x1, y2))  # 保持X坐标一致

        # 去除重复点并重构多边形
        rect_points = list(dict.fromkeys(rect_points))
        if len(rect_points) >= 4:
            rect_approx = np.array(rect_points, dtype=np.int32).reshape(-1,1,2)
            cv2.drawContours(result, [rect_approx], -1, (255, 0, 255), 2)
    
    # 添加返回语句
    return result     
def detect_walls(img):
    """墙体检测专用函数（简化版）"""
    # 第一步：转换红绿区域为白色
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, (0,150,50), (10,255,255)) | cv2.inRange(hsv, (170,150,50), (180,255,255))
    green_mask = cv2.inRange(hsv, (40,100,100), (70,255,255))
    converted = img.copy()
    converted[red_mask == 255] = (255, 255, 255)
    converted[green_mask == 255] = (255, 255, 255)

    # 第二步：仅对白色区域进行定向膨胀
    white_mask = cv2.inRange(converted, (200, 200, 200), (255, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilated_white = cv2.dilate(white_mask, kernel, iterations=2)
    
    # 应用膨胀后的白色区域并直接返回
    dilated = converted.copy()
    dilated[dilated_white == 255] = (255, 255, 255)
    
    

    
    return dilated
    

def smart_point_sampling(img):
    """右半区智能点采样（完整迭代版本）"""
    # 初始化参数
    height, width = img.shape[:2]
    right_half = img[:, width//2:]
    
    # 将红绿色直接转为白色（新增）
    right_half_processed = right_half.copy()
    # 红色检测（BGR格式）
    red_mask = cv2.inRange(right_half_processed, (0,0,200), (50,50,255))
    # 绿色检测（BGR格式）
    green_mask = cv2.inRange(right_half_processed, (0,200,0), (50,255,50))
    # 替换为白色
    right_half_processed[red_mask == 255] = (255, 255, 255)
    right_half_processed[green_mask == 255] = (255, 255, 255)
    
    # 生成白色掩膜（使用处理后的图像）
    hsv = cv2.cvtColor(right_half_processed, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0,0,220), (179,15,255))
    # 步骤1：初始采样
    y_coords, x_coords = np.where(white_mask == 255)
    sample_count = min(len(x_coords), 400)
    indices = np.random.choice(len(x_coords), sample_count, replace=False)
    base_points = [(int(x + width//2), int(y)) for x, y in zip(x_coords[indices], y_coords[indices])]

    # 迭代优化参数
    optimized_points = base_points.copy()
    for _ in range(3):  # 进行3次迭代优化
        # 步骤2：位移优化
        temp_points = []
        for (x, y) in optimized_points:
            rel_x = x - width//2  # 转换为右半区坐标
            # 水平扫描找边界
            left = right = rel_x
            while left > 0 and white_mask[y, left-1] == 255: left -= 1
            while right < white_mask.shape[1]-1 and white_mask[y, right+1] == 255: right += 1
            # 垂直扫描找边界
            top = bottom = y
            while top > 0 and white_mask[top-1, rel_x] == 255: top -= 1
            while bottom < white_mask.shape[0]-1 and white_mask[bottom+1, rel_x] == 255: bottom += 1
            # 计算新坐标
            new_x = (left + right) // 2 + width//2
            new_y = (top + bottom) // 2
            temp_points.append((new_x, new_y))
        
        # 步骤3：异常点过滤（移动距离>50像素 且 必须位于白色区域）
        filtered_points = []
        for (new, old) in zip(temp_points, optimized_points):
            rel_x = new[0] - width//2  # 转换为右半区坐标
            # 检查是否越界且在白色区域
            in_white = (0 <= rel_x < white_mask.shape[1] and 
                    0 <= new[1] < white_mask.shape[0] and 
                    white_mask[new[1], rel_x] == 255)
            if np.sqrt((new[0]-old[0])**2 + (new[1]-old[1])**2) < 50 and in_white:
                filtered_points.append(new)
        # 步骤4：网格聚类融合（替换原DBSCAN）
        grid_size = 15  # 合并网格尺寸
        grid_dict = defaultdict(list)
        
        # 将点分配到网格
        for point in filtered_points:
            grid_x = point[0] // grid_size
            grid_y = point[1] // grid_size
            grid_dict[(grid_x, grid_y)].append(point)
        
        # 生成合并后的点
        optimized_points = [
            (int(np.mean([p[0] for p in pts])), 
             int(np.mean([p[1] for p in pts])))
            for pts in grid_dict.values() if len(pts) >= 1
        ]

    # 最终可视化
    result = img.copy()
    # 绘制优化点（绿色）
    for (x, y) in optimized_points:
        cv2.circle(result, (x, y), 3, (0,255,0), -1)
    
    return result, base_points, optimized_points
    
def draw_white_rectangles(img, points, white_mask):
    """以采样点为中心绘制最长白色区域矩形框"""
    result = img.copy()
    width = img.shape[1]
    right_half_width = white_mask.shape[1]
    rectangles = []
    # 新增：创建包含红绿色的扩展掩膜
    right_half = img[:, width - right_half_width:]
    hsv = cv2.cvtColor(right_half, cv2.COLOR_BGR2HSV)
    # 红色检测
    red_mask = cv2.inRange(hsv, (0,150,50), (10,255,255)) | cv2.inRange(hsv, (170,150,50), (180,255,255))
    # 绿色检测
    green_mask = cv2.inRange(hsv, (40,100,100), (70,255,255))
    # 合并原始白膜与红绿掩膜
    expanded_mask = cv2.bitwise_or(white_mask, cv2.bitwise_or(red_mask, green_mask))

    for (x, y) in points:
        # 转换为右半区坐标
        rel_x = x - (width - right_half_width)
        
        # 使用扩展后的掩膜进行边界扫描
        left = right = rel_x
        while left > 0 and expanded_mask[y, left-1] == 255: left -= 1
        while right < right_half_width-1 and expanded_mask[y, right+1] == 255: right += 1
        
        top = bottom = y
        while top > 0 and expanded_mask[top-1, rel_x] == 255: top -= 1
        while bottom < expanded_mask.shape[0]-1 and expanded_mask[bottom+1, rel_x] == 255: bottom += 1
        
        abs_left = left + (width - right_half_width)
        abs_right = right + (width - right_half_width)
        
        rect = (abs_left, top, abs_right - abs_left, bottom - top)
        rectangles.append(rect)
        cv2.rectangle(result, 
                     (abs_left, top), 
                     (abs_right, bottom), 
                     (0, 165, 255), 2)
    # 新增：矩形框去重逻辑
    def filter_duplicate_rects(rects, threshold=0.8):
        """基于交并比(IOU)的矩形框去重"""
        final_rects = []
        for rect in rects:
            x, y, w, h = rect
            current_rect = (x, y, x+w, y+h)
            
            # 检查是否与已保留的矩形重叠
            keep = True
            for kept_rect in final_rects:
                kx, ky, kw, kh = kept_rect
                kept_rect_coords = (kx, ky, kx+kw, ky+kh)
                
                # 计算交并比
                xi1 = max(current_rect[0], kept_rect_coords[0])
                yi1 = max(current_rect[1], kept_rect_coords[1])
                xi2 = min(current_rect[2], kept_rect_coords[2])
                yi2 = min(current_rect[3], kept_rect_coords[3])
                
                if xi2 <= xi1 or yi2 <= yi1:
                    continue
                
                intersection = (xi2 - xi1) * (yi2 - yi1)
                area_current = (current_rect[2]-current_rect[0])*(current_rect[3]-current_rect[1])
                area_kept = (kept_rect_coords[2]-kept_rect_coords[0])*(kept_rect_coords[3]-kept_rect_coords[1])
                union = area_current + area_kept - intersection
                
                if intersection / union > threshold:
                    keep = False
                    break
                    
            if keep:
                final_rects.append(rect)
        return final_rects

    # 在返回前添加过滤
    filtered_rects = filter_duplicate_rects(rectangles)
    
    return result, filtered_rects
def draw_floorplan_rectangles(img, points, white_mask):
    """以采样点为中心绘制最长白色区域矩形框"""
    result = img.copy()
    width = img.shape[1]
    right_half_width = white_mask.shape[1]
    
    # 新增：创建包含红绿色的扩展掩膜
    right_half = img[:, width - right_half_width:]
    hsv = cv2.cvtColor(right_half, cv2.COLOR_BGR2HSV)
    # 红色检测
    red_mask = cv2.inRange(hsv, (0,150,50), (10,255,255)) | cv2.inRange(hsv, (170,150,50), (180,255,255))
    # 绿色检测
    green_mask = cv2.inRange(hsv, (40,100,100), (70,255,255))
    # 合并原始白膜与红绿掩膜
    expanded_mask = cv2.bitwise_or(white_mask, cv2.bitwise_or(red_mask, green_mask))
    
    for (x, y) in points:
        # 转换为右半区坐标
        rel_x = x - (width - right_half_width)
        
        # 使用扩展后的掩膜进行边界扫描
        left = right = rel_x
        while left > 0 and expanded_mask[y, left-1] == 255: left -= 1
        while right < right_half_width-1 and expanded_mask[y, right+1] == 255: right += 1
        
        top = bottom = y
        while top > 0 and expanded_mask[top-1, rel_x] == 255: top -= 1
        while bottom < expanded_mask.shape[0]-1 and expanded_mask[bottom+1, rel_x] == 255: bottom += 1
        
        abs_left = left + (width - right_half_width)
        abs_right = right + (width - right_half_width)
        
        cv2.rectangle(result, 
                     (abs_left, top), 
                     (abs_right, bottom), 
                     (0, 165, 255), 2)

    return result
if __name__ == "__main__":
    # 示例用法
    result=cv2.imread("E:/github/autolabel/example/input/combined_000026_floorplan.png")
    #result, windows, doors= annotate_color_blocks(result)#(x,y,w,h)
    
    #print(windows)
    #result=detect_walls(result)

    # 获取采样点和白膜
    result_img, base_points, opt_points = smart_point_sampling(result)
    right_half = result[:, result.shape[1]//2:]
    hsv = cv2.cvtColor(right_half, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0,0,220), (179,15,255))
    
    # 绘制严格在白色区域内的十字线
    result,walls = draw_white_rectangles(result_img, opt_points, white_mask)
    print(walls)
    cv2.imwrite("E:/github/autolabel/example/result.jpg", result)
