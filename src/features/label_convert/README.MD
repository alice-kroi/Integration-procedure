在label_convert中，数据转换功能全部写于data_loaders.dataload.py中，可以自动检测voc，coco，yolo，labelme格式数据
我应该考虑后续用AI识别
参考使用范例
from dataload import DataLoader
if __name__ == "__main__":
    loader = DataLoader("path/to/dataset")
    dataset = loader.load()
    print(f"成功加载 {len(dataset['images'])} 张图像")
    # 导出为指定格式（支持任意选择）
    loader.export(dataset, "output/yolo", "YOLO")  # 导出为YOLO格式
    loader.export(dataset, "output/voc", "VOC")    # 导出为VOC格式

或者使用可视化方法
需要streamlit
streamlit run src/features/label_convert/main_paign.py