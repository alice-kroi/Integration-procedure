import streamlit as st
from data_loaders.dataload import DataLoader
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

def main():
    st.set_page_config(page_title="标注格式转换工具", layout="wide", page_icon=":camera:")
    
    # 初始化会话状态
    if 'dataset_stats' not in st.session_state:
        st.session_state.dataset_stats = None
    
    # 侧边栏导航
    st.sidebar.title("导航菜单")
    st.sidebar.header("功能选择")
    page = st.sidebar.radio("", ["格式转换", "数据集分析", "历史记录", "生成测试数据"])  # 新增测试数据生成选项

    if page == "格式转换":
        show_conversion_interface()
    elif page == "数据集分析":
        show_dataset_analysis()
    elif page == "历史记录":
        show_conversion_history()
    elif page == "生成测试数据":  # 新增测试数据生成页面
        show_testdata_generator()

def show_testdata_generator():
    """显示测试数据生成界面"""
    st.title("🧪 测试数据生成工具")
    
    with st.form("test_data_config"):
        col1, col2 = st.columns(2)
        
        with col1:
            num_images = st.number_input("生成图像数量", 1, 1000, 50)
            img_size = st.selectbox("图像尺寸", ["640x480", "800x600", "1024x768"], index=0)
            output_dir = st.text_input("输出目录", "./test_data")
            
        with col2:
            class_names = st.text_area("类别列表", "cat\ndog\ncar", help="每行一个类别名称")
            target_format = st.selectbox("目标格式", ["YOLO", "COCO", "VOC", "Labelme"])
        
        # 高级选项
        with st.expander("高级设置"):
            seed = st.number_input("随机种子", value=42)
            clean_output = st.checkbox("清空输出目录", value=True)
        
        if st.form_submit_button("开始生成", type="primary"):
            try:
                generate_test_data(
                    num_images=num_images,
                    img_size=tuple(map(int, img_size.split('x'))),
                    class_names=[n.strip() for n in class_names.split('\n') if n.strip()],
                    output_dir=output_dir,
                    format=target_format.upper(),
                    seed=seed,
                    clean_output=clean_output
                )
                st.success(f"✅ 成功生成 {num_images} 张测试数据到目录: `{output_dir}`")
                st.balloons()
            except Exception as e:
                st.error(f"生成失败: {str(e)}")

def generate_test_data(num_images, img_size, class_names, output_dir, format, seed, clean_output):
    """调用测试数据生成器"""
    from features.label_convert.test_data.test_data_generater import process_dataset
    from pathlib import Path
    import shutil
    import tempfile
    
    # 创建临时图像目录
    with tempfile.TemporaryDirectory() as tmp_img_dir:
        # 生成虚拟图像文件
        for i in range(num_images):
            (Path(tmp_img_dir) / f"image_{i:04d}.jpg").touch()
        
        # 清空输出目录
        if clean_output and Path(output_dir).exists():
            shutil.rmtree(output_dir)
        
        # 调用生成器
        process_dataset(
            src_dir=tmp_img_dir,
            dest_dir=output_dir,
            class_names=class_names,
            format=format
        )

def show_conversion_interface():
    """显示格式转换主界面"""
    st.title("📁 智能标注格式转换工具")
    
    with st.expander("🚀 快速开始", expanded=True):
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # 修改点2：添加默认值处理
            default_path = st.session_state.get("selected_input_dir", "")
            input_dir = st.text_input(
                "输入数据集路径",
                value=default_path,
                help="示例: E:/datasets/coco_dataset"
            )
            # 添加目录选择组件
            if st.button("📁 浏览文件夹"):
                import threading
                from queue import Queue
                from tkinter import Tk, filedialog
                
                # 使用队列传递结果
                result_queue = Queue()
                
                def file_picker(q):
                    try:
                        root = Tk()
                        root.attributes('-topmost', True)
                        root.withdraw()
                        folder = filedialog.askdirectory(parent=root)
                        q.put(folder.replace("\\", "/") if folder else None)
                        root.destroy()
                    except Exception as e:
                        q.put(None)

                thread = threading.Thread(target=file_picker, args=(result_queue,))
                thread.start()
                thread.join()
                
                # 在主线程更新会话状态
                selected_dir = result_queue.get()
                if selected_dir:
                    st.session_state.selected_input_dir = selected_dir
                    st.rerun()

            
            
            if st.button("检测数据格式"):
                detect_and_show_format(input_dir)
            
        with col2:
            target_format = st.selectbox("目标格式", 
                                       ["COCO", "YOLO", "VOC", "Labelme"],
                                       index=1,
                                       help="选择需要转换的目标格式")

    # 高级设置面板
    with st.expander("⚙️ 高级设置", expanded=False):
        # 添加输出路径浏览功能
        output_col1, output_col2 = st.columns([4, 1])
        with output_col1:
            output_dir = st.text_input(
                "输出路径", 
                st.session_state.get("selected_output_dir", "./converted_data"),
                key="output_path"
            )
        with output_col2:
            if st.button("📁 浏览输出路径", use_container_width=True):
                import threading
                from queue import Queue
                from tkinter import Tk, filedialog
                
                result_queue = Queue()
                def output_picker(q):
                    try:
                        root = Tk()
                        root.attributes('-topmost', True)
                        root.withdraw()
                        folder = filedialog.askdirectory(parent=root)
                        q.put(folder.replace("\\", "/") if folder else None)
                        root.destroy()
                    except: 
                        q.put(None)

                thread = threading.Thread(target=output_picker, args=(result_queue,))
                thread.start()
                thread.join()
                
                selected_dir = result_queue.get()
                if selected_dir:
                    st.session_state.selected_output_dir = selected_dir
                    st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            clean_output = st.checkbox("清空输出目录", value=True)
        with col2:
            show_preview = st.checkbox("转换前预览", value=True)

    # 操作按钮区
    st.divider()
    if st.button("✨ 开始智能转换", type="primary", use_container_width=True):
        handle_conversion(input_dir, target_format, output_dir, clean_output, show_preview)

def detect_and_show_format(path):
    """检测并显示数据格式"""
    try:
        loader = DataLoader(path)
        detected_format = loader.detect_format()
        st.success(f"检测成功！当前数据格式为: **{detected_format}**")
        st.session_state.current_format = detected_format
    except Exception as e:
        st.error(f"格式检测失败: {str(e)}")

def handle_conversion(input_dir, target_format, output_dir, clean_output, show_preview):
    """处理转换流程"""
    try:
        loader = DataLoader(input_dir)
        
        with st.spinner("🔄 正在加载数据集..."):
            dataset = loader.load()
            st.session_state.dataset_stats = {
                'total_images': len(dataset['images']),
                'categories': dataset['categories'],
                'annotations_count': sum(len(img['annotations']) for img in dataset['images'])
            }

        if show_preview:
            show_data_preview(dataset)

        with st.spinner("🚀 正在转换格式..."):
            if clean_output and Path(output_dir).exists():
                shutil.rmtree(output_dir)
            loader.export(dataset, output_dir, target_format)

        st.success(f"✅ 转换完成！输出目录: `{output_dir}`")
        show_conversion_summary(dataset, target_format)
        st.balloons()
        
    except Exception as e:
        st.error(f"❌ 转换失败: {str(e)}")
        st.stop()

def show_data_preview(dataset):
    """显示数据集预览"""
    st.subheader("📊 数据集预览")
    
    # 统计信息
    col1, col2, col3 = st.columns(3)
    col1.metric("图像数量", len(dataset['images']))
    col2.metric("标注总数", sum(len(img['annotations']) for img in dataset['images']))
    col3.metric("类别数量", len(dataset['categories']))
    
    # 类别分布图表
    st.write("### 类别分布")
    category_counts = {}
    for img in dataset['images']:
        for ann in img['annotations']:
            cat = ann['category_name']
            category_counts[cat] = category_counts.get(cat, 0) + 1
            
    fig, ax = plt.subplots()
    ax.barh(list(category_counts.keys()), list(category_counts.values()))
    plt.xlabel("出现次数")
    st.pyplot(fig)
    
    # 图像标注预览
    st.write("### 标注示例（随机采样）")
    sample_images = np.random.choice(dataset['images'], size=3, replace=False)
    for img in sample_images:
        with st.expander(f"图像: {Path(img['file_name']).name}"):
            col1, col2 = st.columns([2, 3])
            with col1:
                st.write("**基本信息**")
                st.json({
                    "尺寸": f"{img['width']}x{img['height']}",
                    "标注数量": len(img['annotations']),
                    "主要类别": max([(ann['category_name'], ann['category_id']) for ann in img['annotations']], 
                                  key=lambda x: x[1])[0]
                })
            with col2:
                try:
                    image = Image.open(img['file_name'])
                    draw = ImageDraw.Draw(image)
                    for ann in img['annotations']:
                        bbox = ann['bbox']
                        draw.rectangle([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]], 
                                     outline="red", width=3)
                    st.image(image, caption="标注预览", use_column_width=True)
                except Exception as e:
                    st.warning(f"无法加载图像: {str(e)}")

def show_conversion_summary(dataset, target_format):
    """显示转换摘要"""
    st.subheader("📝 转换摘要")
    st.write(f"目标格式配置要求：")
    
    format_requirements = {
        "COCO": "需要生成annotations.json文件，包含完整的类别映射",
        "YOLO": "自动生成labels目录和classes.txt类别文件",
        "VOC": "创建Annotations目录存放XML文件，JPEGImages存放图像",
        "Labelme": "为每张图像生成对应的JSON标注文件"
    }
    
    st.info(f"**{target_format}格式**：{format_requirements.get(target_format, '')}")
    st.write("前5个转换后的标注示例：")
    st.json(dataset['images'][0]['annotations'][:2])

def show_dataset_analysis():
    """显示数据集分析页面"""
    st.title("📊 数据集深度分析")
    # 实现分析功能...

def show_conversion_history():
    """显示历史记录页面"""
    st.title("📜 转换历史追溯")
    # 实现历史记录功能...

if __name__ == "__main__":
    main()