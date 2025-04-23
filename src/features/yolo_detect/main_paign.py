import streamlit as st
import matplotlib.pyplot as plt
from models.model_load import load_yolo_model
from data_loaders.dataload import create_dataloader
from postprocessing import plot_results
import yaml

# 界面配置
st.set_page_config(layout="wide")

def main():
    st.title("YOLO模型管理平台")
    
    # 侧边栏控制面板
    with st.sidebar:
        st.header("控制面板")
        mode = st.radio("选择模式", ["实时检测", "模型训练"])
        model_type = st.selectbox("模型类型", ["yolov5s", "yolov5m", "yolov5l"])
    
    if mode == "实时检测":
        # 实时检测模式
        st.subheader("实时目标检测")
        model = load_yolo_model(model_type)
        
        uploaded_file = st.file_uploader("上传检测图片", type=['jpg', 'png', 'jpeg'])
        if uploaded_file:
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="原始图像", use_column_width=True)
            
            # 执行推理
            results = model(uploaded_file)
            plotted_img = plot_results(results)
            
            with col2:
                st.image(plotted_img, caption="检测结果", use_column_width=True)
                st.json({
                    "检测统计": results.pandas().xyxy[0].name.value_counts().to_dict(),
                    "置信度阈值": model.conf
                })
                
    elif mode == "模型训练":
        # 模型训练模式
        st.subheader("模型训练监控")
        config_file = st.file_uploader("上传配置文件", type=['yaml'])
        
        if config_file:
            config = yaml.safe_load(config_file)
            progress_bar = st.progress(0)
            chart_placeholder = st.empty()
            
            if st.button("开始训练"):
                train_loader = create_dataloader(config['train'], batch_size=16)
                model = load_yolo_model(model_type)
                
                # 初始化训练参数
                losses = []
                for epoch in range(100):
                    epoch_loss = 0
                    for i, (images, targets) in enumerate(train_loader):
                        # 训练逻辑...
                        loss = model(images, targets).sum()
                        epoch_loss += loss.item()
                        
                        # 更新进度
                        progress = (i + 1) / len(train_loader)
                        progress_bar.progress(progress)
                    
                    # 绘制损失曲线
                    losses.append(epoch_loss/len(train_loader))
                    fig, ax = plt.subplots()
                    ax.plot(losses)
                    chart_placeholder.pyplot(fig)

if __name__ == "__main__":
    main()