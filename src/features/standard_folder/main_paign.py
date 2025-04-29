import streamlit as st
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# 在main函数开头添加CSS样式
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* 固定定位导航栏 */
    .m-top {
        position: fixed;
        top: 0;
        width: 100%;
        height: 70px;
        background: #242424;
        z-index: 1000;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    
    /* 导航容器布局 */
    .wrap {
        width: 1100px;
        margin: 0 auto;
        display: flex;
        align-items: center;
        height: 100%;
    }
    
    /* LOGO样式 */
    .logo a {
        font-size: 24px;
        color: #fff !important;
        font-weight: bold;
        text-decoration: none;
        margin-right: 50px;
    }
    
    /* 导航项样式 */
    .m-nav {
        display: flex;
        height: 100%;
    }
    .m-nav li {
        padding: 0 20px;
        height: 100%;
        display: flex;
        align-items: center;
        position: relative;
    }
    .m-nav a {
        color: #ccc !important;
        font-size: 16px;
        text-decoration: none;
        position: relative;
    }
    .m-nav a:hover em {
        color: #fff;
    }
    .m-nav .z-slt em {
        color: #fff;
    }
    .m-nav .cor {
        position: absolute;
        bottom: -8px;
        left: 50%;
        transform: translateX(-50%);
        width: 24px;
        height: 2px;
        background: #c62f2f;
    }
</style>
""", unsafe_allow_html=True)

def main():

    # 初始化多级导航状态
    if 'top_nav' not in st.session_state:
        st.session_state.top_nav = "data"
    if 'left_nav' not in st.session_state:
        st.session_state.left_nav = "realtime"

    # 顶部导航栏
    

    # 添加导航点击处理
    st.markdown("""
    <style>
        /* 隐藏默认按钮样式 */
        .stButton > button {
            background: none !important;
            border: none !important;
            box-shadow: none !important;
            border-radius: 0 !important;
            width: 100%;
            margin: 0;
            padding: 0.5rem 1rem;
        }
        /* 激活状态样式 */
        .stButton > .z-slt {
            background: rgba(198,47,47,0.1) !important;
            border-bottom: 2px solid #c62f2f !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # 紧凑布局
    cols = st.columns([1,1,1,0.1])  # 最后一个参数设为0.1消除空白
    with cols[0]:
        clicked_data = st.button("数据看板", key="nav_data")
    with cols[1]:
        clicked_config = st.button("系统配置", key="nav_config")
    with cols[2]:
        clicked_dev = st.button("开发工具", key="nav_dev")
    
    # 处理点击事件
    if clicked_data: st.session_state.top_nav = "data"
    if clicked_config: st.session_state.top_nav = "config"
    if clicked_dev: st.session_state.top_nav = "dev"

    # 添加顶部padding防止内容被遮挡
    st.markdown("<div style='padding-top: 80px;'></div>", unsafe_allow_html=True)

    # 左侧子导航栏
    with st.sidebar:
        if st.session_state.top_nav == "data":
            st.session_state.left_nav = st.radio(
                "数据模块",
                ["realtime", "history", "analysis"],
                format_func=lambda x: {
                    "realtime": "📈 实时数据",
                    "history": "🕰️ 历史趋势",
                    "analysis": "🔍 深度分析"
                }[x]
            )
        elif st.session_state.top_nav == "config":
            st.session_state.left_nav = st.radio(
                "配置模块",
                ["base", "security", "advanced"],
                format_func=lambda x: {
                    "base": "⚙️ 基础设置",
                    "security": "🔒 安全配置", 
                    "advanced": "💡 高级参数"
                }[x]
            )

    # 主内容路由
    if st.session_state.top_nav == "data":
        show_data_page()
    elif st.session_state.top_nav == "config":
        show_config_page()

def show_home():
    """显示主页面内容"""
    st.title("欢迎页面")
    
    with st.expander("📌 系统状态概览", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("在线设备", "8台", "+2")
        with col2:
            st.metric("任务队列", "3项", "-1")
        with col3:
            st.metric("存储空间", "64%", "8GB")
            
    st.divider()
    st.write("当前系统运行状态监测...")

def show_dashboard():
    """显示数据看板"""
    st.title("数据可视化面板")
    
    with st.form("data_filter"):
        col1, col2 = st.columns([3, 1])
        with col1:
            time_range = st.select_slider(
                "时间范围",
                options=["最近1小时", "最近24小时", "本周", "本月"]
            )
        with col2:
            if st.form_submit_button("应用筛选"):
                st.rerun()
    
    # 实时数据图表
    with st.expander("📈 实时数据流", expanded=True):
        t = np.linspace(0, 10, 100)
        fig, ax = plt.subplots()
        ax.plot(t, np.sin(t))
        st.pyplot(fig)

def show_config():
    """显示系统配置"""
    st.title("系统参数设置")
    
    with st.form("system_config"):
        st.subheader("基础设置")
        api_key = st.text_input("API密钥", type="password")
        cache_dir = st.text_input("缓存目录", "./cache")
        
        st.subheader("高级设置")
        col1, col2 = st.columns(2)
        with col1:
            log_level = st.selectbox("日志级别", ["DEBUG", "INFO", "WARNING"])
        with col2:
            auto_clean = st.checkbox("自动清理旧缓存", value=True)
        
        if st.form_submit_button("保存配置", type="primary"):
            st.success("配置已更新")
def show_data_page():
    """显示数据相关页面"""
    if st.session_state.left_nav == "realtime":
        st.title("实时数据看板")
        # ...原有实时数据代码...
    elif st.session_state.left_nav == "history":
        st.title("历史趋势分析")
        # ...新增历史分析代码...

def show_config_page():
    """显示配置相关页面"""
    if st.session_state.left_nav == "base":
        st.title("基础参数设置")
        # ...原有配置代码...
    elif st.session_state.left_nav == "security":
        st.title("安全认证配置")
        # ...新增安全配置代码...

        
if __name__ == "__main__":
    main()