import os
import shutil
import streamlit as st
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

def select_directory():
    """使用tkinter选择目录并返回路径"""
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    path = filedialog.askdirectory()
    root.destroy()  # 关闭tkinter窗口
    return path.replace('\\', '/')  # 统一路径分隔符
def get_subfolders_with_readme():
    """只获取当前脚本所在目录的直接子文件夹"""
    current_dir = Path(__file__).parent  # 获取当前脚本所在目录
    subfolders = []
    for item in current_dir.iterdir():  # 遍历当前脚本所在目录
        if item.is_dir() and item != current_dir:  # 添加目录过滤条件
            readme = item / "README.MD"
            desc = ""
            if readme.exists():
                with open(readme, 'r', encoding='utf-8') as f:
                    desc = f.read().strip()
            subfolders.append({
                "name": item.name,
                "path": str(item),
                "description": desc
            })
    return subfolders

def main():
    st.title("课程作业管理器")
    
    # 获取当前目录下的子文件夹
    subfolders = get_subfolders_with_readme()
    
    # 目录选择部分
    if st.button("📁 选择存储路径"):
        try:
            selected_path = select_directory()
            st.session_state.dest_path = selected_path
        except Exception as e:
            st.error(f"路径选择失败: {str(e)}")
    
    dest_path = st.text_input(
        "存储地址：",
        value=st.session_state.get('dest_path', 'C:/MyDocuments/'),
        key='dest_path_input'
    )

    
    
    # 显示所有子文件夹
    for idx, folder in enumerate(subfolders):
        col1, col2 = st.columns([1,4])
        with col1:
            if st.button(f"复制 {folder['name']}", key=f"btn_{idx}"):
                try:
                    shutil.copytree(folder['path'], os.path.join(dest_path, folder['name']))
                    st.success(f"成功复制 {folder['name']} 到 {dest_path}")
                except Exception as e:
                    st.error(f"复制失败: {str(e)}")
        with col2:
            st.caption(folder['description'] or "暂无项目说明")

if __name__ == "__main__":
    main()