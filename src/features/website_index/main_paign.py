import yaml
import streamlit as st
from pathlib import Path

def read_website_config():
    """读取当前目录下的website_index.yaml配置文件"""
    config_path = Path(__file__).parent / "website_index.yaml"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error("配置文件website_index.yaml未找到")
        return []
    except yaml.YAMLError as e:
        st.error(f"YAML解析错误: {str(e)}")
        return []

def main():
    st.title("网站导航中心")
    
    # 读取配置文件
    websites = read_website_config()
    
    # 显示网站导航
    st.subheader("快速访问")
    for site in websites:
        # 创建两列布局：按钮在左，描述在右
        col1, col2 = st.columns([1, 4])
        with col1:
            # 使用新式link_button并设置样式
            st.link_button(
                f"🌐 {site['name']}", 
                url=site['url'],
                use_container_width=True
            )
        with col2:
            # 显示描述信息，支持Markdown格式
            st.markdown(f"**{site.get('description', '暂无描述')}**")
            if 'tags' in site:
                st.caption(f"标签：{', '.join(site['tags'])}")

if __name__ == "__main__":
    main()