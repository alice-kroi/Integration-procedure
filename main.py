from flask import Flask, send_file
import threading
import subprocess

app = Flask(__name__)

# 主界面路由
@app.route('/')
def index():
    return send_file('index.html')

# 功能模块路由
@app.route('/label_convert')
def label_convert():
    """启动格式转换功能"""
    subprocess.Popen([
        'streamlit', 'run', 
        'src/features/label_convert/main_paign.py',
        '--server.port', '8502'
    ])
    return '<script>window.location.href="http://localhost:8502";</script>'

@app.route('/yolo_detect')
def yolo_detect():
    """启动YOLO检测功能"""
    subprocess.Popen([
        'streamlit', 'run', 
        'src/features/yolo_detect/main_paign.py',
        '--server.port', '8501'
    ])
    return '<script>window.location.href="http://localhost:8501";</script>'

if __name__ == '__main__':
    # 启动Flask主服务
    threading.Thread(
        target=app.run,
        kwargs={'port': 8000, 'use_reloader': False}
    ).start()
    
    # 自动打开浏览器
    import webbrowser
    webbrowser.open('http://localhost:8000')