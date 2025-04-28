mod_assistant/
├── main.py              # 主程序入口
├── rag_module/          # 检索增强模块
│   ├── knowledge_base/  # MOD制作规范/API文档
│   ├── vector_db/       # Chroma向量数据库
│   └── retriever.py     # 文档检索器
├── agent_module/        # 智能体模块
│   ├── dialog_manager.py # 对话管理
│   └── toolkits/        # 工具集
├── mcp_integration/     # MCP集成
│   ├── decompiler.py    # 反编译处理
│   └── build_tools.py   # 构建工具
└── templates/           # MOD模板文件