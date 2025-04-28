class MCPCompiler:
    def build(self, code: str) -> dict:
        """编译MOD并返回日志"""
        # 实现代码生成->编译->测试的流水线
        return {
            "success": os.path.exists("./build/libs/mod.jar"),
            "output": self._read_build_log()
        }