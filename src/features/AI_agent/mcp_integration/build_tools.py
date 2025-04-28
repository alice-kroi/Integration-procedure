import os
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Dict

class BuildError(Exception):
    """自定义编译异常"""
    pass

class MCPCompiler:
    def __init__(self, project_type: str = "forge"):
        self.project_type = project_type
        self.temp_dir = Path(tempfile.mkdtemp())
        self.dependencies = {
            "forge": ["net.minecraftforge:forge:1.20.1-47.1.0"],
            "fabric": ["net.fabricmc:fabric-loader:0.14.22"]
        }

    def build(self, code: str) -> Dict:
        """执行完整构建流程"""
        try:
            # 1. 创建临时构建环境
            self._setup_build_env()
            
            # 2. 代码验证（类似label_convert中的验证逻辑）
            self._validate_code_structure(code)
            
            # 3. 生成构建文件
            self._generate_gradle_build()
            
            # 4. 写入源代码文件
            (self.temp_dir / "src/main/java/mod/ExampleMod.java").write_text(code)
            
            # 5. 执行Gradle构建
            self._run_gradle_task("build")
            
            return {
                "success": self._validate_build_result(),
                "output": self._read_build_log(),
                "artifacts": list(self._find_output_jars())
            }
            
        except subprocess.CalledProcessError as e:
            raise BuildError(f"编译失败: {e.stderr.decode()[:500]}")
        finally:
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _setup_build_env(self):
        """创建临时构建目录结构（类似测试数据生成逻辑）"""
        dirs = [
            "src/main/java/mod",
            "src/main/resources",
            "gradle"
        ]
        for d in dirs:
            (self.temp_dir / d).mkdir(parents=True, exist_ok=True)

    def _validate_code_structure(self, code: str):
        """验证代码基础结构（类似YOLO文件验证）"""
        required_elements = [
            "@Mod(",
            "public class",
            "import net.minecraft"
        ]
        for element in required_elements:
            if element not in code:
                raise BuildError(f"代码缺少必要元素: {element}")

    def _generate_gradle_build(self):
        """生成gradle.build文件"""
        build_template = f"""
        plugins {{
            id 'java'
            id 'net.minecraftforge.gradle' version '5.1.+'
        }}
        
        dependencies {{
            implementation {self.dependencies[self.project_type]!r}
        }}
        """
        (self.temp_dir / "build.gradle").write_text(build_template)

    def _run_gradle_task(self, task: str):
        """执行Gradle命令（兼容Windows环境）"""
        gradlew = "gradlew.bat" if os.name == 'nt' else "./gradlew"
        subprocess.run(
            [gradlew, task],
            cwd=self.temp_dir,
            check=True,
            capture_output=True
        )

    def _find_output_jars(self):
        """查找生成的JAR文件（类似测试数据查找）"""
        return (self.temp_dir / "build/libs").glob("*.jar")

    def _validate_build_result(self) -> bool:
        """验证构建结果（扩展原有简单检查）"""
        jars = list(self._find_output_jars())
        return len(jars) > 0 and any("mod" in jar.name for jar in jars)

    def _read_build_log(self) -> str:
        """读取详细构建日志"""
        log_file = self.temp_dir / "build/outputs/logs/build.log"
        return log_file.read_text(encoding='utf-8') if log_file.exists() else ""
    


if __name__ == "__main__":
    compiler = MCPCompiler(project_type="forge")
    try:
        result = compiler.build("""
            @Mod("examplemod")
            public class ExampleMod {
                // MOD初始化代码...
            }
        """)
        print(f"构建结果: {result['success']}")
    except BuildError as e:
        print(f"构建失败: {str(e)}")