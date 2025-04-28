from langchain.agents import AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from rag_module.retriever import ModRetriever
from mcp_integration.build_tools import MCPCompiler
import os

class ModAssistant:
    def __init__(self):
        # 初始化核心组件
        self.retriever = ModRetriever()  # RAG模块
        self.compiler = MCPCompiler()    # MCP集成
        self.agent = self._init_agent()  # 智能体
        
    def _init_agent(self):
        # 组合工具集
        tools = [
            self._create_retrieval_tool(),
            self._create_build_tool(),
            self._create_decompile_tool()
        ]
        
        # 初始化Agent执行器（具体实现见agent_module）
        return AgentExecutor(...)
    
    def _create_retrieval_tool(self):
        # 创建基于MCP文档的检索工具
        return create_retriever_tool(
            self.retriever.mcp_retriever,
            "search_mcp_docs",
            "查询MCP映射表和MOD开发规范"
        )

    def generate_mod(self, user_query):
        # 执行完整工作流
        agent_response = self.agent.invoke(user_query)
        compiled_result = self.compiler.build(agent_response["code"])
        return {
            "code": agent_response["code"],
            "build_log": compiled_result
        }

if __name__ == "__main__":
    assistant = ModAssistant()
    result = assistant.generate_mod(
        "创建能在1.20.1版本生成随机地形的生物群系MOD"
    )
    print(f"生成的MOD代码：\n{result['code']}")