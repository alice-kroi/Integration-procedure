from typing import Dict, List
import requests
from langchain.text_splitter import MarkdownHeaderTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import os
from typing import Dict
from pathlib import Path

class BuildError(Exception):
    """自定义构建异常"""
    pass
class MCPCompiler:
    def __init__(self, mcp_config: dict = None):
        self.mcp_config = mcp_config or {
            "protocol_version": "2.1",
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size": 1000,
            "model_cache_dir": "./models"  # 新增缓存目录配置
        }
        # 创建模型存储目录
        Path(self.mcp_config["model_cache_dir"]).mkdir(exist_ok=True)
         # 加载语义模型时指定缓存路径
        self.encoder = SentenceTransformer(
            self.mcp_config["embedding_model"],
            cache_folder=self.mcp_config["model_cache_dir"]
        )
        # 初始化向量数据库
        self.chroma_client = chromadb.PersistentClient()
        self.collection = self.chroma_client.get_or_create_collection(
            "knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )
        
        # 加载语义模型
        self.encoder = SentenceTransformer(self.mcp_config["embedding_model"])
        print('模型加载完成')
        # 上下文管理
        self.context_handlers = {
            "web": self._handle_web_context,
            "local": self._handle_local_context
        }

    def build_knowledge(self, sources: List[str]) -> Dict:
        """构建知识库的核心流程"""
        documents = []
        
        # 多源数据采集
        for source in sources:
            if source.startswith("http"):
                documents.extend(self._crawl_website(source))
            else:
                documents.extend(self._read_local_file(source))
        
        # 知识增强处理
        processed = self._process_documents(documents)
        
        # 建立多级索引
        self._build_vector_index(processed)
        self._build_entity_graph(processed)
        
        return {
            "status": "success",
            "doc_count": len(documents),
            "index_size": len(processed)
        }

    def query(self, question: str, context_type: str = "web") -> List[Dict]:
        """混合检索接口"""
        # 上下文感知处理
        context = self.context_handlers[context_type](question)
        
        # 向量检索
        vector_results = self._vector_search(question)
        
        # 语义增强
        expanded_queries = self._query_expansion(question)
        
        return self._rerank_results(vector_results + expanded_queries, context)

    def _process_documents(self, documents: List[str]) -> List[Dict]:
        """文档处理流水线"""
        splitter = MarkdownHeaderTextSplitter()
        return [{
            "content": chunk,
            "embedding": self.encoder.encode(chunk),
            "metadata": {"source": doc.metadata["source"]}
        } for doc in documents for chunk in splitter.split_text(doc)]

    def _build_vector_index(self, documents: List[Dict]):
        """建立向量索引"""
        self.collection.upsert(
            ids=[str(i) for i in range(len(documents))],
            embeddings=[doc["embedding"] for doc in documents],
            documents=[doc["content"] for doc in documents]
        )

    def _vector_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """向量相似度搜索"""
        query_embedding = self.encoder.encode(query).tolist()
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
    def _handle_web_context(self, query: str) -> Dict:
        """网络上下文处理"""
        related_sites = self._find_related_resources(query)
        return {
            "search_engines": related_sites,
            "time_window": "recent_year"
        }
    def _query_expansion(self, query: str) -> List[str]:
        """使用LLM扩展查询"""
        return self.encoder.encode([
            query,
            f"{query} 详细解释",
            f"{query} 实现步骤"
        ])
    def _rerank_results(self, results: List[Dict], context: Dict) -> List[Dict]:
        """结合语义和上下文重排序"""
        return sorted(
            results,
            key=lambda x: self._score_result(x, context),
            reverse=True
        )
    def generate_response(self, query: str):
        compiler = MCPCompiler()
        knowledge = compiler.query(query)
        return self.llm.generate(
            prompt=query,
            knowledge_graph=knowledge
        )
    def _handle_local_context(self, query: str) -> Dict:
        """本地上下文处理"""
        return {
            "local_files": ["./knowledge_base"],
            "cache_ttl": 3600
        }
    def _crawl_website(self, url: str) -> List[str]:
        """网页内容抓取基础实现"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return [response.text]
        except requests.RequestException as e:
            raise BuildError(f"抓取失败: {str(e)}")

    def _read_local_file(self, path: str) -> List[str]:
        """本地文件读取"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return [f.read()]
        except IOError as e:
            raise FileNotFoundError(f"文件读取失败: {str(e)}")

    def _build_entity_graph(self, documents: List[Dict]):
        """实体图谱构建占位实现"""
        # TODO: 实现完整的实体关系分析
        pass

    def _score_result(self, result: Dict, context: Dict) -> float:
        """简单评分占位实现"""
        return len(result['content']) * 0.1  # 基础评分逻辑
def main():
    # 初始化编译器
    compiler = MCPCompiler()
    
    # 构建知识库
    sources = [
        "https://github.com/yzfly/Awesome-MCP-ZH/blob/main/README.md",
        "local_knowledge.txt"
    ]
    print('知识库构建中...')
    build_result = compiler.build_knowledge(sources)
    print(f"知识库构建完成，收录文档：{build_result['doc_count']}篇")
    
    # 执行查询
    questions = [
        "MCP协议的核心功能是什么？",
        "如何实现上下文感知检索？"
    ]
    
    for q in questions:
        results = compiler.query(q)
        print(f"\n问题：{q}")
        for i, res in enumerate(results[:3], 1):
            print(f"结果{i}: {res['content'][:100]}...")

if __name__ == "__main__":
    import unittest
    
    class TestMCPCompiler(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.compiler = MCPCompiler()
            cls.compiler.build_knowledge([
                "sample_docs/mcp_api.md",
                "https://raw.githubusercontent.com/mcp-docs/examples/main/basic_usage.md"
            ])
        
        def test_basic_query(self):
            results = self.compiler.query("协议版本号")
            self.assertGreaterEqual(len(results), 1)
            self.assertIn("2.1", results[0]['content'])
        
        def test_context_search(self):
            web_results = self.compiler.query("实时数据同步", "web")
            local_results = self.compiler.query("本地缓存配置", "local")
            self.assertNotEqual(web_results[0]['content'], local_results[0]['content'])
        
        def test_edge_cases(self):
            # 测试空查询
            with self.assertRaises(ValueError):
                self.compiler.query("")
            
            # 测试无效数据源
            with self.assertRaises(FileNotFoundError):
                self.compiler.build_knowledge(["invalid_source.xyz"])
    
    # 运行演示和测试
    print("=== 知识管理系统演示 ===")
    main()
    print("\n=== 执行单元测试 ===")
    unittest.main(argv=[''], exit=False)