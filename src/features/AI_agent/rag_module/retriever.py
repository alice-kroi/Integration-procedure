from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from typing import List
from langchain_core.documents import Document  # 新增导入
import logging

class ModRetriever:
    def __init__(self):
        # 初始化嵌入模型（参考您之前的YOLO数据处理方式）
        self.embedding = HuggingFaceEmbeddings(
            model_name="text-embedding-3-small",
            model_kwargs={'device': 'cuda'}
        )
        
        # 增强向量数据库配置
        self.vector_db = Chroma(
            persist_directory="./mcp_knowledge",
            embedding_function=self.embedding
        )
        
        # 优化检索参数
        self.retriever = self.vector_db.as_retriever(
            search_kwargs={
                "k": 5,  # 增加返回结果数量
                "score_threshold": 0.82,  # 调整相似度阈值
                "fetch_k": 20  # 扩大候选池
            }
        )
        
        self.logger = logging.getLogger("ModRetriever")

    def load_documents(self, file_path: str):
        """批量加载文档方法（类似您之前的测试数据加载）"""
        try:
            loader = TextLoader(file_path)
            documents = loader.load()
            
            # 文本分割（参考label_convert中的数据处理）
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=300,
                length_function=len,
                add_start_index=True
            )
            splits = text_splitter.split_documents(documents)
            
            self.vector_db.add_documents(splits)
            self.logger.info(f"成功加载文档: {file_path}")
            
        except Exception as e:
            self.logger.error(f"文档加载失败: {file_path} - {str(e)}")
            raise

    def search_mcp(self, query: str) -> List[Document]:
        """增强检索方法"""
        return self.retriever.get_relevant_documents(query)
    

if __name__ == "__main__":
    retriever = ModRetriever()
    retriever.load_documents("mcp_docs/1.20.1_mappings.txt")
    results = retriever.search_mcp("如何创建生物群系")