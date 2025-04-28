class ModRetriever:
    def __init__(self):
        self.vector_db = Chroma(persist_directory="./mcp_knowledge")
        self.retriever = self.vector_db.as_retriever(
            search_kwargs={"score_threshold": 0.85}
        )
    
    def search_mcp(self, query: str) -> List[Document]:
        """检索MCP映射表和开发规范"""
        return self.retriever.get_relevant_documents(query)