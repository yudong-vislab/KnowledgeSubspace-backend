import time
from typing import List, Optional, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os

class PDFRAGSystem:
    """PDF文档的检索增强生成(RAG)系统"""
    
    def __init__(self):
        """初始化RAG系统"""
        # 从环境变量中读取配置
        self.api_key = os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        # 验证API密钥是否存在
        if not self.api_key:
            print("警告: 未设置API密钥，请确保.env文件中有正确的配置")
        
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.api_key,
            base_url=self.base_url
        )
        self.llm = ChatOpenAI(
            openai_api_key=self.api_key,
            model_name="gpt-3.5-turbo",  # 可以根据需要更改为gpt-4
            temperature=0.0,
            base_url=self.base_url
        )
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # 初始化向量存储和检索器
        self.vector_store = None
        self.retriever = None
        
        # 初始化RAG链
        self.rag_chain = None
        
    def load_pdf(self, pdf_path: str) -> bool:
        """加载PDF文档并创建向量存储
        
        Args:
            pdf_path: PDF文件路径
        
        Returns:
            是否加载成功
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(pdf_path):
                print(f"错误: 文件 {pdf_path} 不存在")
                return False
            
            # 加载PDF文档
            print(f"正在加载PDF文档: {pdf_path}")
            start_time = time.time()
            
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # 分割文本
            print(f"文档加载完成，共{len(documents)}页，正在分割文本...")
            splits = self.text_splitter.split_documents(documents)
            
            # 创建向量存储
            print(f"文本分割完成，共{len(splits)}个片段，正在创建向量存储...")
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
            
            # 创建检索器
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # 创建RAG链
            self._create_rag_chain()
            
            end_time = time.time()
            print(f"PDF文档处理完成，耗时{end_time - start_time:.2f}秒")
            return True
        except Exception as e:
            print(f"加载PDF文档时出错: {str(e)}")
            return False
    
    def _create_rag_chain(self):
        """创建RAG链"""
        # 定义提示模板
        template = """
        你是一个问答助手，你的任务是基于提供的上下文信息回答用户的问题。
        请严格根据上下文信息回答，不要添加上下文之外的信息。
        如果上下文信息不足以回答问题，请如实告知用户。
        
        上下文信息:
        {context}
        
        用户问题:
        {question}
        
        回答:
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # 创建RAG链
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def ask_question(self, question: str) -> str:
        """向RAG系统提问
        
        Args:
            question: 用户问题
        
        Returns:
            系统回答
        """
        if self.rag_chain is None:
            return "请先加载PDF文档"
        
        try:
            start_time = time.time()
            print(f"正在回答问题: {question}")
            
            # 获取相关文档
            relevant_docs = self.retriever.get_relevant_documents(question)
            print(f"找到{len(relevant_docs)}个相关文档片段")
            
            # 获取回答
            answer = self.rag_chain.invoke(question)
            
            end_time = time.time()
            print(f"回答生成完成，耗时{end_time - start_time:.2f}秒")
            
            return answer
        except Exception as e:
            print(f"回答问题时出错: {str(e)}")
            return f"回答问题时出错: {str(e)}"
    
    def save_vector_store(self, save_path: str):
        """保存向量存储到本地
        
        Args:
            save_path: 保存路径
        """
        if self.vector_store is None:
            print("没有可保存的向量存储，请先加载PDF文档")
            return
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            
            # 保存向量存储
            self.vector_store.save_local(save_path)
            print(f"向量存储已保存到: {save_path}")
        except Exception as e:
            print(f"保存向量存储时出错: {str(e)}")
    
    def load_vector_store(self, load_path: str):
        """从本地加载向量存储
        
        Args:
            load_path: 加载路径
        """
        try:
            # 加载向量存储
            self.vector_store = FAISS.load_local(
                load_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            
            # 创建检索器
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # 创建RAG链
            self._create_rag_chain()
            
            print(f"向量存储已从: {load_path} 加载")
        except Exception as e:
            print(f"加载向量存储时出错: {str(e)}")

# 示例用法
if __name__ == "__main__":
    # 提示用户输入API密钥和PDF路径
    print("欢迎使用PDF RAG系统！")
    # api_key = input("请输入您的OpenAI API密钥: ")
    # pdf_path = input("请输入PDF文件路径: ")
    pdf_path = "NeuroSync.pdf"
    # 初始化RAG系统
    try:
        rag_system = PDFRAGSystem()
        
        # 加载PDF文档
        if rag_system.load_pdf(pdf_path):
            # 保存向量存储（可选）
            save_vectors = input("是否保存向量存储？(y/n): ")
            if save_vectors.lower() == "y":
                vector_path = input("请输入向量存储保存路径: ")
                rag_system.save_vector_store(vector_path)
            
            # 进入问答循环
            print("PDF文档已加载完成，可以开始提问了！输入'退出'结束程序。")
            while True:
                question = input("请输入问题: ")
                if question.lower() == "退出":
                    break
                
                answer = rag_system.ask_question(question)
                print(f"\n回答: {answer}\n")
    except Exception as e:
        print(f"程序运行出错: {str(e)}")