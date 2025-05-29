import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class ShortMemoryRetriever:
    def __init__(self, agent_memory_file, model_name="/path/to/all-MiniLM-L6-v2"):
        """
        初始化短期记忆检索器
        :param agent_memory_file: 指定 Agent 的短期记忆 JSON 文件路径
        :param model_name: 预训练向量模型 (默认使用 MiniLM-L6-v2)
        """
        if not os.path.exists(agent_memory_file):
            raise FileNotFoundError(f"⚠️ 文件 {agent_memory_file} 不存在！请检查路径是否正确。")

        self.memory_file = agent_memory_file
        self.model = SentenceTransformer(model_name)  # 加载 MiniLM 本地模型
        self.short_memory = self._load_memory()
        self.memory_embeddings = self._encode_memories()

    def _load_memory(self):
        """ 加载 JSON 文件中的短期记忆 """
        with open(self.memory_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "short_memory" not in data or not isinstance(data["short_memory"], list):
            raise ValueError(f"⚠️ 文件 {self.memory_file} 格式不正确，缺少 `short_memory` 字段！")

        return data["short_memory"]

    def _encode_memories(self):
        """ 计算所有短期记忆的向量表示 """
        texts = [
            f"{mem['summarize']} {mem.get('comment', '')} Stance: {mem.get('stance', 'Unknown')}"
            for mem in self.short_memory
        ]
        return np.array(self.model.encode(texts))  # 计算所有记忆的嵌入向量

    def retrieve_most_similar(self, query):
        """
        计算查询与所有短期记忆的相似度，并返回最相似的索引
        :param query: 输入的查询文本
        :return: 最相似的记忆索引
        """
        if not self.short_memory:
            raise ValueError("⚠️ 短期记忆为空，无法进行检索！")

        query_vector = self.model.encode([query])  # 计算查询向量
        similarities = cosine_similarity(query_vector, self.memory_embeddings)[0]  # 计算余弦相似度
        most_similar_idx = int(np.argmax(similarities))  # 获取最大相似度索引
        return most_similar_idx
    
    def retrieve_similarity(self, query):
        """
        计算查询与所有短期记忆的相似度，并返回最相似的索引
        :param query: 输入的查询文本
        :return: 最相似的记忆索引
        """
        if not self.short_memory:
            raise ValueError("⚠️ 短期记忆为空，无法进行检索！")

        query_vector = self.model.encode([query])  # 计算查询向量
        similarities = cosine_similarity(query_vector, self.memory_embeddings)[0]  # 计算余弦相似度

        return any(similarity > 0.5 for similarity in similarities)
        

