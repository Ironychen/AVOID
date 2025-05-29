import faiss
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer

class ShortMemoryDB:
    def __init__(self, index_file="short_memory.index", meta_file="short_memory_meta.json"):
        self.model = SentenceTransformer('/usr/gao/gubincheng/LLM/all-MiniLM-L6-v2')  # 加载 MiniLM 本地模型
        self.vector_dim = self.model.get_sentence_embedding_dimension()  # 获取模型向量维度
        self.index_file = index_file  # FAISS 索引文件
        self.meta_file = meta_file  # 存储 name -> FAISS 索引映射
        self.index = self._load_or_create_index()
        self.memory_meta = self._load_or_create_meta()

    def _load_or_create_index(self):
        """ 加载或创建 FAISS 索引 """
        if os.path.exists(self.index_file):
            print("📥 Loading existing FAISS index...")
            return faiss.read_index(self.index_file)
        else:
            print("🆕 Creating new FAISS index...")
            return faiss.IndexFlatL2(self.vector_dim)  # L2 距离索引

    def _load_or_create_meta(self):
        """ 加载或创建 memory_id -> FAISS 索引 """
        if os.path.exists(self.meta_file):
            with open(self.meta_file, "r") as f:
                return json.load(f)
        return {}

    def _generate_embedding(self, text):
        """ 使用 MiniLM 生成文本的向量表示 """
        return np.array(self.model.encode(text), dtype=np.float32)

    def add_memory(self, memory):
        """
        添加短期记忆并存储到 FAISS
        :param memory: 短期记忆项 (dict)，包含 name, summarize, comment, stance
        """
        name = memory["name"]
        combined_text = f"{memory['summarize']} {memory['comment']} Stance: {memory['stance']}"
        embedding = self._generate_embedding(combined_text)

        # 记录索引范围
        idx = self.index.ntotal
        self.index.add(np.array([embedding]))  # 添加向量
        self.memory_meta[name] = idx

        # 保存数据
        self._save_index()
        self._save_meta()
        print(f"✅ Added memory: {name}")

    def search_similar(self, query, top_k=2):
        """
        通过 KNN 搜索最相似的短期记忆
        :param query: 需要查询的文本
        :param top_k: 取前 K 个最相似的记忆
        :return: List of (memory_name, similarity_score)
        """
        query_vector = self._generate_embedding(query).reshape(1, -1)
        distances, indices = self.index.search(query_vector, top_k)

        # 解析索引并匹配名称
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            for name, stored_idx in self.memory_meta.items():
                if stored_idx == idx:
                    results.append((name, distance))
                    break

        return sorted(results, key=lambda x: x[1])  # 按距离排序

    def _save_index(self):
        """ 保存 FAISS 索引 """
        faiss.write_index(self.index, self.index_file)

    def _save_meta(self):
        """ 保存 memory_id -> FAISS 索引映射 """
        with open(self.meta_file, "w") as f:
            json.dump(self.memory_meta, f, indent=4)

    def load_index(self):
        """ 重新加载索引 """
        self.index = faiss.read_index(self.index_file)
        with open(self.meta_file, "r") as f:
            self.memory_meta = json.load(f)
        print("🔄 Index and metadata reloaded.")

# 实例化数据库
# memory_db = ShortMemoryDB()

# # 从 JSON 文件中读取 short_memory 字段
# json_path = "/usr/gao/gubincheng/article_rep/Agent/simplest_agent/Env_Rumor_Test/agent_1/agent_1.json"
# with open(json_path, "r", encoding="utf-8") as f:
#     data = json.load(f)
# short_memory = data.get("short_memory", [])

# # 添加所有记忆项
# for memory in short_memory:
#     memory_db.add_memory(memory)

# query_text = "The Chinese Space Program officials doubt the authenticity of the American moon landings."
# similar_memories = memory_db.search_similar(query_text, top_k=2)

# print("🔍 Most similar memories:", similar_memories)
# memory_db.load_index()
# print("🔄 已加载索引，向量数:", memory_db.index.ntotal)

memory_db = ShortMemoryDB()
memory_db.load_index()  # 重新加载索引

query_text = "Some new query about the moon landing."
similar_memories = memory_db.search_similar(query_text, top_k=2)

print("🔍 Most similar memories:", similar_memories)

