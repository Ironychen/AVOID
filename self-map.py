import numpy as np
from typing import List, Dict, Union
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from LLM_prompt import get_content_response

# -------------------------
# 初始化本地的 MiniLM-L6-v2 模型
encoder = SentenceTransformer('/usr/gao/gubincheng/LLM/all-MiniLM-L6-v2')

# 使用本地模型生成文本嵌入，支持单个字符串或字符串列表
def get_embedding(text: Union[str, List[str]]) -> np.array:
    return encoder.encode(text)

# -------------------------
# 定义记忆片段，每个记忆片段包含：姓名、摘要、评论和立场
class MemorySnippet:
    def __init__(self, name: str, summarize: str, comment: str, stance: str):
        self.name = name
        self.summarize = summarize
        self.comment = comment
        self.stance = stance
        
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "summarize": self.summarize,
            "comment": self.comment,
            "stance": self.stance
        }

# Self-MAP 框架实现
class SelfMAP:
    def __init__(self, memory_bank: List[MemorySnippet] = None):
        self.memory_bank = memory_bank if memory_bank is not None else []
    
    def add_memory(self, snippet: MemorySnippet):
        self.memory_bank.append(snippet)
    
    # 记忆检索：利用向量化方式计算当前指令与记忆库中所有摘要之间的余弦相似度，
    # 返回相似度最高的 top_k 个记忆片段。
    def retrieve_relevant_memory(self, current_instruction: str, top_k: int = 3) -> List[MemorySnippet]:
        query = current_instruction
        query_embedding = get_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)  # 转换为二维向量
        # 获取记忆库中所有摘要
        summaries = [snippet.summarize for snippet in self.memory_bank]
        if not summaries:
            return []
        summaries_embeddings = get_embedding(summaries)  # 批量编码，形状 (n, d)
        # 计算当前指令与各候选摘要的余弦相似度
        similarities = sk_cosine_similarity(query_embedding, summaries_embeddings)[0]
        # 选取相似度最高的 top_k 索引
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.memory_bank[i] for i in top_indices]
    
    # 记忆简化：简化
    def memory_simplification(self, snippet: MemorySnippet) -> str:
        """Simplifies the news summary by extracting key information and updating snippet.simplified_summary."""

        if not snippet.summarize:
            raise ValueError(f"MemorySnippet {snippet.name} has an empty summary. Unable to simplify.")

        prompt = f"""
        Read the news summary of {snippet.name}: "{snippet.summarize}". Identify the key event, the people or organizations involved, and the core controversy. 
        Then, analyze the given stance ("{snippet.stance}") and comment ("{snippet.comment}") to infer {snippet.name}'s deeper attitude toward the event, considering both explicit and implicit perspectives. 
        Finally, based on these insights, generate a concise yet advanced summary that captures the overall meaning, reflecting both factual information and the underlying stance in a coherent manner. 
        The response no more than 100 words.
        """

        simplified = get_content_response(prompt)

        if simplified is None:
            raise RuntimeError(f"Failed to generate a simplified summary for {snippet.name}. Check `get_content_response`.")

        setattr(snippet, "simplified_summary", simplified)  # Ensure the field is dynamically added
        return simplified

    
    # 记忆精炼：基于评论和立场生成简单的推理理由
    def infer_motivation(self, snippet: MemorySnippet) -> str:
        """Analyzes the potential reasons behind the stance taken in the simplified summary."""

        if not hasattr(snippet, "simplified_summary") or not snippet.simplified_summary:
            raise ValueError(f"MemorySnippet {snippet.name} does not have a simplified summary. Run memory_simplification first.")

        prompt = f"""
        Given the simplified summary: "{snippet.simplified_summary}", analyze the possible motivations behind this stance. 
        Consider:
        - Political, economic, or social influences that might shape this viewpoint.
        - The potential audience or stakeholders being addressed.
        - Whether this perspective aligns with a broader narrative or agenda.
        Provide a coherent reasoning for why this stance was taken.
        """

        reasoning = get_content_response(prompt)

        if reasoning is None:
            raise RuntimeError(f"Failed to generate reasoning for {snippet.name}. Check `generate_advanced_description`.")

        setattr(snippet, "reasoning", reasoning)  # 存储推理结果
        return reasoning
        
    # 生成自我反思记忆：先检索相关记忆，再对检索到的k个记忆进行简化和精炼，返回带有附加信息的字典列表
    def prepare_self_reflective_memory(self, current_instruction: str, top_k: int = 3) -> List[Dict]:
        relevant_snippets = self.retrieve_relevant_memory(current_instruction, top_k)
        reflective_memories = []
        for snippet in relevant_snippets:
            self.memory_simplification(snippet)
            self.memory_refinement(snippet)
            mem_dict = snippet.to_dict()
            mem_dict["simplified_summary"] = snippet.simplified_summary
            mem_dict["rationale"] = snippet.rationale
            reflective_memories.append(mem_dict)
        return reflective_memories
    
    # 动作规划：结合当前指令、自我反思记忆与当前环境候选元素，生成下一步操作的描述
    def plan_next_action(self, current_instruction: str, current_env_candidates: List[str]) -> str:
        memories = self.prepare_self_reflective_memory(current_instruction)
        plan_text = f"当前指令：{current_instruction}\n"
        plan_text += "检索到的相关记忆：\n"
        if memories:
            for mem in memories:
                plan_text += (
                    f"- 摘要：{mem['summarize']}\n"
                    f"  简化摘要：{mem['simplified_summary']}\n"
                    f"  推理理由：{mem['rationale']}\n"
                )
        else:
            plan_text += "无相关记忆。\n"
        plan_text += "当前环境候选元素：" + ", ".join(current_env_candidates) + "\n"
        if current_env_candidates:
            next_action = f"执行操作：选择 {current_env_candidates[0]}"
        else:
            next_action = "未找到候选元素。"
        plan_text += f"规划的下一步动作：{next_action}"
        return plan_text

# -------------------------
# 示例：构建记忆库并进行下一步规划
if __name__ == "__main__":
    # 创建 SelfMAP 实例
    self_map = SelfMAP()
    
    # 模拟添加对话历史中的记忆片段（按照 agent_2.json 中 short_memory 的形式）
    snippet1 = MemorySnippet(
        name="Ismail",
        summarize="The news article reports on the leakage of government documents, resulting in a seizure of a reporter's communications and charges against a former Senate aide for lying to investigators. The story presents a credible account of ongoing investigations into breaches of privacy and journalistic ethics. An accompanying image features the Fox News Channel logo, symbolizing its modern and dynamic brand. The public's comments are not visible, indicating a lack of feedback on the issue.",
        comment="The situation outlined is deeply concerning, as it involves breaches of privacy and ethical journalism standards. The government's seizure of a reporter's communications without her knowledge raises serious questions about transparency and the role of the press in a democratic society. It's crucial that the actions of the Senate aide are thoroughly investigated and that any violations are addressed appropriately. The integrity of our democratic processes depends on a free and independent press, and such incidents undermine that principle.",
        stance="Oppose"
    )
    snippet2 = MemorySnippet(
        name="Derek",
        summarize="The event depicted in the news likely features a public speaking or political gathering. The image shows a man in a dark suit jacket and light blue shirt addressing an audience from a podium with a microphone, suggesting a speech delivery. The clock indicating 10:10 in the background could denote the event's timing. There are no comments available to gauge opinions on the news article.",
        comment="It seems there's an issue with the interactive transcript not being loaded. This could be a temporary technical glitch. I recommend trying to reload the page or checking for any updates on the platform's status page. If the problem persists, contacting customer support might be the next step.",
        stance="Neutral"
    )
    snippet3 = MemorySnippet(
        name="Helen",
        summarize="The news article reports that Chinese Space Program officials and experts have petitioned the American government for explanations about the authenticity of the American moon landings, based on an analysis of moon rover images. While the article lacks substantial evidence and authoritative sources to support the claim, it describes an image showing a lunar lander with solar panels, a dish antenna, and a red flag with yellow stars, indicating a Chinese mission. The image suggests a successful landing and exploration, reflecting technological achievement and space exploration advancements.",
        comment="The claims made by top officials of the Chinese Space Program and Russian engineers regarding the American moon landings as a 'hoax' are intriguing, but they lack substantial evidence to support such a significant claim. The widely accepted historical accounts of the moon landings, along with the wealth of physical evidence on the Moon, have not been refuted by the recent analysis. It's important to approach such allegations with a critical eye and await further credible evidence before reaching conclusions. The involvement of conspiracy theorists in this discussion only adds to the complexity of the matter.",
        stance="Neutral"
    )
    
    self_map.add_memory(snippet1)
    self_map.add_memory(snippet2)
    self_map.add_memory(snippet3)
    
    # 当前对话上下文
    current_instruction = "预定一张价格在 $50 到 $100 之间的 WWE 门票。"
    current_env_candidates = ["$50 门票", "$70 门票", "$90 门票"]
    
    # 规划下一步动作
    action_plan = self_map.plan_next_action(current_instruction, current_env_candidates)
    print("动作规划结果：\n")
    print(action_plan)
