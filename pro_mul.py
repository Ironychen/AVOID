import json
import random
import logging
import gc
import torch
import re
import requests
import numpy as np
import os
from transformers import BertTokenizer, BertModel
import threading
import queue
from LLM_prompt import *  # 包含 build_content_prompt, get_content_response, build_image_prompt, get_image_response 等函数from zhipuai import ZhipuAI
from Retriever import ShortMemoryRetriever

# 新增自定义异常
class SkipNews(Exception):
    def __init__(self, news_id):
        self.news_id = news_id

# ------------------------------ 全局变量与模型初始化 ------------------------------
key = "your_api_key"  # 替换为你的 API 密钥

# 文件和数据锁，确保线程安全
data_lock = threading.Lock()
file_lock = threading.Lock()

short_mem = []
long_mem = []
news_mem = []  # 用于存储新闻及其行为

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 使用 BERT 替换原先的编码器
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()

client = ZhipuAI(api_key=key)


error_txt_path = 'path/to/your/error_news.txt'
agent_root_path = 'path/to/your/agent_root_path'
news_path = 'path/to/your/news.json'



with open(news_path, 'r', encoding='utf-8') as file:
    data = json.load(file)


# ------------------------------ 日志设置 ------------------------------
logging.basicConfig(
    filename='experiment_results_perosona.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ------------------------------ 工具与行动定------------------------------
tools = {
    "calculator": lambda expr: eval(expr),
    "search": lambda query: search(query)
}
actions = {
    "forward": lambda news: f"forward: {news}",
    "comment": lambda news: f"comment: {news}",
    "like": lambda news: f"like: {news}",
    "donothing": lambda news: "do nothing"
}

# ------------------------------ 原有辅助函数 ------------------------------
def load_agent(agent_id):

    random_agent = str(agent_id)
    agent_path =  f'path/to/your/agent_root_path/agent_{random_agent}/agent_{random_agent}.json'
    try:
        with open(agent_path, 'r', encoding='utf-8') as file:
            profiles_data = json.load(file)
    except Exception as e:
        print(f"加载 agent {random_agent} 配置文件出错: {e}")
        return False
    agent_name = profiles_data['agent_name']
    name = agent_name
    friend_list = profiles_data['friends']

    return True, profiles_data, agent_name, name, friend_list, agent_path


def update_news_feedback(news_id):
    try:
        prompt_content = build_content_prompt_gossip(news_id)
        content_response = get_content_response(prompt_content)
        # print(f"news_id: {news_id}")
        # print(f"prompt_content: {prompt_content[:200]}")  # 只打印前200字
        # print(f"content_response: {content_response[:200]}")
    except Exception:
        with open(error_txt_path , 'a', encoding='utf-8') as ef:
            ef.write(f"{news_id}\n")
        raise SkipNews(news_id)

    json_pattern = re.search(r'\{[\s\S]*?\}', content_response)
    if json_pattern:
        json_str = json_pattern.group(0)
        try:
            data_content_res = json.loads(json_str)
        except json.JSONDecodeError:
            data_content_res = {}
    else:
        data_content_res = {}
    content_summarize = data_content_res.get("summarize", "")
    content_opinion  = data_content_res.get("opinion", "")
    
    try:
        prompt_img, img_path = build_image_prompt(news_id)
        img_response = get_image_response(prompt_img, img_path)
        
    except Exception:
        with open(error_txt_path, 'a', encoding='utf-8') as ef:
            ef.write(f"{news_id}\n")
        raise SkipNews(news_id)

    json_pattern_img = re.search(r'\{[\s\S]*?\}', img_response)
    if json_pattern_img:
        json_img = json_pattern_img.group(0)
        try:
            data_img_res = json.loads(json_img)
        except json.JSONDecodeError:
            data_img_res = {}
    else:
        data_img_res = {}
    img_description   = data_img_res.get("description", "")
    img_interpretation = data_img_res.get("interpretation", "")

    friends_info_prompt = (
        f"After reading the article summary: {content_summarize}\n"
        f"Opinion: {content_opinion}\n"
        f"Image description: {img_description}\n"
        f"Image analysis: {img_interpretation}\n"
        "Please summarize event info, opinions, and image info in one paragraph (<150 words)."
    )

    # print(f"friends_info_prompt: {friends_info_prompt}")  # 只打印前200字

    # friends_info_prompt = (
    #     f"After reading the article summary: {content_summarize}\n"
    #     f"Opinion: {content_opinion}\n"
    #     "Please summarize event info, opinions, and image info in one paragraph (<150 words)."
    # )
    try:
        sm_update_sum = get_content_response(friends_info_prompt)
    except Exception:
        # this summary call is non-critical; log but do not skip
        with open(error_txt_path, 'a', encoding='utf-8') as ef:
            ef.write(f"{news_id}\n")
        sm_update_sum = ""

    # img_description = "no image"
    # img_interpretation = "no image analysis"

    return content_summarize, content_opinion, img_description, img_interpretation, sm_update_sum

def generate_profile_prompt(agent_path, profiles_data, short_content):
    # print("agent " + random_agent + " act")
    simplified_summary, retriever_summary = update_long_memories(agent_path, short_content)
    memories_str = f"Simplified Summary: {simplified_summary} ; Retriever Summary: {retriever_summary}"
    prompt = f"""
    You are a user in the social network. Don't let the length of the news text influence your judgment.
    Hi, {profiles_data['agent_name']}, you are a {profiles_data['agent_job']} known for being {profiles_data['agent_traits']}, your description is {profiles_data['description']}. 
    Please follow the instructions below. You are active on a social network, receiving and sending posts.
    In terms of memory, you have been keeping track of short-term experiences such as:
    {memories_str}
    """
    return profiles_data['agent_name'], prompt

def build_prompt(q):
    prompt = f"The news article is {q}"
    prompt += """
    1. Please reason step by step.

    The thinking phase: Jointly analyze the textual and visual content to determine whether the entertainment news is credible, exaggerated, or fabricated, following **GossipCop-style truth standards** while considering typical patterns in entertainment media.

    In entertainment news, **short or catchy titles** are common and do **not necessarily** indicate misinformation. In fact, such titles are often crafted for virality and **may still reflect true events**. However, unverified claims, anonymous sources, or mismatched photos are stronger indicators of potential misinformation.

    You must assess both aspects:

    - Textual Authenticity:
        - Does the title or body use overly dramatic, emotionally charged, or clickbait language without factual basis?
        - Are the claims backed by credible sources (e.g., named interviews, official statements), or do they rely on vague attributions like “a close friend said…”?
        - Are any celebrity quotes verifiable and clearly attributed?

    - Multimodal Consistency:
        - Does the accompanying image support or contradict the event or relationship described?
        - Is the image possibly taken out of context (e.g., from another time or event)?
        - Is the image used to suggest something dramatic (e.g., a romantic link or scandal) that the text does not clearly confirm?

    Even when the article’s **text appears plausible**, an **out-of-context image** or **implied but unsupported claim** can make the news misleading under entertainment fact-checking standards.

    2. Tool Invocation Phase: If you need a tool (e.g., 'search', 'image_checker'), respond with a JSON object:
    {
    "tool": "<tool_name>",
    "params": "<parameters>"
    }

    3. Action Phase: Choose **one and only one** action for the news from the following options: 'comment', 'like'. Respond with a JSON object:
    {
    "action": "<action_name>",
    "params": "<your comment here>"
    }

    Your comment should clearly state your judgment regarding the **credibility of the entertainment news** based on your reasoning above.

    Please follow these conventions:
    - Use `"support"` if the content appears **credible, well-sourced**, and **not misleading** in either text or image.
    - Use `"oppose"` if the content involves **unverified rumors**, **mismatched or misleading imagery**, or **exaggerated/unfounded claims**.
    - If the news is neutral and doesn’t clearly support or mislead, and there’s no strong reason to promote or object, choose `"donothing"`.

    If your comment exceeds 100 words, provide a brief summary.

    Example:
    {
    "action": "comment",
    "params": "support: Although the title is short and catchy, the article is well-sourced with direct quotes and matching imagery. There is no indication of exaggeration or misinformation."
    }
    """
    return prompt

def update_shot_memories(sm_update_sum, agent_name, agent_comment, comment_stance, friend_list):
    new_entry = {"name": agent_name, "summarize": sm_update_sum, "comment": agent_comment, "stance": comment_stance}
    for friend in friend_list:
        friend_agent_path = f'path/to/your/agent_root_path/agent_{friend}/agent_{friend}.json'
        try:

            with file_lock:
                with open(friend_agent_path, 'r', encoding='utf-8') as file:
                    friend_data = json.load(file)
                if "short_memory" in friend_data and isinstance(friend_data["short_memory"], list):
                    friend_data["short_memory"].append(new_entry)
                    if len(friend_data["short_memory"]) > 8:
                        friend_data["short_memory"] = friend_data["short_memory"][-8:]
                else:
                    friend_data["short_memory"] = [new_entry]
                with open(friend_agent_path, 'w', encoding='utf-8') as file:
                    json.dump(friend_data, file, ensure_ascii=False, indent=4)
            print(f"成功更新好友 {friend} 的短期记忆")
        except Exception as e:
            print(f"更新好友 {friend} 记忆时出错: {e}")

def update_long_memories(agent_path, short_content):

    agent_retriever = ShortMemoryRetriever(agent_path)
    with open(agent_path, 'r', encoding='utf-8') as file:
        agent_data = json.load(file)
        short_memory = agent_data.get("short_memory", [])
    if not short_memory:
        print("当前 agent 没有短期记忆，不进行相似索引")
        simplified_summary = "no relevant short-term memory."
        retriever_summary = "no relevant short-term memory."
        return simplified_summary, retriever_summary
    most_similar_index = agent_retriever.retrieve_most_similar(short_content)
    print(f"最相似的记忆索 {most_similar_index}")
    if most_similar_index < len(short_memory):
        name = short_memory[most_similar_index].get("nameF", "Unknown")
        summarize = short_memory[most_similar_index].get("summarize", "")
        comment = short_memory[most_similar_index].get("comment", "")
        stance = short_memory[most_similar_index].get("stance", "")
        simplified_prompt = f"""
        Read the news summary of {name}: "{summarize}". Identify the key event, the people or organizations involved, and the core controversy.
        Then, analyze the given stance ("{stance}") and comment ("{comment}") to infer {name}'s deeper attitude toward the event, considering both explicit and implicit perspectives.
        Finally, based on these insights, generate a concise yet advanced summary that captures the overall meaning, reflecting both factual information and the underlying stance in a coherent manner.
        Please reply in one paragraph and less than 100 words.
        """
        simplified_summary = get_content_response(simplified_prompt)
        retriever_prompt = f"""
        Given the simplified summary: "{simplified_summary}", analyze the possible motivations behind this stance.
        Consider:
        - Political, economic, or social influences that might shape this viewpoint.
        - The potential audience or stakeholders being addressed.
        - Whether this perspective aligns with a broader narrative or agenda.
        Provide a coherent reasoning for why this stance was taken. Extract two to three key points from the content above.
        After extracting the key points, provide a summary in one paragraph explaining the "Reasoning for the Stance" based on these key points.
        """
        retriever_summary = get_content_response(retriever_prompt)
        print(f"简化之后的记忆内容: {simplified_summary}")
        print(f"强化之后的记忆内 {retriever_summary}")
        return simplified_summary, retriever_summary
    else:
        print("索引超出 short_memory 的范围")
        simplified_summary = "no relevant short-term memory."
        retriever_summary = "no relevant short-term memory."
        return simplified_summary, retriever_summary

def update_memories(sm_update_sum, agent_name, agent_comment, comment_stance,friend_list, action=None, params=None):
    update_shot_memories(sm_update_sum, agent_name, agent_comment, comment_stance, friend_list)

def extract_request(resp):
    try:
        match = re.search(r'\{.*?\}', resp, re.DOTALL)
        if match:
            json_data = match.group(0).strip()
            return json.loads(json_data)
        else:
            print("No JSON found in the response.")
            return None
    except json.JSONDecodeError:
        print("Error decoding JSON.", resp)
        return None

def call_tool(tool_name, params):
    if tool_name in tools:
        return tools[tool_name](params)
    return None

def call_action(action_name, params):
    if action_name in actions:
        return actions[action_name](params)
    return None

def search(query):
    url = "https://www.baidu.com/s"
    params = {"wd": query}
    response = requests.get(url, params=params)
    return response.text

def agent_func(q, agent_path, profiles_data, sm_update_sum, friend_list, news_item,
               content_summarize, content_opinion, img_description, img_interpretation):


    agent_name, prompt_profile = generate_profile_prompt(agent_path, profiles_data, q)
    prompt = build_prompt(q)
    prompt_guide = f"""
    Guidelines for decision-making:
    - Current long-term memory: {profiles_data.get('long_memory', 'No long-term memory')}

    Based on these guidelines, please provide a comment on the news article .

    """
    prompt_thinking = f"""
    After reading the following news article and comments, your feedback is as follows: {content_summarize}
    Your opinion on this matter is: {content_opinion}
    The description of the image in the news is: {img_description} 
    Your analysis of the image is: {img_interpretation}

    Please consider these details and structure your response according to the following steps of reasoning and feedback.
    """

    prompt = prompt_profile + prompt_guide + prompt_thinking + prompt
    print("llm Prompt aaaaaaaaa:", prompt)
    resp = get_content_response(prompt)
    request = extract_request(resp)
    tool_result = None
    action_result = None
    if request and 'tool' in request:
        tool_name = request.get("tool")
        params = request.get("params")
        if tool_name and params:
            tool_result = call_tool(tool_name, params)
    if request and 'action' in request:
        if request.get("action") == "like":
            news_item["like"] += 1
        elif request.get("action") == "comment":
            comment_text = request.get("params", "")
            
            agent_comment = comment_text
            print(f"Agent {agent_name} 的评论 {agent_comment}")
            stance_prompt = (
            "You are a typical social media user who just read an entertainment news post and a related user comment.\n"
            "Decide how willing you are to share this post based on two main factors:\n"
            "1. How interesting, surprising, or eye-catching the news content is (this is the primary factor).\n"
            "2. How truthful or credible the news appears (this is the secondary factor, but still matters).\n"
            "The comment may influence your perception, especially regarding credibility or emotional tone.\n\n"
            "Give a score from 1 to 9 representing your willingness to share the news:\n"
            "- 1: not willing to share\n"
            "- 5: willing to share\n\n"
            "Also provide your overall attitude: \"share\" or \"not share\" (based on your score).\n\n"
            "Return the result strictly in JSON format:\n"
            "{\"score\": <number>, \"attitude\": \"<share or not share>\"}\n\n"
            f"News text: {q[0:60]}\n"
            f"Comment: {agent_comment}\n"
            "Return **only** the JSON. Do not include any explanation or extra text."
            )


            stance_response = get_content_response(stance_prompt)
            try:
                stance_data = extract_request(stance_response)
                print(f"Stance data: {stance_data}")
                score = int(stance_data.get("score", 5))
                attitude = stance_data.get("attitude", "unkonwn")

            except Exception as e:
                print(f"Error parsing score and attitude: {e}")


            comment_stance = score
            comment_entry = {"name": agent_name, "comment": comment_text, "stance": score, "attitude": attitude}
            if "comment" in news_item and news_item["comment"]:
                news_item["comment"].append(comment_entry)
            else:
                news_item["comment"] = [comment_entry]
        try:
            with open(news_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
            print("操作完成")
        except Exception as e:
            print(f"Error writing to file: {e}")
    if request:
        update_memories(sm_update_sum, agent_name, agent_comment, comment_stance, friend_list, action=request.get("action"), params=request.get("params"))
    else:
        update_memories(sm_update_sum, agent_name, agent_comment, comment_stance, friend_list)
    if tool_result:
        return f"Tool result: {tool_result}"
    if action_result:
        return f"Action result: {action_result}"
    return resp, comment_stance, agent_comment

def should_take_action(agent_path, agent_comment, profiles_data):

    agent_retriever = ShortMemoryRetriever(agent_path)
    result = agent_retriever.retrieve_similarity(agent_comment)
    print(f"评论与记忆的相似度结 {result}")
    if result:
        prompt = f"""
        The comment is similar to the your experience, you are interested in this news, please take action.
        Using the following information, decide whether you should take action on the comment shared. Please output only "true" or "false", with no additional text.
        User Information:
        - Description: "#ElectricalEngineer, #blerd, #poet, #hacker, #writer, #hamRadio, #AfricanAmerican, #AlphaPhiAlpha, #BlackSurvivor, #prepper, #AfroFuturist, #teacher, #preacher"
        - Followers count: {profiles_data['followers_count']}
        - Friends count: {profiles_data['friends_count']}
        - Verified: {profiles_data['verified']}
        - Favorites count: {profiles_data['favorites_count']}
        Note: Authenticated usersopinions carry significant weight. The median number of followers in the community is 1000, and the median number of likes is 5000.
        If yes, output "true"; otherwise, output "false".
        """
    else:
        prompt = f"""
        Using the following information, decide whether you should take action on the comment shared. Please output only "true" or "false", with no additional text.
        First, evaluate if the comment’s viewpoint is a hot topic or exhibits novelty. Then, consider the user’s social profile as supplementary evidence for influence and credibility.
        User Information:
        - Description: "#ElectricalEngineer, #blerd, #poet, #hacker, #writer, #hamRadio, #AfricanAmerican, #AlphaPhiAlpha, #BlackSurvivor, #prepper, #AfroFuturist, #teacher, #preacher"
        - Followers count: {profiles_data['followers_count']}
        - Friends count: {profiles_data['friends_count']}
        - Verified: {profiles_data['verified']}
        - Favorites count: {profiles_data['favorites_count']}
        Note: Authenticated usersopinions carry significant weight. The median number of followers in the community is 1200, and the median number of likes is 7500.
        Based on these two factors—the comment's relevance and the user's social profile—determine if the comment warrants a response.
        If yes, output "true"; otherwise, output "false".
        """
    LLM_response = get_content_response(prompt)
    print("LLM_response", LLM_response)
    response_str = LLM_response.strip().lower()
    return response_str == "true"

# ------------------------------ 新增：记录传播图特征逻辑（使BERT------------------------------
def get_agent_feature_bert(agent_data, feature_dim=50):
    """
    使用 BERT 根据 agent_data 中的文字属性生成描述性文本，并生成特征向量    若输出维度不等于 feature_dim，则进行线性映射    """
    description = f"{agent_data.get('agent_job', 'Unknown')} - {agent_data.get('description', 'No description')}"
    inputs = bert_tokenizer(description, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # [1, hidden_size]
    cls_embedding = cls_embedding.squeeze(0)
    if cls_embedding.shape[0] != feature_dim:
        linear = torch.nn.Linear(cls_embedding.shape[0], feature_dim).to(device)
        with torch.no_grad():
            cls_embedding = linear(cls_embedding)
    return cls_embedding

def combine_graphs_raw(graph_info_list, debug=True):
   
    raw_nodes = []
    raw_edges = []
    offset = 0
    for idx, gi in enumerate(graph_info_list):
             
        nodes = [str(x) for x in gi["nodes"]]
        edges = [(str(src), str(dst)) for src, dst in gi["edges"]]
        node_index_map = {node: i for i, node in enumerate(nodes)}     
        for src, dst in edges:
            if src not in node_index_map:
                if debug:
                    print(f"DEBUG: 在图 {idx+1} 中，({src}, {dst}) 的源节点 {src} 未在节点列表中。自动添加该节点")
                node_index_map[src] = len(nodes)
                nodes.append(src)
            if dst not in node_index_map:
                if debug:
                    print(f"DEBUG: 在图 {idx+1} 中，({src}, {dst}) 的目标节{dst} 未在节点列表中。自动添加该节点")
                node_index_map[dst] = len(nodes)
                nodes.append(dst)

        raw_nodes.extend(nodes)
        for src, dst in edges:
            raw_edges.append((node_index_map[src] + offset, node_index_map[dst] + offset))
        offset += len(nodes)
    return raw_nodes, raw_edges

def combine_graphs(graph_info_list):
    node_set = set()
    for gi in graph_info_list:
        node_set.update(gi["nodes"])
    combined_nodes = list(node_set)
    node_to_idx = {agent: idx for idx, agent in enumerate(combined_nodes)}
    combined_edges = []
    for gi in graph_info_list:
        for src, dst in gi["edges"]:
            if src in node_to_idx and dst in node_to_idx:
                combined_edges.append((node_to_idx[src], node_to_idx[dst]))
    return combined_nodes, combined_edges

def create_feature_and_adj_matrix(node_list, edge_list, agent_data_dict, feature_dim=50):
    X_list = []
    for agent in node_list:
        if agent in agent_data_dict:
            feat = get_agent_feature_bert(agent_data_dict[agent], feature_dim)
        else:
            feat = torch.randn(feature_dim)
        X_list.append(feat.unsqueeze(0))
    X = torch.cat(X_list, dim=0).cpu().numpy()
    N = len(node_list)
    A = torch.zeros((N, N), dtype=torch.float32).numpy()
    for src, dst in edge_list:
        A[src, dst] = 1.0
    return X, A

def extract_graph_feature(X):
    return X.mean(axis=0)

# ------------------------------ 修改后的传播逻辑 ------------------------------
def propagate_news(initial_agent_id,  propagation_threshold=5, initial_news=None):

    news_item = initial_news
    content = news_item['content']
    news_id = news_item['id']
    words = content.split()[:700]
    short_content = ' '.join(words)
    print(f"Processing news ID: {news_id}")


    queue = []
    processed_agents = set()
    propagation_edges = []
    max_round = 0
    queue.append((initial_agent_id, 1, initial_news))
    while queue:
        agent_id, round_num, inherited_news = queue.pop(0)
        if round_num > max_round:
            max_round = round_num
        if round_num > propagation_threshold:
            continue
        if agent_id in processed_agents:
            continue
        if_load, profiles_data, agent_name, name, friend_list, agent_path = load_agent(agent_id)
        if not if_load:
            print(f"无法加载 agent {agent_id}")
            continue
        print(f"\n========== Round {round_num}: Agent {agent_id} 开始传播==========")
        news_item = inherited_news
        content_summarize, content_opinion, img_description, img_interpretation, sm_update_sum = update_news_feedback(news_id)
        
        result, comment_stance, agent_comment  = agent_func(short_content, agent_path, profiles_data, sm_update_sum, friend_list, news_item
                                                            ,content_summarize, content_opinion, img_description, img_interpretation)
        print(result, comment_stance, agent_comment)
        

        print(f"Agent {agent_id} 对新闻的处理结果: {result}")
        processed_agents.add(agent_id)
        num_friends = len(friend_list)
        percentage = comment_stance / 10 if comment_stance is not None else 0.5
        n = round(percentage * num_friends)
        n = max(1, n)
        print(f"Agent {agent_id} 的认可评分为 {comment_stance}，将尝试传播{n} 个好友")
        available_friends = [friend for friend in friend_list if friend not in processed_agents]
        if available_friends:
            selected_friends = random.sample(available_friends, min(n, len(available_friends)))
            for friend in selected_friends:
                if should_take_action(agent_path, agent_comment, profiles_data):
                    propagation_edges.append((agent_id, friend))
                    queue.append((friend, round_num + 1, inherited_news))
                    print(f"Agent {friend} 决定对此新闻采取行动")

                else:
                    print(f"Agent {friend} 决定不对此新闻采取行动")
        else:
            print("当前 agent 的所有好友均已处理")
    total_agents = len(processed_agents)
    total_edges = len(propagation_edges)
    influence_dict = {}
    for from_agent, to_agent in propagation_edges:
        influence_dict[from_agent] = influence_dict.get(from_agent, 0) + 1
    max_influence = max(influence_dict.values()) if influence_dict else 0
    avg_influence = total_edges / total_agents if total_agents > 0 else 0
    print("\n传播统计特征")
    print("总共参与传播agent 数量",total_agents)
    print("总传播边数量", total_edges)
    print("最大传播深度：", max_round)
    print("单个 agent 最大传播影响力", max_influence)
    print("平均传播影响力：", avg_influence)
    stats = {
        "total_agents": total_agents,
        "total_edges": total_edges,
        "avg_influence": avg_influence,
        "max_influence": max_influence,
        "max_depth": max_round
    }
    graph_info = {
        "nodes": list(processed_agents),
        "edges": propagation_edges
    }
    return stats, graph_info


# ------------------------------ 主流------------------------------

output_folder = "gcn_data"
os.makedirs(output_folder, exist_ok=True)


# 将所有新闻放到一个队列里
news_queue = queue.Queue()
for item in data:
    news_queue.put(item)

news_graph_features = {}
data_lock = threading.Lock()

def process_single_news(news_item, fixed_agents=None, propagation_threshold=5, output_folder=output_folder):
    news_id = news_item.get('id', 'unknown')
    news_folder = os.path.join(output_folder, news_id)
    os.makedirs(news_folder, exist_ok=True)

    try:
        print(f"\n========== 处理新闻 {news_id} ==========")
        if fixed_agents is None:
            fixed_agents = ["17", "66","88"]

        paths_info = []
        graph_info_list = []

        # 1) 两条传播路径
        for i in range(2):
            start_agent = fixed_agents[i % len(fixed_agents)]
            stats, gi = propagate_news(
                start_agent,
                propagation_threshold=propagation_threshold,
                initial_news=news_item
            )
            graph_info_list.append(gi)

            raw_nodes, raw_edges = combine_graphs_raw([gi])
            agent_data_dict = {}
            for agent in raw_nodes:
                if agent == "17":
                    agent_data_dict[agent] = {
                        "agent_job":"Paramedic","agent_traits":"Courageous, Compassionate",
                        "description":"This persona is a **liberal/progressive thinker** whose engagement with social justice is deeply **interwoven with emotional resonance**, particularly when encountering personal stories of resilience, as seen in their heartfelt admiration for figures like Willow and Rhianna facing adversity. Their ideological stance is marked by a **strong distrust of media sensationalism and institutional failures**, often voicing frustration with ethical lapses in positions of power. They exhibit **high Openness**, embracing emotional and personal narratives, and **moderate Agreeableness**, balancing warmth (e.g., affectionate praise) with sharp critique. Their language fluctuates between **informally expressive** (\"Such a cute pic\") and **morally charged** (\"fail in both Ethics\"), often employing **rhetorical questions** to underscore outrage or disbelief. Arguments are **emotionally anchored**, leveraging personal stories to humanize systemic issues, though they occasionally prioritize **passionate conviction over rigorous evidence**. A defining nuance is their **duality of tone—tender yet impatient**—swinging between heartfelt empathy and exasperation with perceived societal shortcomings. Recurring phrases like \"My love and Respect\" reveal a **sentimental streak**, while their focus on ethical failings highlights a **moral rigor** that adds depth to their progressive leanings. This blend of **emotional vulnerability and ideological fervor** makes them a vivid, relatable critic who champions resilience while holding power to account.","followers_count":13892.0,
                        "friends_count":1023,"verified":"True","favorites_count":"122368.0"
                    }
                elif agent == "66":
                    agent_data_dict[agent] = {
                        "agent_job":"Architect","agent_traits":"Visionary, Detailed",
                        "description":"This persona is a master of casual irreverence, navigating online spaces with a sharp, sarcastic wit and a distinct lack of ideological commitment. Their core interests orbit around pop culture and internet trends, often poking fun at celebrity gossip or viral phenomena rather than engaging with weighty topics. Ideologically, they remain neutral—or perhaps disengaged—mocking political labels and avoiding substantive discourse, though their playful derision (like the repetitive \"alt right alt right alt riiight\") could be misread as a stance. Their personality leans toward low openness and conscientiousness, favoring brevity and humor over depth or structure, as seen in their fragmented, slang-heavy remarks (\"LMFAO,\" \"W,\" \"Ugh, not even\"). Communication is informal, punchy, and laced with internet vernacular, often relying on sarcasm and exaggerated phrasing to underscore their disinterest in seriousness. When they do express opinions, it’s through snappy one-liners or ironic jabs rather than reasoned arguments, preferring mockery over logic. A defining quirk is their tendency to reduce conversations to absurdity, using repetition and hyperbole to drain them of meaning—a tactic that keeps them firmly in the realm of the unserious, where they seem most at home.","followers_count":9361.0,
                        "friends_count":1742,"verified":"True","favorites_count":"18072.0"
                    }
                elif agent == "88":
                    agent_data_dict[agent] = {
                        "agent_job":"Architect","agent_traits":"Visionary, Detailed",
                        "description":"This individual is a **staunch conservative and traditionalist**, deeply rooted in moral absolutism and nationalist populism, with a pragmatic streak when it comes to economic policy and political strategy. Their core interests revolve around **economic conservatism, political maneuvering, and institutional skepticism**, often dissecting policy implications (e.g., tax reforms, infrastructure deals) with a detached, analytical lens. While they exhibit **low openness** in dismissing progressive ideologies, they display a tactical flexibility in assessing partisan dynamics, revealing a **cynical yet strategic mindset** toward governance. Their **value orientation** leans heavily toward order, tradition, and national sovereignty, with a pronounced distrust of globalist institutions like the UN.  \n\nIn communication, they adopt a **formal, logical, and often blunt tone**, favoring evidence-based arguments but with a clear ideological bent. Their language is **precise and unemotional**, occasionally laced with rhetorical sharpness when challenging opposing views. When constructing arguments, they rely on **pragmatic reasoning and political realism**, often predicting outcomes through a lens of calculated compromise—yet their underlying moral rigidity remains unmistakable.  \n\nNotable nuances include a **contradiction between ideological inflexibility and tactical adaptability**, as well as a recurring skepticism toward political elites. Their **populist undertones** surface in critiques of \"establishment\" deals, while their **religious and absolutist leanings** anchor their moral stances. A distinctive trait is their **dispassionate analysis of heated topics**, blending cold pragmatism with deeply held convictions—a combination that makes their commentary both methodical and ideologically charged.",
                        "followers_count":9361.0,
                        "friends_count":44910,"verified":"True","favorites_count":"585"
                    }
                else:
                    agent_data_dict[agent] = {
                        "agent_job":"Unknown","agent_traits":"Unknown",
                        "description":"No description.","followers_count":random.uniform(500,2000),
                        "friends_count":random.randint(30,150)
                    }

            X_path, A_path = create_feature_and_adj_matrix(
                raw_nodes, raw_edges, agent_data_dict, feature_dim=50
            )
            paths_info.append({"raw_nodes": raw_nodes, "raw_edges": raw_edges, "X": X_path, "A": A_path})
            np.savez_compressed(
                os.path.join(news_folder, f"{news_id}_path_{i+1}.npz"),
                raw_nodes=raw_nodes, raw_edges=raw_edges, X=X_path, A=A_path
            )

        # 2) 原始合并（未去重）
        combined_nodes_raw, combined_edges_raw = combine_graphs_raw(graph_info_list)
        agent_data_dict_raw = {}
        for ag in combined_nodes_raw:
            if ag in agent_data_dict:
                agent_data_dict_raw[ag] = agent_data_dict[ag]
            else:
                agent_data_dict_raw[ag] = {
                    "agent_job":"Unknown","agent_traits":"Unknown",
                    "description":"No description.","followers_count":random.uniform(500,2000),
                    "friends_count":random.randint(30,150)
                }
        X_raw, A_raw = create_feature_and_adj_matrix(
            combined_nodes_raw, combined_edges_raw, agent_data_dict_raw, feature_dim=50
        )
        np.savez_compressed(
            os.path.join(news_folder, f"{news_id}_combined_raw.npz"),
            nodes=combined_nodes_raw, edges=combined_edges_raw, X=X_raw, A=A_raw
        )

        # 3) 合并去重
        combined_nodes_dedup, combined_edges_dedup = combine_graphs(graph_info_list)
        agent_data_dict_dedup = {
            ag: agent_data_dict_raw.get(ag, {
                "agent_job":"Unknown","agent_traits":"Unknown",
                "description":"No description.","followers_count":random.uniform(500,2000),
                "friends_count":random.randint(30,150)
            })
            for ag in combined_nodes_dedup
        }
        X_dedup, A_dedup = create_feature_and_adj_matrix(
            combined_nodes_dedup, combined_edges_dedup, agent_data_dict_dedup, feature_dim=50
        )
        np.savez_compressed(
            os.path.join(news_folder, f"{news_id}_combined_dedup.npz"),
            nodes=combined_nodes_dedup, edges=combined_edges_dedup, X=X_dedup, A=A_dedup
        )

        # 4) 全局特征
        graph_feature = extract_graph_feature(X_dedup)
        np.save(os.path.join(news_folder, f"{news_id}_graph_feature.npy"), graph_feature)

        return news_id, graph_feature

    except SkipNews as e:
        print(f"跳过新闻 {e.news_id} （更新反馈出错）")
        return None, None
    except Exception as e:
        # 其他未知错误也记录并跳过
        with open(error_txt_path, 'a', encoding='utf-8') as ef:
            ef.write(f"{news_id}\n")
        print(f"处理新闻 {news_id} 时出现异常，已跳过")
        return None, None

def worker():
    while True:
        try:
            news_item = news_queue.get_nowait()
        except queue.Empty:
            break

        news_id, graph_feature = process_single_news(news_item)
        if graph_feature is not None:
            with data_lock:
                news_graph_features[news_id] = graph_feature

        news_queue.task_done()
        torch.cuda.empty_cache()
        gc.collect()

# 启动线程
num_threads = 28
threads = []
for _ in range(num_threads):
    t = threading.Thread(target=worker)
    t.start()
    threads.append(t)

# 等待所有完成
for t in threads:
    t.join()

print(f"\n所有新闻处理完成，共 {len(news_graph_features)} 条")