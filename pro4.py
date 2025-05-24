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

# ------------------------------ 全局变量与模型初始化 ------------------------------
key = "653d34becfd62e46c1a02e7642e43c02.eqLfDKBjGJdnOgH"

# 文件和数据锁，确保线程安全
data_lock = threading.Lock()
file_lock = threading.Lock()

short_mem = []
long_mem = []
news_mem = []  # 用于存储新闻及其行为

# 指定设备为第二张 GPU（cuda:1device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 使用 BERT 替换原先的编码器
bert_tokenizer = BertTokenizer.from_pretrained("/usr/gao/gubincheng/article_rep/ENDEF-SIGIR2022/ENDEF-SIGIR2022-main/bert-base-uncased")
bert_model = BertModel.from_pretrained("/usr/gao/gubincheng/article_rep/ENDEF-SIGIR2022/ENDEF-SIGIR2022-main/bert-base-uncased").to(device)
bert_model.eval()

client = ZhipuAI(api_key=key)

# 读取新闻数据
news_path = '/usr/gao/gubincheng/article_rep/Agent/data/gossip/extracted_news.json'
with open(news_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 全局变量（略agent_name = ""
name = ""
friend_list = []      
agent_comment = ""
comment_stance = ""
comment_attitude = ""
profiles_data = {}
agent_path = ""
news_item = {}
content = ""
news_id = ""
short_content = ""
content_summarize = ""
content_opinion = ""
img_description = ""
img_interpretation = ""
sm_update_sum = ""
comment_summarize = ""

# ------------------------------ 日志设置 ------------------------------
logging.basicConfig(
    filename='experiment_results_gossip418.log',
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
    global random_agent, profiles_data, agent_name, name, friend_list, agent_comment, agent_path
    random_agent = str(agent_id)
    agent_path = f'/usr/gao/gubincheng/article_rep/Agent/data/pro/Env_Rumor_gossip_content_img/agent_{random_agent}/agent_{random_agent}.json'
    try:
        with open(agent_path, 'r', encoding='utf-8') as file:
            profiles_data = json.load(file)
    except Exception as e:
        print(f"加载 agent {random_agent} 配置文件出错: {e}")
        return False
    agent_name = profiles_data['agent_name']
    name = agent_name
    friend_list = profiles_data['friends']
    agent_comment = ""
    return True

def load_random_news():
    global news_item, content, news_id, short_content
    news_item = random.choice(data)
    content = news_item['content']
    news_id = news_item['id']
    words = content.split()[:600]
    short_content = ' '.join(words)
    print(f"选中的新闻ID {news_id}")

def process_news(news):
    global news_item, content, news_id, short_content
    news_item = news
    content = news['content']
    news_id = news['id']
    words = content.split()[:600]
    short_content = ' '.join(words)
    print(f"Processing inherited news, ID: {news_id}")
    update_news_feedback()
    result = agent_func(short_content)
    return result

def build_comment_prompt():
    if "comment" in news_item and news_item["comment"]:
        summaries = []
        for c in news_item["comment"]:
            author = c.get("name", "Unknown")
            comment_text = c.get("comment", "").strip()
            stance = c.get("stance", "unknown").strip()
            summaries.append(f"Author: {author}, Comment: {comment_text}, Stance: {stance}")
        comments_summary = " ; ".join(summaries)
        return f"Summaries of Existing Comments: {comments_summary} return with an paragraph under 150 words."
    else:
        return "No existing comments."

def update_news_feedback():
    global content_summarize, content_opinion, comment_summarize, sm_update_sum, img_description, img_interpretation
    prompt_content = build_content_prompt_gossip(news_id)
    content_response = get_content_response(prompt_content)
    json_pattern = re.search(r'\{[\s\S]*?\}', content_response)
    if json_pattern:
        json_str = json_pattern.group(0)
        try:
            data_content_res = json.loads(json_str)
        except json.JSONDecodeError:
            print("Error decoding content JSON, using default values.")
            data_content_res = {}
    else:
        print("无法解析文本反馈JSON, 使用默认空字符串")
        data_content_res = {}
    content_summarize = data_content_res.get("summarize", "")
    content_opinion = data_content_res.get("opinion", "")

    # 获取图片反馈
    prompt_img, img_path = build_image_prompt(news_id)
    img_response = get_image_response(prompt_img, img_path)
    json_pattern_img = re.search(r'\{[\s\S]*?\}', img_response)
    if json_pattern_img:
        json_img = json_pattern_img.group(0)
        try:
            data_img_res = json.loads(json_img)
        except json.JSONDecodeError:
            print("Error decoding image JSON, using default values.")
            data_img_res = {}
    else:
        print("无法解析图片反馈JSON, 使用默认空字符串")
        data_img_res = {}
    img_description = data_img_res.get("description", "")
    img_interpretation = data_img_res.get("interpretation", "")

    friends_info_prompt = f"""
    After reading the following news article and comments, your feedback is as follows: {content_summarize}
    Your opinion on this matter is: {content_opinion}
    The description of the image in the news is: {img_description} 
    Your analysis of the image is: {img_interpretation}

    According to the above details, please summarize the event information, opinions and image information of the news respectively. And organize it into a paragraph, no more than 150 words
    """
    print("friends_info_prompt", friends_info_prompt)
    sm_update_sum = get_content_response(friends_info_prompt)
    print("sm_update_sum", sm_update_sum)

def generate_profile_prompt():
    print("agent " + random_agent + " act")
    simplified_summary, retriever_summary = update_long_memories()
    memories_str = f"Simplified Summary: {simplified_summary} ; Retriever Summary: {retriever_summary}"
    prompt = f"""
    You are a user in the social network. Don't let the length of the news text influence your judgment.
    Hi, {profiles_data['agent_name']}, you are a {profiles_data['agent_job']} known for being {profiles_data['agent_traits']}. Please follow the instructions below.
    You are active on a social network, receiving and sending posts.
    In terms of memory, you have been keeping track of short-term experiences such as:
    {memories_str}
    """
    return profiles_data['agent_name'], prompt

def build_prompt(q):
    prompt = f"The news article is {q}"
    prompt += """
    1. Please reason step by step.

    The thinking phase: Jointly analyze the textual and visual content to determine whether the entertainment news is credible, exaggerated, or fabricated, following **GossipCop-style truth standards** while considering typical patterns in entertainment media.

    📌 In entertainment news, **short or catchy titles** are common and do **not necessarily** indicate misinformation. In fact, such titles are often crafted for virality and **may still reflect true events**. However, unverified claims, anonymous sources, or mismatched photos are stronger indicators of potential misinformation.

    You must assess both aspects:

    - Textual Authenticity:
        - Does the title or body use overly dramatic, emotionally charged, or clickbait language without factual basis?
        - Are the claims backed by credible sources (e.g., named interviews, official statements), or do they rely on vague attributions like “a close friend said…”?
        - Are any celebrity quotes verifiable and clearly attributed?

    - Multimodal Consistency:
        - Does the accompanying image support or contradict the event or relationship described?
        - Is the image possibly taken out of context (e.g., from another time or event)?
        - Is the image used to suggest something dramatic (e.g., a romantic link or scandal) that the text does not clearly confirm?

    ⚠️ Even when the article’s **text appears plausible**, an **out-of-context image** or **implied but unsupported claim** can make the news misleading under entertainment fact-checking standards.

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

def update_shot_memories():
    global sm_update_sum, agent_name, agent_comment, comment_stance
    new_entry = {"name": agent_name, "summarize": sm_update_sum, "comment": agent_comment, "stance": comment_stance}
    for friend in friend_list:
        friend_agent_path = f'/usr/gao/gubincheng/article_rep/Agent/data/pro/Env_Rumor_gossip_content_img/agent_{friend}/agent_{friend}.json'
        try:
            with open(friend_agent_path, 'r', encoding='utf-8') as file:
                friend_data = json.load(file)
        except Exception as e:
            print(f"读取好友 {friend} 的文件时出错: {e}")
            continue
        if "short_memory" in friend_data and isinstance(friend_data["short_memory"], list):
            friend_data["short_memory"].append(new_entry)
            if len(friend_data["short_memory"]) > 8:
                friend_data["short_memory"] = friend_data["short_memory"][-8:]
        else:
            friend_data["short_memory"] = [new_entry]
        try:
            with open(friend_agent_path, 'w', encoding='utf-8') as file:
                json.dump(friend_data, file, ensure_ascii=False, indent=4)
            print(f"成功更新好友 {friend} 的短期记忆")
        except Exception as e:
            print(f"写入好友 {friend} 的文件出 {e}")

def update_long_memories():
    global agent_path
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
    print(f"🔍 最相似的记忆索 {most_similar_index}")
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
        print(f"🔍 简化之后的记忆内容: {simplified_summary}")
        print(f"🔍 强化之后的记忆内 {retriever_summary}")
        return simplified_summary, retriever_summary
    else:
        print("⚠️ 索引超出 short_memory 的范围")
        simplified_summary = "no relevant short-term memory."
        retriever_summary = "no relevant short-term memory."
        return simplified_summary, retriever_summary

def update_memories(action=None, params=None):
    update_shot_memories()

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

def agent_func(q):
    global agent_comment, comment_stance
    agent_name, prompt_profile = generate_profile_prompt()
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
            print(f"Agent {agent_name} 的评 {agent_comment}")
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
            f"News text: {short_content[0:60]}\n"
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
        update_memories(action=request.get("action"), params=request.get("params"))
    else:
        update_memories()
    if tool_result:
        return f"Tool result: {tool_result}"
    if action_result:
        return f"Action result: {action_result}"
    return resp

def should_take_action():
    global agent_path
    agent_retriever = ShortMemoryRetriever(agent_path)
    result = agent_retriever.retrieve_similarity(agent_comment)
    print(f"🔍 评论与记忆的相似度结 {result}")
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
    description = f"{agent_data.get('agent_job', 'Unknown')} - {agent_data.get('agent_traits', 'Unknown')} - {agent_data.get('description', 'No description')}"
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
    """
    将多个传播图信息直接串联（不去重），返回原始节点列表和边列表    如果在处理过程中，发现边中有节点未在当前节点列表中，则打印警告信息并将其加入    """
    raw_nodes = []
    raw_edges = []
    offset = 0
    for idx, gi in enumerate(graph_info_list):
        # 确保节点均为字符        
        nodes = [str(x) for x in gi["nodes"]]
        # 边转换为字符串形式，假设边存储的也是agent id
        edges = [(str(src), str(dst)) for src, dst in gi["edges"]]
        # 构造从节点到索引的映射
        node_index_map = {node: i for i, node in enumerate(nodes)}
        # 检查边中是否存在节点未nodes        
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
        # 将当前图节点添加raw_nodes，并更新边的索引
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
def propagate_news(initial_agent_id, propagation_threshold=5, initial_news=None):
    global news_item, short_content, comment_stance, content, news_id
    if initial_news is None:
        load_random_news()
        initial_news = news_item
    else:
        news_item = initial_news
        content = news_item['content']
        news_id = news_item['id']
        words = content.split()[:500]
        short_content = ' '.join(words)
        print(f"Processing news ID: {news_id}")

    key_extract_prompt = build_keyword_extraction_prompt(short_content)
    key_extract_response = get_content_response(key_extract_prompt)
    print("key_extract_response", key_extract_response)
    input("Press Enter to continue...")
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
        if not load_agent(agent_id):
            print(f"无法加载 agent {agent_id}")
            continue
        print(f"\n========== Round {round_num}: Agent {agent_id} 开始传==========")
        news_item = inherited_news
        update_news_feedback()
        result = agent_func(short_content)
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
                if should_take_action():
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

def backup_all_agents_short_memory(agent_ids):
    backup = {}
    for agent_id in agent_ids:
        agent_file = f'/usr/gao/gubincheng/article_rep/Agent/data/pro/Env_Rumor_gossip_content_img/agent_{agent_id}/agent_{agent_id}.json'
        try:
            with open(agent_file, 'r', encoding='utf-8') as file:
                data_agent = json.load(file)
                backup[agent_id] = data_agent.get("short_memory", [])
        except Exception as e:
            print(f"读取 agent {agent_id} 文件时出 {e}")
    return backup

def restore_all_agents_short_memory(backup):
    for agent_id, short_memory in backup.items():
        agent_file = f'/usr/gao/gubincheng/article_rep/Agent/data/pro/Env_Rumor_gossip_content_img/agent_{agent_id}/agent_{agent_id}.json'
        try:
            with open(agent_file, 'r', encoding='utf-8') as file:
                data_agent = json.load(file)
            data_agent["short_memory"] = short_memory
            with open(agent_file, 'w', encoding='utf-8') as file:
                json.dump(data_agent, file, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"恢复 agent {agent_id} 文件时出 {e}")

# ------------------------------ 主流------------------------------
# 保存结果到文件夹 "gcn_data"
output_folder = "gcn_data_img_gossip111"
os.makedirs(output_folder, exist_ok=True)


news_graph_features = {}  # key: news_id, value: 保存图特征及各矩阵信
# 对于每条新闻，生成三个传播路径，并将各路径单独保存，再合并后保存结果
for news in data:
    news_id = news.get('id', 'unknown')
    # 创建以新闻id命名的子文件    
    news_folder = os.path.join(output_folder, news_id)
    os.makedirs(news_folder, exist_ok=True)
    
    print("\n====================================")
    print(f"开始处理新{news['id']}")
    fixed_agents = ["17", "66"]
    paths_info = []  # 保存2个独立路径的结果
    graph_info_list = []  # 保存2个传播路径的 graph_info
    for i in range(2):
        start_agent = fixed_agents[i % len(fixed_agents)]
        stats, gi = propagate_news(start_agent, propagation_threshold=5, initial_news=news)
        graph_info_list.append(gi)
        raw_nodes, raw_edges = combine_graphs_raw([gi])
        agent_data_dict = {}
        for agent in raw_nodes:
            if agent in ["17", "66"]: 
                if agent == "17":
                    agent_data_dict[agent] = {
                        "agent_job": "Paramedic",
                        "agent_traits": "Courageous, Compassionate",
                        "description": "Mechanical Engineer in a past life. Full time investor since 2007. Focused on deep value micro cap stocks selling below tangible book value.",
                        "followers_count": 13892.0,
                        "friends_count": 1023,
                        "verified": "True",
                        "favorites_count": "122368.0"
                    }
                elif agent == "66":
                    agent_data_dict[agent] = {
                    "agent_job": "Architect",
                    "agent_traits": "Visionary, Detailed",
                    "description": "Here are all my thoughts off tangent",
                    "followers_count": 936.0,
                    "friends_count": 1742,
                    "verified": "False",
                    "favorites_count": "18072.0"
                }
            else:
                agent_data_dict[agent] = {
                    "agent_job": "Unknown",
                    "agent_traits": "Unknown",
                    "description": "No description available.",
                    "followers_count": random.uniform(500, 2000),
                    "friends_count": random.randint(30, 150)
                }
        X_path, A_path = create_feature_and_adj_matrix(raw_nodes, raw_edges, agent_data_dict, feature_dim=50)
        path_info = {
            "raw_nodes": raw_nodes,
            "raw_edges": raw_edges,
            "X": X_path,
            "A": A_path
        }
        paths_info.append(path_info)
        # 保存每个单独路径
        np.savez_compressed(os.path.join(news_folder, f"{news_id}_path_{i+1}.npz"),
                            raw_nodes=raw_nodes, raw_edges=raw_edges, X=X_path, A=A_path)
    print("单独三个路径的结果：")
    for idx, path in enumerate(paths_info):
        print(f"路径 {idx+1}: 节点{len(path['raw_nodes'])}, X 形状 {path['X'].shape}, A 形状 {path['A'].shape}")
    # 合并三个路径（原始：未去重）
    combined_nodes_raw, combined_edges_raw = combine_graphs_raw(graph_info_list)
    # 合并三个路径（去重）
    combined_nodes_dedup, combined_edges_dedup = combine_graphs(graph_info_list)
    print("原始合并（未去重）节点数:", len(combined_nodes_raw), "边数:", len(combined_edges_raw))
    print("去重合并节点", len(combined_nodes_dedup), "边数:", len(combined_edges_dedup))
    # 构造矩阵（原始合并    
    agent_data_dict_raw = {}
    for agent in combined_nodes_raw:
        if agent in ["17", "66"]: 
            if agent == "17":
                agent_data_dict_raw[agent] = {
                    "agent_job": "Paramedic",
                    "agent_traits": "Courageous, Compassionate",
                    "description": "Mechanical Engineer in a past life. Full time investor since 2007. Focused on deep value micro cap stocks selling below tangible book value.",
                    "followers_count": 13892.0,
                    "friends_count": 1023,
                    "verified": "True",
                    "favorites_count": "122368.0"
                }
            
            elif agent == "66":
                agent_data_dict_raw[agent] = {
                    "agent_job": "Architect",
                    "agent_traits": "Visionary, Detailed",
                    "description": "Here are all my thoughts off tangent",
                    "followers_count": 936.0,
                    "friends_count": 1742,
                    "verified": "False",
                    "favorites_count": "18072.0"
                }
        else:
            agent_data_dict_raw[agent] = {
                "agent_job": "Unknown",
                "agent_traits": "Unknown",
                "description": "No description available.",
                "followers_count": random.uniform(500, 2000),
                "friends_count": random.randint(30, 150)
            }
    X_raw, A_raw = create_feature_and_adj_matrix(combined_nodes_raw, combined_edges_raw, agent_data_dict_raw, feature_dim=50)
    # 构造矩阵（去重合并    
    agent_data_dict_dedup = {}
    for agent in combined_nodes_dedup:
        if agent in ["17", "66"]: 
            if agent == "17":
                agent_data_dict_dedup[agent] = {
                    "agent_job": "Paramedic",
                    "agent_traits": "Courageous, Compassionate",
                    "description": "Mechanical Engineer in a past life. Full time investor since 2007. Focused on deep value micro cap stocks selling below tangible book value.",
                    "followers_count": 13892.0,
                    "friends_count": 1023,
                    "verified": "True",
                    "favorites_count": "122368.0"
                }
            elif agent == "66":
                agent_data_dict_dedup[agent] = {
                    "agent_job": "Architect",
                    "agent_traits": "Visionary, Detailed",
                    "description": "Here are all my thoughts off tangent",
                    "followers_count": 1362.0,
                    "friends_count": 7421,
                    "verified": "False",
                    "favorites_count": "18072.0"
                }
        else:
            agent_data_dict_dedup[agent] = {
                "agent_job": "Unknown",
                "agent_traits": "Unknown",
                "description": "No description available.",
                "followers_count": random.uniform(500, 2000),
                "friends_count": random.randint(30, 150)
            }
    X_dedup, A_dedup = create_feature_and_adj_matrix(combined_nodes_dedup, combined_edges_dedup, agent_data_dict_dedup, feature_dim=50)
    print("原始合并矩阵 X_raw 形状:", X_raw.shape, "A_raw 形状:", A_raw.shape)
    print("去重合并矩阵 X_dedup 形状:", X_dedup.shape, "A_dedup 形状:", A_dedup.shape)
    # 保存合并后的结果
    np.savez_compressed(os.path.join(news_folder, f"{news_id}_combined_raw.npz"),
                        nodes=combined_nodes_raw, edges=combined_edges_raw, X=X_raw, A=A_raw)
    np.savez_compressed(os.path.join(news_folder, f"{news_id}_combined_dedup.npz"),
                        nodes=combined_nodes_dedup, edges=combined_edges_dedup, X=X_dedup, A=A_dedup)
    # 提取图全局特征（这里采用去重后 X_dedup 的均值）
    graph_feature = extract_graph_feature(X_dedup)
    print("当前新闻图全局特征向量形状:", graph_feature.shape)
    np.save(os.path.join(news_folder, f"{news['id']}_graph_feature.npy"), graph_feature)
    news_graph_features[news['id']] = {
        "paths": paths_info,  # 三条独立路径的详细矩阵信        "combined_raw": {"nodes": combined_nodes_raw, "edges": combined_edges_raw, "X": X_raw, "A": A_raw},
        "combined_dedup": {"nodes": combined_nodes_dedup, "edges": combined_edges_dedup, "X": X_dedup, "A": A_dedup},
        "graph_feature": graph_feature
    }

    torch.cuda.empty_cache()
    gc.collect()

# 最后打印所有新闻的图全局特征记录
print("\n所有新闻的图特征记")
for nid, info in news_graph_features.items():
    print(f"新闻 {nid}: 图全局特征向量 {info['graph_feature']}")

