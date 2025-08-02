import os
import json
from zhipuai import ZhipuAI

# 设置你的 ZhipuAI API Key
key = "653d34becfd62e46c1a02e7642e43c02.eqLfDKBj4GJdnOgH"

def get_content_response(prompt):
    client = ZhipuAI(api_key=key)
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[
            {"role": "system", "content": "You are an active user on social media."},
            {"role": "user", "content": prompt}
        ],
        stream=False
    )
    return response.choices[0].message.content

def summarize_short_memory_to_long_memory(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if "short_memory" not in data:
        print(f"No short_memory in {json_path}")
        return

    # 构造 prompt
    memory_entries = data["short_memory"]
    prompt = "Here are several pieces of short-term memory from social media interactions:\n"
    for m in memory_entries:
        prompt += f"- Summary: {m['summarize']}\n  Comment: {m['comment']}\n\n"
    prompt += "Based on these, summarize 2 or 3 key long-term memories or hot social events this user might have formed."

    # 调用 API 获取 long memory 总结
    try:
        result = get_content_response(prompt).strip()
        # 尝试用换行拆分
        long_memory = [item.strip('- ').strip() for item in result.split('\n') if item.strip()]
        data["long_memory"] = long_memory
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Updated long_memory in: {json_path}")
    except Exception as e:
        print(f"Failed to process {json_path}: {e}")

def process_all_agents(base_path="/usr/gao/gubincheng/article_rep/Agent/data/pro/Env_Rumor_politic_persona1"):
    for agent_dir in os.listdir(base_path):
        agent_path = os.path.join(base_path, agent_dir)
        json_file = os.path.join(agent_path, f"{agent_dir}.json")
        if os.path.isfile(json_file):
            summarize_short_memory_to_long_memory(json_file)

# 调用入口
process_all_agents()
