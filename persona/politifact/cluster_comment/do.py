import json
import os

# 路径配置
persona_json = '/usr/gao/gubincheng/article_rep/Agent/simplest_agent/cluster/gossipcop/clustered_comments/all_cluster_personas_number.json'
agent_root = '/usr/gao/gubincheng/article_rep/Agent/data/pro/Env_Rumor_gossip_persona1'

# 读取persona列表
with open(persona_json, 'r', encoding='utf-8') as f:
    personas = json.load(f)

assert len(personas) == 100, "persona数量不是100，请检查！"

for i, persona_obj in enumerate(personas):
    agent_dir = os.path.join(agent_root, f'agent_{i}')
    agent_json = os.path.join(agent_dir, f'agent_{i}.json')
    if not os.path.isfile(agent_json):
        print(f"未找到: {agent_json}，跳过")
        continue
    # 读取agent文件
    with open(agent_json, 'r', encoding='utf-8') as f:
        agent_data = json.load(f)
    # 替换description
    agent_data['description'] = persona_obj['persona']
    # 保存
    with open(agent_json, 'w', encoding='utf-8') as f:
        json.dump(agent_data, f, ensure_ascii=False, indent=2)
    print(f"已更新: {agent_json}")

print("全部替换完成。")