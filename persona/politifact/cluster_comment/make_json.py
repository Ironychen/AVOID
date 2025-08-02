import json
import re

input_path = '/usr/gao/gubincheng/article_rep/Agent/simplest_agent/cluster/gossipcop/clustered_comments/all_cluster_personas.json'
output_path = '/usr/gao/gubincheng/article_rep/Agent/simplest_agent/cluster/gossipcop/clustered_comments/all_cluster_personas_number.json'

with open(input_path, 'r', encoding='utf-8') as f:
    personas = json.load(f)

new_personas = []
for idx, item in enumerate(personas):
    persona = item['persona']
    # 只保留 This persona is a staunchly 后面的内容
    match = re.search(r'(?:\*\*Final Persona:\*\*|### Final Persona:)\s*\n*\s*(This persona is a staunchly.*)', persona, re.DOTALL)
    if match:
        persona = match.group(1)
    else:
        # 去掉前缀“**Final Persona:**  \n\n”或“### Final Persona:  \n\n”
        persona = re.sub(r'^(?:\*\*Final Persona:\*\*|### Final Persona:)\s*\n*\s*', '', persona)
    new_personas.append({
        "id": idx,
        "cluster": item.get("cluster", ""),
        "persona": persona.strip()
    })

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(new_personas, f, ensure_ascii=False, indent=2)

print(f"已处理并保存到 {output_path}")