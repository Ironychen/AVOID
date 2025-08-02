import os
import json
import torch
import numpy as np

from transformers import BertTokenizer, BertModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)

bert_model.to(device)
bert_model.eval()

def get_agent_feature_bert(agent_data, feature_dim=50):
    """
    使用 BERT 根据 agent_data 中的文字属性生成描述性文本，并生成特征向量。
    若输出维度不等于 feature_dim，则进行线性映射。
    """
    description = f"{agent_data.get('agent_job', 'Unknown')} - {agent_data.get('agent_traits', 'Unknown')} - {agent_data.get('description', 'No description')}"
    print("aaaaaaaaaa:", description)
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

def encode_all_agents(feature_dim=50):
    """
    遍历编号 0-99 的 agent，读取对应 JSON 文件，并利用 BERT 编码生成特征向量，
    最后返回形状为 (100, feature_dim) 的 tensor。
    """
    features = []
    for i in range(100):
        json_path = f"/Env_Rumor_gossip_persona/agent_{i}/agent_{i}.json"
        if not os.path.exists(json_path):
            print(f"文件不存在: {json_path}, 使用空数据")
            agent_data = {}
        else:
            with open(json_path, 'r', encoding='utf-8') as f:
                agent_data = json.load(f)
        # 编码得到特征向量
        feature_vec = get_agent_feature_bert(agent_data, feature_dim=feature_dim)
        features.append(feature_vec.unsqueeze(0))  # 保留 batch 维度

    # 拼接为 (100, feature_dim) 的 tensor
    all_features = torch.cat(features, dim=0)
    return all_features

# 示例调用：
if __name__ == '__main__':
    feature_dim = 50
    all_agent_features = encode_all_agents(feature_dim=feature_dim)
    print("所有 agent 的特征向量形状:", all_agent_features.shape)

    # 打印前 5 个特征向量
    print("前 5 个 agent 的特征向量:")
    print(all_agent_features[:5])

    # 打印随机一个特征向量
    random_index = torch.randint(0, all_agent_features.shape[0], (1,)).item()
    print(f"随机选择的 agent_{random_index} 的特征向量:")
    print(all_agent_features[random_index])

    # 保存方式2：转换为 numpy 数组后使用 np.save 保存为 .npy 文件
    np_save_path = "agent_features_gossip_persona.npy"
    np.save(np_save_path, all_agent_features.cpu().numpy())
    print(f"特征向量已保存为 numpy 格式：{np_save_path}")
