import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool, global_max_pool, SAGEConv
import torch.nn as nn
import os

# --------------------------
# 定义 GCN 模型
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, pool_ratio=0.8):
        super(GCN, self).__init__()
        # self.conv1 = GCNConv(in_channels, hidden_channels)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.pool = TopKPooling(hidden_channels, ratio=pool_ratio)
        # 固定线性层: 将 2*hidden_channels 压缩到 50 维
        self.linear = nn.Linear(hidden_channels * 2, 50)
        for param in self.linear.parameters():
            param.requires_grad = False
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x, edge_index, batch, perm, score, _ = self.pool(x, edge_index, batch=batch)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        x = self.linear(x)
        return x  # 输出形状 (batch, 50)

# --------------------------
# 针对每条新闻处理，返回形状为 (50,) 的特征向量
def process_news(id_str, model, device):
    # 构造 npz 文件路径（假设该路径格式用于所有新闻）
    npz_file = f"/usr/gao/gubincheng/article_rep/Agent/simplest_agent/merged1_gcn_data_img_gossip_8/{id_str}/new_{id_str}_combined_dedup.npz"
    if not os.path.exists(npz_file):
        print(f"文件不存在: {npz_file}")
        return None
    data_npz = np.load(npz_file, allow_pickle=True)
    X = data_npz['X']  # 节点特征矩阵，形状： (num_nodes, 50)
    A = data_npz['A']  # 邻接矩阵，形状： (num_nodes, num_nodes)
    x = torch.tensor(X, dtype=torch.float).to(device)
    rows, cols = np.nonzero(A)
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long).to(device)
    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(x.size(0), dtype=torch.long).to(device)
    
    with torch.no_grad():
        graph_feature = model(data)  # 形状 (1, 50)
    return graph_feature.squeeze(0)  # shape: (50,)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 假设输入特征维度为50，隐藏层设为64
    model = GCN(in_channels=50, hidden_channels=64, pool_ratio=0.8).to(device)
    model.eval()
    
    features = []
    ids_file = "/usr/gao/gubincheng/article_rep/Agent/simplest_agent/feature_enr/extracted_ids_gossip10.txt"
    with open(ids_file, 'r') as f:
        id_list = [line.strip() for line in f if line.strip()]
    
    for id_str in id_list:
        print(f"Processing {id_str}...")
        feature = process_news(id_str, model, device)
        if feature is not None:
            features.append(feature.cpu().numpy())
            
    if not features:
        print("没有生成任何特征，请检查路径及数据文件。")
        return
    
    # 堆叠所有新闻的特征，最终 shape: (286, 50)
    features_np = np.stack(features, axis=0)
    print("最终特征矩阵形状:", features_np.shape)
    
    output_file = "/usr/gao/gubincheng/article_rep/Agent/simplest_agent/feature_enr/gcn_features_gossip_img_8sage_mer1.npy"
    np.save(output_file, features_np)
    print(f"特征已保存到 {output_file}")

if __name__ == '__main__':
    main()