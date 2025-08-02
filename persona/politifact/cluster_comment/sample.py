import pandas as pd
import numpy as np
import random
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from typing import List
import torch
from tqdm import tqdm
import math
from sklearn.decomposition import PCA

# --------------------------
# 配置：输入和输出文件路径，以及采样参数
INPUT_TSV = '/usr/gao/gubincheng/article_rep/Agent/simplest_agent/cluster/cluster_comment/clu7/cluster_7_with_clusters.tsv'  # 输入文件，需包含 'text' 和 'cluster' 列
OUTPUT_TSV = 'input_sampled.tsv'  # 输出采样结果，保留原列
OUTPUT_IMG = 'sampled_pca.png'   # PCA 可视化图

MODEL_NAME = "/usr/gao/gubincheng/article_rep/ENDEF-SIGIR2022/ENDEF-SIGIR2022-main/bert-base-uncased"
SAMPLE_RATIO = 0.1           # 总采样比例，针对所有簇合计
ALPHA = 1.03                 # 算法权重，典型性 vs 多样性
BATCH_SIZE = 16              # BERT 编码批量大小

# 设备配置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_data(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep='\t', dtype=str)
    if 'text' not in df.columns or 'cluster' not in df.columns:
        raise KeyError("输入文件必须包含 'text' 和 'cluster' 两列")
    df['text'] = df['text'].fillna('').astype(str)
    return df


def embed_texts(texts: pd.Series,
                model_name: str,
                batch_size: int = BATCH_SIZE,
                device: str = DEVICE) -> torch.Tensor:
    """
    使用 BERT pooler_output 生成文本嵌入，输出 Tensor 到 GPU 或 CPU
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    emb_list = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts.iloc[i:i+batch_size].tolist()
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=512, return_tensors='pt')
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            emb_list.append(out.pooler_output)
    return torch.cat(emb_list, dim=0)  # [N, D]


def get_allocation(cluster_sizes: List[int], total_budget: int) -> List[int]:
    """
    3.2 采样预算分配：按簇大小升序，先让小簇满配，再均匀分配剩余预算
    返回按原簇顺序的预算列表
    """
    m = len(cluster_sizes)
    # 确保预算至少为簇数
    if total_budget < m:
        total_budget = m
    sizes = np.array(cluster_sizes)
    idx_sorted = np.argsort(sizes)
    alloc = [0] * m
    B = total_budget
    r = m
    # 第一轮分配
    for i in idx_sorted:
        q = B // r
        take = min(int(sizes[i]), q)
        alloc[i] = take
        B -= take
        r -= 1
    # 分配剩余
    for i in idx_sorted:
        if B <= 0:
            break
        if alloc[i] < sizes[i]:
            alloc[i] += 1
            B -= 1
    return alloc


def select_samples(points: torch.Tensor,
                   alpha: float,
                   num_samples: int,
                   device: str = DEVICE) -> List[int]:
    """
    3.3 簇内子序列抽样，基于“典型性 + 多样性”的贪心算法
    返回本簇选中点的局部索引列表
    """
    center = points.mean(dim=0, keepdim=True)
    wp = alpha ** (-10)
    wd = 1 - wp
    selected_idx = []
    selected_pts = torch.zeros((0, points.size(1)), device=device)
    for _ in range(num_samples):
        # 计算典型性
        d_cent = torch.norm(points - center, dim=1)
        gp = wp / (1 + d_cent)
        # 计算多样性
        if selected_pts.size(0) == 0:
            gd = torch.zeros_like(gp)
        else:
            dist_matrix = torch.cdist(points, selected_pts, p=2)
            avg_dist = dist_matrix.mean(dim=1)
            gd = wd * avg_dist
        scores = gp + gd
        scores[selected_idx] = -float('inf')
        j = int(torch.argmax(scores).item())
        selected_idx.append(j)
        selected_pts = torch.cat([selected_pts, points[j:j+1]], dim=0)
    return selected_idx


def main():
    # 1. 加载数据 & 文本嵌入
    df = load_data(INPUT_TSV)
    print(f"加载 {len(df)} 条记录，开始 BERT 嵌入（GPU）……")
    embeddings = embed_texts(df['text'], MODEL_NAME)  # Tensor[N, D]

    # 2. 计算总预算 & 每簇分配
    total = len(df)
    total_budget = math.ceil(total * SAMPLE_RATIO)
    cluster_ids = sorted(df['cluster'].unique(), key=lambda x: int(x))
    cluster_sizes = [int((df['cluster']==cid).sum()) for cid in cluster_ids]
    allocations = get_allocation(cluster_sizes, total_budget)
    print(f"总评论数={total}, 总预算={total_budget}")
    for cid, size, alloc in zip(cluster_ids, cluster_sizes, allocations):
        print(f"簇 {cid}: size={size}, alloc={alloc}")

    # 3. 各簇抽样
    selected_global = []
    for cid, alloc in zip(cluster_ids, allocations):
        idxs = df.index[df['cluster']==cid].to_numpy()
        pts = embeddings[idxs]
        k = alloc
        local_idxs = select_samples(pts, ALPHA, k)
        sel = idxs[local_idxs]
        selected_global.extend(sel.tolist())

    # 4. 保存结果
    df_selected = df.loc[selected_global].reset_index(drop=True)
    df_selected.to_csv(OUTPUT_TSV, sep='\t', index=False)
    print(f"已保存采样结果到 {OUTPUT_TSV}，共 {len(df_selected)} 条。")

    # 5. PCA 可视化
    coords = embeddings.cpu().numpy()
    coords2 = PCA(n_components=2, random_state=42).fit_transform(coords)
    plt.figure(figsize=(8,6))
    plt.scatter(coords2[:,0], coords2[:,1], c='lightgray', s=10, alpha=0.5)
    sel2 = coords2[selected_global]
    plt.scatter(sel2[:,0], sel2[:,1], c='red', s=30, marker='x', label='Selected')
    plt.title('Representative Samples PCA')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300)
    plt.show()
    print(f"PCA 保存到 {OUTPUT_IMG}")

if __name__ == "__main__":
    main()
