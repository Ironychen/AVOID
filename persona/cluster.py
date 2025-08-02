import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


def load_data(tsv_path: str) -> pd.DataFrame:
    """
    加载 TSV 文件，包含列：clean_title, content, id, 2_way_label
    并将 title 与 content 拼接成新的 text 列
    """
    df = pd.read_csv(tsv_path, sep='\t', dtype={'clean_title': str, 'content': str, 'id': str})
    df['text'] = (df['clean_title'].fillna('') + ' ' + df['content'].fillna('')).astype(str)
    return df


def embed_texts(texts: pd.Series,
                model_name: str,
                batch_size: int = 16,
                device: str = None) -> np.ndarray:
    """
    使用 BERT pooler_output 作为句向量
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch_texts = [str(x) for x in texts.iloc[i:i+batch_size]]
            enc = tokenizer(batch_texts, padding=True, truncation=True,
                            max_length=512, return_tensors='pt')
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            embs = out.pooler_output.cpu().numpy()
            all_embs.append(embs)
    return np.vstack(all_embs)


def cluster_embeddings(embs: np.ndarray, n_clusters: int = 10) -> np.ndarray:
    """
    对向量进行层次聚类，返回每条文本的簇标签
    """
    clustering = AgglomerativeClustering(
    n_clusters=n_clusters
    )
    labels = clustering.fit_predict(embs)
    return labels


def allocate_budget(cluster_sizes, k):
    """
    根据簇大小和总预算 k，按升序优先照顾小簇，再均匀分配剩余预算
    返回与 cluster_sizes 顺序对应的分配列表
    """
    m = len(cluster_sizes)
    idx_sorted = sorted(range(m), key=lambda i: cluster_sizes[i])
    A = [0] * m
    B = k
    r = m
    for i in idx_sorted:
        q = B // r
        alloc = min(cluster_sizes[i], q)
        A[i] = alloc
        B -= alloc
        r -= 1
    while B > 0:
        for i in idx_sorted:
            if A[i] < cluster_sizes[i] and B > 0:
                A[i] += 1
                B -= 1
            if B == 0:
                break
    return A


def visualize_clusters(embs: np.ndarray, labels: np.ndarray, save_path='cluster_vis_gossip.png'):
    """
    使用 PCA 将高维 embeddings 降到 2 维，并绘制带簇标签的散点图
    """
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embs)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        coords[:, 0], coords[:, 1],
        c=labels,
        cmap='tab10',
        s=10,
        alpha=0.8
    )
    plt.legend(*scatter.legend_elements(), title="Cluster", loc="upper right")
    plt.title("PCA Visualization of Clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def main():
    # 参数配置
    tsv_path = '/usr/gao/gubincheng/article_rep/Agent/simplest_agent/cluster/gossipcop/news_gossip_8_intersection.tsv'
    model_name = '/usr/gao/gubincheng/article_rep/ENDEF-SIGIR2022/ENDEF-SIGIR2022-main/bert-base-uncased'
    n_clusters = 12
    sampling_budget = 30  # 总采样预算，可根据需求调整

    # 1. 加载并拼接文本
    df = load_data(tsv_path)

    # 2. BERT 编码
    print("正在计算文本向量……")
    embeddings = embed_texts(df['text'], model_name=model_name, batch_size=16)

    # 3. 聚类
    print(f"正在进行 {n_clusters} 个簇的层次聚类……")
    df['cluster'] = cluster_embeddings(embeddings, n_clusters=n_clusters)

    # 只保存id和cluster到csv
    df[['id', 'cluster']].to_csv('news_id_cluster.csv', index=False)
    print("已保存聚类结果到 news_id_cluster.csv")
    
    # 5. 统计每簇大小，并分配预算
    grouped = df.groupby('cluster')['id'].apply(list)
    cluster_sizes = [len(grouped[cid]) for cid in range(n_clusters)]
    allocations = allocate_budget(cluster_sizes, sampling_budget)

    # 6. 按分配结果抽样
    sampled_ids = []
    for cid, alloc in enumerate(allocations):
        ids = grouped[cid]
        if alloc < len(ids):
            sampled = random.sample(ids, alloc)
        else:
            sampled = ids
        sampled_ids.extend(sampled)
        print(f"Cluster {cid}: size={len(ids)}, allocated={alloc}")

    # 7. 保存抽样结果
    sample_df = df[df['id'].isin(sampled_ids)][['id', 'cluster']]
    sample_df.to_csv('sampled_ids_gossip.csv', index=False)
    print(f"已保存抽样结果 sampled_ids.csv，共 {len(sampled_ids)} 条记录。")

    # 8. 可视化聚类
    print("正在绘制聚类可视化图……")
    visualize_clusters(embeddings, df['cluster'].values)

if __name__ == "__main__":
    main()
