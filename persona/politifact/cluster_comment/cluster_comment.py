import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import Tuple
import torch
from tqdm import tqdm

# --------------------------
# 只需修改这里的簇编号，即可处理对应 cluster_X.tsv
CLUSTER_ID = 0
# --------------------------

# 使用本地或远程的 BERT 模型路径
MODEL_NAME = "/usr/gao/gubincheng/article_rep/ENDEF-SIGIR2022/ENDEF-SIGIR2022-main/bert-base-uncased"
K_MIN = 3
K_MAX = 5


def load_comments(tsv_path: str) -> Tuple[pd.Series, pd.DataFrame]:
    df = pd.read_csv(tsv_path, sep='\t', dtype=str)
    if 'text' not in df.columns:
        raise KeyError(f"{tsv_path} 中未找到 'text' 列")
    df['text'] = df['text'].fillna('').astype(str)
    return df['text'], df


def embed_texts(texts: pd.Series,
                model_name: str,
                batch_size: int = 16,
                device: str = None) -> np.ndarray:
    """
    使用 BERT pooler_output 作为文本嵌入
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts.iloc[i:i+batch_size].tolist()
            enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            emb = out.pooler_output.cpu().numpy()
            all_embs.append(emb)
    return np.vstack(all_embs)


def find_best_k(embs: np.ndarray, k_min: int, k_max: int) -> int:
    best_k, best_score = k_min, -1.0
    for k in range(k_min, min(k_max, embs.shape[0]) + 1):
        labels = KMeans(n_clusters=k, random_state=42).fit_predict(embs)
        score = silhouette_score(embs, labels)
        print(f"k={k:2d}, silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score = k, score
    print(f"最佳簇数: k={best_k} (silhouette={best_score:.4f})\n")
    return best_k


def process_cluster(cluster_id: int):
    in_tsv = f'cluster_{cluster_id}.tsv'
    out_tsv = f'cluster_{cluster_id}_with_clusters.tsv'
    out_img = f'cluster_{cluster_id}_pca.png'

    print(f"\n=== 处理 cluster_{cluster_id} ===")
    # 1. Load
    texts, df = load_comments(in_tsv)

    # 2. Embed
    print("生成文本嵌入中…")
    embeddings = embed_texts(texts, MODEL_NAME, batch_size=16)

    # 3. Find best k
    print("搜索最优簇数…")
    best_k = find_best_k(embeddings, K_MIN, K_MAX)

    # 4. Cluster
    print(f"进行 KMeans(k={best_k}) 聚类…")
    df['cluster'] = KMeans(n_clusters=best_k, random_state=42).fit_predict(embeddings)

    # 5. Save clustered TSV
    df.to_csv(out_tsv, sep='\t', index=False)
    print(f"已保存聚类结果到 {out_tsv}")

    # 6. Visualize with PCA
    print("降维并绘制 PCA 可视化图…")
    coords = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        coords[:, 0], coords[:, 1],
        c=df['cluster'], cmap='tab10', s=15, alpha=0.8
    )
    plt.legend(*sc.legend_elements(), title="Cluster", loc="best")
    plt.title(f"cluster_{cluster_id} PCA Visualization (k={best_k})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_img, dpi=300)
    plt.show()
    print(f"已保存可视化图到 {out_img}")

if __name__ == "__main__":
    process_cluster(CLUSTER_ID)
