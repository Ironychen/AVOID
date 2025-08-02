import os
import pandas as pd

# 配置路径
clustered_ids_path = '/usr/gao/gubincheng/article_rep/Agent/simplest_agent/cluster/gossipcop/news_id_cluster.csv'
comments_path      = '/usr/gao/gubincheng/article_rep/Agent/simplest_agent/cluster/gossipcop/comments_gossip_all.tsv'
output_dir         = '/usr/gao/gubincheng/article_rep/Agent/simplest_agent/cluster/gossipcop/clustered_comments'

# 1. 读取新闻 -> 簇 映射
clusters_df = pd.read_csv(clustered_ids_path, dtype={'id': str})
# 确保列名为 id, cluster
if 'id' not in clusters_df.columns or 'cluster' not in clusters_df.columns:
    raise ValueError("clustered_ids1.csv 中应包含 'id' 和 'cluster' 两列")

# 2. 读取所有评论
comments_df = pd.read_csv(comments_path, sep='\t', dtype=str)

# 假设评论文件中有一列 'news_id' 与 clustered_ids 中的 'id' 对应
if 'news_id' not in comments_df.columns:
    raise ValueError("politic_comments_all.tsv 中未找到 'news_id' 列")

# 3. 合并评论与簇信息
merged = comments_df.merge(
    clusters_df.rename(columns={'id': 'news_id'}),
    on='news_id',
    how='inner'      # 只保留那些在 clustered_ids1.csv 中的 news_id
)

# 4. 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 5. 按簇分组并分别写入文件
for cluster_label, group in merged.groupby('cluster'):
    out_path = os.path.join(output_dir, f'cluster_{cluster_label}.tsv')
    # 将该簇的所有评论保存，保留所有原始列
    group.to_csv(out_path, sep='\t', index=False)
    print(f"Cluster {cluster_label}: {len(group)} 条评论，已保存到 {out_path}")

print("✅ 已完成按簇分割评论并保存到目录：", output_dir)
