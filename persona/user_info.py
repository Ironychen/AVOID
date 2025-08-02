import os
import pandas as pd

# 输入/输出路径配置
comments_path = 'path/to/your/comments.tsv'
output_dir    = 'path/to/your/output_dir'
counts_path   = 'path/to/your/user_counts.tsv'

def main():
    # 1. 读取所有评论
    df = pd.read_csv(comments_path, sep='\t', dtype=str)
    if 'user_id' not in df.columns:
        raise KeyError("输入文件中未找到 'user_id' 列")

    # 2. 按 user_id 分组，并统计每人的评论数
    counts = df.groupby('user_id').size().reset_index(name='comment_count')
    counts.to_csv(counts_path, index=False)
    print(f"{counts_path}")
    print(f"   共 {len(counts)} 位用户，评论总数 {len(df)}，")
    print(f"   平均每人 {counts['comment_count'].mean():.2f} 条，"
          f"中位数 {counts['comment_count'].median():.0f} 条，"
          f"最多 {counts['comment_count'].max()} 条，"
          f"最少 {counts['comment_count'].min()} 条。")

    # 3. 为每个用户创建单独文件
    os.makedirs(output_dir, exist_ok=True)
    for user_id, group in df.groupby('user_id'):
        # 安全化文件名
        safe_id = user_id.replace('/', '_').replace('\\', '_')
        out_path = os.path.join(output_dir, f'user_{safe_id}.tsv')
        group.to_csv(out_path, sep='\t', index=False)
    print(f"{output_dir}")

if __name__ == "__main__":
    main()
