import os
import pandas as pd

def read_tsv_fallback(path: str, **kwargs) -> pd.DataFrame:
    """
    先尝试 utf-8，再尝试 latin1，失败则抛出。
    """
    try:
        return pd.read_csv(path, encoding='utf-8', **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='latin1', **kwargs)

def is_english(text: str) -> bool:
    """
    检查 text 是否全为 ASCII 字符（近似判断英文）。
    """
    return all(ord(c) < 128 for c in text)

def extract_comments(input_root: str, output_path: str):

    records = []
    for dirpath, _, files in os.walk(input_root):
        if 'tweets_retweets_comments_news.tsv' not in files:
            continue

        file_path = os.path.join(dirpath, 'tweets_retweets_comments_news.tsv')
        rel = os.path.relpath(file_path, input_root)
        news_id = rel.split(os.sep)[0]

        df = read_tsv_fallback(
            file_path,
            sep='\t',
            dtype=str,
            usecols=['text', 'user_id', 'type']
        )
        if df is None or df.empty:
            continue

        df['type'] = df['type'].str.strip().str.lower()
        comments = df[df['type'] == 'comment'].copy()
        if comments.empty:
            continue

        comments['text'] = comments['text'].fillna('').astype(str)

        comments = comments[~comments['text'].str.contains('@', na=False)]
        if comments.empty:
            continue

        comments = comments[comments['text'].str.strip() != '']
        if comments.empty:
            continue

        comments = comments[comments['text'].apply(is_english)]
        if comments.empty:
            continue

        comments = comments[['text', 'user_id']].copy()
        comments['news_id'] = news_id
        records.append(comments)

    if records:
        result = pd.concat(records, ignore_index=True)
    else:
        result = pd.DataFrame(columns=['text', 'user_id', 'news_id'])

    result.to_csv(output_path, sep='\t', index=False)
    print(f"提取完成，共 {len(result)} 条符合条件的评论，已保存至 {output_path}")

if __name__ == "__main__":
    input_root = "path/to/your/input_root"
    output_path = "path/to/your/output_comments.tsv"
    extract_comments(input_root, output_path)
