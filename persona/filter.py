
import os

# —— 配置区 —— #
IDS_PATH    = "path/to/your/intersection_ids.txt"
INPUT_TSV   = "path/to/your/comments.tsv"
OUTPUT_TSV  = "path/to/your/news_filtered.tsv"

def load_ids(path: str) -> set:
    """读取每行一个 ID，返回一个 set。"""
    with open(path, "r", encoding="utf-8") as f:
        return { line.strip() for line in f if line.strip() }

def filter_tsv_by_ids(input_tsv: str, ids: set, output_tsv: str):
    """
    读取 input_tsv，动态找到 header 中 'id' 列的下标，然后把 id 在 ids 集合里的行写入 output_tsv。
    """
    with open(input_tsv, "r", encoding="utf-8") as fin, \
         open(output_tsv, "w", encoding="utf-8") as fout:

        # 读取并写出表头
        header = fin.readline().rstrip("\n")
        fout.write(header + "\n")

        # 找到 'id' 列的索引
        cols = header.split("\t")
        try:
            id_idx = cols.index("id")
        except ValueError:
            raise RuntimeError(f"在表头中找不到 'id' 列：{header}")

        # 逐行检查并写入
        kept = 0
        for line in fin:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > id_idx and parts[id_idx] in ids:
                fout.write(line)
                kept += 1

    print(f"✅ 完成过滤，共保留 {kept} 条记录，输出到：{output_tsv}")

if __name__ == "__main__":
    ids = load_ids(IDS_PATH)
    print(f"加载到 {len(ids)} 个交集 ID")
    filter_tsv_by_ids(INPUT_TSV, ids, OUTPUT_TSV)
