import os
import shutil

# 定义两个文件夹路径
folder1 = "/usr/gao/gubincheng/article_rep/Agent/simplest_agent/gcn_data_img_politic"
folder2 = "/usr/gao/gubincheng/article_rep/Agent/simplest_agent/gcn_data_politic_content"

# 获取两个文件夹中的子文件夹名称
subfolders_in_folder1 = set(os.listdir(folder1))
subfolders_in_folder2 = set(os.listdir(folder2))

# 找到只在第二个文件夹中出现的子文件夹
subfolders_only_in_folder2 = subfolders_in_folder2 - subfolders_in_folder1

# 将只在第二个文件夹中的子文件夹复制到第一个文件夹
for subfolder_name in subfolders_only_in_folder2:
    src_path = os.path.join(folder2, subfolder_name)
    dest_path = os.path.join(folder1, subfolder_name)
    if os.path.isdir(src_path):  # 确保是文件夹而不是文件
        shutil.copytree(src_path, dest_path)
        print(f"已复制文件夹: {subfolder_name} 到 {folder1}")

print(f"操作完成！已将 {len(subfolders_only_in_folder2)} 个子文件夹从 {folder2} 复制到 {folder1}")