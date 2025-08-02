import os
import json
import glob

# 定义源目录路径
source_base_path = "/usr/gao/gubincheng/article_rep/Agent/data/politic/Env_Rumor_Test_content"

# 定义目标目录路径
target_base_path = "/usr/gao/gubincheng/article_rep/Agent/data/pro/Env_Rumor_politic_updated"

# 记录处理情况
processed_count = 0
success_count = 0

# 获取目标目录中的所有agent目录
target_agent_dirs = glob.glob(os.path.join(target_base_path, "agent_*"))
print(f"找到 {len(target_agent_dirs)} 个目标agent目录")

# 遍历每个agent目录并更新short_memory
for target_agent_dir in target_agent_dirs:
    agent_name = os.path.basename(target_agent_dir)  # 例如 "agent_1"
    target_json_file = os.path.join(target_agent_dir, f"{agent_name}.json")
    source_json_file = os.path.join(source_base_path, agent_name, f"{agent_name}.json")
    
    processed_count += 1
    
    try:
        # 检查源文件是否存在
        if not os.path.exists(source_json_file):
            print(f"警告: 源文件不存在 {source_json_file}，跳过处理")
            continue
            
        # 读取源文件中的short_memory
        with open(source_json_file, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
            source_short_memory = source_data.get('short_memory', [])
            
        if not source_short_memory:
            print(f"警告: 源文件 {source_json_file} 中的short_memory为空，跳过处理")
            continue
        
        # 检查目标文件是否存在
        if not os.path.exists(target_json_file):
            print(f"警告: 目标文件不存在 {target_json_file}，跳过处理")
            continue
            
        # 读取目标JSON文件
        with open(target_json_file, 'r', encoding='utf-8') as f:
            target_data = json.load(f)
        
        # 备份原始数据（可选）
        backup_file = target_json_file + ".bak"
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(target_data, f, ensure_ascii=False, indent=4)
        
        # 更新short_memory
        target_data['short_memory'] = source_short_memory
        
        # 保存更新后的数据
        with open(target_json_file, 'w', encoding='utf-8') as f:
            json.dump(target_data, f, ensure_ascii=False, indent=4)
        
        success_count += 1
        
        # 每处理10个文件打印一次进度
        if processed_count % 10 == 0:
            print(f"已处理 {processed_count} 个文件，当前: {target_json_file}")
    
    except Exception as e:
        print(f"处理 {agent_name} 时出错: {str(e)}")

print(f"处理完成: 总共处理 {processed_count} 个文件，成功更新 {success_count} 个文件")