import os
import argparse
import torch
import pickle
import glob
from antiberty import AntiBERTyRunner
import time


# ----------------------------------------------------------------------
# 核心文件处理逻辑 (处理单个 FASTA 文件)
# ----------------------------------------------------------------------

def process_single_fasta(fasta_path, output_subfolder, antiberty_runner):
    """
    处理单个 FASTA 文件：读取序列，检查有效性，生成 AntiBERTy 嵌入并保存。
    """
    # 准备输出文件路径
    fasta_filename = os.path.basename(fasta_path)
    pkl_filename = fasta_filename.replace('.fasta', '_antiberty.pkl')
    pkl_path = os.path.join(output_subfolder, pkl_filename)

    # 如果输出文件已存在，跳过（可选优化）
    # if os.path.exists(pkl_path):
    #     print(f"[Skip] {fasta_filename} -> already exists.")
    #     return

    try:
        with open(fasta_path, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2:
                raise ValueError("FASTA file has no valid sequence line.")
            sequence = lines[1].strip()

        # 检查 NaN 或空序列
        if sequence.lower() == 'nan' or len(sequence) == 0:
            raise ValueError("Sequence is NaN or empty.")

        # 检查序列长度限制 (AntiBERTy 限制)
        if len(sequence) > 512:
            print(f"[Skip] {fasta_path} — sequence length {len(sequence)} exceeds 512.")
            # 保存 None 嵌入
            with open(pkl_path, 'wb') as pkl_file:
                pickle.dump(None, pkl_file)
            return

        # 正常生成 embedding
        # 注意: embed([sequence]) 返回的是一个列表，我们取第一个元素
        embedding = antiberty_runner.embed([sequence])[0]
        with open(pkl_path, 'wb') as pkl_file:
            pickle.dump(embedding, pkl_file)

        print(f"[Success] {fasta_filename} -> {os.path.basename(output_subfolder)}/{pkl_filename}")

    except Exception as e:
        # 如果出错/无效内容，保存 None 嵌入
        with open(pkl_path, 'wb') as pkl_file:
            pickle.dump(None, pkl_file)
        print(f"[Empty] {fasta_filename} — reason: {e}")


# ----------------------------------------------------------------------
# 迭代主函数 (类似于 process_multiple_directories)
# ----------------------------------------------------------------------

def process_antiberty_embeddings(input_base, output_base, suffix_list, antiberty_runner):
    """
    遍历 suffix_list 中的所有子目录，并处理其中的 FASTA 文件。
    """
    time_start = time.time()
    total_suffixes = len(suffix_list)
    print(f"AntiBERTy Runner initialized. Starting processing {total_suffixes} subdirectories...")

    for i, subfolder in enumerate(suffix_list):
        print(f"\n[{i + 1}/{total_suffixes}] Processing Folder: {subfolder}")

        # 动态构建输入路径
        input_subfolder_path = os.path.join(input_base, subfolder)

        if not os.path.isdir(input_subfolder_path):
            print(f"[Warning] Input folder not found, skipping: {input_subfolder_path}")
            continue

        # 动态构建输出路径
        output_subfolder_path = os.path.join(output_base, subfolder)
        os.makedirs(output_subfolder_path, exist_ok=True)
        print(f"Saving results to: {output_subfolder_path}")

        # 查找所有 FASTA 文件
        fasta_files = glob.glob(os.path.join(input_subfolder_path, '*.fasta'))

        for fasta_path in fasta_files:
            process_single_fasta(fasta_path, output_subfolder_path, antiberty_runner)

    time_end = time.time()
    print(f"\nProcessed all files.")
    print('AntiBERTy embedding generation time cost:', time_end - time_start, 's')


# ----------------------------------------------------------------------
# 主程序入口 (用于命令行参数解析)
# ----------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 输入基础路径
    parser.add_argument('-f', '--fasta_dir', type=str,
                        default='./data/S4169_fasta',
                        help="Base directory containing subdirectories with FASTA files")

    # 输出基础路径
    parser.add_argument('-o', '--output_dir', type=str,
                        default='./data/S4169_antiberty',
                        help="Base directory to save generated AntiBERTy embeddings")

    # 字典列表
    parser.add_argument('--suffix_list', type=str,
                        # default='["ab_h", "ab_h_m", "ab_l", "ab_l_m"]',
                        default='["R1", "R2", "R3", "R4", "R5", "R6", "L1", "L2", "L3","Rm1", "Rm2", "Rm3", "Rm4", "Rm5", "Rm6", "Lm1", "Lm2", "Lm3"]',
                        # default='["a", "a_m"]',
                        help="List of suffixes to create subdirectories and process")

    args = parser.parse_args()

    # 解析 suffix_list 为列表
    try:
        suffix_list = eval(args.suffix_list)
        if not isinstance(suffix_list, list):
            raise ValueError("suffix_list must be a Python list represented as a string.")
    except Exception as e:
        print(f"Error parsing suffix_list: {e}")
        exit(1)

    # 初始化 AntiBERTyRunner (仅进行一次)
    print("Initializing AntiBERTy Runner...")
    try:
        antiberty_runner = AntiBERTyRunner()
    except Exception as e:
        print(f"Error initializing AntiBERTyRunner: {e}")
        print("Please ensure your environment has the necessary model files.")
        exit(1)

    # 调用函数处理多个目录
    process_antiberty_embeddings(args.fasta_dir, args.output_dir, suffix_list, antiberty_runner)