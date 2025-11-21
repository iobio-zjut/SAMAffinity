import os
import argparse
import time
import glob
from multiprocessing import Pool
import logging
import sys


# ----------------------------------------------------------------------
# PSSMGenerator 类 (修改 __init__ 和 seq2pssm)
# ----------------------------------------------------------------------
class PSSMGenerator:
    def __init__(self, input_fasta_dir, output_pssm_dir, psiblast_path, db_path):
        self.input_fasta_dir = input_fasta_dir
        self.output_pssm_dir = output_pssm_dir
        # 新增：接收 psiblast 和 db 路径
        self.psiblast_path = psiblast_path
        self.db_path = db_path

        # 创建输出目录（如果不存在）
        os.makedirs(self.output_pssm_dir, exist_ok=True)
        # 初始化日志文件（如果日志文件不存在）
        if not os.path.exists('./PSSM.log'):
            open('./PSSM.log', 'w').close()

    def is_nan_fasta(self, fasta_file):
        """检查 fasta 文件内容是否包含 'nan' 行"""
        try:
            with open(fasta_file, 'r') as file:
                lines = file.readlines()
            # 检查是否有 'nan' 行
            for line in lines:
                if line.strip().lower() == "nan":
                    return True
            return False
        except Exception as e:
            print(f"Error reading fasta file {fasta_file}: {e}")
            return False

    def create_empty_output(self, fasta_file):
        """创建空的 .pssm 和 .txt 文件"""
        fasta_basename = os.path.basename(fasta_file)
        pssm_output_file = os.path.join(self.output_pssm_dir, fasta_basename.replace('.fasta', '.pssm'))
        txt_output_file = os.path.join(self.output_pssm_dir, fasta_basename.replace('.fasta', '.txt'))

        # 创建空文件
        open(pssm_output_file, 'w').close()
        open(txt_output_file, 'w').close()
        print(f"Empty PSSM and TXT files created for {fasta_file}")

    def seq2pssm(self, fasta_file):
        # 检查文件是否为 'nan'
        if self.is_nan_fasta(fasta_file):
            self.create_empty_output(fasta_file)
            return None

        # 定义 PSSM 输出文件名
        fasta_basename = os.path.basename(fasta_file)
        pssm_output_file = os.path.join(self.output_pssm_dir, fasta_basename.replace('.fasta', '.pssm'))
        txt_output_file = os.path.join(self.output_pssm_dir, fasta_basename.replace('.fasta', '.txt'))

        # 如果 PSSM 已存在，跳过处理
        if os.path.exists(pssm_output_file):
            print(f"✓ Skip (PSSM already exists): {fasta_file}")
            return pssm_output_file

        # 运行 psiblast 命令 (使用传入的路径)
        command = (
            f"{self.psiblast_path} "
            f"-query {fasta_file} "
            f"-db {self.db_path} "
            f"-num_iterations 3 "
            f"-out {txt_output_file} "
            f"-out_ascii_pssm {pssm_output_file}"
        )

        # 使用 os.system 执行命令
        os.system(command)

        # 检查 PSSM 文件是否生成成功
        if not os.path.exists(pssm_output_file):
            message = f"✗ Failed to generate PSSM for {fasta_file}. Command: {command}"
            print(message)
            # 将信息写入日志文件
            with open('./PSSM.log', 'a') as log_file:
                log_file.write(message + '\n')
            return None

        return pssm_output_file

    def generate_all_pssms(self):
        """
        处理 input_fasta_dir 下的所有 FASTA 文件
        """
        fasta_files = [os.path.join(self.input_fasta_dir, f)
                       for f in os.listdir(self.input_fasta_dir) if f.endswith('.fasta')]

        # 确保输入目录有文件
        if not fasta_files:
            print(f"Warning: No FASTA files found in {self.input_fasta_dir}. Skipping.")
            return

        # 使用 for 循环逐个处理该子目录下的所有文件
        for fasta_file in fasta_files:
            pssm = self.seq2pssm(fasta_file)
            if pssm is not None:
                pass
            else:
                pass

        print(f"Finished processing files in: {self.input_fasta_dir}")


# ----------------------------------------------------------------------
# 迭代主函数 (修改 PSSMGenerator 的实例化)
# ----------------------------------------------------------------------

def process_multiple_directories(base_fasta_dir, base_output_dir, suffix_list, psiblast_path, db_path):
    """
    负责迭代 suffix_list 并调用 PSSMGenerator。
    新增：接收 psiblast_path 和 db_path。
    """
    time_start = time.time()
    total_suffixes = len(suffix_list)
    print(f"Start processing {total_suffixes} subdirectories...")

    for i, suffix in enumerate(suffix_list):
        print(f"\n[{i + 1}/{total_suffixes}] Starting directory: {suffix}")

        # 动态构建输入路径和输出路径
        input_fasta_dir = os.path.join(base_fasta_dir, suffix)
        output_pssm_dir = os.path.join(base_output_dir, suffix)

        # 检查输入目录是否存在
        if not os.path.exists(input_fasta_dir):
            print(f"Warning: Input directory not found, skipping: {input_fasta_dir}")
            continue

        # 创建 PSSMGenerator 实例并生成 PSSM (传入新的路径参数)
        pssm_generator = PSSMGenerator(
            input_fasta_dir,
            output_pssm_dir,
            psiblast_path,  # 新增
            db_path  # 新增
        )
        pssm_generator.generate_all_pssms()

    time_end = time.time()
    print(f"\nProcessed all files.")
    print('PSSM generation time cost:', time_end - time_start, 's')


# ----------------------------------------------------------------------
# 主程序入口 (新增命令行参数)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 输入基础路径
    parser.add_argument('-f', '--fasta_dir', type=str,
                        default='./data/S4169_fasta',
                        help="Base directory containing subdirectories with FASTA files")

    # 输出基础路径
    parser.add_argument('-o', '--output_dir', type=str,
                        default='./data/S4169_PSSM',
                        help="Base directory to save generated PSSM files")

    # 字典列表
    parser.add_argument('--suffix_list', type=str,
                        default='["R1", "R2", "R3", "R4", "R5", "R6", "L1", "L2", "L3","Rm1", "Rm2", "Rm3", "Rm4", "Rm5", "Rm6", "Lm1", "Lm2", "Lm3"]',
                        help="List of suffixes to create subdirectories and process")

    # 新增参数 1: psiblast 可执行文件路径
    parser.add_argument('--psiblast_path', type=str, required=True,
                        default='./ncbi-blast-2.12.0+/bin/psiblast',
                        help="Full path to the psiblast executable.")

    # 新增参数 2: swissprot 数据库路径
    parser.add_argument('--db_path', type=str, required=True,
                        default='./ncbi-blast-2.12.0+/bin/swissprot',
                        help="Full path to the BLAST database (e.g., swissprot).")

    args = parser.parse_args()

    # 检查 psiblast 可执行文件是否存在
    if not os.path.exists(args.psiblast_path):
        print(f"FATAL ERROR: psiblast executable not found at {args.psiblast_path}")
        sys.exit(1)

    # 检查数据库文件是否存在（或者至少是数据库的前缀）
    # 由于 BLAST 数据库由多个文件组成，这里仅进行基本警告
    # if not os.path.exists(args.db_path + '.psq') and not os.path.exists(args.db_path + '.phr'):
    #     print(f"Warning: BLAST database file not found at {args.db_path}. Ensure the path is correct.")

    # 解析 suffix_list 为列表
    try:
        suffix_list = eval(args.suffix_list)
        if not isinstance(suffix_list, list):
            raise ValueError("suffix_list must be a Python list represented as a string.")
    except Exception as e:
        print(f"Error parsing suffix_list: {e}")
        exit(1)

    # 调用函数处理多个目录 (传入新的路径参数)
    process_multiple_directories(
        args.fasta_dir,
        args.output_dir,
        suffix_list,
        args.psiblast_path,  # 传入
        args.db_path  # 传入
    )