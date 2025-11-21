import argparse
import subprocess
import sys
import os
import shlex
import json
import pandas as pd
import time  # 添加 time 模块用于计时


# ----------------------------------------------------------------------
# FASTA 生成逻辑 (来自脚本 2)
# ----------------------------------------------------------------------

def create_fasta_from_csv(data_df, output_dir, fasta_suffix):
    """
    根据传入的DataFrame生成fasta文件。
    """
    # 遍历每一行，生成fasta文件
    for index, row in data_df.iterrows():
        # 构建fasta文件名
        fasta_name = f"{row['PDB']}_{row['mutation_clean']}_{fasta_suffix}.fasta"
        fasta_path = os.path.join(output_dir, fasta_name)

        # 获取序列内容 (注意: 列名就是 fasta_suffix 的值)
        # 确保序列列存在
        if fasta_suffix not in row:
            print(f"Error: Column '{fasta_suffix}' not found in CSV row for {row['PDB']}.")
            continue

        sequence = row[fasta_suffix]

        # 写入fasta文件
        with open(fasta_path, 'w') as fasta_file:
            fasta_file.write(f">{row['PDB']}_{row['mutation_clean']}_{fasta_suffix}\n")
            # 确保序列被视为字符串写入
            fasta_file.write(str(sequence).strip())
            # print(f"Saved {fasta_path}") # 太多输出，只在完成时打印


def process_fasta_generation(csv_file, output_dir_base, suffix_list):
    """
    负责迭代 suffix_list，并协调 FASTA 文件的生成。
    """
    time_start = time.time()
    print("-" * 50)
    print(f"Starting FASTA generation from CSV: {csv_file}")
    print(f"Outputting to base directory: {output_dir_base}")
    print("-" * 50)

    try:
        data_df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"FATAL ERROR: CSV file not found at {csv_file}")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL ERROR reading CSV: {e}")
        sys.exit(1)

    for fasta_suffix in suffix_list:
        output_dir = os.path.join(output_dir_base, fasta_suffix)  # 创建每个后缀的输出文件夹路径
        os.makedirs(output_dir, exist_ok=True)  # 创建保存目录（如果不存在）

        # 生成fasta文件
        create_fasta_from_csv(data_df, output_dir, fasta_suffix)
        print(f"Finished generating FASTA files for suffix: {fasta_suffix} (Saved in {output_dir})")

    time_end = time.time()
    print(f"FASTA generation time cost: {time_end - time_start:.2f} s")
    print("-" * 50)


# ----------------------------------------------------------------------
# 特征提取逻辑 (原脚本 1 的 run_feature_script 保持不变)
# ----------------------------------------------------------------------

def run_feature_script(script_path, fasta_dir, output_dir, suffix_list_str, extra_args=""):
    """
    通过 subprocess 调用外部 Python 脚本。
    (逻辑与原脚本1保持一致)
    """
    # 构造命令行参数列表
    command = [
        sys.executable,  # Python 解释器路径
        script_path,
        '-f', fasta_dir,
        '-o', output_dir,
        '--suffix_list', suffix_list_str,
    ]

    # 如果有额外参数，将其添加到命令中
    if extra_args:
        command.extend(shlex.split(extra_args))

    print("-" * 50)
    print(f"Executing: {os.path.basename(script_path)}")
    print(f"Command: {' '.join(command)}")
    print("-" * 50)

    try:
        subprocess.run(
            command,
            check=True,  # 检查退出码
            capture_output=False,
            text=True
        )
        print(f"SUCCESS: {os.path.basename(script_path)} finished.")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {os.path.basename(script_path)} failed with code {e.returncode}.")
        sys.exit(1)


# ----------------------------------------------------------------------
# 主程序入口 (更新 main 函数)
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Unified script for generating FASTA files and subsequent protein features using a config file.')

    parser.add_argument('--config_file', type=str, required=True,
                        help="Path to the JSON configuration file (e.g., config.json)")

    args = parser.parse_args()

    # 1. 加载配置
    try:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {args.config_file}")
        sys.exit(1)

    # ------------------------------------------
    # 提取公共参数和路径
    # ------------------------------------------

    # 新增: CSV 文件路径
    csv_file = config['common']['csv_file']
    # FASTA 输出路径 (也是特征脚本的输入路径)
    fasta_dir = config['common']['fasta_dir']

    suffix_list = config['common']['suffix_list']
    suffix_list_str = json.dumps(suffix_list)  # 供特征脚本使用

    script_paths = config['script_paths']
    output_dirs = config['output_dirs']

    # ------------------------------------------
    # STEP 1: FASTA 文件生成 (运行预处理脚本)
    # ------------------------------------------

    # 运行 FASTA 生成逻辑
    process_fasta_generation(csv_file, fasta_dir, suffix_list)

    # ------------------------------------------
    # STEP 2: 特征提取 (运行原脚本 1 的逻辑)
    # ------------------------------------------

    print("\n" + "=" * 60)
    print("STARTING FEATURE EXTRACTION (STEP 2)")
    print("=" * 60)

    # 1. AntiBERTy
    run_feature_script(
        script_path=script_paths['antiberty'],
        fasta_dir=fasta_dir,
        output_dir=output_dirs['antiberty'],
        suffix_list_str=suffix_list_str,
    )

    # 2. ESM2
    esm_config = config.get('esm2_specific', {})
    esm_extra_args = f"-emp {esm_config.get('esm_model_path', '')}"
    if esm_config.get('nogpu', False):
        esm_extra_args += " --nogpu"

    run_feature_script(
        script_path=script_paths['esm2'],
        fasta_dir=fasta_dir,
        output_dir=output_dirs['esm2'],
        suffix_list_str=suffix_list_str,
        extra_args=esm_extra_args
    )

    # 3. ProtT5
    prott5_config = config.get('prott5_specific', {})
    prott5_extra_args = f"--model {prott5_config.get('model_path', '')} "
    prott5_extra_args += f"--per_protein {prott5_config.get('per_protein', 0)} "
    prott5_extra_args += f"--half {prott5_config.get('half', 0)} "
    prott5_extra_args += f"--is_3Di {prott5_config.get('is_3Di', 0)}"

    run_feature_script(
        script_path=script_paths['prott5'],
        fasta_dir=fasta_dir,
        output_dir=output_dirs['prott5'],
        suffix_list_str=suffix_list_str,
        extra_args=prott5_extra_args
    )

    # 4. PSSM
    pssm_config = config.get('pssm_specific', {})
    pssm_extra_args = f"--psiblast_path {pssm_config.get('psiblast_path', '')} "
    pssm_extra_args += f"--db_path {pssm_config.get('db_path', '')}"

    run_feature_script(
        script_path=script_paths['pssm'],
        fasta_dir=fasta_dir,
        output_dir=output_dirs['pssm'],
        suffix_list_str=suffix_list_str,
        extra_args=pssm_extra_args  # 传入路径参数
    )

    print("\nAll tasks (FASTA Generation and Feature Extraction) executed successfully!")


if __name__ == '__main__':
    # 检查 pandas 依赖
    try:
        import pandas as pd
    except ImportError:
        print("Error: The 'pandas' library is required for CSV processing.")
        print("Please install it using: pip install pandas")
        sys.exit(1)
    print("use example: python run_all_embeddings.py --config_file config.json")

    main()