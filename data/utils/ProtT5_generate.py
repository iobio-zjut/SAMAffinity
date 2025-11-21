#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refactored ProtT5 Embedding Script
Logic aligned with ESM script for batch directory processing.
"""

import argparse
import time
import os
import glob
import pickle
from pathlib import Path
import torch
from transformers import T5EncoderModel, T5Tokenizer

# --------------------------------------------------------------------------
# Device Configuration
# --------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print("Using device: {}".format(device))


# --------------------------------------------------------------------------
# Helper Functions (Copied/Adapted from Script 1 & 2)
# --------------------------------------------------------------------------

def is_nan_fasta(fasta_file):
    """检查 fasta 文件内容是否为 'nan' (来自脚本1)"""
    with open(fasta_file, 'r') as file:
        lines = file.readlines()
        return len(lines) > 1 and lines[1].strip().lower() == 'nan'


def create_empty_pkl(output_file):
    """创建一个空的 .pkl 文件 (来自脚本1)"""
    with open(output_file, 'wb') as f:
        pickle.dump({}, f)
    print(f"Created empty .pkl file: {output_file}")


def get_T5_model(model_dir):
    """加载 T5 模型 (只在程序开始时运行一次)"""
    print("Loading T5 from: {}".format(model_dir))
    model = T5EncoderModel.from_pretrained(model_dir).to(device)
    model = model.eval()
    vocab = T5Tokenizer.from_pretrained(model_dir, do_lower_case=False)
    return model, vocab


def read_fasta(fasta_path, split_char, id_field, is_3Di):
    '''
    读取包含序列的 fasta 文件 (来自脚本2)
    '''
    sequences = dict()
    with open(fasta_path, 'r') as fasta_f:
        for line in fasta_f:
            if line.startswith('>'):
                # get uniprot ID from header
                uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
                # replace tokens that are mis-interpreted when loading h5
                uniprot_id = uniprot_id.replace("/", "_").replace(".", "_")
                sequences[uniprot_id] = ''
            else:
                if is_3Di:
                    sequences[uniprot_id] += ''.join(line.split()).replace("-", "").lower()
                else:
                    sequences[uniprot_id] += ''.join(line.split()).replace("-", "")
    return sequences


def process_single_fasta(fasta_path, output_path, model, vocab, args):
    """
    处理单个 FASTA 文件并保存为 .pkl
    (由原 get_embeddings 函数改造，移除了内部模型加载，改为接收 model 对象)
    """

    # 参数解包
    split_char = args.split_char
    id_field = args.id
    per_protein = False if int(args.per_protein) == 0 else True
    half_precision = False if int(args.half) == 0 else True
    is_3Di = False if int(args.is_3Di) == 0 else True

    max_residues = 4000
    max_seq_len = 1000
    max_batch = 100

    seq_dict = dict()
    emb_dict = dict()

    # 读取 fasta
    seq_dict = read_fasta(fasta_path, split_char, id_field, is_3Di)
    prefix = "<fold2AA>" if is_3Di else "<AA2fold>"

    if half_precision:
        # 注意：这里不能直接 model.half()，因为会影响主模型对象。
        # 实际上应该在加载模型时处理，或者在这里使用 autocast。
        # 简单起见，假设外部已处理或在此处转换上下文，但为防止副作用，建议在外部加载时决定精度。
        # 此处仅打印提示，实际精度由加载时的配置决定。
        pass

        # 预处理和排序
    # sort sequences by length to trigger OOM at the beginning
    seq_dict = sorted(seq_dict.items(), key=lambda kv: len(seq_dict[kv[0]]), reverse=True)

    batch = list()
    for seq_idx, (pdb_id, seq) in enumerate(seq_dict, 1):
        # replace non-standard AAs
        seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
        seq_len = len(seq)
        seq = prefix + ' ' + ' '.join(list(seq))
        batch.append((pdb_id, seq, seq_len))

        # count residues in current batch
        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            token_encoding = vocab.batch_encode_plus(seqs,
                                                     add_special_tokens=True,
                                                     padding="longest",
                                                     return_tensors='pt'
                                                     ).to(device)
            try:
                with torch.no_grad():
                    embedding_repr = model(token_encoding.input_ids,
                                           attention_mask=token_encoding.attention_mask
                                           )
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                continue

            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]
                # account for prefix in offset
                emb = embedding_repr.last_hidden_state[batch_idx, 1:s_len + 1]

                if per_protein:
                    emb = emb.mean(dim=0)

                # 保存到字典
                emb_dict[identifier] = emb.detach().cpu().clone()
                # 移除了 numpy 转换以保持 Tensor 格式 (如果你需要 numpy，可以在这里加 .numpy())
                # 脚本1输出的是 Tensor (clone().cpu())，这里保持一致。

    # 保存结果
    with open(output_path, 'wb') as handle:
        pickle.dump(emb_dict, handle, protocol=4)

    # 简略日志
    # print(f"Saved {output_path}")


# --------------------------------------------------------------------------
# Main Processing Logic (Similar to Script 1)
# --------------------------------------------------------------------------

def process_multiple_directories(base_fasta_dir, base_output_dir, suffix_list, args):
    # 1. 加载模型 (循环外加载，极大提升效率)
    model, vocab = get_T5_model(args.model)
    if int(args.half) == 1:
        model = model.half()
        print("Using model in half-precision!")

    # 2. 开始计时
    time_start = time.time()

    for suffix in suffix_list:
        # 动态构建输入路径和输出路径
        fasta_dir = os.path.join(base_fasta_dir, suffix)
        output_dir = os.path.join(base_output_dir, suffix)

        # 检查并创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # 处理指定目录中的所有 FASTA 文件
        fasta_files = glob.glob(os.path.join(fasta_dir, '*.fasta'))
        print(f"Found {len(fasta_files)} FASTA files to process in {fasta_dir}.")

        for fasta_file in fasta_files:
            # 根据输入文件名构建输出文件名
            base_name = os.path.basename(fasta_file)
            file_prefix = os.path.splitext(base_name)[0]
            # 保持与脚本1一致的命名习惯 (_emb.pkl)
            output_file = os.path.join(output_dir, f"{file_prefix}_prott5.pkl")

            # 检测 fasta 是否为 'nan'
            if is_nan_fasta(fasta_file):
                print(f"FASTA file {fasta_file} contains 'nan'. Skipping processing.")
                create_empty_pkl(output_file)
                continue

            # 检查是否已存在 (可选，防止重复跑)
            # if os.path.exists(output_file): continue

            # 调用生成嵌入的函数 (传入已加载的模型)
            try:
                process_single_fasta(fasta_file, output_file, model, vocab, args)
            except Exception as e:
                print(f"Error processing {fasta_file}: {e}")

    # 结束计时
    time_end = time.time()
    print(f"Processed all files.")
    print('Embedding generation time cost:', time_end - time_start, 's')


def main():
    parser = argparse.ArgumentParser(description='ProtT5 embedding script with directory batch processing')

    # ------------------------------------------
    # 路径参数 (对齐脚本1)
    # ------------------------------------------
    parser.add_argument('-f', '--fasta_dir', type=str,
                        default='./data/S4169_fasta',
                        help="Base directory containing subdirectories with FASTA files")

    parser.add_argument('-o', '--output_dir', type=str,
                        default='./data/S4169_protT5',
                        help="Base directory to save generated ProtT5 embeddings")

    parser.add_argument('--suffix_list', type=str,
                        default='["R1", "R2", "R3", "R4", "R5", "R6", "L1", "L2", "L3","Rm1", "Rm2", "Rm3", "Rm4", "Rm5", "Rm6", "Lm1", "Lm2", "Lm3"]',
                        # default='["ab_h", "ab_h_m", "ab_l", "ab_l_m", "ag_a", "ag_a_m", "ag_b", "ag_b_m"]',
                        # default='["a", "a_m", "b", "b_m"]',
                        help="List of suffixes to create subdirectories and process")


    # ------------------------------------------
    # 模型参数 (保留脚本2的特性)
    # ------------------------------------------
    parser.add_argument('--model', required=False, type=str,
                        default="./prott5",
                        help='Path to pre-trained model checkpoint.')

    parser.add_argument('--split_char', type=str, default='!',
                        help="Character for splitting FASTA header")

    parser.add_argument('--id', type=int, default=0,
                        help="Index for uniprot identifier in header")

    parser.add_argument('--per_protein', type=int, default=0,
                        help="0 for per-residue embeddings, 1 for mean-pooled per-protein")

    parser.add_argument('--half', type=int, default=0,
                        help="1 for half_precision, 0 for full-precision")

    parser.add_argument('--is_3Di', type=int, default=0,
                        help="1 for 3Di input, 0 for AA input")

    args = parser.parse_args()

    # 解析 suffix_list
    suffix_list = eval(args.suffix_list)

    # 执行主逻辑
    process_multiple_directories(args.fasta_dir, args.output_dir, suffix_list, args)


if __name__ == '__main__':
    main()