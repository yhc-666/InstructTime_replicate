#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import argparse
import sys
import os
import numpy as np
import pandas as pd
from pprint import pprint

def load_pickle(file_path):
    """加载pickle文件并返回其内容"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return None

def inspect_data(data):
    """检查数据类型并适当地显示信息"""
    print("\n数据类型:", type(data))
    
    if isinstance(data, dict):
        print("\n字典键:")
        for i, key in enumerate(data.keys()):
            print(f"{i+1}. {key} (类型: {type(data[key])})")
        
        while True:
            key_input = input("\n请输入要查看的键 (输入'q'退出, 'all'查看所有键): ")
            if key_input.lower() == 'q':
                break
            elif key_input.lower() == 'all':
                for key, value in data.items():
                    print(f"\n键: {key}")
                    print_value(value)
            elif key_input in data:
                print(f"\n键: {key_input}")
                print_value(data[key_input])
            else:
                try:
                    idx = int(key_input) - 1
                    key = list(data.keys())[idx]
                    print(f"\n键: {key}")
                    print_value(data[key])
                except (ValueError, IndexError):
                    print("无效输入，请重试")
    
    elif isinstance(data, list) or isinstance(data, tuple):
        print(f"\n列表/元组长度: {len(data)}")
        if len(data) > 0:
            print(f"第一个元素类型: {type(data[0])}")
        
        while True:
            idx_input = input("\n请输入要查看的索引 (输入'q'退出, 'len'查看长度, 'all'查看所有): ")
            if idx_input.lower() == 'q':
                break
            elif idx_input.lower() == 'len':
                print(f"长度: {len(data)}")
            elif idx_input.lower() == 'all':
                if len(data) > 100:
                    confirm = input(f"列表包含 {len(data)} 个元素，确定要显示全部吗? (y/n): ")
                    if confirm.lower() != 'y':
                        continue
                for i, item in enumerate(data):
                    print(f"\n索引 {i}:")
                    print_value(item)
            else:
                try:
                    idx = int(idx_input)
                    if 0 <= idx < len(data):
                        print(f"\n索引 {idx}:")
                        print_value(data[idx])
                    else:
                        print(f"索引超出范围，应在 0 到 {len(data)-1} 之间")
                except ValueError:
                    print("无效输入，请输入数字")
    
    elif isinstance(data, np.ndarray):
        print(f"\nNumPy数组形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
        
        while True:
            action = input("\n操作 ('q'退出, 'shape'查看形状, 'sample'查看样本, 'stats'查看统计信息): ")
            if action.lower() == 'q':
                break
            elif action.lower() == 'shape':
                print(f"形状: {data.shape}")
            elif action.lower() == 'sample':
                if data.ndim == 1:
                    print(f"样本数据 (前10个): {data[:10]}")
                elif data.ndim == 2:
                    print(f"样本数据 (前5行5列):\n{data[:5, :5]}")
                else:
                    print(f"高维数组，第一个切片:\n{data[0]}")
            elif action.lower() == 'stats':
                try:
                    print(f"最小值: {data.min()}")
                    print(f"最大值: {data.max()}")
                    print(f"平均值: {data.mean()}")
                    print(f"标准差: {data.std()}")
                except:
                    print("无法计算统计信息")
    
    elif isinstance(data, pd.DataFrame):
        print(f"\nDataFrame形状: {data.shape}")
        print("\n列名:")
        for col in data.columns:
            print(f"- {col}")
        
        while True:
            action = input("\n操作 ('q'退出, 'head'查看头部, 'info'查看信息, 'describe'查看描述): ")
            if action.lower() == 'q':
                break
            elif action.lower() == 'head':
                print(data.head())
            elif action.lower() == 'info':
                data.info()
            elif action.lower() == 'describe':
                print(data.describe())
    
    else:
        print_value(data)

def print_value(value):
    """打印值的适当表示"""
    if isinstance(value, (np.ndarray, pd.DataFrame)):
        print(f"类型: {type(value)}")
        if isinstance(value, np.ndarray):
            print(f"形状: {value.shape}")
            print(f"数据类型: {value.dtype}")
            if value.size < 100:
                print(value)
            else:
                print(f"前10个元素: {value.flatten()[:10]}")
        else:  # DataFrame
            print(f"形状: {value.shape}")
            print(value.head())
    else:
        try:
            if isinstance(value, (list, dict, tuple)) and len(value) > 100:
                print(f"大型数据结构，长度: {len(value)}")
            else:
                pprint(value)
        except:
            print(f"无法显示值: {type(value)}")

def main():
    parser = argparse.ArgumentParser(description='检查PKL文件内容')
    parser.add_argument('file_path', help='要检查的pkl文件路径')
    args = parser.parse_args()
    
    if not os.path.exists(args.file_path):
        print(f"错误: 文件 '{args.file_path}' 不存在")
        sys.exit(1)
    
    print(f"加载文件: {args.file_path}")
    data = load_pickle(args.file_path)
    
    if data is not None:
        inspect_data(data)

if __name__ == "__main__":
    main() 


# python inspect_pkl.py ts_tokenized_datasets/ihm/tokenized_v1_patch64/test.pkl