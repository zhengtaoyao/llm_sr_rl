#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析原版LLM-SR进化搜索的错误统计脚本
用于查看legal/illegal function数量和错误类型
"""

import os
import json
import re
import numpy as np
import argparse
from typing import Dict, List, Tuple
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def analyze_log_file(log_file: str) -> Dict:
    """分析主日志文件中的错误信息"""
    print(f'🔍 分析日志文件: {log_file}')
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # 统计不同类型的错误
    execution_errors = re.findall(r'Execution Error: (.+)', content)
    score_none_count = content.count('Score        : None')
    successful_scores = re.findall(r'Score        : (-?\d+\.?\d*)', content)
    
    # 统计错误类型
    error_types = {}
    for error in execution_errors:
        if error in error_types:
            error_types[error] += 1
        else:
            error_types[error] = 1
    
    # 处理成功分数
    scores = []
    if successful_scores:
        scores = [float(s) for s in successful_scores if s != 'None']
    
    total_samples = len(execution_errors) + len(scores)
    success_rate = len(scores) / total_samples if total_samples > 0 else 0
    
    return {
        'total_samples': total_samples,
        'execution_errors': len(execution_errors),
        'score_none_count': score_none_count,
        'successful_evaluations': len(scores),
        'success_rate': success_rate,
        'error_types': error_types,
        'scores': scores
    }

def analyze_samples_json(samples_dir: str) -> Dict:
    """分析samples目录中的JSON文件"""
    print(f'📊 分析样本目录: {samples_dir}')
    
    total_samples = 0
    success_samples = 0
    failed_samples = 0
    scores = []
    
    if not os.path.exists(samples_dir):
        print(f'⚠️ 样本目录不存在: {samples_dir}')
        return {}
    
    for filename in os.listdir(samples_dir):
        if filename.endswith('.json'):
            total_samples += 1
            try:
                with open(os.path.join(samples_dir, filename), 'r') as f:
                    data = json.load(f)
                    if data.get('score') is not None:
                        success_samples += 1
                        scores.append(float(data['score']))
                    else:
                        failed_samples += 1
            except Exception as e:
                print(f'❌ 读取文件 {filename} 失败: {e}')
                failed_samples += 1
    
    return {
        'total_samples': total_samples,
        'success_samples': success_samples,
        'failed_samples': failed_samples,
        'success_rate': success_samples / total_samples if total_samples > 0 else 0,
        'scores': scores
    }

def analyze_tensorboard(log_dir: str) -> Dict:
    """分析TensorBoard数据"""
    print(f'📈 分析TensorBoard数据: {log_dir}')
    
    try:
        ea = EventAccumulator(log_dir)
        ea.Reload()
        
        result = {
            'available_tags': ea.Tags()['scalars'],
            'legal_illegal_data': {},
            'time_data': {},
            'best_score_data': {}
        }
        
        # 获取Legal/Illegal Function统计
        if 'Legal_Illegal Function/legal function num' in ea.Tags()['scalars']:
            legal_data = ea.Scalars('Legal_Illegal Function/legal function num')
            illegal_data = ea.Scalars('Legal_Illegal Function/illegal function num')
            
            result['legal_illegal_data'] = {
                'legal_functions': legal_data[-1].value if legal_data else 0,
                'illegal_functions': illegal_data[-1].value if illegal_data else 0,
                'legal_history': [(s.step, s.value) for s in legal_data],
                'illegal_history': [(s.step, s.value) for s in illegal_data]
            }
        
        # 获取时间统计
        if 'Total Sample_Evaluate Time/sample time' in ea.Tags()['scalars']:
            sample_time_data = ea.Scalars('Total Sample_Evaluate Time/sample time')
            eval_time_data = ea.Scalars('Total Sample_Evaluate Time/evaluate time')
            
            result['time_data'] = {
                'total_sample_time': sample_time_data[-1].value if sample_time_data else 0,
                'total_evaluate_time': eval_time_data[-1].value if eval_time_data else 0
            }
        
        # 获取最佳分数
        if 'Best Score of Function' in ea.Tags()['scalars']:
            best_score_data = ea.Scalars('Best Score of Function')
            result['best_score_data'] = {
                'current_best': best_score_data[-1].value if best_score_data else None,
                'history': [(s.step, s.value) for s in best_score_data]
            }
        
        return result
    
    except Exception as e:
        print(f'❌ TensorBoard分析失败: {e}')
        return {}

def print_error_summary(experiment_path: str):
    """打印错误摘要报告"""
    print('\n' + '='*60)
    print('🧬 LLM-SR 原版进化搜索 - 错误分析报告')
    print('='*60)
    
    # 分析主日志
    log_files = [f for f in os.listdir('llmsr_logs') if experiment_path in f and f.endswith('.log')]
    if log_files:
        latest_log = sorted(log_files)[-1]
        log_path = os.path.join('llmsr_logs', latest_log)
        log_results = analyze_log_file(log_path)
        
        print(f'\n📋 主日志分析 ({latest_log}):')
        print(f'  总样本数: {log_results["total_samples"]}')
        print(f'  成功评估: {log_results["successful_evaluations"]} ({log_results["success_rate"]:.1%})')
        print(f'  执行失败: {log_results["execution_errors"]} ({(1-log_results["success_rate"]):.1%})')
        
        print(f'\n🔍 错误类型详情:')
        for error_type, count in log_results['error_types'].items():
            print(f'  - {error_type}: {count}次')
    
    # 分析样本JSON
    samples_path = f'logs/{experiment_path}/samples'
    if os.path.exists(samples_path):
        json_results = analyze_samples_json(samples_path)
        
        print(f'\n📊 样本JSON分析:')
        print(f'  总样本数: {json_results["total_samples"]}')
        print(f'  成功样本: {json_results["success_samples"]} ({json_results["success_rate"]:.1%})')
        print(f'  失败样本: {json_results["failed_samples"]} ({(1-json_results["success_rate"]):.1%})')
    
    # 分析TensorBoard数据
    tb_path = f'logs/{experiment_path}'
    if os.path.exists(tb_path):
        tb_results = analyze_tensorboard(tb_path)
        
        if 'legal_illegal_data' in tb_results and tb_results['legal_illegal_data']:
            legal_data = tb_results['legal_illegal_data']
            print(f'\n📈 TensorBoard统计:')
            print(f'  Legal functions: {legal_data["legal_functions"]}')
            print(f'  Illegal functions: {legal_data["illegal_functions"]}')
            
            total = legal_data["legal_functions"] + legal_data["illegal_functions"]
            if total > 0:
                legal_rate = legal_data["legal_functions"] / total
                print(f'  Legal rate: {legal_rate:.1%}')
        
        if 'best_score_data' in tb_results and tb_results['best_score_data']:
            best_data = tb_results['best_score_data']
            if best_data['current_best'] is not None:
                print(f'\n🏆 最佳分数: {best_data["current_best"]:.8f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='分析LLM-SR错误统计')
    parser.add_argument('--experiment', type=str, default='oscillator1_qwen3_8b_evolution',
                        help='实验名称 (例如: oscillator1_qwen3_8b_evolution)')
    
    args = parser.parse_args()
    print_error_summary(args.experiment)
    
    print(f'\n💡 查看方法:')
    print(f'  1. 查看主日志: tail -f llmsr_logs/{args.experiment}_*.log')
    print(f'  2. 启动TensorBoard: tensorboard --logdir=logs/{args.experiment} --port=6006')
    print(f'  3. 查看样本JSON: ls logs/{args.experiment}/samples/')
    print('='*60)
