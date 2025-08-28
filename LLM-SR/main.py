# 🔥 修复 vLLM 内存池兼容性问题：在导入PyTorch/vLLM之前设置CUDA分配器配置
import os
# 移除 expandable_segments 以避免 vLLM 内存池断言失败
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,garbage_collection_threshold:0.6"

from argparse import ArgumentParser
import numpy as np
import torch
import pandas as pd

from llmsr import pipeline
from llmsr import config
from llmsr import sampler
from llmsr import evaluator


parser = ArgumentParser()
parser.add_argument('--port', type=int, default=None)
parser.add_argument('--use_api', type=bool, default=False)
# parser.add_argument('--api_model', type=str, default="gpt-3.5-turbo")
parser.add_argument('--api_model', type=str, default="gpt-4o-mini")

parser.add_argument('--spec_path', type=str)
parser.add_argument('--log_path', type=str, default="./logs/oscillator1")
parser.add_argument('--problem_name', type=str, default="oscillator1")
parser.add_argument('--run_id', type=int, default=1)

# 🔥 NEW: GRPO-related arguments
parser.add_argument('--use_rl', action='store_true', default=False, 
                    help='Use GRPO instead of evolutionary search')
parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-1.5B-Instruct',
                    help='Path or name of the language model for GRPO')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of training epochs for GRPO')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size for GRPO training')
parser.add_argument('--learning_rate', type=float, default=1e-6,
                    help='Learning rate for GRPO training')
parser.add_argument('--rollout_n', type=int, default=5,
                    help='Group size for GRPO (number of samples per prompt)')
parser.add_argument('--gpus', type=int, default=2,
                    help='Number of GPUs to use for GRPO training')
parser.add_argument('--output_dir', type=str, default=None,
                    help='Output directory for GRPO training results')
parser.add_argument('--use_rl_v2', action='store_true', default=False,
                    help='Use GRPO v2 pipeline with dense multi-component reward and memory retrieval')
parser.add_argument('--grid_train_data', action='store_true', default=False,
                    help='Enable grid-based sampling for dataset (v2)')
parser.add_argument('--num_grid_groups', type=int, default=10,
                    help='Number of grid groups for dataset (v2)')
parser.add_argument('--kl_coef', type=float, default=1e-3,
                    help='KL coefficient for v2 GRPO')
parser.add_argument('--max_prompt_length', type=int, default=4096,  # 🔥 增加到4096
                    help='Max prompt length for v2 GRPO')
parser.add_argument('--max_new_tokens', type=int, default=8192,  # 🔥 增加到8192
                    help='Max new tokens for v2 GRPO')
parser.add_argument('--max_model_len', type=int, default=16384,  # 🔥 增加到16384
                    help='Max model len for v2 GRPO')
parser.add_argument('--max_num_batched_tokens', type=int, default=8192,  # 🔥 增加到8192
                    help='Max number of batched tokens for GRPO')
parser.add_argument('--few_shot_k', type=int, default=3,
                    help='Few-shot examples from memory for v2 dataset builder')
parser.add_argument('--num_islands', type=int, default=4,
                    help='Number of memory islands for v2 (default: 4)')
parser.add_argument('--top_k_per_island', type=int, default=8,
                    help='Top-k samples to keep per island for v2 (default: 8)')

# 🌐 HTTP GRPO-related arguments
parser.add_argument('--use_http', action='store_true', default=False,
                    help='Use local HTTP LLM service instead of direct model loading')
parser.add_argument('--http_url', type=str, default='http://localhost:5000',
                    help='URL of the local HTTP LLM service')
parser.add_argument('--tokenizer_path', type=str, default='mistralai/Mixtral-8x7B-Instruct-v0.1',
                    help='HuggingFace tokenizer path for HTTP mode')

args = parser.parse_args()




if __name__ == '__main__':
    
    if args.use_rl or args.use_rl_v2:
        # 🚀 GRPO Training Mode
        print("=" * 60)
        print("🔥 LLM-SR GRPO Training Mode" + (" (v2)" if args.use_rl_v2 else ""))
        print("=" * 60)
        
        # Validate required arguments for GRPO
        if not args.spec_path:
            raise ValueError("--spec_path is required for GRPO training")
        
        # Set up paths
        template_path = args.spec_path
        data_path = f'./data/{args.problem_name}/train.csv'
        output_dir = args.output_dir or f'./llmsr_grpo_outputs/{args.problem_name}'
        # 传递输出目录给奖励函数
        os.environ["LLMSR_OUTPUT_DIR"] = output_dir
        
        # Check if files exist
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found: {template_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        print(f"📋 Problem: {args.problem_name}")
        print(f"📄 Template: {template_path}")
        print(f"📊 Data: {data_path}")
        print(f"🤖 Model: {args.tokenizer_path if args.use_http else args.model_path}")
        print(f"📁 Output: {output_dir}")
        print(f"⚙️ Epochs: {args.epochs}, Batch Size: {args.batch_size}")
        print(f"🎯 Learning Rate: {args.learning_rate}, Group Size: {args.rollout_n}")
        
        # v2 直连优先（不支持 HTTP）
        if args.use_rl_v2:
            print("🚀 Using GRPO v2 pipeline (direct mode only)")
            from llmsr.rl.grpo_runner_v2 import train_llmsr_grpo_v2
            train_llmsr_grpo_v2(
                template_path=template_path,
                data_path=data_path,
                model_path=args.model_path,
                output_dir=output_dir,
                grid_train_data=args.grid_train_data,
                num_grid_groups=args.num_grid_groups,
                gpus=args.gpus,
                rollout_n=args.rollout_n,
                kl_coef=args.kl_coef,
                max_prompt_length=args.max_prompt_length,
                max_new_tokens=args.max_new_tokens,
                max_model_len=args.max_model_len,
                max_num_batched_tokens=args.max_num_batched_tokens,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                few_shot_k=args.few_shot_k,
                num_islands=args.num_islands,
                top_k_per_island=args.top_k_per_island,
            )
        # 🌐 HTTP GRPO Training Mode (优先检查)
        elif args.use_http:
            print("🌐 Using local HTTP LLM service for GRPO training")
            
            # Import HTTP GRPO runner
            from llmsr.rl.grpo_runner import train_llmsr_grpo_http
            
            train_llmsr_grpo_http(
                template_path=template_path,
                data_path=data_path,
                http_url=args.http_url,
                tokenizer_path=args.tokenizer_path,
                output_dir=output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                rollout_n=args.rollout_n,
                experiment_name=f'llmsr_grpo_http_{args.problem_name}',
                gpus=args.gpus,
                log_dir=args.log_path
            )
        else:
            # 🔥 普通GRPO模式（仅在非HTTP时执行）
            print("🚀 Switching to GRPO training mode...")
            
            # Load specification for compatibility
            with open(template_path, encoding="utf-8") as f:
                specification = f.read()
            
            # Create dummy inputs (not used in GRPO mode, but required for interface)
            df = pd.read_csv(data_path)
            data = np.array(df)
            X = data[:, :-1]
            y = data[:, -1].reshape(-1)
            if 'torch' in args.spec_path:
                X = torch.Tensor(X)
                y = torch.Tensor(y)
            data_dict = {'inputs': X, 'outputs': y}
            dataset = {'data': data_dict}
            
            # Create dummy config (not used in GRPO mode)
            config_obj = config.Config(use_api=args.use_api, api_model=args.api_model)
            class_config = config.ClassConfig(llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox)
            
            # Call pipeline with GRPO mode
            pipeline.main(
                specification=specification,
                inputs=dataset,
                config=config_obj,
                max_sample_nums=None,  # Not used in GRPO
                class_config=class_config,
                use_rl=True,  # 🔥 Enable GRPO mode
                template_path=template_path,
                data_path=data_path,
                model_path=args.model_path,
                output_dir=output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                rollout_n=args.rollout_n,
                experiment_name=f'llmsr_grpo_{args.problem_name}',
                gpus=args.gpus,
                log_dir=args.log_path,
                # 🔥 添加大token长度参数
                max_prompt_length=args.max_prompt_length,
                max_new_tokens=args.max_new_tokens,
                max_model_len=args.max_model_len,
                max_num_batched_tokens=args.max_num_batched_tokens
            )
        
    else:
        # 🧬 Original Evolutionary Mode
        print("=" * 60)
        print("🧬 LLM-SR Evolutionary Search Mode")
        print("=" * 60)
        
        # Load config and parameters
        class_config = config.ClassConfig(llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox)
        config_obj = config.Config(
            use_api=args.use_api, 
            api_model=args.api_model,
            azure_endpoint="https://chhws-mbuye1am-eastus2.cognitiveservices.azure.com/",
            azure_api_key="55EaaZ9XTafHDunCJkTbEt0alnd2sySEu5VViAUrSctsF2Q50HpMJQQJ99BFACHYHv6XJ3w3AAAAACOGiiLi",
            azure_deployment="gpt-4o-mini"
        )
        global_max_sample_num = 10000 
        # global_max_sample_num = 3

        # Load prompt specification
        with open(
            os.path.join(args.spec_path),
            encoding="utf-8",
        ) as f:
            specification = f.read()
        
        # Load dataset
        problem_name = args.problem_name
        df = pd.read_csv('./data/'+problem_name+'/train.csv')
        data = np.array(df)
        X = data[:, :-1]
        y = data[:, -1].reshape(-1)
        if 'torch' in args.spec_path:
            X = torch.Tensor(X)
            y = torch.Tensor(y)
        data_dict = {'inputs': X, 'outputs': y}
        dataset = {'data': data_dict} 
        
        pipeline.main(
            specification=specification,
            inputs=dataset,
            config=config_obj,
            max_sample_nums=global_max_sample_num,
            class_config=class_config,
            use_rl=False,  # 🧬 Use evolutionary search
            log_dir=args.log_path,
        )
