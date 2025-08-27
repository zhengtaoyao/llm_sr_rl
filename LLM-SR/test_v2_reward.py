#!/usr/bin/env python3
"""
测试v2版本奖励函数的提取功能
"""

import sys
sys.path.append('llmsr/rl')
from simple_verl_reward_v2_fixed import extract_function_body_v2

# 测试用例1：带EDIT指令的输出（来自日志）
test_case1 = """
<think>
Okay, so the user wants a mathematical function skeleton for acceleration in a damped nonlinear oscillator with a driving force. Let me think about the physics here. 
...
</think>

EDIT ADD -params[0]*v
EDIT ADD -params[1]*x
EDIT ADD -params[2]*x**3
EDIT ADD params[3]

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    return -params[0]*v - params[1]*x - params[2]*x**3 + params[3]
"""

# 测试用例2：带Python代码块的输出
test_case2 = """
<think>
Let's tackle this problem...
</think>

The acceleration in a damped nonlinear oscillator with a driving force can be modeled as...

```python
def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    return -params[0] * x - params[1] * v - params[2] * x**3 + params[3] * np.sin(params[4] * x + params[5])
```

**Explanation:**
...
"""

# 测试用例3：多行return语句
test_case3 = """
<think>
...
</think>

EDIT ADD -params[0]*v
EDIT ADD -params[1]*x
EDIT ADD params[2]*x**3
EDIT ADD params[3]*v**2
EDIT ADD params[4]*np.sin(params[5]*x)

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    return (
        -params[0]*v
        - params[1]*x
        + params[2]*x**3
        + params[3]*v**2
        + params[4]*np.sin(params[5]*x)
    )
"""

def test_extraction():
    print("🧪 测试v2版本的函数体提取功能")
    print("="*50)
    
    test_cases = [
        ("带EDIT指令和单行return", test_case1),
        ("带Python代码块", test_case2),
        ("带EDIT指令和多行return", test_case3)
    ]
    
    for i, (name, test_input) in enumerate(test_cases, 1):
        print(f"\n测试用例{i}: {name}")
        print("-"*30)
        
        result = extract_function_body_v2(test_input)
        
        if result:
            print(f"✅ 成功提取函数体:")
            print(f"```\n{result}```")
        else:
            print(f"❌ 提取失败")
        
        print("-"*30)

if __name__ == "__main__":
    test_extraction()
