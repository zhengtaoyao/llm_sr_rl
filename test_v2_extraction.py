#!/usr/bin/env python3
"""测试V2版本的函数提取功能"""

import sys
sys.path.append('/storage/home/westlakeLab/zhangjunlei/llm_sr_rl/LLM-SR')

from llmsr.rl.simple_verl_reward_v2_fixed import extract_function_body_v2

# 测试案例1：EDIT DSL格式
test1 = """<think>
Okay, so the user wants a mathematical function skeleton for acceleration in a damped nonlinear oscillator with a driving force. Let me think about the physics here.
</think>

EDIT ADD -params[0]*v
EDIT ADD -params[1]*x
EDIT ADD -params[2]*x**3
EDIT ADD params[3]

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    return -params[0]*v - params[1]*x - params[2]*x**3 + params[3]"""

# 测试案例2：带EDIT但格式略有不同
test2 = """<think>
思考过程...
</think>

EDIT ADD -params[0]*v
EDIT ADD -params[1]*x
EDIT ADD params[2]*x**3
EDIT ADD params[3]*v**2
EDIT ADD params[4]*np.sin(params[5]*x)
EDIT ADD params[6]*np.cos(params[7]*x)
EDIT ADD params[8]*np.sin(params[9]*v)

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    return (
        -params[0]*v
        - params[1]*x
        + params[2]*x**3
        + params[3]*v**2
        + params[4]*np.sin(params[5]*x)
        + params[6]*np.cos(params[7]*x)
        + params[8]*np.sin(params[9]*v)
    )"""

# 测试案例3：Markdown代码块格式
test3 = """<think>
思考过程...
</think>

The acceleration in a damped nonlinear oscillator with a driving force can be modeled as a combination of linear damping, nonlinear restoring forces, and an external driving term. Here's the mathematical skeleton:

```python
def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    return -params[0] * x - params[1] * v - params[2] * x**3 + params[3] * np.sin(params[4] * x + params[5])
```

**Explanation:**
- `params[0]`: Linear stiffness coefficient (restoring force).
- `params[1]`: Damping coefficient (proportional to velocity).
- `params[2]`: Nonlinear stiffness coefficient (e.g., cubic term for hardening/softening nonlinearity).
- `params[3]`: Amplitude of the driving force.
- `params[4]`: Angular frequency of the driving force.
- `params[5]`: Phase shift of the driving force."""

# 测试案例4：直接的函数定义（无代码块）
test4 = """def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    return -params[0]*x**3 - params[1]*v + params[2]*np.sin(params[3]*x + params[4])"""

# 测试案例5：只有return语句
test5 = """return -params[0] * x - params[1] * v + params[2]"""

print("=" * 80)
print("测试V2版本的函数提取功能")
print("=" * 80)

test_cases = [
    ("EDIT DSL格式", test1),
    ("带多个EDIT ADD的格式", test2),
    ("Markdown代码块格式", test3),
    ("直接函数定义", test4),
    ("只有return语句", test5)
]

for name, test_str in test_cases:
    print(f"\n测试案例: {name}")
    print("-" * 40)
    result = extract_function_body_v2(test_str)
    if result:
        print(f"✅ 提取成功!")
        print(f"函数体:\n{result}")
    else:
        print(f"❌ 提取失败!")
        print(f"返回结果: {repr(result)}")
    print("-" * 40)

print("\n测试完成!")
