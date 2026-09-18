# Validate 行为验证

这些测试遵循 [遗产行为测试的编写要求](../../legacy/tests/behavior/README.md)，但行为依据是 [proposal.md](../../proposal.md) 和用户对新 validate 的要求。测试通过公开的 `load_task`、`validate_task`、`expand_expression` 接口，以及实际 CLI 或集合验证入口观察结果。

每个主题保留 5–8 个测试函数，直接展示任务 YAML、变更输入和预期输出；参数化仅用于同一行为的代表性写法和边界。没有 mock、网络调用或 LLM 评判。数学结果来自手工推导，不能把求解器自己的输出当成正确答案。已有普通测试中重复的探针、常数和单位示例已合并到这里；公式解析安全性、YAML 重复字段等细节回归仍保留在 `tests/test_validation.py`。

## 如何运行和阅读

```sh
python -m pytest tests/behavior -v
```

`python -m pytest` 也会自动发现这些测试。规范示例在 pytest 临时目录中生成；化学任务测试直接读取仓库中的新任务族。测试不改动任务库，自动验证不能代替科学审查。

| 主题 | 文件 | 应当观察到的行为 |
| --- | --- | --- |
| 任务文件与集合 | [test_task_validation.py](test_task_validation.py) | 完整任务 CLI 返回成功 JSON；文件名与 task_name 一致；跨目录重名失败；同族描述不同失败；旧字段、不合法角色/变量名、重复变量、零个或多个 target 失败；OOD 边界不能等于 min/max；target/internal 不能采样；不读取 legacy 与 `.yaml.archived`。 |
| 机制代数求解 | [test_mechanism_solving.py](test_mechanism_solving.py) | 两组耦合方程最终得到 `a=2*x, b=x, c=3*x, d=2*x, y=5*x`；公式和变量顺序不影响结果；允许残差形式与等价常数定义；公式数量足够仍不保证内部状态唯一；不能用标准答案猜测机制的多分支；显式 sqrt 可以指定分支；不一致机制失败；唯象公式必须以 target 为左侧、仅依赖 inputs、且被机制蕴含。 |
| 单位约束 | [test_unit_validation.py](test_unit_validation.py) | `h=3*x+3` 中两个数值 3 可分别作为无量纲乘数和有长度单位的偏置，且不报警；缺失 internal 单位可以由整体关系推导；已知长度与时间不能相加；代数消去不能掩盖原式的量纲错误；`exp(x)` 对有长度单位的 x 无效，`exp(x/2.0)` 可以通过给数值尺度赋单位而有效。 |
| 机制探针 | [test_probe_validation.py](test_probe_validation.py) | answer 可以是 internal 名称、表达式或等式，并展开为 input-only；input/target 不能当探针；错误标准答案失败；能求出的无关 internal 也不能当探针；auxiliary 发警告但不直接否决任务，仍求出所有状态；依赖 auxiliary 的探针失败；常数探针、未声明派生探针发人工审查警告。 |
| 八个化学任务族 | [test_chemical_tasks.py](test_chemical_tasks.py) | 每族包含蓝本及全部一至五重组合；描述、变量与采样范围稳定；八个蓝本符合独立手算结果；五重组合改变目标和至少一个探针；真实 ID/OOD 数据有限；朗缪尔的七个平凡探针候选归档。构造与验证记录见 [任务族说明](../../playground/chemical_families/README.md)。 |

## 关键可核验例子

耦合机制不要求每条公式的左侧都是新变量：

```text
a + b - 3*x = 0
a - b = x
c + d = 5*b
c - d = b
y = c + d
```

手工推导先得到 `a=2*x, b=x`，再得到 `c=3*x, d=2*x`，最终 `y=5*x`。测试明确核验所有 internal 和 target 的完整表达式。

对于 `a+b=x; y=a+b; 2*y=2*x`，target 可以唯一写成 `y=x`，且公式数等于 internal 数加一，但 a、b 各自仍不唯一，因此必须失败。对于 `a^2=x; y=a`，声明 `phenomenal_model: y=sqrt(x)` 不能替机制消除负分支；把机制改为 `a=sqrt(x)` 才可通过。

auxiliary 示例为 `h=x+a; s=h-a; y=s*x`。即使 h 依赖辅助变量 a，所有内部表达式仍应被保存：`h=x+a, s=x, y=x^2`。选择 s 为探针可通过并产生 auxiliary 警告；选择 h 为探针必须因不能消去 a 而失败。

这次行为测试发现了一个真实偏差：原实现只识别 `k=2` 形式的未声明数值常数，把 `2=k`、`k-2=0` 和 `2*k=4` 错判成未声明变量。新实现按方程的唯一数值解识别这些等价定义，四种写法都有可执行验证。

## 自动验证的边界

通过这些测试意味着上述契约在这些代表性输入上成立，不是对任意数学方程的证明。求解器只接受唯一的显式代数解，无法显式求解的任务应人工审查或归档；不通过数值拟合或目标标准答案替未观测状态选择分支。

单位检查不能证明机制科学正确；数值常量可携带任意单位，约束欠定时也不保证能唯一推导内部单位。结构有关联性的检查不能证明探针具有独立科学意义或非平凡性。英文描述的科学合理性、采样范围的科学意义、变体是否泄露答案、机制可识别性和领域非平凡性仍需人工审查。

这里没有 Fundamentality 或 DAG 恢复评分，也没有把这些已移除功能重新引入 validate。
