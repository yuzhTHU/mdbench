# MDBench

[![PyPI](https://img.shields.io/pypi/v/mdbench.svg?logo=pypi&logoColor=white&label=PyPI&color=3775A9&cacheSeconds=300)](https://pypi.org/project/mdbench/)
[![Python](https://img.shields.io/pypi/pyversions/mdbench.svg?logo=python&logoColor=white&label=Python&color=3776AB&cacheSeconds=300)](https://pypi.org/project/mdbench/)
[![Documentation](https://img.shields.io/badge/Documentation-online-0A7B83?logo=readthedocs&logoColor=white)](https://yuzhthu.github.io/mdbench/)
[![License](https://img.shields.io/pypi/l/mdbench.svg?label=License&color=2E8B57&cacheSeconds=300)](LICENSE)

[English](README.md)

MDBench 用于评估 AI 从方程或观测数据中恢复科学规律及其生成机制的能力。

## 对机制发现的理解

MDBench 将唯象方程理解为若干简单、相互一致的关系共同作用后的可观测结果。唯象方程描述变量之间“有什么关系”，机制则通过物理关系、假设和中间变量解释这种关系“为什么成立”。

例如，圆轨道下的开普勒第三定律可以由万有引力定律、牛顿第二定律和匀速圆周运动关系导出。示例见 [`problems/demo_problem.yaml`](problems/demo_problem.yaml)。

每条机制关系使用 `variable = formula`，其中公式必须能被 [nd2py](https://pypi.org/project/nd2py/) 解析。显式关系构成 DAG：

```text
a = f1(x)
b = f2(x, a)
y = f3(x, a, b)
```

系统也支持隐式方程组。程序会收集相关方程，直到未知变量构成闭合系统，再进行符号求解或数值求根：

```text
a = f1(x, a, b)
b = f2(x, a, b)
y = f3(x, a, b)
```

所有变量在 `variable_description` 下声明为 `target`、`inputs`、`intermediates` 或 `auxiliary_inputs`。最后一类是仅被机制使用、并在最终唯象方程中被消去的外部输入。原始关系保存在 `Problem.mechanism`，可执行的求解步骤保存在 `Problem.solution`。

## 任务与评估

MDBench 包含三类任务：

1. **符号回归**：`(X, y) → 唯象方程`。
2. **机制解释**：唯象方程 → 机制方程。
3. **机制发现**：`(X, y) → 机制方程`。

机制评估只汇报相互独立的分项，不提供总分：

- **预测准确度**：用于符号回归和机制发现；在 Feedback 的公开训练集或 Final 的 Train/ID/OOD 数据上汇报 Pearson r、R²、MAE、RMSE、sMAPE 和误差容限准确率。
- **导出方程等价性**：仅在 final 中通过 SymPy、私有数值点和 LLM 交叉判断导出方程是否等价于私有唯象方程。
- **机制基础性**：LLM 评价，主要由最不基础的机制关系决定；不依赖参考答案。
- **Ground Truth 结构还原度**：对参考机制进行公式 AST 与依赖图的软匹配；忽略变量名和数值常量的具体取值。
- **机制描述复杂度**：每条关系的平均/最大 AST 节点数及总节点数；不依赖任何参考答案，数值越低越简洁。

安装要求 Python 3.12 或更高版本：

```bash
pip install -e ".[dev]"
mdbench --help
```

Sphinx 文档位于 [`docs/`](docs/)，构建方式如下：

```bash
cd docs
make html
```

## 命令

### 导出内置问题

MDBench 的发行包包含参考问题库，无需下载仓库文件即可将其导出到本地：

```bash
mdbench export --output-dir problems/
```

覆盖已有文件前需要确认；非交互式运行可以指定 `--force`。目标目录中的其它文件不会被删除。

下面的生命周期命令通过 `--problems` 接收一个或多个 YAML 文件或目录，默认使用 `./problems`。

### 验证问题

检查配置格式、变量使用、物理单位、采样设置、显式或隐式方程求解，以及机制能否导出目标唯象方程：

```bash
mdbench validate
mdbench validate --problems problems/demo_problem.yaml
```

可以选择调用 LLM 判断每条机制关系是否足够基础。API 或响应错误会直接报告，不会退回启发式规则：

```bash
mdbench validate --check-fundamentality \
  --llm-provider deepseek --llm-model deepseek-v4-flash
```

### 合成数据

生成可复现的训练集、样本内测试集和样本外测试集：

```bash
mdbench synthetic --problems problems/ --output-dir data/synthetic_data/
```

每个 NPZ 保存三组数据、由 `variables` 指定的行顺序，以及记录随机种子和样本数量的 JSON `generation_config`。辅助输入在此阶段一并生成，在准备任务时可以选择隐藏。

### 准备任务

运行前必须已经生成 synthetic data。默认不保存私有答案：

```bash
mdbench prepare \
  --problems problems/ \
  --synthetic-data-dir data/synthetic_data/ \
  --task mechanism_discovery \
  --format directory
```

`--save-answer` 会保存答案和测试集；`--reveal-auxiliary` 会在机制任务中公开辅助输入；`--force` 表示同意覆盖计划中的文件。程序不会清空已有目录，并会提示未被覆盖的冗余文件。`--format directory` 将内容平铺为多个文件，`--format file` 将相同内容打包为一个 NPZ。

### 评估提交

提交可以是直接输入的公式、由分号分隔的多条机制方程，或者每个非空行包含一条方程的纯文本文件。JSON 和 YAML submission 不受支持。

```bash
mdbench evaluate \
  --evaluation-mode feedback \
  --problem data/problem/PREPARED_TASK \
  --submission submission.txt \
  --verbose
```

feedback 模式只使用公开任务和训练数据。Benchmark 管理者使用
`--evaluation-mode final --answer answer.json` 进行最终评价，此时还会使用
隐藏的 ID/OOD 测试集并评价参考机制结构的恢复程度。运行 Agent 时，只将准备
好的公开任务复制到独立的临时工作目录，并要求 Agent 不得离开该目录。由于其中
没有源 problem YAML 和私有 answer，其它生命周期命令与 final evaluation 都无
法取得所需材料。`--verbose` 会以简洁的等式链打印显式或隐式求解结果。

机制基础性会自动调用配置的 LLM，并在结果中打印 provider 和 model：

```bash
mdbench evaluate \
  --evaluation-mode feedback \
  --problem data/problem/PREPARED_TASK \
  --submission submission.txt \
  --llm-provider deepseek \
  --llm-model deepseek-v4-flash
```

`scripts/` 中提供了功能相同的独立入口：

```text
validate_problem_main.py   验证问题定义
synthetic_data_main.py     生成合成数据
prepare_problem_main.py    准备公开及私有任务文件
evaluate_result_main.py    评估提交结果
```

`scripts/visualize_mechanism_main.py` 可以将求解后的机制渲染为 DOT、SVG、PNG 或 PDF；非 DOT 格式需要安装 Graphviz。

## 目录约定

```text
problems/                  原始问题 YAML
data/
  synthetic_data/         生成的 train/ID/OOD 数据
  problem/                准备后的 Benchmark 任务
src/
  core/                   基础数据模型
  features/               项目专用的读取、求解、验证与采样功能
  metrics/                公式与机制评价指标
  utils/                  通用工具和 LLM 客户端
scripts/                   独立命令入口
tests/                     单元测试和验证样例
```

目录格式的问题包含：

```text
problem.json               公开的问题描述
data_train.npy             公开的训练数据
answer.json                可选的私有答案
data_id_test.npy           可选的私有样本内测试数据
data_ood_test.npy          可选的私有样本外测试数据
```

不使用 `--save-answer` 时只写入 `problem.json` 和 `data_train.npy`。公共接口保持基础数据格式：单位使用 `Dict[str, int | float]`，公式使用 nd2py 可解析字符串，数据使用 NumPy 格式。
