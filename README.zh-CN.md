# 机制发现基准

[English](README.md)

Mechanism Discovery Benchmark（`mdbench`）用于评估智能体能否同时恢复可观测的输入—输出规律以及产生该规律的隐藏机制。可观测规律在训练集、同分布测试集和分布外测试集上评测；隐藏状态通过相互独立的机制探针评测。所有分数均来自客观的符号等价性与数值精度评估结果，无需引入 LLM 裁判。

有关动机和方法论设计原则，可参见 [`proposal.md`](proposal.md)。

## 快速开始

需要 Python 3.12 或更高版本。

```sh
pip install -e '.[dev]'

# 验证任务实例
# `--tasks` 接受任务文件或目录。指定目录时，目录会被递归扫描。
# 因此，可以通过 `--tasks 'tasks/'` 直接选择完整的内置任务库（`.yaml.archived` 文件会被忽略）
mdbench validate \
  --tasks 'tasks/astronomy/binary_mass_function/Astronomy Binary Mass Function - Original.yaml'

# 将任务实例转换为问题文件、答案文件、数据文件
mdbench export \
  --tasks 'tasks/astronomy/binary_mass_function/Astronomy Binary Mass Function - Original.yaml' \
  --output-dir data/tasks \
  --seed 0

# 将问题文件提供给 algorithm，并自动评测提交结果
# 改为 `--algorithm codex` 即可运行 Codex 基线（需要已安装 `codex` CLI，并完成模型提供商配置）
mdbench run \
  --algorithm dummy \
  --problem-file 'data/tasks/Astronomy Binary Mass Function - Original/problem/problem.json' \
  --answer-file 'data/tasks/Astronomy Binary Mass Function - Original/answer/answer.json' \
  --save-path logs/demo
```

完整参数见 `mdbench <command> --help`。

`validate` 默认使用精确符号运算的加速路径。复杂任务可用
`pip install -e '.[dev,fast]'` 安装可选的 FLINT 多项式后端；未安装时自动使用
SymPy。加速不跳过方程、探针或量纲检查，也不用数值抽样代替符号验证。


## 基准流程

1. **定义任务：** YAML 文件描述观测变量、隐藏变量、代数机制、可观测规律、采样范围、单位和隐藏状态探针；可从 [`demo_task.yaml`](demo_task.yaml) 开始。
2. **验证任务：** `mdbench validate` 检查结构、方程可解性、可观测等价性、量纲、采样规则与探针一致性。
3. **导出数据：** `mdbench export` 生成公开的智能体输入，以及包含可复现 train/ID/OOD 数据的私有评测输入。
4. **运行评测：** `mdbench run` 运行算法并冻结最终状态，随后评测提交的可观测模型；每个机制探针都从该状态的独立副本开始。

导出目录严格区分公开与私有信息：

```text
<task>/problem/problem.json       公开的变量元数据
<task>/problem/train.npy          公开的训练观测
<task>/answer/answer.json       私有的任务与探针定义
<task>/answer/{train,id_test,ood_test}.npy
```

数组布局为 `(变量数, 样本数)`，行顺序由 `data_columns` 指定，target 位于第一行。
评测时只能向智能体提供 `problem/` 下的文件。

## 反馈服务

```sh
mdbench feedback --port 8000 --workers 4
```

`POST /evaluate` 接收 `problem`、`train_data` 和 `submission` 三个 multipart 文件，返回显式解和训练集指标。

该服务包装了方程组求解、数值计算等基础设施服务，可供智能体在发现过程中使用，且不会泄露私有答案。

`mdbench run` 默认会自动启动隔离的本地服务，也可通过 `--feedback-server-url` 复用已有服务。

## 仓库结构

- `src/`：验证、数据生成、导出、反馈、评分和算法适配器
- `tasks/`：当前任务库
- `tests/`：单元、行为、流水线与 Runner 测试
- `legacy/`：已归档且不再支持的旧实现

运行测试：

```sh
python -m pytest
```

本项目采用 [MIT License](LICENSE)。
