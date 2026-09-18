# 机制发现 Benchmark

本实现以 [proposal.md](proposal.md) 为唯一设计依据，通过符号等价性和数值精度分别评估唯象模型恢复与机制探针恢复。新流程不包含 Fundamentality、DAG 恢复或 LLM 主观打分。

安装：Python 3.12+，执行 `pip install -e '.[dev]'`。Codex 算法另外需要已安装且可认证的 `codex` CLI。

任务实例直接存放在 `problems/` 下，文件名必须与 `task_name` 完全一致。`demo_problem.yaml` 是模板；对应可验证实例是 `problems/Electrical Dissipation - Variant 1-2.yaml`。旧任务保留在 `problems/legacy/`，不会被扫描。

```sh
mdbench validate --problems problems
mdbench export --problems problems --output-dir data/tasks --seed 0
python run.py --algorithm codex \
  --problem-file 'data/tasks/Electrical Dissipation - Variant 1-2/agent/problem.json' \
  --train-data-npy-file 'data/tasks/Electrical Dissipation - Variant 1-2/agent/train.npy' \
  --answer 'data/tasks/Electrical Dissipation - Variant 1-2/answer/answer.json' \
  --timeout 600 --probe-timeout 120 --save-path logs/demo
```

公开的 `agent/` 只包含可观测变量描述与训练数据，私有的 `answer/` 包含机制模型、唯象模型、机制探针及 train / ID / OOD 数据。NPY 使用 `(变量数, 样本数)` 布局，行顺序由 `data_columns` 给出，第一行是 target。

机制公式允许任意等价写法和任意顺序，但必须能唯一求出所有内部变量与 target 关于 input / auxiliary 的显式代数表达式。欠定、多分支和不一致的模型会明确报错。auxiliary 会产生警告；探针必须最终只依赖 input。常数通过机制方程定义，不存在独立 ConstantSpec。数值可携带任意物理单位，不产生单位警告。

`mdbench feedback --port 8000` 启动并发、缓存的 POST 服务。向 `/evaluate` 上传 multipart 文件字段 `problem`、`train_data`、`submission`，得到提交模型的显式解与训练精度，不需要标准答案。模型无效时返回 `ok: false`。本机访问应绕过 HTTP 代理。

提交文件每行一个等式，可以用分号分隔并添加 `#` 注释。Codex 运行后保存 submission、events、session checkpoint 与 model；runner 保存耗时及去除 ANSI 的 `performance.json`。超时后保留已写出的提交与现场，没有完整产物则记录失败。

离线评测分别计算 train / ID / OOD 的 MAE、MSE、RMSE、MAPE、R²、NRMSE、最大误差、有限预测比例，以及符号和数值等价性。每个探针都从同一个结束 checkpoint 的独立副本恢复会话，尽量禁用工具，并提示不得修改原机制。展开探针回答只使用提交模型，不能使用标准内部状态补全回答。失败探针仍计入恢复率分母。

旧代码、测试、脚本和 baseline 已归档到 `legacy/`，新代码不兼容旧接口。任务的科学意义、探针非平凡性和机制可识别性仍需人工审查。[迁移记录](playground/rewrite_plan.md) 说明代码边界；[英文 README](README.md) 给出完整字段、命令及实现限制。

运行 `python -m pytest` 验证新流程。真实 Codex CLI 恢复测试使用本机模拟 API，不调用付费模型。
