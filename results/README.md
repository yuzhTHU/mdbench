# 运行结果汇总

通过 `mdbench run` 启动的运行结果默认会存储在 `./logs` 中。然而，由于可能涉及冒烟测试、预实验、多进程、补跑和重测，`./logs` 中的结果往往比较杂乱，不容易共享、汇总、和统计。为此，在本目录（`./results`）中构造一个用于统计实验结果的抽象层，以便以易于批量检查和处理的格式导入 `./logs` 中真正有用的结果。

本目录约定如下：

* `./logs` 中的原始日志不得随意修改。
* 每个实验都应该创建一个 `./results/{exp_name}/` 目录
    - setup 不同的结果应当放在不同的实验目录中。
    - 例如，`./results/codex_deepseek-v4-flash-0731/` 或者 `./results/codex_gpt-5.5-sol-run1/`
    - 对于多次独立运行的实验，将每次运行结果分散到不同的 `exp_name` 中。
* 运行结果存储到 `./results/{exp_name}/{physics,chemistry,biology,material,...}/{task_name_wo_suffix}/{task_name}.json` 中
    - 例如，`./results/codex_deepseek-v4-flash-0731/biology/Allosteric Regulation/Allosteric Regulation - Original.json`
  "total_seconds": 989.3036813475192,
    - 该结果 JSON 文件应当以 `mdbench run` 生成的 `performance.json` 为基础（其中通常包含了 algorithm, agent_seconds, evaluation_seconds, evaluation, total_seconds, ok 等字段），
    - 结果 JSON 文件中还应当添加 `source: {hostname: ..., path: /path/to/corresponding/performance.json, file_sha256: ...}` 字段与 `update: [{updated_at: ..., update_message: ...}]` 字段，以供追溯原始运行结果。
    - `performance.json` 中默认没有记录 Agent 运行的 `cost_in_USD` 和 `used_token`。如果可能的话，建议尽量将这两个字段也补全到结果 JSON 中。
    - `performance.json` 中默认没有记录 submission。如果存在的话，将此字段也补全到结果 JSON 中。
    - 缺失结果不得创建占位文件；实际生成的错误结果可以收录。
* 对于 `./results/{exp_name}/` 目录，其中包括的运行结果应该基本采用同质化的 setup（允许随机种子等运行参数不同），并使用 `./results/{exp_name}/setup.json` 记录实验 setup。
    - setup JSON 中应当包括但不限于 `algorithm`、`model`，以及 `src/algorithm/*.py` 中 `update_parser` 所设定的参数。
