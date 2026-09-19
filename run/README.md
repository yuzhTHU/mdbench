# 可审计的 OpenRouter 批量实验

`openrouter_experiment.py` 是现有 `run.py` 的批量控制器。它只从仓库 `.env` 的唯一 `OPENROUTER_API_KEY` 赋值读取真实密钥，不使用继承的同名环境变量，也不执行 shell 配置文件。Codex 使用隔离的 `openrouter` profile；真实密钥仅由本地计费网关持有，Codex 得到的是无外部权限的本地凭证。

直接运行 `run.py` 时通过 `--exp_name NAME` 指定实验名，默认结果目录为 `logs/run/NAME/<task_name>/`。批量控制器也将每道题的 dataset、运行产物和评估结果放在实验根目录下以 `task_name` 命名的独立目录中；实验级账本和汇总仍位于实验根目录。

一次实验的 `experiment.json` 固定随机种子、问题族、问题清单、模型、时间上限和预算。先运行两题 smoke；只有生成 submission、可恢复会话、evaluation 后，才写出 `budget_plan.json`。预算估计使用预跑最大单题成本的三倍，且设置最低每题估计；必要时按问题族交替删减题数。默认每题发现阶段 900 秒，每个探针独立恢复 checkpoint、120 秒；普通 `run.py` 调用默认串行执行探针。

```sh
venv/bin/python -u run/openrouter_experiment.py --root logs/run/<experiment> --phase smoke --concurrency 2
venv/bin/python -u run/openrouter_experiment.py --root logs/run/<experiment> --phase batch --concurrency 10
```

长期批量通过 tmux 运行。batch 在首次启动时执行同并发数的短请求预检，结果保存在 `capacity_probe.json`。已完成或失败的题目不会在重启时重复提交；未完成任务可继续调度，但应先归档该任务的旧输出目录，避免混入旧 submission。

日志目录包含：

- `controller.log`：复用 legacy logger/tag2ansi，文件中去掉 ANSI。
- `usage.jsonl`：每次请求开始、响应 ID、费用、input/output/cache tokens、错误、预算熔断均追加并 fsync。
- `usage.json`：原子写入累计费用、在途预留、token、时间和每题费用。优先使用供应商报告的 cost；缺失时按价格估算，未完成请求保守计入预留上限，明确标注来源。此值可能高于实际账单。
- `api/<request-id>/`：请求 JSON、HTTP 状态、原始 SSE 和错误。没有真实密钥或 Authorization 头。
- `tasks.json`、`summary.json`：原子更新题目状态和性能；失败题仍计入准确率分母，Original 对照与变体分别统计。
- `<task_name>/`：该题的 prompt、submission、performance、审计记录、冻结 checkpoint、探针问答和自包含的 `dataest/`。后者包含 `problem.json`、`answer.json`、`train.npy`、`id_test.npy` 和 `ood_test.npy`。Codex 工具被禁止读取私有 answer、仓库题库、私有日志、proposal 生成现场和 `.env`；仅当前 workspace 可写。
- `source_snapshots/`：启动时的关键源码快照，不通过 git 操作。

网关限制模型、并发和单响应最大生成 token；费用加在途请求预留不得超过预算。401/402/403 立即熔断，连续三次网关系统错误或连续三题基础设施失败也停止整批；429 有限重试。单题失败跳过继续。收到中断信号时停止派题，通知正在运行的 runner 退出并持久化部分结果。

`tests/test_experiment.py` 使用离线 provider 验证密钥来源、崩溃后的计费恢复、并发预算、原始 usage 落盘、profile 隔离；其中真实 Codex/沙箱集成测试只连接本机模拟服务，验证 numpy 可用且私有文件不可读，不消费外部 API。

运行工具在 workspace/.tmp 中创建临时文件，通过 TMPDIR/TMPPREFIX/MPLCONFIGDIR 支持 heredoc 和科学计算缓存，禁止向共享 /tmp 写入。正常时间预算耗尽但没有 submission 的情况属于算法失败，保留 checkpoint、计 0 分并继续派题；tests/test_experiment.py 还验证连续四题这种失败不会导致误熔断。

`playground/experiments/report_experiment.py` 生成可读 report.md 与逐题 metrics.csv；`playground/experiments/watch_experiment.py` 在 tmux 中持续更新，批量完成时退出。`playground/experiments/reconcile_usage.py` 使用同一仓库 .env 密钥查询中断请求的 generation 账单，写入独立 billing_reconciliation.json，不修改正在运行的预算账本。不要累加恢复后 Codex 事件中的累计 usage；计费网关逐请求记录的 usage 才是本次新增消耗。每题费用包含其全部预跑和重试，相关尝试在 attempts/ 中保留。


当前代码边界：`src/algorithms/codex.py` 提供显式文件密钥读取、持久化用量账本、可配置上游地址的 Responses 网关、并发预检、子进程环境隔离、Codex 进程清理和失败分类。`openrouter_experiment.py` 负责 OpenRouter 模型元数据查询、选题、smoke/repair/batch 调度、预实验预算估计、状态汇总和源码快照。现有算法 `run`/`resume` 接口保持不变，普通单题调用不会自动启动计费网关。

观察、汇报和账单核查工具的用法见 [playground/experiments/README.md](../playground/experiments/README.md)。历史日志中记录的旧命令和源码快照保留当时内容，不随文件迁移改写。
