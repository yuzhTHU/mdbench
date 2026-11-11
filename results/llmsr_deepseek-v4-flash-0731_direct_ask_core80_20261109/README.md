# llmsr + DeepSeek Direct-Ask — Core80 (2026-11-09)

本目录代表同一次 Core80 实验，80 题均已结束。实际 performance.json 共 69 份，严格完整原生评分 41 题；完整评分不代表答对。

遵循 results/README.md，不为缺失 performance 创建占位结果。setup.json 的 planned_runs 列出全部80题、结果路径或缺失原因；原始错误终态和冻结哈希见 ../_summaries/rebuttal-sr-core80-20261109/provenance/llmsr.jsonl。原始运行不重跑、不择优、不拼接不同尝试。

新增 submission 字段来自原生 evaluation.submitted_solution 的规范化方程，不冒充原始冻结文件字节。source.file_sha256 是服务器原 performance.json 的 SHA256。费用仅使用已保存的原始 usage，金额/Token未知保持 null；未用账户余额差估算。

汇总同时提供论文整题 SA(M) 与 Legacy 探针通过比例，二者不能混用；缺失和失败在固定分母汇总中贡献0，原生指标不补造为 False。详见 ../_summaries/rebuttal-sr-core80-20261109/README.md。
