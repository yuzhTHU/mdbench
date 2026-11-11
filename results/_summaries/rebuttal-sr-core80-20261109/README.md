# 三个新增符号回归 + Direct-Ask 基线（Core80）

2026-11-09 正式实验，2026-11-11 整理。每算法独立运行80题，四领域各20题，共240次；不是 Full512。全部已结束，所有尝试保留，没有补跑或按最好成绩选择。

## 结果范围

| 基线 | 计划/终态 | performance文件 | 严格完整原生评分 | 原生无效/探针缺有效回答 | 评分基础设施超时 | 生成网关故障 |
|---|---:|---:|---:|---:|---:|---:|
| AI-Feynman |80/80|75|26|49|5|0|
| E2ESR |80/80|75|27|48|5|0|
| LLM-SR (DeepSeek) |80/80|69|41|28|2|9|

依 results/README.md，不为21次缺失performance生成占位结果。每个实验目录的setup.json含完整80题清单和缺失原因；本目录provenance/*.jsonl另存全部240次原始终态、冻结及discovery回执的解析内容和原文件SHA256。219份实际performance原生字段均保留。

严格完整表示P及全部计划探针的ID/OOD数值、符号四指标齐全且为布尔值，False也属于有效评分。native_invalid并非全部是方程ValidationError，也包括API返回无最终assistant内容、输出无法解析等。12次2700秒超时是评分基础设施限时，9次LLM-SR生成网关故障另列。Direct-Ask的180秒是网络请求超时，不是此前agent实验的120秒回答预算。

## 两种M口径分别保留

- **P**：符号等价的现象方程题数 / 全部计划题数。
- **paper_task_SA_M**：全部计划探针均符号等价的题数 / 全部计划题数，不以P通过为前提。
- **legacy_probe_M**：符号等价的探针条数 / 全部计划探针数，与用户桌面Excel的Legacy Core一致。
- 缺失、错误、不可评估结果按用户要求在固定分母汇总中贡献0，原生JSON的缺失、null、错误信息仍保留，不篡改为科学判错。汇总不只使用有分的子集。
- 原生numerically_equivalent与论文Acc_0.1不是同一阈值，本汇总不混用。所有现存train/ID/OOD符号标志已核对一致。

| 基线 | P（80题） | 论文整题M（80题） | Legacy探针M（165探针） |
|---|---:|---:|---:|
| AI-Feynman |0/80 = 0%|0/80 = 0%|0/165 = 0%|
| E2ESR |0/80 = 0%|1/80 = 1.25%|1/165 ≈ 0.6061%|
| LLM-SR |0/80 = 0%|1/80 = 1.25%|2/165 ≈ 1.2121%|

四领域P均为0。Legacy物理M分别0/36=0%、1/36≈2.78%、2/36≈5.56%；化学、生物、材料均0。E2ESR及LLM-SR的整题M成功均为Hall Response - Original；LLM-SR另有Hall Response - Variant 4-5的mobile探针通过，但整题未全部通过。P=0表示未满足严格符号等价，不表示未拟合方程。P全0使1−M|P未定义，保留null。

Core80包含8道Original、16道单变体、56道多变体；每领域分别2/4/14题。各层任务数与探针数均在CSV明确列出，避免混用分母。

## 文件与复算

- summary.json：两种口径、逐组统计、输入SHA。
- domain_variant_breakdown.csv：三算法×(合计+四领域)×(合计+Original+单变体+多变体)，60行。
- per_task.csv：全部240次逐题数据，明确原生未知值与政策判零指标。
- provenance/*.jsonl：真实终态和冻结/发现回执，不是performance占位文件。
- source_code_record.json：本批实际使用的原生源码Git blob与SHA256。
- build_export.py：由既有只读原始快照构建归档的脚本，需本地原始batch副本；不发起实验。
- recompute.py：仅依赖本仓库已导出的结果和setup，Python标准库，无模型或公式求值调用。

从仓库根目录运行：

```bash
python3 results/_summaries/rebuttal-sr-core80-20261109/recompute.py --results-root results
```

## 实验设置与可比性

发现阶段900秒；正式种子20261109+task_index；只用公开1000训练样本选择候选。三算法的机制问答统一OpenRouter DeepSeek-v4-flash-0731，minimal、temperature0、max_output_tokens2048、180秒请求超时、最多4次有限传输尝试、probe_workers1。LLM-SR候选生成另用同模型low/temperature1；其数学AST适配限制、仿射种子和BFGS拟合等参数见setup.json，种子允许获胜，不将其冒称LLM发现。

原生科学源码固定 f133fd197f38135e748350670bf18b4acfb4491a；三个算法各自官方版本、模型权重及适配参数在setup.json。旧PySR是不同版本的历史实验，本次未修改或重新求值其结果；只保证所列聚合定义一致，不宣称运行版本完全相同。每种算法仅一次运行，不据此声称显著差异。

新增submission文本来自原生submitted_solution规范化方程，不冒充冻结原文件字节；冻结SHA另存。服务器此时连接超时，归档依据2026-11-09T17:13:12.217207+00:00已下载只读快照，原始performance内容与其SHA可追溯。费用仅取已保存原始usage，未知金额/Token为null；不把账户余额差当成本。
