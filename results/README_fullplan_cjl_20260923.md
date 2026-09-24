# Chengjialin：Core / Full 逻辑实验结果

每个实验目录对应一个模型的一次 Core80 或 Full512 逻辑实验，不再按调度批次、补丁批次或恢复批次分散成小目录。

## 六个目录

| 实验 | 计划题数 | 实际原生成绩 JSON | 缺失文件 |
|---|---:|---:|---:|
| [DeepSeek Core r1](./codex-cjl-deepseek-v4-flash-0731-low-coreset-r1/setup.json) | 80 | 80 | 0 |
| [DeepSeek Core r2](./codex-cjl-deepseek-v4-flash-0731-low-coreset-r2/setup.json) | 80 | 79 | 1 |
| [DeepSeek Core r3](./codex-cjl-deepseek-v4-flash-0731-low-coreset-r3/setup.json) | 80 | 80 | 0 |
| [GLM Full r1](./codex-cjl-glm-5.3-flash-high-fullset-r1/setup.json) | 512 | 507 | 5 |
| [GLM Core r2](./codex-cjl-glm-5.3-flash-high-coreset-r2/setup.json) | 80 | 79 | 1 |
| [GLM Core r3](./codex-cjl-glm-5.3-flash-high-coreset-r3/setup.json) | 80 | 79 | 1 |
| 合计 | 912 | 904 | 8 |

**GLM 首次 Core 的 80 道题已经包含在 Full512 中**，不另建重复目录。GLM Core 三次重复统计 = Full 中首次 Core80 + Core r2 + Core r3。
每个目录内仍按 `领域/题目组/题目.json` 存放，另有 `setup.json`。缺少原生 performance 的 8 项只列清单，不创建占位成绩；实际错误 JSON 可以收录，文件存在不等于有效评分或答对。

## 选用口径与来源

- 每个计划身份只保留既有总统计中固定选用的一整个尝试；不按最好分重新选择、不跨尝试拼分。独立重复不混为一次。
- 904 个选用成绩文件按字节原样搬移：原生评分、submission、source/SHA、费用、token 及 update 历史全部不变。文件中的 experiment_provenance.experiment_name 是原来源 setup 标识，由本目录 setup.json 的 source_setups 映射解析。
- 按用户要求将历史 setup 片段合为逻辑实验：模型、推理档及文件名保持对应关系，但原版/补丁代码、NumPy/FLINT、历史 BigModel/OpenRouter、恢复协议并不统一。完整原 setup 元数据保存在 source_setups 中，不能把新目录称为单一同质环境实验。
- 43 次事后日志恢复明确标注，不冒充原 900 秒内正式提交。submission 保留 runner 的规范化送评分式，冻结文本/候选仍是独立证据。
- 用户的超时/ValidationError 后处理判错只用于派生统计，不回写原生评分；API、基础设施及缺证仍不强行填零。
- 费用只在原账本能支持完整数值时提供；目录内选用尝试费用不等于包含所有历史尝试的总费用。历史 BigModel 金额未知不填零。

## 缺失原生文件的 8 项

- `g3-r1-t169` → `codex-cjl-glm-5.3-flash-high-fullset-r1`
- `g3-r1-t192` → `codex-cjl-glm-5.3-flash-high-fullset-r1`
- `g3-r1-t486` → `codex-cjl-glm-5.3-flash-high-fullset-r1`
- `g3-r1-t490` → `codex-cjl-glm-5.3-flash-high-fullset-r1`
- `g3-r1-t503` → `codex-cjl-glm-5.3-flash-high-fullset-r1`
- `g4-r2-t009` → `codex-cjl-deepseek-v4-flash-0731-low-coreset-r2`
- `g5-r2-t049` → `codex-cjl-glm-5.3-flash-high-coreset-r2`
- `g5-r3-t057` → `codex-cjl-glm-5.3-flash-high-coreset-r3`

## 统计与历史保留

[24 条 P%/M% 与 72 条 Original/单变体/多变体细分](./_summaries/cjl-fullplan-20260924/README.md) 的数值、分母和结果选用完全不变；原文件路径引用已同步。
912 项固定选用关系、904 文件来源及 8 项缺失见 [逐题索引](./fullplan_cjl_20260923_export_index.json)。
原 32 个批次片段目录的全部 996 份成绩（含当前未选用的 92 份历史成绩）、setup 和旧统计保存在可恢复的本地 layout32_archive_20260924 备份中；Git checkpoint `2746ac1` 保留原成绩归档，`5e8af55` 保留旧布局及统计。原始远端日志及所有尝试未改、未删除。
索引的 historical_attempt_archive 留存历史批次计数、未选用文件来源及 158 次历史尝试无 performance 的清单；这些不是新增独立题目，也不能和当前 8 项缺失直接相加。
上游 results/README.md 保持原样；本次仅按用户明确要求调整逻辑目录粒度，并充分披露混合历史 setup。未重新运行实验、评分或发出模型请求。
