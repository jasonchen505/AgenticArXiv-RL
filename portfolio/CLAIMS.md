# 简历结论与证据对应

先用 [INTERVIEW.md](INTERVIEW.md) 讲清问题，再按此表回答具体证据。表中的训练与成绩来自历史记录，本轮没有训练或生成新答案。

| 可以使用的结论 | 代码入口 | 公开证据 | 必须一起说明的边界 |
|---|---|---|---|
| 完成 Qwen2.5-1.5B QLoRA/SFT，4,776 行、597 步 | [train_sft.py](../AgenticArxiv/rl/train_sft.py) | [原始训练配置](evidence/training-records/r003-sft-training-manifest.json)、[曝光与权重来源](evidence/training-records/r003-sft-provenance.json) | 当前训练代码已有后续修改；精确历史 source SHA、数据和权重在记录/服务器中。4,776 是语言行，137 是语义实例 |
| DPO 采用匹配 chosen-only 继续 SFT 和两种子 | [train_round010_preferences.py](../scripts/train_round010_preferences.py) | [原始训练审计](evidence/training-records/r010-training-audit.json)、[完整评测汇总](evidence/training-records/r010-outcome.json) | 448 是池大小，计划曝光 204；51 步不是完整一轮。合成 TRAIN 负例；不声称同 FLOPs。DPO 45/66、45/66，控制 51/66、52/66 |
| 真实 GRPO、参考冻结、lr=0 控制，拒绝无稳定收益候选 | [train_round029_online.py](../scripts/train_round029_online.py)、[原生训练器](../AgenticArxiv/rl/train_grpo.py) | [原审计字段摘录](evidence/training-records/r030-grpo-audit-excerpt.json)、[摘录来源与 SHA](evidence/training-records/provenance.json) | 每种子学习组 64 步/256 rollout；lr=0 调用不是学习更新。摘录保留训练、比较与门槛，省略监控数组/完整逐题审计；原件在服务器 |
| 评测完整自然请求与真实版本原句 | [execution.py](paper_agent/execution.py)、[protocol.py](paper_agent/protocol.py) | [冻结问题](evidence/round056-public-tasks.json)、[选句及缺项](evidence/round056-selected-evidence.json) | 助手语义审阅；六篇已知论文、抽取式答案、离线注入恢复。公开 SHA 检查不能证明全文语义支持 |
| 公共身份可确定时减少模型规划，复杂请求回退 | [identity.py](paper_agent/identity.py)、[execution.py](paper_agent/execution.py) | [逐题与成本](evidence/round056-outcome.json) | 5 条一次调用、3 条两次调用；8 题同为 5/8，16→11 调用、94,116→82,580 token。不是普遍质量或统计非劣保证 |
| 记录 GPU、版本、预算和全部调用成本 | [run.py](paper_agent/run.py)、[native.py](paper_agent/native.py) | [冻结源绑定](paper_agent/provenance.json)、[历史闭环摘要](evidence/round056-closure-audit.json) | 多人共享 GPU 的模型秒数不能作为生产吞吐；没有 DDP 或分布式系统实测结论 |

## 原始记录与摘录

`training-records/` 中五个记录逐字节复制原始文件。GRPO 文件为明确列出省略项的字段摘录，不伪称原件；所有原件、公开文件 SHA 与转换方式见 provenance。原始训练 log、论文全文、完整 token IDs、轨迹和权重仍由服务器保存。

## 对外展示流程

1. 阅读 README 和两条简历版，说明主问题与个人工作。
2. 在仓库根执行 `python -m portfolio.review_results --html /tmp/paper-agent-review.html`。只用 Python 标准库，输出新文件；选择 R56-02 看完整完成的选句，再选 R56-06 看问项遗漏，选 R56-01 看未闭合输出和已计入的成本。
3. 用本表定位 SFT、DPO、GRPO 原记录及训练代码；说明 DPO/GRPO 未满足候选推广条件。
4. 对方需要模型重跑时，按 REPRODUCE 恢复原资产、环境和顺序。公开浏览器不会启动模型或模拟一次新 Agent 调用。

本地点击上层代码相对链接需要完整仓库；GitHub 分支中这些链接直接可用。
