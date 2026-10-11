# 第 001 轮：Agentic RL 复现引导与指标审计

日期：2026-10-02（America/Los_Angeles）。目标岗位：Agent 算法，后训练 + 部分工程 / infra。
原始代码来自 GitHub clone；本轮不把原作者的训练成果当成个人贡献。

## 完成与限制

- 奖励基线覆盖 81 个任务、5 类合成策略，总体 / 类目闸门通过。
- 14 条冻结坏例：13 条 fixed 保持，1 条 known open 仍复现。
- 选定测试：123 passed，40 subtests passed；不是全仓测试。
- 从 24 份归档评测、1,134 次历史运行的严格指标重算 pass^k。
- 新增可靠性 JSON 导出与回归测试；纠正 README 三语言和方法文档的 pass^1 标签。
- GPU 4 运行缓存 Qwen2.5-1.5B-Instruct 的三条首步生成，保存响应 / token / 时延 / 显存；未执行工具，不计算任务成功率。
- 原快照、原训练数据和 outputs 权重缺失。尚未完成 SFT / GRPO 训练复现。

## 指标结论

- DAPO IID 的 0.4815 是单次严格成功率；pass^3 为 0.4444，与 GRPO 持平。
- DrGRPO train 的 0.6566 是单次严格成功率；pass^3 为 0.6364，与 GRPO 持平。
- SFT train pass^3 为 0.0606；单次率为 0.0808。
- Dev 只有 8 个任务，OOD 只有 4 个任务；不要把 24 / 12 次重复当成独立任务。
- 缺少原快照无法恢复 expected_paper_ids。对裸任务重新评分出现的差异只是诊断，不能据此宣布历史成绩错误或代码回归。

## 证据和运行方式

HTML：`artifacts/reproduction/round-001-2026-10-02/report.html`。
同目录保留 environment.json、artifact_audit.json、pytest.log/xml、badcase.log、gpu_smoke.json、changes.patch 等。

本轮只读复用 `/data/home/chenyizhou/my-research-project-v11/.venv/bin/python`，通过仓库 `.repro-deps` 覆盖 arxiv / loguru / fire / TRL 0.29.1，没有修改其他项目环境。
这仅用于复现引导验证；正式训练前建立独立、锁定依赖的环境。

```bash
cd /data/home/chenyizhou/AgenticArXiv-RL
export REPRO_PYTHON=/data/home/chenyizhou/my-research-project-v11/.venv/bin/python
export PYTHONPATH="$PWD/.repro-deps:$PWD/AgenticArxiv"
"$REPRO_PYTHON" -m pytest -q AgenticArxiv/tests/test_baselines.py AgenticArxiv/tests/test_splits.py AgenticArxiv/tests/test_reliability.py AgenticArxiv/tests/test_grpo_reward.py AgenticArxiv/tests/test_report_metric_export.py
"$REPRO_PYTHON" -m AgenticArxiv.benchmark.run_baselines --task-set expanded --output artifacts/reproduction/round-001-2026-10-02/reward_baseline
"$REPRO_PYTHON" eval/badcase_replay.py
"$REPRO_PYTHON" scripts/audit_archived_evaluations.py --output artifacts/reproduction/round-001-2026-10-02/artifact_audit.json
```

## 下一阶段

1. 独立训练环境、依赖锁定；重建并冻结新的快照和语义标准答案，不冒充原快照。
2. 81 任务 gold 轨迹检查，记录图表答案的 VLM 来源与覆盖。
3. GPU4 上进行 SFT / GRPO 小预算试验，测量显存、零方差组与正确 token mask，再扩展预算。
4. 同快照 / 切分 / 解码协议下重新评测 Base、SFT、GRPO；signal split 随新模型重新测量。
5. 围绕工具不支持请求、虚假完成、奖励对齐或信号池做单变量改进与多种子对照。
6. 每轮更新 HTML；只把亲自复现或改进的结果写入个人求职材料。
