# 复现说明

## 环境与范围

1.5B 学习环境：Torch 2.5.1+cu124、Transformers 4.57.6、TRL 0.29.1、PEFT 0.17.1、bitsandbytes 0.45.5。QLoRA NF4/bf16，LoRA r16/alpha32，原 SFT 一轮 597 优化器步、batch1/accum8、seed42、max4096。

4B 原生证据环境：Torch 2.5.1、Transformers 5.3.0，Qwen/Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，原生 Torch 路径、未用 FLA/causal-conv/MTP。两套环境不要混装。模型在服务器通过镜像下载并锁定版本。

历史数据/原文/模型在服务器留存；仓库包含轻量证据索引、原始代码和新独立接口，未上传大模型、全部论文与 GB 轨迹。精确重放需恢复相同快照/资产/运行库并验证 manifest；重新生成数据或换库属于新的实验，不能回填旧结果。

## CPU 检查

```bash
python -m unittest portfolio.test_protocol
python -m unittest portfolio.test_review_results
python -m portfolio.review_results --html /tmp/paper-agent-review.html --receipt /tmp/paper-agent-review.json
python -m portfolio.check_closed_traces --assets /path/to/round-056-2026-10-13
```

`test_protocol` 使用自包含合同测试。`check_closed_traces` 需要保存的真实研究资产，重放实际事件、句子跨度、提示和渲染结果，无模型调用，不作新的能力成绩。轻量接口的生成函数与冻结源绑定在 `paper_agent/provenance.json`。

`test_review_results` 和 `review_results` 仅使用仓库中的轻量证据及 Python 标准库，不依赖 Transformers、Torch、GPU 或服务器资产。六项检查包含一项完整记录通过和五项篡改拒收：丢失败题、改选句、换版本、改配对成本和训练权重声明。这些检查证明记录一致性，不验证语义评分正确性。HTML/receipt 拒绝覆盖已有文件，重跑时选择新输出路径。

## 离线 GPU 接口

在具备上述 4B 环境、对应本地模型及论文 bank 的服务器，从仓库根运行：

```bash
timeout 10800 python -m portfolio.paper_agent.run \
  --bank /path/to/paper-bank.json --catalog /path/to/public-catalog.json \
  --tasks /path/to/public-tasks.json --model-dir /path/to/Qwen3.5-4B \
  --gpu 4 --arm on_compiled --out /path/to/new-run/traces.jsonl \
  --inject-failure R56-03 --inject-failure R56-08
```

`--arm on_two` 为同模式两调用；`off_compiled` 为关闭思考成本参照。每组独立输出，拒绝覆盖现存轨迹。GPU 只允许 4/7、显存额度 0.5、至少 13000MiB 空闲准入，其他人的作业保持运行。该单臂接口不自动安排三组对照；要复核历史结论，必须还恢复冻结的分片、顺序和全套评测规则。

输入仅公共任务/目录与实际工具结果；私有评分不参与运行。开启思考每次 8192 生成 token，关闭思考 256，最终可见≤256、上下文20480、最多8句/工具。所有失败 token、规划回退、无答案轨迹都保留。默认离线读取，未将本地快照工具包装成真实网页请求。

## 训练代码与审计索引

上游通用训练入口为 `AgenticArxiv/rl/train_sft.py`、`train_dpo.py`、`train_grpo.py`；历史匹配配方保留于 `scripts/train_round010_preferences.py`、`train_round014_preferences.py`、`train_round029_online.py`、`train_round036_plan.py` 等实际源文件。R010 对应 448 条数据池、51 步、有效批量4、计划曝光204；它不同于 R014 真实错误锚定配方。准确文件名、优化器步与证据 SHA 以 `evidence/results.json` 及 `evidence/training-records/` 为准。

先按 repo 的 snapshot/SFT 数据构建流程恢复资产，审计 TRAIN 父任务、语义实例、句式变体和 completion-only 监督。恢复历史配方时固定论文环境与工具版本；目前上游为10工具，早期学习是9工具，不能跳过旧 source SHA 失败后仍称精确重放。

不要仅凭训练 loss、权重变化或工具运行成功声明用户任务改进。DPO 用同 chosen 继续 SFT 控制，GRPO 用同种子 lr=0 控制；参考冻结、策略变化、全部 final 在确认推理前冻结的顺序都需核验。
