# 2026-10-14 上游更新

来源：[PR #108](https://github.com/Algorineko/AgenticArXiv-RL/pull/108)，fork 与上游均为提交 `088f129a8bd4690479dc2697ae36451205070717`。相对上次 `0263207` 新增4个含合并的commit，5文件，284新增/10删除行；服务器无冲突合并并保留之前的复现修改。

## 值得采用

- `self_evolve mix`：保留基础 TRAIN 混合，追加定向行；核验 messages fingerprint、去重、标注 mixture_source、固定 shuffle，写含输入/输出 SHA 的训练 manifest。
- `--wrapper-set targeted`：采用与基础集不同的措辞框架，让定向行不因相同指纹全部被去掉；参数增广默认 split 改为 v3_81。
- `trials_by_task`：拒绝同一 summary 混合多个 agent，防止把不同策略的试次合成一个 pass^k。

合并后的完整 `AgenticArxiv/tests`：856 passed、7 skipped、1058 subtests passed；无 GPU 或外部路由 API 调用。

## 对本项目的启发与限制

基础行为回放加定向样本，比只拿少量坏例继续训练更适合保持旧能力。messages SHA 只能证明字节去重：不同 wrapper 仍可能是同一任务/决策，需要额外报告语义实例、父任务、工具程序与论文集合的重合；不能把新增行数当作新增能力。

必须保持用户任务质量与训练样本质量的联系。我们已经观察到参考证据超容量和取证缺口，这些不是新增语言 wrapper 可补回的问题；应先修好任务可完成性，再将可见证据充分而模型选错的 TRAIN 决策纳入后训练。

PR 标题提到 T4 上一个真实回合 gate passed，但本次差异只交付代码和测试，未取得可独立审计的该回合完整学习/留出原始轨迹。因此记录为上游报告，不引用为我们复现的增益，更不把它推广到这批自然正文任务。

网络记录：服务器 github.com:443 直连超时；官方 api.github.com 可读，服务器公开 Git 镜像可用。镜像取得的提交与官方 fork/上游 SHA 一致；origin URL 未改，未通过 Mac VPN 中转模型，也未向镜像发送凭据。
