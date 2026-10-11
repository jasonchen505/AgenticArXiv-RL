# Round018：数据门槛通过，论文迁移对照运行
新世界共69篇准入元数据，排除原世界231个base IDs，40篇真实PDF、206个图表元数据及618条caption工具记录。全部22条gold通过，资源与任务SHA在模型预测前冻结。

修复仅影响未完成的数据阶段：有效后端值extractive、三个关键词池canonical keys，以及gold执行的独立scratch路径。已有论文内容/顺序、任务程序/参数、核心评分与模型均保持一致。原始快照、失败脚本及修订计划归档；没有确认案例训练。

有限控制器PID3112775使用原脚本，GPU4 only，198次/6作业/3模型，bf16、seed45、repeat3、baseline prompt，无新增训练。报告快照为0完整作业，源SFT IID正在运行。最终性能必须等待全部闭环。

下一步先查远端round017/transfer-status.json与round018/transfer-controller.log。完成后运行scripts/audit_round018_transfer.py，独立核对198条strict与false finish。保留CE43、CE44两种子和旧Dev10/11/9，不自动推广。真实错误DPO+CE分支已关闭。

报告：artifacts/reproduction/round-018-2026-10-06/report.html；历史报告与QA保留。完整原作者VLM实验复现仍未完成。
