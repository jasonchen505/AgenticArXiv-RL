#!/usr/bin/env python3
"""Outcome closure plus train-only real-error collection; no false gain claims."""
from pathlib import Path
import html,json,re
HERE=Path(__file__).resolve().parent
ROOT=HERE if (HERE/'round-012-2026-10-04/report.html').exists() else HERE.parent/'artifacts/reproduction'
R=ROOT/'round-013-2026-10-05';R.mkdir(exist_ok=True);OLD=ROOT/'round-012-2026-10-04'
def read(p,default=None):
    try:return json.loads(Path(p).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default
out=read(OLD/'mainline-outcome.json');audit=read(R/'round012-closure-audit.json',{});plan=read(R/'collection-plan.json',{});gold=read(R/'collection-gold-validation.json',{});status=read(R/'collection-status.json',{});done=read(R/'collection-completion.json',{})
esc=html.escape;css=re.search(r'<style>(.*?)</style>',(OLD/'report.html').read_text(),re.S).group(1)
models=['sft','ct43','ct44','dpo43','dpo44','anchor43','anchor44'];names={'sft':'原SFT','ct43':'继续SFT43','ct44':'继续SFT44','dpo43':'纯DPO43','dpo44':'纯DPO44','anchor43':'DPO＋监督43','anchor44':'DPO＋监督44'}
rows=''.join(f'<tr><th>{names[a]}</th><td>{out["strict_totals"]["fresh"][a]}/66</td><td>{out["strict_totals"]["previous"][a]}/66</td><td>{out["metrics"][a]["dev"]["strict_successes"]}/24</td><td>{out["metrics"][a]["fresh-iid_test"]["false_finish_count"]}/54</td></tr>' for a in models)
trace=ROOT.parents[1]/'outputs/repro_round013/collection/rollout_traces.jsonl' if ROOT.name=='reproduction' else R/'rollout_traces.jsonl'
live=read(R/'collection-live-progress.json',{})
if trace.exists():
    n=sum(1 for s in trace.read_text().splitlines() if s.strip().endswith('}'))
    live={'recorded_model_rollouts':n,'scope':'Written raw trajectories only; final coverage/weight-invariance audit still required.'}
    (R/'collection-live-progress.json').write_text(json.dumps(live,indent=2)+'\n')
count=done.get('actual_model_rollouts',live.get('recorded_model_rollouts',0))
state={'running':'真实训练集采样进行中','waiting_for_gpu4':'等待GPU4空闲','finalizing':'采样完成，正在核验','completed':'采样与冻结权重核验完成','failed_or_wait_timeout':'流程停止，失败记录保留'}.get(status.get('phase'),'准备完成，等待启动')
cases=read(OLD/'task-level-audit.json')['strict_counts']['fresh-iid_test'];labels={'fresh012_chain_cl5_dl3_tr3':'检索→下载→缓存检查→翻译','fresh012_infeasible_index_out_of_range':'4篇上下文请求第6篇','fresh012_qa_cr5_abstract':'读取abstract章节'}
case_rows=''.join('<tr><th>'+label+'</th>'+''.join(f'<td>{cases[a][t]}/3</td>' for a in ['sft','ct43','ct44','anchor43','anchor44'])+'</tr>' for t,label in labels.items())
support=read(R/'existing-failure-support.json',{});sig=audit.get('same_seed_action_signatures',{})
sig_text='；'.join(f'seed{s}有{x["same_action"]}/{x["compared_trials"]}条工具动作序列一致，动作与思考文本均相同{x["same_action_and_thought"]}条' for s,x in sig.items())
collection_result=f'<p>采样1096条，严格失败{done["strict_failures"]}条，截断{done["clipped"]}条。覆盖137语义/51父任务，每语义8条；原SFT与保存参考392个张量均保持完全等值。严格失败尚未自动转成训练负例。</p>' if done else '<p>采样尚未完成，当前不宣称覆盖达标、权重等值已通过或负例已有效。结束后自动核验完整1096条、每语义8条，以及392个源适配器和参考张量保持等值；不自动开始新训练。</p>'
links=[]
for f,label in [('round012-closure-audit.json','上一轮完成核验'),('collection-plan.json','真实失败采样计划'),('collection-gold-validation.json','137条标准路径核验'),('existing-failure-support.json','原失败数据支持'),('collection-status.json','采样状态'),('collection-live-progress.json','已记录轨迹数量'),('collection-completion.json','采样完成核验'),('failure-inventory.json','失败轨迹与指标'),('PROGRESS.md','累计进度')]:
    if (R/f).exists():links.append('<a href="'+f+'">'+label+'</a>')
page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AgenticArXiv-RL · 第013轮回到有效训练数据</title><style>{css}.card{{min-width:0;overflow-wrap:anywhere}}.table-wrap td{{white-space:nowrap}}.kpi{{min-width:0}}main{{overflow-wrap:anywhere}}@media(max-width:400px){{.kpi{{padding:12px}}.kpi strong{{font-size:25px}}}}</style></head><body><main>
<div class="eyebrow">REAL ERROR DATA / ROUND013</div><h1>收束无收益目标，<br>回到有效训练数据。</h1><p class="lead">组合目标的642次新评测已完成：DPO＋监督两个种子仍45/66，继续SFT为54/66、51/66，原SFT46/66。按预定标准结束这条合成负例系数路线。下一步收集并核验模型在训练任务上的真实错误，保留监督继续训练作为必要对照。</p><p class="meta">2026-10-05 · America/Los_Angeles<br>/data/home/chenyizhou/AgenticArXiv-RL</p><span class="tag">仅GPU4</span><span class="tag">GPU5–7不使用</span><span class="tag">新阶段lr=0 · 数据采样</span>
<div class="kpis"><div class="kpi"><strong>642</strong><span>上一轮新评测全部完成</span></div><div class="kpi"><strong>0 / 0</strong><span>组合目标相对纯DPO · 两种子</span></div><div class="kpi"><strong>{gold.get('passed','待核验')}/137</strong><span>训练语义标准路径通过</span></div><div class="kpi"><strong>{count}/1096</strong><span>新阶段已记录模型轨迹</span></div></div>
<nav><a href="#results">已完成结果</a><a href="#behavior">行为与对照</a><a href="#data">数据改进</a><a href="#collection">实际执行</a><a href="#next">主线下一步</a><a href="#evidence">证据</a></nav>
<section id="results"><h2>01 / 组合目标没有额外收益</h2><div class="table-wrap"><table id="closure-metrics"><thead><tr><th>模型</th><th>第012轮新22请求</th><th>第010轮请求 · 已观察</th><th>旧Dev · 已观察</th><th>新IID提前结束</th></tr></thead><tbody>{rows}</tbody></table></div><p>组合目标相对继续SFT为−9、−6/66，相对纯DPO两个种子均0；相对原SFT均−1/66。450次哈希锁定开发记录复用、642次新运行全部覆盖，无异常或权威严格重计分漂移。两组均真实51步，392适配器张量发生变化，392参考张量保持冻结；不能把loss下降或权重变化写成任务收益。</p><div class="callout"><strong>按预先声明的止损标准，结束这个合成负例目标分支。保留全部结果，不继续搜索监督权重、beta或挑选最好种子。</strong></div></section>
<section id="behavior"><h2>02 / 继续SFT是必须保留的强对照</h2><div class="table-wrap"><table><thead><tr><th>新请求中成绩有差异的全部3任务</th>{''.join('<th>'+names[a]+'</th>' for a in ['sft','ct43','ct44','anchor43','anchor44'])}</tr></thead><tbody>{case_rows}</tbody></table></div><p>组合目标仍跳过CL链的缓存检查，对无效序号尝试下载、工具失败后声称完成；未恢复abstract参数。{esc(sig_text)}。这只比较模型写出的动作和思考，不要求工具观察或计时字节相同，也不意味着权重相同。</p><p>第010轮新参数确认：原SFT50/66，继续SFT51/66、52/66；第012轮新参数确认：46/66，对照54/66、51/66。继续SFT在这两个集合的两种子均有正收益，是目前应保留的监督候选；但旧Dev均10→9/24，收益集中在少量任务，不自动升为默认模型。原SFT作为起点参考继续保留。</p><p class="note">同一请求组内比较；不能把原SFT在不同新集合的50→46当作权重退化。仓库false_finish包含省略步骤、错误拒绝和失败后完成宣称，不是专门的错误拒绝率。本轮新IID原SFT6/54、组合目标9/54，CL漏步骤也在原SFT出现，不全部归因于DPO。</p></section>
<section id="data"><h2>03 / 从预设拒绝文本转向真实模型错误</h2><div class="grid"><div class="card"><h3>已有失败覆盖太窄</h3><p>原冻结SFT训练采样408条，严格失败38条，只涉及{support.get('failure_semantic_tasks',7)}个训练场景。前面提到的7个是两组都有训练信号的筛选子集，不是全部失败场景。语言包装和重复采样不是独立能力支持；直接大量重复这些失败，容易重新走向狭窄难例分布。</p></div><div class="card"><h3>采样原有全部训练语义</h3><p>51原训练父任务＋86已有参数变体＝137语义，与原SFT语义集合完全一致。每语义2组×4次，共1096条；seed46、temperature1。全部137条标准路径通过，训练父任务与留出集合交叉0。没有引入测试失败请求。</p></div></div><p>这次负例来源改为实际模型写出的回答，不再预设三种拒绝句子。下一步必须恢复对应的可见状态，在同一状态验证正确动作，再判定错误回答确实违反任务或环境事实。优先从首个错误决策或单独验证的恢复分支建立可审计配对。</p><p>严格计分失败不自动等于有效负例：正确论文ID别名、无害额外参数、可接受的解释、状态无法恢复或截断轨迹都需要审计。不能把无效负例“纠正”成手写错误文本继续训练；保留排除原因、唯一决策数、语义与父任务覆盖。</p></section>
<section id="collection"><h2>04 / 已启动固定策略采样，尚无新学习收益</h2><div class="callout good"><strong>{esc(state)} · 学习率0 · 计划1096条真实模型轨迹</strong></div><p>状态更新时间 {esc(status.get('updated_at_utc','未登记'))}。使用原SFT，沿用真实多轮离线环境，2048 completion预算、最多6turn、每组4次，不做动态重采样或失败筛选。原生执行274个lr0循环步骤，是为记录轨迹运行的控制流程，不能称为274次有效学习更新。这是训练采样协议，不能把其成功率与bf16 Benchmark成绩直接横比。</p>{collection_result}<p>只有GPU4空闲时启动，单控制器锁避免重复，5–7留给其他人。当前采样期间GPU4可能占用；本轮没有新后训练候选，也没有对新的测试集宣称提升。</p></section>
<section id="next"><h2>05 / 先验证数据，再比较学习方法</h2><p>下一项训练须先通过负例真实性、同状态正确动作、去重与覆盖检查。再保留广泛专家回放，用相同正确回答和预算的继续SFT控制来判断偏好学习的额外价值。现有请求均为已观察开发证据；新的模型确认设计须在输出前冻结。不使用测试轨迹训练，不靠单例修复规则或系数搜索堆出分数。</p><p>求职主线现在有完整SFT、真实GRPO、两种子任务分布与奖励对照、DPO及能力保持目标相对监督控制的负结果；下一项贡献是可审计的真实错误数据管线。收益只能由训练后的完整对照确认。</p><p class="note">已知论文caption世界与已观察留出父任务；原确认组16/22工具程序已在训练出现，原OOD组1/4也出现。一个SFT种子、两个后训练种子不足以支持广泛统计结论。作者VLM、快照与权重仍未恢复，原作者完整实验复现仍未完成。</p></section>
<section id="evidence"><h2>06 / 完整报告与证据</h2><p>{' · '.join(links)}</p><p><a href="../round-012-2026-10-04/report.html">第012轮完整完成报告</a> · <a href="../round-004-2026-10-03/train-failures.json">原训练采样失败记录</a></p></section>
</main></body></html>'''
(R/'report.html').write_text(page);print(R/'report.html')
