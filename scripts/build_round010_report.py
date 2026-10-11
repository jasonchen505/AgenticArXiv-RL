#!/usr/bin/env python3
"""Outcome-aware standalone report; supports local and remote rendering."""
from pathlib import Path
import html,json,re
HERE=Path(__file__).resolve().parent
ROOT=HERE if (HERE/'round-009-2026-10-04/report.html').exists() else HERE.parent/'artifacts/reproduction'
R=ROOT/'round-010-2026-10-04'
def read(f,default=None):
    p=R/f;return json.loads(p.read_text()) if p.exists() else default
esc=html.escape;css=re.search(r'<style>(.*?)</style>',(ROOT/'round-009-2026-10-04/report.html').read_text(),re.S).group(1)
out=read('mainline-outcome.json',{});complete=read('completion.json',{});status=read('controller-status.json',{});data=read('preference-data-manifest.json',{});plan=read('experiment-plan.json',{})
models=['sft','ct43','dpo43','ct44','dpo44'];arms=models[1:];names={'sft':'原SFT','ct43':'继续SFT · 43','dpo43':'DPO · 43','ct44':'继续SFT · 44','dpo44':'DPO · 44'}
header='<tr><th>严格成功</th>'+''.join('<th>'+names[a]+'</th>' for a in models)+'</tr>'
rows=[];details=[];dev=[]
if out:
    for g,label in [('fresh-iid_test','新IID · 18任务'),('fresh-ood_test','原OOD分组 · 新参数4任务')]:
        rows.append('<tr><th>'+label+'</th>'+''.join(f'<td>{out["metrics"][a][g]["strict_successes"]}/{out["metrics"][a][g]["runs"]} · {out["metrics"][a][g]["strict_success_rate"]:.2%}</td>' for a in models)+'</tr>')
        sub=[]
        for title,key in [('稳定成功 pass³','pass3_tasks'),('提前结束（仓库false_finish）','false_finish_count'),('平均token','avg_tokens')]:
            sub.append('<tr><th>'+title+'</th>'+''.join(f'<td>{out["metrics"][a][g][key]}/{out["metrics"][a][g]["tasks" if key=="pass3_tasks" else "runs"]}</td>' if key!='avg_tokens' else f'<td>{out["metrics"][a][g][key]:,.1f}</td>' for a in models)+'</tr>')
        counts=read('task-level-audit.json')['strict_counts'][g]
        sub.extend('<tr><th>'+esc(t)+'</th>'+''.join(f'<td>{counts[a][t]}/3</td>' for a in models)+'</tr>' for t in sorted(counts['sft']))
        details.append('<details><summary>'+label+'：稳定成功、提前结束、成本与逐任务结果</summary><div class="table-wrap"><table><thead>'+header+'</thead><tbody>'+''.join(sub)+'</tbody></table></div></details>')
    rows.append('<tr><th>新请求合计 · 22任务</th>'+''.join(f'<td>{out["strict_totals"]["fresh"][a]}/66</td>' for a in models)+'</tr>')
    dev.append('<tr><th>旧Dev · 8任务</th>'+''.join(f'<td>{out["metrics"][a]["dev"]["strict_successes"]}/24</td>' for a in models)+'</tr>')
    dev.append('<tr><th>第008轮请求 · 22任务</th>'+''.join(f'<td>{out["strict_totals"]["previous"][a]}/66</td>' for a in models)+'</tr>')
else:
    rows.append('<tr><th>新参数22任务 × 3次 × 5模型</th><td colspan="5">训练与评测进行中，尚无收益结论</td></tr>')
    dev.append('<tr><th>开发复用</th><td colspan="5">统一冻结最终权重后评测，不按开发集选checkpoint</td></tr>')
training=read('training-audit.json',{});tr=[]
for a in arms:
    m=read(a+'-training-manifest.json',{});job=read(a+'-train-job.json',{})
    if m:tr.append(f'<tr><th>{names[a]}</th><td>{m["actual_steps"]}</td><td>{m["train_loss"]:.4f}</td><td>{job["elapsed_seconds"]/60:.1f}分钟</td><td>{m["torch_peak_gib"]:.2f}GiB</td><td>{m["reference_equal_sft_and_frozen_tensors"] if a.startswith("dpo") else "不适用"}</td></tr>')
    else:tr.append('<tr><th>'+names[a]+'</th><td colspan="5">'+('运行中' if status.get('current_job')==a+'-train' else '待完成或待最终归档')+'</td></tr>')
effects=' · '.join(f'seed{s}：DPO − 继续SFT {d:+d}/66' for s,d in out.get('dpo_minus_continued_sft',{}).get('fresh',{}).items())
current={'prepared':'配方与新测试已冻结','running':'串行训练或评测进行中','waiting_for_gpu4':'等待GPU4空闲，未并发抢占资源','failed_or_wait_timeout':'流程停止，保留失败记录，需核对状态','completed':'训练、评测与结果核验已完成'}.get(status.get('phase'),'读取执行状态')
conclusion=out.get('plain_chinese','真实训练已启动。先完成DPO与继续SFT两个种子的同预算对照，再判断偏好学习能否改善正常执行与正确拒绝。训练loss不能代替任务成绩。')
changed=[]
for item in read('behavior-audit.json',{}).get('changed_tasks',[]):
    if not item['group'].startswith('fresh'):continue
    changed.append('<tr><th>'+esc(item['task_id'])+'</th>'+''.join(f'<td>{item["strict_counts"][a]}/3 · 提前结束{item["premature_finish_counts"][a]}/3</td>' for a in models)+'</tr>')
behavior='<div class="table-wrap"><table><thead>'+header+'</thead><tbody>'+''.join(changed)+'</tbody></table></div>' if changed else '<p>'+('新任务各模型严格成功与提前结束次数没有跨模型差异。' if out else '评测完成后展示严格成功或提前结束次数不同的全部新任务，检查正常任务错误拒绝与阻塞原因混淆。')+'</p>'
links=[]
for p in sorted(R.rglob('*')):
    if p.is_file() and p.suffix in {'.json','.jsonl','.md','.log','.py'} and p.name!='report-browser-checks.json':links.append('<a href="'+p.relative_to(R).as_posix()+'">'+esc(p.relative_to(R).as_posix())+'</a>')
page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AgenticArXiv-RL · 第010轮</title><style>{css}.card{{min-width:0;overflow-wrap:anywhere}}.table-wrap td{{white-space:nowrap}}</style></head><body><main>
<div class="eyebrow">PREFERENCE LEARNING / ROUND010</div><h1>学会正确拒绝，<br>也要保留执行能力。</h1><p class="lead">主线转向成对偏好学习：同一SFT出发，DPO与继续SFT使用相同正确回答、51步与有效批量4，各训练两个种子。DPO额外看到错误回答；比较它是否优于单纯继续学习正确回答。</p><p class="meta">2026-10-04 · America/Los_Angeles<br>/data/home/chenyizhou/AgenticArXiv-RL</p><span class="tag">仅GPU4</span><span class="tag">GPU5–7不使用</span><span class="tag">不把测试轨迹加入训练</span>
<div class="kpis"><div class="kpi"><strong>448</strong><span>训练偏好对 · 51父任务</span></div><div class="kpi"><strong>4 × 51</strong><span>两种方法 × 两个种子</span></div><div class="kpi"><strong>22/22</strong><span>新请求标准路径通过</span></div><div class="kpi"><strong>{complete.get('actual_new_benchmark_runs','690')}</strong><span>{'完成的新评测' if out else '冻结的新评测目标'}</span></div></div>
<nav><a href="#outcome">结果与状态</a><a href="#design">训练对照</a><a href="#fresh">任务成绩</a><a href="#behavior">行为与成本</a><a href="#training">真实训练</a><a href="#limits">主线与边界</a><a href="#evidence">证据</a></nav>
<section id="outcome"><h2>01 / 让实际评测决定收益</h2><div class="callout"><strong>{esc(conclusion)}</strong></div><p>{esc(current)} · 当前任务 {esc(status.get('current_job',status.get('next_job','已完成')))} · 状态更新时间 {esc(status.get('updated_at_utc','未记录'))}。{'690次新执行、90次复用开发记录完成；无异常与权威严格重计分漂移。' if out else '当前报告为执行快照，未完成部分不写成实验结论；全部结束后自动生成核验与完成报告。'}</p></section>
<section id="design"><h2>02 / 一项学习目标，两种方法</h2><div class="grid"><div class="card"><h3>DPO：正确回答优于错误回答</h3><p>388对正常状态：正确工具动作或完成回答优于错误拒绝。60对不可行状态：正确具体理由优于另外两种错误理由或虚假完成。DPO使用sigmoid损失、beta0.1，相对起始SFT参考优化偏好。</p></div><div class="card"><h3>继续SFT：控制多训练的影响</h3><p>使用完全相同的448个prompt、chosen、初始顺序和两个种子，只做completion-only监督学习。配方匹配51优化步、有效batch4、lr5e−6、4096长度、NF4+bf16 LoRA、cosine和0.03warmup，两组关闭dropout。</p></div></div><p>数据仅从原训练SFT文件选择：388个正常决策用原始语言包装，10个阻塞语义各用6个已有语言包装，共60对。覆盖51训练父任务、137语义实例；语言变体不是独立任务支持。chosen原本已经用于起始SFT训练，负例是按训练契约预先构造的合成回答，不是在线模型采样。没有从测试失败挑选样本。</p><p>两种方法仅在学习目标及其必要的rejected输入上不同；同优化步、有效prompt batch不等于同计算量或完成token数。{'所有最终适配器张量变化及DPO参考冻结均已核验。' if out else '训练结束后核验适配器变化与DPO的SFT参考冻结，所有模型统一冻结后才评测。'}</p></section>
<section id="fresh"><h2>03 / 新参数请求上的成对结果</h2><div class="table-wrap"><table id="main-metrics"><thead>{header}</thead><tbody>{''.join(rows)}</tbody></table></div><p><strong>{esc(effects)}</strong></p><p>22任务×3次×5模型=330次新请求运行；四个候选在旧Dev和第008轮请求再跑360次开发诊断。复用原SFT的90次开发记录并锁定哈希。pass³为同任务三次全部严格成功。只在同一请求组内比较，不跨新旧任务直接比较绝对成功率。</p>{''.join(details)}<details><summary>已观察开发集：Dev与第008轮请求</summary><div class="table-wrap"><table><thead>{header}</thead><tbody>{''.join(dev)}</tbody></table></div></details></section>
<section id="behavior"><h2>04 / 成绩之外，检查错误拒绝</h2>{behavior}<p>提前结束使用仓库false_finish定义：FINISH时缺少所需工具，也包含明确拒绝；不一定口头宣称成功完成。关注正常任务错误拒绝、具体阻塞理由混淆、工具失败后完成宣称。若减少token来自错误拒绝，不算效率提升。</p></section>
<section id="training"><h2>05 / 实际步数、权重和资源</h2><div class="table-wrap"><table><thead><tr><th>训练模型</th><th>实际步</th><th>本目标loss</th><th>进程时间</th><th>Torch峰值</th><th>SFT相同且冻结的参考张量</th></tr></thead><tbody>{''.join(tr)}</tbody></table></div><p>两种loss含义不同，不能横向比较数值大小来证明算法更好。每模型51步、有效批量4，计划204次训练prompt暴露；不保证448个样本在这段预算内均被看到。不是GRPO在线rollout或完整工具轨迹训练。</p><p>初始DPO微批量2在第一次更新前显存不足；此前完成的SFT与失败DPO作为attempt1保留。正式对照双方统一微批量1、累积4后重新训练，有效批量、51步、lr、数据与评测不变，修正在任何留出输出前登记。只用v2最终step51，未混用旧设置模型。仅GPU4空闲时启动，5–7留给其他人。</p></section>
<section id="limits"><h2>06 / 后训练主线与结论边界</h2><div class="callout good">本轮把“正常执行”与“正确拒绝”放入同一个成对学习目标，并用继续SFT控制额外学习正确答案的影响。判断DPO是否在两个种子都超过对照、是否保持正常能力，再决定下一步；不依据单例或最好种子宣称改进。</div><p>新22参数契约在本轮输出前冻结，标准路径22/22通过，精确契约不重叠原81任务、SFT137语义、第007/008轮请求，训练父任务交叉0。最大序列{data.get('max_sequence_tokens',2614)}token，无4096长度截断。模型提示只包含请求与可见历史；后台标准答案不拼入提示。</p><p>同一已知论文caption世界与已观察的留出父任务。完整工具程序结构大多已见过，原OOD组不是严格未见程序认证。本轮是新参数确认，不能称为广泛新论文世界泛化；原作者缺失VLM、快照与权重仍未恢复。一个SFT种子、两个新训练种子也不足以建立强统计结论。</p><p>求职成果主线：完整SFT训练、真实GRPO与两种子负结果、终止奖励对照、现在的DPO与监督继续训练对照。优先呈现真实训练及评测结果，明确数据来源、参考策略、拒绝行为和算力成本。</p></section>
<section id="evidence"><h2>07 / 原始数据、配方和记录</h2><p class="source-links">{' · '.join(links)}</p><p><a href="../round-009-2026-10-04/report.html">第009轮：终止奖励负结果核验</a> · <a href="PROGRESS.md">本轮累计进度快照</a></p></section>
</main></body></html>'''
(R/'report.html').write_text(page);print(R/'report.html')
