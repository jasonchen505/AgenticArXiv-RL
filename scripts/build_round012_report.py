#!/usr/bin/env python3
"""Outcome-aware report for one fixed capability-preservation objective."""
from pathlib import Path
import html,json,re
HERE=Path(__file__).resolve().parent
ROOT=HERE if (HERE/'round-011-2026-10-04/report.html').exists() else HERE.parent/'artifacts/reproduction'
R=ROOT/'round-012-2026-10-04';R.mkdir(exist_ok=True)
def read(p,default=None):
    try:return json.loads(Path(p).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default
esc=html.escape;css=re.search(r'<style>(.*?)</style>',(ROOT/'round-011-2026-10-04/report.html').read_text(),re.S).group(1)
out=read(R/'mainline-outcome.json',{});status=read(R/'controller-status.json',{});plan=read(R/'experiment-plan.json',{});done=read(R/'completion.json',{})
models=['sft','ct43','dpo43','ct44','dpo44','anchor43','anchor44'];names={'sft':'原SFT','ct43':'继续SFT43','dpo43':'纯DPO43','ct44':'继续SFT44','dpo44':'纯DPO44','anchor43':'DPO＋监督43','anchor44':'DPO＋监督44'}
header='<tr><th>评测组</th>'+''.join('<th>'+names[a]+'</th>' for a in models)+'</tr>'
rows=[];more=[]
if out:
    for g,label in [('fresh-iid_test','新IID18任务'),('fresh-ood_test','原OOD分组 · 新4任务')]:
        rows.append('<tr><th>'+label+'</th>'+''.join(f'<td>{out["metrics"][a][g]["strict_successes"]}/{out["metrics"][a][g]["runs"]}</td>' for a in models)+'</tr>')
        details=[]
        for label,key,den in [('稳定成功pass³','pass3_tasks','tasks'),('提前结束false_finish','false_finish_count','runs')]:details.append('<tr><th>'+label+'</th>'+''.join(f'<td>{out["metrics"][a][g][key]}/{out["metrics"][a][g][den]}</td>' for a in models)+'</tr>')
        more.append('<details><summary>'+g+'的稳定成功与提前结束</summary><div class="table-wrap"><table><thead>'+header+'</thead><tbody>'+''.join(details)+'</tbody></table></div></details>')
    rows.append('<tr><th>新22请求合计</th>'+''.join(f'<td>{out["strict_totals"]["fresh"][a]}/66</td>' for a in models)+'</tr>')
else:rows.append('<tr><th>新22请求 ×3次</th><td colspan="7">评测尚未完成，暂无收益结论；不将局部结果当作最终成绩。</td></tr>')
tr=[]
for a in ['anchor43','anchor44']:
    m=read(R/(a+'-training-manifest.json'),{});j=read(R/(a+'-train-job.json'),{});f=read(R/(a+'-candidate-freeze.json'),{})
    if m:tr.append(f'<tr><th>{names[a]}</th><td>{m["actual_steps"]}</td><td>{m["train_loss"]:.4f}</td><td>{j.get("elapsed_seconds",0)/60:.1f}分钟</td><td>{m["torch_peak_gib"]:.2f}GiB</td><td>{f.get("changed_tensors","待核验")}</td><td>{m["reference_equal_sft_and_frozen_tensors"]}/392</td></tr>')
    else:tr.append('<tr><th>'+names[a]+'</th><td colspan="6">'+('训练中' if status.get('current_job')==a+'-train' else '待完成')+'</td></tr>')
changed=[]
for t in read(R/'behavior-audit.json',{}).get('changed_tasks',[]):
    if not t['group'].startswith('fresh'):continue
    changed.append('<tr><th>'+esc(t['task_id'])+'</th>'+''.join(f'<td>{t["strict_counts"][a]}/3 · 终止{t["premature_finish_counts"][a]}/3</td>' for a in models)+'</tr>')
behavior='<div class="table-wrap"><table><thead>'+header+'</thead><tbody>'+''.join(changed)+'</tbody></table></div>' if changed else '<p>完整评测后逐项展示新请求的严格成绩或提前结束差异，保留所有任务和原始输出。</p>'
state={'prepared':'数据与配方已冻结','running':'训练或评测正在执行','waiting_for_gpu4':'等待GPU4空闲','finalizing':'完整推理完成，正在核验','completed':'训练、评测和核验已完成','failed_or_wait_timeout':'流程已停止，失败记录保留'}.get(status.get('phase'),'准备冻结实验')
finished_runs=sum(read(p,{}).get('sample_count',0) for p in R.glob('*/summary.json') if not any(p.parent.name==a+'-dev' or p.parent.name.startswith('previous-'+a+'-') for a in models[:5]))
gold=read(R/'fresh-gold-validation.json',{}).get('passed','待核验')
links=[]
for f,label in [('experiment-plan.json','冻结实验配方'),('fresh-data-manifest.json','新请求及数据边界'),('fresh-gold-validation.json','新请求标准路径'),('objective-integration-check.json','组合损失CPU核验'),('training-completion-check.json','两个种子训练完成核验'),('training-audit.json','最终训练核验'),('freeze-chronology-audit.json','冻结与评测时序'),('mainline-outcome.json','完整结果'),('behavior-audit.json','逐任务原始轨迹'),('completion.json','完整运行计数'),('controller-status.json','执行状态'),('PROGRESS.md','累计进度')]:
    if (R/f).exists():links.append('<a href="'+f+'">'+label+'</a>')
text=out.get('plain_chinese','真实训练与成对评测按冻结计划推进。目标是判断增加正确回答监督约束能否保留工具执行与正确终止能力，尚无算法收益结论。')
page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AgenticArXiv-RL · 第012轮能力保持</title><style>{css}.card{{min-width:0;overflow-wrap:anywhere}}.table-wrap td{{white-space:nowrap}}code{{overflow-wrap:anywhere}}</style></head><body><main>
<div class="eyebrow">CAPABILITY PRESERVATION / ROUND012</div><h1>偏好学习，<br>加上正确回答的约束。</h1><p class="lead">上一轮纯DPO没有超过继续SFT。本轮只增加chosen监督损失：同448训练对、同起始SFT、51步和两个种子；保留纯DPO、继续SFT、原SFT共五个模型作对照，统一比较新请求。</p><p class="meta">2026-10-04 · America/Los_Angeles<br>/data/home/chenyizhou/AgenticArXiv-RL</p><span class="tag">仅GPU4</span><span class="tag">GPU5–7不使用</span><span class="tag">权重固定1 · 不搜索系数</span>
<div class="kpis"><div class="kpi"><strong>2 × 51</strong><span>新候选计划优化步</span></div><div class="kpi"><strong>7</strong><span>同一确认请求上的模型</span></div><div class="kpi"><strong>{gold}/22</strong><span>输出前标准路径核验</span></div><div class="kpi"><strong>{done.get('actual_new_benchmark_runs',finished_runs)}/642</strong><span>完成的新运行 · 450开发记录复用</span></div></div>
<nav><a href="#status">结果与状态</a><a href="#design">单项改进</a><a href="#results">新请求成绩</a><a href="#training">实际训练</a><a href="#behavior">行为核验</a><a href="#scope">结论边界</a><a href="#evidence">证据</a></nav>
<section id="status"><h2>01 / 以完整对照决定收益</h2><div class="callout"><strong>{esc(text)}</strong></div><p>{esc(state)} · 当前任务：{esc(status.get('current_job',status.get('next_job','已完成' if out else '待启动')))} · 状态更新时间 {esc(status.get('updated_at_utc','未登记'))}。未完成部分不写成实验结论；训练loss下降不能替代Agent任务成绩。</p></section>
<section id="design"><h2>02 / 唯一变化：加入正确回答监督损失</h2><div class="grid"><div class="card"><h3>保持正确回答的概率</h3><p>总损失 = sigmoid DPO损失 + chosen completion的token平均交叉熵。原DPO beta0.1，两个损失权重固定[1,1]。安装版本TRL0.29.1原生支持sigmoid与sft组合，不加入任务专用规则。</p></div><div class="card"><h3>三类必要对照</h3><p>同种子纯DPO：检验能力保持项的作用；同种子继续SFT：判断偏好学习是否有额外收益；原SFT：检查是否真的超过起点。只恢复纯DPO的退化不能称为超过监督对照。</p></div></div><p>与第010轮相同的448对、51训练父任务、137语义，max2614token、4096长度无截断。chosen已用于起始SFT，rejected为训练契约产生的合成文本，不是在线模型采样。没有把测试失败轨迹转成训练数据。</p><p>训练固定seed43/44，51步、微批量1、累积4、有效批量4、lr5e−6、NF4＋bf16 LoRA、cosine、warmup0.03、关闭dropout。每模型计划204次prompt暴露，不保证见完448对。只选最终step51，两个模型统一冻结后才开始本轮模型推理。</p></section>
<section id="results"><h2>03 / 七个冻结模型的相同请求对照</h2><div class="table-wrap"><table id="main-metrics"><thead>{header}</thead><tbody>{''.join(rows)}</tbody></table></div>{''.join(more)}<p>新确认22×3×7=462次；两个新候选在旧Dev24和第010轮请求66再运行180次开发诊断，共642次新执行。五个对照的450次旧Dev＋第010轮记录按哈希复用。默认提示、bf16、seed45、session前缀和每组scratch保持一致；不跨新旧请求绝对分数判断进步。</p><p>新请求保留全部22个原留出父任务。预先固定days26/28/29、关键词max6、摘要140词、无效序号6等参数，不按候选输出删题。标准路径22/22与精确契约重叠检查必须在训练前通过。</p></section>
<section id="training"><h2>04 / 真正更新与参考冻结</h2><div class="table-wrap"><table id="training-metrics"><thead><tr><th>候选</th><th>实际步</th><th>本目标loss</th><th>进程时间</th><th>Torch峰值</th><th>改变张量</th><th>SFT相同的冻结参考</th></tr></thead><tbody>{''.join(tr)}</tbody></table></div><p>训练前后逐张量验证参考仍等于起始SFT，并核验保存权重确有更新。新目标loss不能和纯DPO或SFT数值横比。加入监督项也会增加计算；相同优化步和prompt批量不等于相同FLOPs或token预算。Torch分配峰值不等于整卡物理显存峰值。</p></section>
<section id="behavior"><h2>05 / 正常执行与正确终止都要保持</h2>{behavior}<p>同时检查遗漏步骤、错误参数、无效引用的具体理由，以及工具失败后声称完成。仓库false_finish涵盖提前FINISH、错误拒绝或失败后宣称完成等，不是专门的错误拒绝率。保留全部变化任务，不把token减少直接当作效率收益。</p></section>
<section id="scope"><h2>06 / 有明确止损的算法验证</h2><div class="callout good">必须同时报告两个种子相对纯DPO、继续SFT和原SFT的变化。若组合目标仍不能超过继续SFT，就收束这条合成负例目标路线，转向训练集负例的有效性与覆盖，不继续搜索系数或补测试规则。</div><p>本轮设计受已观察的第010轮开发失败启发，配方及新确认契约在本轮模型输出前冻结。当前旧请求均为开发证据；新参数仍来自同一已知论文caption世界和观察过的留出父任务。16/22完整工具程序已在训练出现，原OOD分组1/4已出现，不代表全新论文或完全未见工具程序泛化。</p><p>求职主线是展示完整训练、受控算法改进、相对监督学习的额外收益检验，以及可恢复执行、冻结参考和逐轨迹分析。原作者VLM、快照及权重仍未恢复，原作者完整实验复现尚未完成；一个SFT种子与两个后训练种子不足以支持广泛统计结论。不提交或推送。</p></section>
<section id="evidence"><h2>07 / 报告与原始证据</h2><p>{' · '.join(links)}</p><p><a href="../round-011-2026-10-04/report.html">第011轮：纯DPO负结果与后续设计</a> · <a href="../round-010-2026-10-04/preference-data-manifest.json">复用训练数据审计</a></p></section>
</main></body></html>'''
(R/'report.html').write_text(page);print(R/'report.html')
