#!/usr/bin/env python3
"""Real-error data admission and a fully matched, finite learning comparison."""
from pathlib import Path
import collections,html,json,re
HERE=Path(__file__).resolve().parent
ROOT=HERE if (HERE/'round-013-2026-10-05/report.html').exists() else HERE.parent/'artifacts/reproduction'
R=ROOT/'round-014-2026-10-05';OLD=ROOT/'round-013-2026-10-05'
def read(p,default=None):
    try:return json.loads(Path(p).read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default
esc=html.escape
data=read(R/'certified-training-data-manifest.json',{});admit=read(R/'negative-admission-audit.json',{});check=read(R/'independent-data-check.json',{});collection=read(OLD/'collection-completion.json',{});plan=read(R/'experiment-plan.json',{});status=read(R/'controller-status.json',{});out=read(R/'mainline-outcome.json',{})
css=re.search(r'<style>(.*?)</style>',(OLD/'report.html').read_text(),re.S).group(1)
models=['sft','ct43','ct44','replay43','replay44','real43','real44'];names={'sft':'原SFT','ct43':'旧继续SFT43','ct44':'旧继续SFT44','replay43':'同数据监督43','replay44':'同数据监督44','real43':'真实偏好＋回放43','real44':'真实偏好＋回放44'}
def good(d):return d['completed'] and d['tool_accurate'] and d['arg_score']==1 and d['ref_score']==1 and d['parse_fail']==0 and d['tool_fail']==0 and d['terminal_semantics_accurate']
def summary(a,g):
    leaf=a+'-dev' if g=='dev' else g.split('-')[0]+'-'+a+'-'+g.split('-',1)[1]
    return read(R/leaf/'summary.json')
def score(a,p):
    if p=='dev':
        x=summary(a,p);return str(sum(good(d) for d in x['details']))+'/24' if x else '待完成'
    xs=[summary(a,p+'-'+g) for g in ['iid_test','ood_test']]
    return str(sum(good(d) for x in xs for d in x['details']))+'/66' if all(xs) else '待完成'
metric_rows=''.join('<tr><th>'+names[a]+'</th>'+''.join('<td>'+score(a,g)+'</td>' for g in ['fresh','previous','dev'])+'</tr>' for a in models)
train_rows=[];trained=0;actual_steps=0
for a in ['replay43','real43','replay44','real44']:
    m=read(R/(a+'-training-manifest.json'),{});f=read(R/(a+'-candidate-freeze.json'),{});j=read(R/(a+'-train-job.json'),{})
    trained+=bool(f);actual_steps+=m.get('actual_steps',0)
    step=str(m.get('actual_steps'))+'/111' if m else '待完成'
    log=R/(a+'-train.log')
    if not m and log.exists():
        found=re.findall(r'(\d+)/111',log.read_text(errors='replace'))
        if found:step=found[-1]+'/111 · 执行中'
    train_rows.append(f'<tr><th>{names[a]}</th><td>{step}</td><td>{m.get("dataset_rows","443计划")}</td><td>{str(f.get("changed_tensors","待核验"))}</td><td>{str(f.get("reference_equal_sft_tensors","待核验"))}</td><td>{str(round(j["elapsed_seconds"],1))+"s" if j.get("exit_code")==0 else esc(j.get("state","待启动"))}</td></tr>')
excluded=collections.Counter(admit.get('exclusions',{}));excluded.update(x['reason'] for x in check.get('removed',[]))
exlabels={'duplicate_decision':'重复记录','ambiguous_or_reason_only_blocked_text':'含糊解释或仅理由计分差异','possible_order_alternative':'可能等价的动作顺序','correct_alias_backend_resolution_failure':'正确引用别名的后端解析缺口','section_alias_or_missing_snapshot':'章节别名或快照资源缺口','possibly_harmless_extra_action':'可能无害的额外动作','semantic_decision_duplicate':'语义相同的决策','redundant_previously_valid_tool_call':'重复执行过的正确调用'}
exrows=''.join(f'<tr><th>{esc(exlabels.get(k,k))}</th><td>{n}</td></tr>' for k,n in excluded.items())
kind_labels={'wrong_paper_reference':'论文引用错误','blocked_invalid_call':'必定失败的调用','premature_finish':'必需步骤未完成就结束','wrong_requested_tool':'请求工具错误','malformed_action':'无法解析的动作','wrong_requested_parameter':'明确请求参数错误','blocked_false_completion':'不可执行却声称完成'}
kindrows=''.join(f'<tr><th>{kind_labels[k]}</th><td>{v}</td></tr>' for k,v in data.get('negative_type_counts',{}).items())
phase={'running':'训练／评测流水线运行中','waiting_for_gpu4':'等待GPU4空闲','waiting_for_gpu4_headroom':'等待GPU4显存余量（至少8GiB）','waiting_for_allowed_gpu':'等待GPU0–4空闲（训练已完成）','prepared':'计划已冻结，等待执行','finalizing':'全部评测结束，核验中','completed':'训练与完整对照核验完成','failed_or_wait_timeout':'流程停止，失败证据保留'}.get(status.get('phase'),'尚未启动')
lead=out.get('plain_chinese','已完成1096条冻结策略采样，并把102次严格失败筛为45个可验证、去重后的错误决策。加入覆盖全部训练父任务的398条专家回放，已启动两个种子的真实偏好与同数据监督对照；任务收益尚待完整评测。')
eval_done=sum(read(p).get('sample_count',0) for p in R.glob('*/summary.json') if not (p.parent.name.endswith('-dev') or p.parent.name.startswith('previous-')))+sum(read(R/(a+'-dev')/'summary.json',{}).get('sample_count',0)+sum(read(R/('previous-'+a+'-'+g)/'summary.json',{}).get('sample_count',0) for g in ['iid_test','ood_test']) for a in ['replay43','real43','replay44','real44'])
evidence=[]
for f,label in [('collection-recovery.json','采样收尾修复记录'),('negative-admission-plan.json','训练负例准入标准'),('negative-admission-audit.json','102次失败逐条审核'),('independent-data-check.json','独立去重与出处核验'),('certified-training-data-manifest.json','最终训练数据清单'),('certified_preferences.jsonl','45条真实偏好样本'),('replay-loss-integration-check.json','专家回放零偏好梯度核验'),('control-loss-integration-check.json','同路径监督损失核验'),('chosen-token-integration-check.json','443条正确回答token与mask核验'),('fresh-gold-validation.json','新22任务标准路径'),('fresh-data-manifest.json','新请求冻结记录'),('experiment-plan.json','完整实验计划'),('controller-status.json','当前执行状态'),('completion.json','完整实验完成核验'),('mainline-outcome.json','完整对照结论'),('PROGRESS.md','累计进度')]:
    if (R/f).exists():evidence.append(f'<a href="{f}">{label}</a>')
page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AgenticArXiv-RL · 第014轮真实错误与回放对照</title><style>{css}</style></head><body><main>
<div class="eyebrow">REAL ERRORS + EXPERT REPLAY / ROUND014</div><h1>把真实错误变成数据，<br>用完整监督对照检验价值。</h1><p class="lead">{esc(lead)}</p><p class="meta">2026-10-05 · America/Los_Angeles<br>/data/home/chenyizhou/AgenticArXiv-RL</p><span class="tag">仅GPU4</span><span class="tag">GPU5–7不使用</span><span class="tag">不再搜索合成负例系数</span>
<div class="kpis"><div class="kpi"><strong>1096</strong><span>真实策略轨迹 · 137语义/51父任务</span></div><div class="kpi"><strong>45</strong><span>验证并去重的实际错误决策</span></div><div class="kpi"><strong>398</strong><span>覆盖全部51父任务的专家回放</span></div><div class="kpi"><strong>{trained}/4</strong><span>新模型完成训练并冻结</span></div></div>
<nav><a href="#data">已完成数据</a><a href="#audit">准入与排除</a><a href="#comparison">真实学习对照</a><a href="#execution">实际执行</a><a href="#results">完整评测</a><a href="#evidence">证据</a></nav>
<section id="data"><h2>01 / 数据采集完成，学习收益另行判断</h2><p>原SFT固定策略采样1096/1096条、137/137语义、51/51训练父任务，每语义8条，无留出父任务交叉、无截断。原策略和参考392个张量分别完全等值，274个lr0执行循环不是学习更新。严格成功994/1096，失败102条，涉及48语义、25父任务；这是原生训练采样协议，不能直接和bf16 Benchmark成绩横比。</p><p>采样进程正常结束，耗时{round(collection.get('process_seconds',0),1)}s，实际设备显存峰值21674MiB。原收尾脚本遗漏评测接口的agent_type与trial参数；第014轮CPU修复完成核验，保留原失败状态和日志，未重跑GPU采样、未改冻结采样脚本。</p><div class="callout good"><strong>最终45个独立错误决策，覆盖35个训练语义、21个父任务。每个保留决策的正确动作及剩余标准步骤均在同状态执行，并通过严格核验。</strong></div></section>
<section id="audit"><h2>02 / 严格失败不自动成为训练负例</h2><div class="grid"><div class="card"><h3>保留的实际模型错误</h3><table id="negative-types"><tbody>{kindrows}</tbody></table></div><div class="card"><h3>排除57条，保留原因和出处</h3><div class="table-wrap"><table id="exclusions"><tbody>{exrows}</tbody></table></div></div></div><p>初步形成52条候选对；独立检查再排除6个仅终止措辞不同的重复决策，以及1次重复执行过的正确搜索，最终45条。没有把正确论文ID别名、可能等价的动作顺序、无害额外调用、含糊拒绝或快照资源缺口当作可验证错误。错误回答来自模型原始输出；格式归一化仅为续写轮添加Thought:标记，不改写内容。</p><p>逐条重放真实前缀，完整工具观察必须与记录完全相同；用模型当时收到的1000字符内观察重建可见历史。正确动作从该状态执行，完整剩余路径必须严格成功；先冻结准入门槛，再产出数据。20决策/15语义/10父任务最低支持要求全部通过，训练父任务与留出交叉0。</p><p class="note">训练提示是Benchmark格式的可见状态重建。原GRPO token IDs和EOS边界未记录，因此不宣称复原原生采样的逐token输入或原生行为概率。45是独立错误决策数，不是45个独立任务；同一源SFT与一个采样种子仍限制数据多样性。</p></section>
<section id="comparison"><h2>03 / 真实偏好的额外价值必须超过同数据监督</h2><div class="table-wrap"><table id="learning-design"><thead><tr><th>条件</th><th>同数据继续SFT · seeds43/44</th><th>真实偏好＋回放 · seeds43/44</th></tr></thead><tbody><tr><th>源模型／正确数据</th><td colspan="2">同一原SFT；398专家决策＋45实际错误状态的正确动作＝443条</td></tr><tr><th>学习损失</th><td>仅正确回答token-mean CE</td><td>同一CE＋sigmoid DPO，权重[1,1]、beta0.1</td></tr><tr><th>数据遍历与预算</th><td colspan="2">一次全443条；111步，micro1/accum4，最后一组3条；lr5e−6</td></tr><tr><th>数值与长度</th><td colspan="2">NF4＋bf16，max4096，实际最长2962，无截断，dropout关闭</td></tr><tr><th>专家回放的偏好项</th><td>不计算偏好损失</td><td>chosen=rejected，偏好项常数log2、梯度0；仍训练正确CE</td></tr></tbody></table></div><p>398回放决策覆盖全部137语义/51父任务，占443条的89.8%；实际偏好占10.2%，没有大量重复狭窄错误。两臂采用同一个原生DPOTrainer的CE分支，控制微批量和累积梯度中的监督权重差异；监督臂loss_type=[sft]，偏好臂=[sigmoid,sft]。监督臂仅正确回答影响学习梯度。相同输入的额外前向计算不是错误文本监督，也不宣称精确等FLOPs。</p><p>CPU FP64验证了回放偏好梯度为0、CE控制与独立正确token平均损失一致；实际443行的正确token序列与监督mask全部相同，移位后监督token共17669。四个模型从相同源适配器出发，参考冻结392张量，只保存最后一个完整数据遍历，全部冻结后才进入评测。</p><p class="note">组合目标包含回放常数log2项，不能把其训练loss与纯CE数值直接排名。当前监督实验也比旧51步CT更长，因此两个新种子必须各自与同443条/111步的监督臂比较；旧CT是额外强控制，不能取代预算匹配控制。</p></section>
<section id="execution"><h2>04 / 已启动有限的训练与完整评测流程</h2><div class="callout good"><strong>{esc(phase)} · {esc(status.get('current_job',status.get('next_job','')))}</strong></div><p>状态更新时间：{esc(status.get('updated_at_utc','待登记'))}。四个新模型冻结后，运行822次新评测并复用270次哈希锁定开发记录。仅GPU4空闲时执行；不使用5–7，单控制器锁防重复。不是周期自动化。当前新训练模型已确认完成步数合计{actual_steps}，未完成项不写成完成或收益。</p><div class="table-wrap"><table id="training-progress"><thead><tr><th>模型</th><th>优化步骤</th><th>训练行数</th><th>变化张量</th><th>等值参考张量</th><th>实际进程</th></tr></thead><tbody>{''.join(train_rows)}</tbody></table></div></section>
<section id="results"><h2>05 / 不用局部成绩代替完整结论</h2><p>新22请求全部标准路径通过，训练父任务交叉0，与原81/SFT137及第007/008/010/012轮契约无精确重复；在当前模型输出前冻结。保留全部18IID/4OOD父任务，days23/24/25、关键词上限7、总结160词、列表4篇却请求第8篇；不根据当前输出筛选任务。七个模型共享同一bf16/prompt/seed/repeat3协议，另外保留旧Dev及第012轮已观察请求。</p><p>当前完整组新评测记录{eval_done}/822；部分组和仍运行任务不计为实验完成。以下“待完成”没有成绩，已观察第012轮请求属于开发证据。</p><div class="table-wrap"><table id="main-metrics"><thead><tr><th>模型</th><th>本轮新22请求</th><th>第012轮已观察请求</th><th>旧Dev</th></tr></thead><tbody>{metric_rows}</tbody></table></div><div class="callout"><strong>{esc(out.get('plain_chinese','暂无新任务收益结论。必须两个种子均超过同数据监督、原SFT及旧CT，并保留旧Dev水平，才满足预先声明的收益标准；不调系数、不选最佳种子或中间checkpoint。'))}</strong></div><p class="note">已知论文caption世界与已观察留出父任务，16/22工具程序在训练出现，原OOD组也有1/4出现；新参数不等于新世界或完全未见程序。一个源SFT种子和两个后训练种子限制统计结论。作者VLM、原快照与权重仍缺失，原作者完整实验复现未完成。</p></section>
<section id="evidence"><h2>06 / 面向求职的可审计贡献与证据</h2><p>本轮贡献是实际模型错误采样、同状态正确动作验证、排除假负例、语义去重、广泛专家回放和梯度一致的监督控制。算法收益由训练后的完整对照决定，不能由失败条数、训练loss或权重变化代替。</p><p>{' · '.join(evidence)}</p><p><a href="../round-013-2026-10-05/report.html">第013轮采样与上一目标收束</a></p></section></main></body></html>'''
if (R.parent/'round-015-2026-10-05/operational-amendment.json').exists():
    page=page.replace('仅GPU4</span>','训练GPU4 · 评测空闲GPU0–4</span>').replace('仅GPU4空闲时执行；','原训练仅GPU4，剩余评测仅在GPU0–4空闲时执行（Round015资源修订）；')
if (R.parent/'round-016-2026-10-06/operational-amendment.json').exists():
    page=page.replace('训练GPU4 · 评测空闲GPU0–4</span>','训练及评测仅GPU4</span>').replace('原训练仅GPU4，剩余评测仅在GPU0–4空闲时执行（Round015资源修订）；','训练及剩余评测仅GPU4，评测准入空闲显存至少8GiB（Round016资源修订）；')
(R/'report.html').write_text(page);print(R/'report.html')
