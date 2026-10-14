import torch,time,json,hashlib
from .presence import GeneratedPresencePenalty
def h(x):return hashlib.sha256(json.dumps(x,separators=(",",":")).encode()).hexdigest()
def call(client,arm,phase,prompt,calls):
 budget=arm['generation_cap'];kwargs={'enable_thinking':arm['thinking']} if arm['arm']!='qwen3' else {}
 messages=[{'role':'user','content':prompt}]
 text=client.tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=False,**kwargs)
 inputs=client.tokenizer(text,return_tensors='pt').to('cuda');ids=inputs['input_ids'][0].tolist();assert len(ids)+budget<=20480,'context cap; no truncation'
 if arm['arm']=='qwen3': assert ids==client.tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=True)
 seed=int(hashlib.sha256((client.task_id+'|'+phase+'|271').encode()).hexdigest()[:8],16)%2147483647
 torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 sampling=arm['sampling'];gkwargs={'do_sample':False} if sampling is None else {k:v for k,v in sampling.items() if k!='presence_penalty'}
 if sampling:
  gkwargs.update(do_sample=True,logits_processor=[GeneratedPresencePenalty(len(ids),sampling['presence_penalty'])],eos_token_id=[248046,248044])
 torch.cuda.synchronize();start=time.monotonic()
 with torch.inference_mode(): out=client.model.generate(**inputs,max_new_tokens=budget,pad_token_id=client.tokenizer.pad_token_id,**gkwargs)
 torch.cuda.synchronize();elapsed=int((time.monotonic()-start)*1000);gen=out[0,len(ids):].tolist();raw=client.tokenizer.decode(gen,skip_special_tokens=False);decoded=client.tokenizer.decode(gen,skip_special_tokens=True)
 reasoning_tokens=0;reasoning=None;channel_error=None;final_ids=gen
 if arm['thinking']:
  endthink=client.tokenizer.convert_tokens_to_ids('</think>')
  if endthink not in gen:
   channel_error='reasoning did not close within registered generation budget';reasoning_tokens=len(gen);reasoning=decoded;content='';final_ids=[]
  else:
   k=gen.index(endthink);reasoning_tokens=k+1;reasoning=client.tokenizer.decode(gen[:k],skip_special_tokens=True);final_ids=gen[k+1:];content=client.tokenizer.decode(final_ids,skip_special_tokens=True)
 else:content=decoded
 eos=[248046,248044] if sampling else client.model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else (eos or [])
 final_visible=sum(i not in client.tokenizer.all_special_ids for i in final_ids)
 if final_visible>256:channel_error='final channel exceeded registered 256 token cap'
 if not gen or gen[-1] not in eos:channel_error=channel_error or 'generation did not finish at EOS; no partial JSON acceptance'
 response={'choices':[{'message':{'role':'assistant','content':content,'reasoning_content':reasoning}}],'usage':{'prompt_tokens':len(ids),'completion_tokens':len(gen),'total_tokens':len(ids)+len(gen),'reasoning_tokens':reasoning_tokens,'final_channel_tokens_including_EOS':len(final_ids),'final_visible_tokens':final_visible},'full_generated_raw':raw,'generated_token_IDs':gen,'channel_error':channel_error,'finished_at_EOS':bool(gen and gen[-1] in eos)}
 calls.append({'phase':phase,'request':{'model':arm['model'],'messages':messages,'temperature':sampling['temperature'] if sampling else 0,'sampling':sampling,'seed':seed,'effective_eos_token_IDs':eos,'presence_scope':'generated IDs only, once per token, before temperature/top-k/top-p','max_tokens':budget,'stream':False,'chat_template_kwargs':kwargs},'response':response,'input_token_ID_sha256':h(ids),'formatted_prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'formatted_prompt_suffix':text[-100:],'input_token_count':len(ids),'llm_ms':elapsed})
 return content,channel_error
