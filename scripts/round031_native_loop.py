from agents.base_agent import *

def guarded_native_loop(
    self,
    task: str,
    agent_model: str = None,
    session_id: str = "default",
    initial_history: str = "",
) -> Dict[str, Any]:
    """Run one task through the ReAct loop.

    ``initial_history`` is model-visible state that existed before the
    current user request.  Benchmark setup actions use it to expose the
    same legitimate session context that GRPO sees, without counting the
    setup as policy actions or placing it in the scored trajectory.
    """
    log.info(f"[{self.__class__.__name__}] 开始执行任务: {task}")
    run_start = time.time()
    self.session_id = session_id
    msg_id = uuid.uuid4().hex

    if agent_model is None:
        agent_model = settings.models.agent_model

    try:
        self.side_effects.create_chat_log(
            session_id, msg_id, "user", task, model=agent_model, agent_type=self.agent_type
        )
    except Exception as e:
        log.warning(f"Failed to log user message: {e}")

    tools = self.discover_tools()
    tools_description = self.format_tools_for_prompt(tools)

    # Benchmark/GRPO 会显式传入同一份、由 TaskSpec.setup 派生的可见状态。
    # 此时不要再从 side-effects 追加另一种格式的论文列表，否则评测 prompt
    # 会比训练 prompt 多出标题，重新造成输入分布错位。普通 Web/API 调用没有
    # initial_history，仍沿用运行时 store 中的真实会话上下文。
    enriched_task = (
        task if initial_history.strip()
        else self._enrich_task_with_context(task, session_id)
    )

    history: List[Dict[str, str]] = []
    step_timings: List[Dict[str, int]] = []
    token_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for iteration in range(self.max_iterations):
        log.info(f"第 {iteration + 1} 次迭代")

        generated_history = self.format_history(history)
        history_text = initial_history.strip()
        if generated_history:
            history_text = (
                f"{history_text}\n\n{generated_history}"
                if history_text else generated_history
            )
        messages, extra = self.build_messages(self._state_task(enriched_task), tools_description, history_text)
        extra = self._merge_llm_extra(extra)

        llm_ms = 0
        tool_ms = 0
        thought = ""
        action_dict = None
        observation = ""

        try:
            t0 = time.time()
            # temperature 优先由调用方（如 DPO 数据生成）通过 llm_extra 指定，
            # 否则回退到默认 0.1。硬编码 0.1 会让 llm_extra 里的 temperature
            # 被静默忽略，导致多次 rollout 近乎确定、DPO 无法产生偏好分歧。
            eff_temperature = float(extra.pop("temperature", 0.1))
            response = self.llm_client.chat_completions(
                model=agent_model,
                messages=messages,
                temperature=eff_temperature,
                max_tokens=1000,
                stream=False,
                extra=extra or None,
            )
            llm_ms = int((time.time() - t0) * 1000)

            # 累计 token 用量
            usage = response.get("usage") or {}
            token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            token_usage["total_tokens"] += usage.get("total_tokens", 0)

            thought, action_dict = self.parse_response(response)
            log.info(f"Thought: {thought}")

            if self._guard_veto(thought, action_dict, history, step_timings, llm_ms, iteration, msg_id):
                continue

            if action_dict is None:
                log.info("任务完成")
                observation = "任务完成"
                history.append({"thought": thought, "action": "FINISH", "observation": observation})
                self._guard_terminal(thought, observation)
                step_timings.append({"llm_ms": llm_ms, "tool_ms": 0})
                self._log_step(msg_id, iteration, thought, "FINISH", "{}", observation, llm_ms, 0, session_id)
                break

            # 带副作用的工具执行
            t1 = time.time()
            observation = self._execute_with_side_effects(action_dict)
            tool_ms = int((time.time() - t1) * 1000)
            log.info(f"Observation: {observation[:200]}...")

            step_timings.append({"llm_ms": llm_ms, "tool_ms": tool_ms})

            action_str = json.dumps(action_dict, ensure_ascii=False)
            history.append({"thought": thought, "action": action_str, "observation": observation})
            self._guard_tool(thought, action_dict, observation)

            self._log_step(
                msg_id, iteration, thought,
                action_dict.get("name", ""), json.dumps(action_dict.get("args", {}), ensure_ascii=False),
                observation[:4000], llm_ms, tool_ms, session_id,
            )

            if iteration == self.max_iterations - 1:
                log.warning("达到最大迭代次数，强制结束")
                history.append({"thought": "达到最大迭代次数", "action": "FORCE_STOP", "observation": "迭代限制"})
                step_timings.append({"llm_ms": 0, "tool_ms": 0})
                self._log_step(msg_id, iteration + 1, "达到最大迭代次数", "FORCE_STOP", "", "迭代限制", 0, 0, session_id)
                break

        except Exception as e:
            error_msg = f"LLM调用失败: {str(e)}"
            log.error(error_msg)
            history.append({"thought": "LLM调用失败", "action": "ERROR", "observation": error_msg})
            step_timings.append({"llm_ms": llm_ms, "tool_ms": 0})
            self._log_step(msg_id, iteration, "LLM调用失败", "ERROR", "", error_msg, llm_ms, 0, session_id)
            break

    final_observation = history[-1]["observation"] if history else "无执行结果"

    reply = ""
    for step in reversed(history):
        if step.get("action") not in ("FINISH", "FORCE_STOP", "ERROR"):
            reply = step.get("observation", "")
            break
    reply = reply or final_observation

    try:
        self.side_effects.create_chat_log(
            session_id, msg_id + "_reply", "assistant", reply,
            model=agent_model, agent_type=self.agent_type,
        )
    except Exception as e:
        log.warning(f"Failed to log assistant reply: {e}")

    total_time_ms = int((time.time() - run_start) * 1000)
    total_llm_ms = sum(s["llm_ms"] for s in step_timings)
    total_tool_ms = sum(s["tool_ms"] for s in step_timings)

    result = {
        "task": task,
        "msg_id": msg_id,
        "history": history,
        "final_observation": final_observation,
        "total_time_ms": total_time_ms,
        "iteration_count": len(history),
        "agent_type": self.agent_type,
        "timing": {
            "total_llm_ms": total_llm_ms,
            "total_tool_ms": total_tool_ms,
            "framework_overhead_ms": total_time_ms - total_llm_ms - total_tool_ms,
            "steps": step_timings,
        },
        "token_usage": token_usage,
    }
    log.info(f"任务执行完成，共 {len(history)} 步, 总耗时 {total_time_ms}ms (LLM {total_llm_ms}ms + Tool {total_tool_ms}ms)")
    log.info("-" * 80)
    return self._guard_output(result)
