"""Agent Loop - 核心 Agent 循环处理逻辑"""

import asyncio
import inspect
import json
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from loguru import logger
from backend.modules.tools.conversation_history import get_conversation_history


def _is_key_rotation_eligible_error(error_text: str) -> bool:
    """判断错误是否适合触发 key 轮换重试。"""
    lower = (error_text or "").lower()
    auth_hints = (
        "401", "unauthorized", "invalid api key", "invalid_api_key",
        "authentication", "invalid token", "token is unusable",
        "api key", "apikey", "access denied",
        "insufficient_quota", "account_deactivated",
    )
    rate_hints = (
        "429", "rate limit", "rate_limit", "quota",
        "too many requests", "capacity", "overloaded",
    )
    return any(hint in lower for hint in auth_hints + rate_hints)


class AgentLoop:
    """Agent 主循环类 - 处理消息、调用 LLM、执行工具、生成响应"""

    MAX_KEY_ROTATION_RETRIES = 3

    def __init__(
        self,
        provider,               # 负责调用大模型
        workspace: Path,        # 本地工作目录
        tools,                  # 工具注册表，负责执行文件，shell，网页等工具
        context_builder=None,   # 负责构造发送给大模型的上下文
        session_manager=None,   # 保存会话历史
        subagent_manager=None,  # 子agent
        model: Optional[str] = None,    # 模型名
        max_iterations: int = 25,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        thinking_enabled: bool = True,
    ):
        self.provider = provider
        self.workspace = workspace
        self.tools = tools
        self.context_builder = context_builder
        self.session_manager = session_manager
        self.subagent_manager = subagent_manager
        self.model = model
        self.max_iterations = max_iterations
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking_enabled = thinking_enabled
        self._key_rotation_count = 0
        
        logger.debug(
            f"AgentLoop initialized: max_iterations={max_iterations}, max_retries={max_retries}"
        )

    @staticmethod
    def _summarize_tool_calls_for_log(tool_calls: List[Any]) -> str:
        parts = []
        for tool_call in tool_calls:
            tool_id = str(getattr(tool_call, "id", "") or "").strip() or "<empty>"
            tool_name = str(getattr(tool_call, "name", "") or "").strip() or "<unknown>"
            parts.append(f"{tool_name}#{tool_id}")
        return ", ".join(parts) if parts else "<none>"


    """
    这次请求用那个provider，模型，token，循环等
    """
    def _resolve_execution_runtime(
        self,
        model_override: Optional[Dict[str, Any]] = None,
    ) -> tuple[Any, Optional[str], float, int, int, bool]:
        """解析当前消息执行应使用的 provider 和模型参数。"""
        base_provider = self.provider
        base_model = self.model
        base_temperature = self.temperature
        base_max_tokens = self.max_tokens
        base_max_iterations = self.max_iterations
        base_thinking_enabled = self.thinking_enabled
        base_api_mode = getattr(base_provider, "api_mode", "chat_completions")

        if not model_override:
            return (
                base_provider,
                base_model,
                base_temperature,
                base_max_tokens,
                base_max_iterations,
                base_thinking_enabled,
            )

        candidate_provider = base_provider
        candidate_model = model_override.get("model", base_model)
        candidate_temperature = model_override.get("temperature", base_temperature)
        candidate_max_tokens = model_override.get("max_tokens", base_max_tokens)
        candidate_max_iterations = model_override.get(
            "max_iterations",
            base_max_iterations,
        )
        candidate_api_mode = model_override.get("api_mode", base_api_mode)
        candidate_thinking_enabled = model_override.get(
            "thinking_enabled",
            base_thinking_enabled,
        )

        override_provider = model_override.get("provider")
        override_api_key = model_override.get("api_key") or None
        override_api_base = model_override.get("api_base") or None

        if override_provider or override_api_key or override_api_base:
            try:
                from backend.modules.providers import create_provider
                from backend.modules.config.loader import config_loader
                from backend.modules.providers.runtime import (
                    build_provider_unavailable_message,
                    get_provider_runtime_state,
                )

                provider_id = override_provider or config_loader.config.model.provider
                runtime_state = get_provider_runtime_state(
                    config_loader.config,
                    provider_id,
                    api_key_override=override_api_key,
                    api_base_override=override_api_base,
                )
                if not runtime_state.selectable:
                    raise ValueError(
                        build_provider_unavailable_message(
                            provider_id,
                            runtime_state.reason,
                        )
                    )

                candidate_provider = create_provider(
                    api_key=runtime_state.api_key or None,
                    api_keys=runtime_state.api_keys or None,
                    api_base=runtime_state.api_base,
                    default_model=candidate_model,
                    api_mode=candidate_api_mode,
                    timeout=getattr(self.provider, "timeout", 120.0),
                    max_retries=getattr(self.provider, "max_retries", self.max_retries),
                    provider_id=provider_id,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to create runtime provider override, falling back to base runtime config: "
                    f"{exc}"
                )
                return (
                    base_provider,
                    base_model,
                    base_temperature,
                    base_max_tokens,
                    base_max_iterations,
                    base_thinking_enabled,
                )

        return (
            candidate_provider,
            candidate_model,
            candidate_temperature,
            candidate_max_tokens,
            candidate_max_iterations,
            candidate_thinking_enabled,
        )


    """
    异步流式返回，处理web信息的核心函数
    """
    async def process_message(
        self,
        message: str,       # 用户发送的信息
        session_id: str,
        context: Optional[List[Dict[str, Any]]] = None, # 历史上下文
        session_summary: Optional[str] = None,  # 会话摘要
        media: Optional[List[str]] = None,      # 附件
        channel: Optional[str] = None,          # 来源渠道，QQ，微信
        chat_id: Optional[str] = None,
        account_id: Optional[str] = None,
        cancel_token=None,                      # 用户中断生成
        yield_intermediate: bool = True,
        model_override: Optional[Dict[str, Any]] = None,    # 本次临时模型配置
        persona_override=None,          # 人格配置
        tool_event_handler=None,        # 工具调用事件回调
        reasoning_event_handler=None,   # 推理内容回调
        prefer_direct_workflow_result: bool = False,
    ) -> AsyncIterator[str]:
        """处理用户消息并生成流式响应"""
        logger.debug(f"Processing message for session {session_id}")

        """
        1. 给工具设置会话环境
            用户是否取消
            工具结果记录哪里？
        """


        # 设置工具注册表的会话ID（用于审计日志）和渠道信息
        if self.tools:
            self.tools.set_session_id(session_id)
            self.tools.set_channel(channel)
            # 将取消令牌传递给支持中断的工具（如 WorkflowTool）
            if cancel_token and hasattr(self.tools, 'set_cancel_token'):
                self.tools.set_cancel_token(cancel_token)

            spawn_tool = self.tools.get_tool("spawn")
            if spawn_tool and hasattr(spawn_tool, 'set_context'):
                spawn_tool.set_context(session_id)

        """
        2. 构造messages
            如果没有就构造，追加用户信息
        """
        if self.context_builder and context is not None:
            messages = self.context_builder.build_messages(
                history=context,
                current_message=message,
                session_summary=session_summary,
                media=media,
                channel=channel,
                chat_id=chat_id,
                account_id=account_id,
                persona_config=persona_override,
            )
        else:
            if context is None:
                context = []
            
            messages = list(context)
            messages.append({
                "role": "user",
                "content": message,
            })

        (
            active_provider,
            runtime_model,
            runtime_temperature,
            runtime_max_tokens,
            runtime_max_iterations,
            runtime_thinking_enabled,
        ) = self._resolve_execution_runtime(model_override)
        
        iteration = 0
        total_tool_calls = 0
        final_content = ""
        direct_result_selected = False
        tool_call_limit_reached = False
        self._key_rotation_count = 0
        request_trace_id = f"{session_id[:8]}-{uuid.uuid4().hex[:8]}"

        logger.info(
            "Agent 循环请求开始: "
            f"trace={request_trace_id}, session={session_id}, model={runtime_model or '<default>'}"
        )

        """
        3. 进入循环
            每一轮循环
            1. 取工具定义
            2. 调用模型
            3. 接收模型流式输出
            4. 判断有没有工具调用
            5. 如果没有工具调用，就结束
            6. 如果有工具调用，就执行工具
            7. 把工具结果追加到 messages
            8. 进入下一轮，让模型继续回答
            模型 → 工具 → 模型 → 工具 → 模型 → 最终回答
        """
        try:
            while iteration < runtime_max_iterations:
                iteration += 1
                
                if cancel_token and cancel_token.is_cancelled:
                    logger.debug(f"Agent loop cancelled at iteration {iteration}")
                    return
                
                logger.debug(f"Iteration {iteration}: {total_tool_calls} tool calls")
                
                tool_definitions = self.tools.get_definitions() if self.tools else []
                
                content_buffer = ""
                tool_calls_buffer = []
                finish_reason = None
                reasoning_buffer = ""
                provider_payload = None
                provider_trace_kwargs: Dict[str, Any] = {}
                if active_provider.__class__.__module__.endswith(".openai_provider"):
                    provider_trace_kwargs["request_trace_id"] = request_trace_id

                """
                4. 调用provider的chat_stream()
                    根据大模型的返回值进行处理
                    1.) 普通文本 chunk.content
                    2.) 工具调用 chunk.tool_call
                    3.) 推理内容 chunk.reasoning_content
                    4.) provider 原始附加信息 chunk.provider_payload
                    5.) 结束原因 chunk.finish_reason
                    6.) 错误 chunk.error
                """
                async for chunk in active_provider.chat_stream(
                    messages=messages,
                    tools=tool_definitions,
                    model=runtime_model,
                    temperature=runtime_temperature,
                    max_tokens=runtime_max_tokens,
                    thinking_enabled=runtime_thinking_enabled,
                    **provider_trace_kwargs,
                ):
                    """
                    5. 返回普通文本
                    """
                    if chunk.is_content and chunk.content:
                        content_buffer += chunk.content
                        if yield_intermediate:
                            yield chunk.content
                    """
                    6. 返回工具调用
                        将工具收集，后面它会去重、检查剩余工具调用次数，
                        然后把这次模型的 assistant 消息和 tool_calls 追加到上下文
                        源码里还会把工具调用转换成 OpenAI function call 风格的结构，模型需要知道这些操作
                    """
                    if chunk.is_tool_call and chunk.tool_call:
                        tool_calls_buffer.append(chunk.tool_call)
                    
                    if chunk.is_reasoning and chunk.reasoning_content:
                        reasoning_buffer += chunk.reasoning_content
                        if reasoning_event_handler:
                            try:
                                maybe_result = reasoning_event_handler(
                                    chunk.reasoning_content
                                )
                                if inspect.isawaitable(maybe_result):
                                    await maybe_result
                            except Exception as exc:
                                logger.warning(
                                    f"Failed to emit reasoning chunk for session {session_id}: {exc}"
                                )
                    
                    if chunk.has_provider_payload and chunk.provider_payload:
                        provider_payload = chunk.provider_payload

                    if chunk.is_done and chunk.finish_reason:
                        finish_reason = chunk.finish_reason
                    
                    if chunk.is_error:
                        error_text = chunk.raw_error or chunk.error or ""
                        rotated_provider = self._try_key_rotation(
                            active_provider, error_text
                        )
                        if rotated_provider is not None:
                            active_provider = rotated_provider
                            iteration -= 1
                            self._key_rotation_count += 1
                            await asyncio.sleep(1.0)
                            continue
                        yield chunk.error
                        return
                
                if content_buffer:
                    final_content = content_buffer
                elif reasoning_buffer and not tool_calls_buffer:
                    final_content = reasoning_buffer
                
                if tool_calls_buffer:
                    deduped_tool_calls = []
                    seen_tool_call_ids = set()
                    for tc in tool_calls_buffer:
                        if tc.id and tc.id in seen_tool_call_ids:
                            logger.warning(
                                f"Skipping duplicate tool call by id in main loop: {tc.name} ({tc.id})"
                            )
                            continue

                        if tc.id:
                            seen_tool_call_ids.add(tc.id)
                        deduped_tool_calls.append(tc)

                    tool_calls_buffer = deduped_tool_calls

                    logger.info(
                        "已接收工具批次: "
                        f"trace={request_trace_id}, iteration={iteration}, count={len(tool_calls_buffer)}, "
                        f"calls=[{self._summarize_tool_calls_for_log(tool_calls_buffer)}]"
                    )

                    remaining_tool_slots = runtime_max_iterations - total_tool_calls
                    if remaining_tool_slots <= 0:
                        logger.warning(
                            "Reached max tool call limit before executing a new tool call batch; "
                            "aborting batch to avoid sending unmatched tool results upstream"
                        )
                        tool_call_limit_reached = True
                        break
                    if len(tool_calls_buffer) > remaining_tool_slots:
                        logger.warning(
                            f"Truncating tool call batch from {len(tool_calls_buffer)} to "
                            f"{remaining_tool_slots} to keep tool_calls/tool_results aligned"
                        )
                        tool_calls_buffer = tool_calls_buffer[:remaining_tool_slots]

                    tool_call_dicts = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in tool_calls_buffer
                    ]
                    
                    if self.context_builder:
                        messages = self.context_builder.add_assistant_message(
                            messages,
                            content_buffer or None,
                            tool_call_dicts,
                            reasoning_content=reasoning_buffer or None,
                            provider_payload=provider_payload,
                        )
                    else:
                        msg = {
                            "role": "assistant",
                            "content": content_buffer or "",
                            "tool_calls": tool_call_dicts,
                        }
                        if reasoning_buffer:
                            msg["reasoning_content"] = reasoning_buffer
                        if provider_payload:
                            msg.update(provider_payload)
                        messages.append(msg)
                    """
                    7. 遍历工具调用
                        执行前，它会通过 tool_event_handler 和 WebSocket 通知机制告诉前端：现在开始执行某个工具。
                        然后进入重试逻辑
                    """
                    for tool_call in tool_calls_buffer:
                        if total_tool_calls >= runtime_max_iterations:
                            logger.warning(
                                f"Reached max tool calls limit ({runtime_max_iterations}), "
                                f"skipping remaining tool calls in this iteration"
                            )
                            break
                        
                        if cancel_token and cancel_token.is_cancelled:
                            logger.debug(f"Agent loop cancelled before tool execution")
                            return

                        total_tool_calls += 1
                        tool_name = tool_call.name
                        tool_args = tool_call.arguments
                        tool_id = tool_call.id

                        logger.info(
                            "开始执行工具: "
                            f"trace={request_trace_id}, seq={total_tool_calls}, "
                            f"name={tool_name}, tool_call_id={tool_id}"
                        )

                        if tool_event_handler:
                            try:
                                maybe_result = tool_event_handler(
                                    "tool_call",
                                    {
                                        "tool_name": tool_name,
                                        "arguments": tool_args,
                                        "session_id": session_id,
                                    },
                                )
                                if inspect.isawaitable(maybe_result):
                                    await maybe_result
                            except Exception as e:
                                logger.warning(f"Tool event handler failed before execution: {e}")
                        
                        try:
                            from backend.ws.tool_notifications import notify_tool_execution
                            await notify_tool_execution(
                                session_id=session_id,
                                tool_name=tool_name,
                                arguments=tool_args,
                            )
                        except Exception as e:
                            logger.warning(f"Failed to send tool notification: {e}")
                        
                        start_time = time.time()
                        result = None
                        last_error = None
                        
                        if self.tools:
                            self.tools.set_tool_event_handler(tool_event_handler)
                        try:
                            for attempt in range(self.max_retries):
                                try:
                                    """
                                    执行工具函数
                                    """
                                    result = await self.execute_tool(tool_name, tool_args)
                                    logger.debug(f"Tool {tool_name} succeeded")
                                    break
                                except Exception as e:
                                    last_error = e
                                    logger.warning(
                                        f"Tool {tool_name} failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                                    )
                                    if attempt < self.max_retries - 1:
                                        await asyncio.sleep(self.retry_delay)
                        finally:
                            if self.tools:
                                self.tools.set_tool_event_handler(None)
                        
                        duration_ms = int((time.time() - start_time) * 1000)
                        
                        if result is not None:
                            try:
                                conversation_history = get_conversation_history()
                                conversation_history.add_conversation(
                                    session_id=session_id,
                                    tool_name=tool_name,
                                    arguments=tool_args,
                                    user_message=message,
                                    result=result,
                                    duration_ms=duration_ms
                                )
                            except Exception as e:
                                logger.warning(f"Failed to record tool conversation: {e}")
                            
                            try:
                                from backend.ws.tool_notifications import notify_tool_execution
                                await notify_tool_execution(
                                    session_id=session_id,
                                    tool_name=tool_name,
                                    arguments=tool_args,
                                    result=result,
                                )
                            except Exception as e:
                                logger.warning(f"Failed to send tool result notification: {e}")

                            if tool_event_handler:
                                try:
                                    maybe_result = tool_event_handler(
                                        "tool_result",
                                        {
                                            "tool_name": tool_name,
                                            "arguments": tool_args,
                                            "result": result,
                                            "session_id": session_id,
                                            "duration_ms": duration_ms,
                                        },
                                    )
                                    if inspect.isawaitable(maybe_result):
                                        await maybe_result
                                except Exception as e:
                                    logger.warning(f"Tool event handler failed after execution: {e}")

                            if tool_name == "workflow_run" and prefer_direct_workflow_result:
                                final_content = result
                                direct_result_selected = True
                                if result:
                                    yield result
                                break
                            
                            if self.context_builder:
                                messages = self.context_builder.add_tool_result(
                                    messages,
                                    tool_id,
                                    tool_name,
                                    result,
                                )
                            else:
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "name": tool_name,
                                    "content": result,
                                })
                            logger.info(
                                "已追加工具结果: "
                                f"trace={request_trace_id}, name={tool_name}, tool_call_id={tool_id}, "
                                f"status=success, duration_ms={duration_ms}"
                            )
                        else:
                            error_msg = f"Tool execution failed after {self.max_retries} attempts: {str(last_error)}"
                            logger.error(f"Tool {tool_name} failed permanently: {error_msg}")
                            
                            try:
                                conversation_history = get_conversation_history()
                                conversation_history.add_conversation(
                                    session_id=session_id,
                                    tool_name=tool_name,
                                    arguments=tool_args,
                                    user_message=message,
                                    error=error_msg,
                                    duration_ms=duration_ms
                                )
                            except Exception as e:
                                logger.warning(f"Failed to record tool conversation: {e}")
                            
                            try:
                                from backend.ws.tool_notifications import notify_tool_execution
                                await notify_tool_execution(
                                    session_id=session_id,
                                    tool_name=tool_name,
                                    arguments=tool_args,
                                    error=error_msg,
                                )
                            except Exception as e:
                                logger.warning(f"Failed to send tool error notification: {e}")

                            if tool_event_handler:
                                try:
                                    maybe_result = tool_event_handler(
                                        "tool_error",
                                        {
                                            "tool_name": tool_name,
                                            "arguments": tool_args,
                                            "error": error_msg,
                                            "session_id": session_id,
                                            "duration_ms": duration_ms,
                                        },
                                    )
                                    if inspect.isawaitable(maybe_result):
                                        await maybe_result
                                except Exception as e:
                                    logger.warning(f"Tool event handler failed on error: {e}")
                            """
                            8. 将工具结果加回messages
                            """
                            if self.context_builder:
                                messages = self.context_builder.add_tool_result(
                                    messages,
                                    tool_id,
                                    tool_name,
                                    error_msg,
                                )
                            else:
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "name": tool_name,
                                    "content": error_msg,
                                })
                            logger.info(
                                "已追加工具结果: "
                                f"trace={request_trace_id}, name={tool_name}, tool_call_id={tool_id}, "
                                f"status=error, duration_ms={duration_ms}"
                            )
                    if direct_result_selected:
                        break
                else:
                    """
                    9. 没有工具调用，只产生文本
                        任务完成
                    """
                    if not yield_intermediate and content_buffer:
                        yield content_buffer
                    break

                if direct_result_selected:
                    break
            """
            10. 限制最大迭代，最大工具调用
            """
            # 检查是否达到限制
            if (
                tool_call_limit_reached
                or iteration >= runtime_max_iterations
                or total_tool_calls >= runtime_max_iterations
            ):
                if tool_call_limit_reached or total_tool_calls >= runtime_max_iterations:
                    logger.warning(f"Max tool calls ({runtime_max_iterations}) reached")
                    warning_msg = f"\n\n[达到最大工具调用次数 {runtime_max_iterations}]"
                else:
                    logger.warning(f"Max iterations ({runtime_max_iterations}) reached")
                    warning_msg = f"\n\n[达到最大迭代次数 {runtime_max_iterations}]"
                yield warning_msg
                final_content += warning_msg
            """
            11. 保存会话和审计日志 
                如果有 session_manager，它会把用户消息和 AI 回复保存到会话里
                如果有工具系统，还会记录 AI 完整响应到审计日志
            """
            # 保存到会话（如果有 session_manager）
            if self.session_manager and final_content:
                try:
                    session = self.session_manager.get_or_create(session_id)
                    session.add_message("user", message)
                    session.add_message("assistant", final_content)
                    self.session_manager.save(session)
                except Exception as e:
                    logger.warning(f"Failed to save session: {e}")
            
            # 记录AI完整响应到审计日志
            if self.tools and final_content:
                try:
                    from backend.modules.tools.file_audit_logger import file_audit_logger
                    file_audit_logger.record_ai_response(
                        session_id=session_id,
                        user_message=message,
                        ai_response=final_content,
                        duration_ms=None  # 暂时不记录耗时
                    )
                except Exception as e:
                    logger.warning(f"Failed to record AI response to audit log: {e}")
                
        except Exception as e:
            logger.exception(f"Error in agent loop: {e}")
            raise

    """
    API key轮换，尝试换备用API，重新创建provider
    """
    def _try_key_rotation(
        self,
        current_provider: Any,
        error_text: str,
    ) -> Optional[Any]:
        """尝试通过 key 轮换恢复请求。

        当错误是认证/限流相关时，切换到下一个 API Key 并返回新的 provider。
        如果无法轮换（只有一个 key 或不适用），返回 None。
        """
        if not _is_key_rotation_eligible_error(error_text):
            return None

        if self._key_rotation_count >= self.MAX_KEY_ROTATION_RETRIES:
            logger.warning(
                f"Key rotation limit reached ({self.MAX_KEY_ROTATION_RETRIES}), "
                f"stopping rotation attempts"
            )
            return None

        provider_id = getattr(current_provider, "provider_id", None)
        if not provider_id:
            return None

        from backend.modules.providers.runtime import get_key_rotator, KeyRotator
        from backend.modules.config.loader import config_loader
        from backend.modules.providers.runtime import get_provider_runtime_state

        config = config_loader.config
        runtime_state = get_provider_runtime_state(config, provider_id)
        api_keys = runtime_state.api_keys

        if len(api_keys) <= 1:
            logger.debug(
                f"Key rotation skipped for {provider_id}: only {len(api_keys)} key(s) available"
            )
            return None

        rotator = get_key_rotator(provider_id, api_keys)
        current_key = getattr(current_provider, "api_key", "") or ""
        next_key = rotator.mark_key_failed(current_key)

        if not next_key or next_key == current_key:
            logger.warning(
                f"Key rotation exhausted for {provider_id}: no alternative key available"
            )
            return None

        logger.info(
            f"Key rotation for {provider_id}: switching from "
            f"{current_key[:8]}... to {next_key[:8]}... "
            f"(error: {error_text[:100]})"
        )

        from backend.modules.providers import create_provider

        try:
            new_provider = create_provider(
                api_key=next_key,
                api_keys=api_keys,
                api_base=runtime_state.api_base,
                default_model=getattr(current_provider, "default_model", None),
                api_mode=getattr(current_provider, "api_mode", "chat_completions"),
                timeout=getattr(current_provider, "timeout", 120.0),
                max_retries=getattr(current_provider, "max_retries", self.max_retries),
                provider_id=provider_id,
            )
            return new_provider
        except Exception as exc:
            logger.warning(f"Failed to create rotated provider: {exc}")
            return None

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> str:
        """
        执行工具调用
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数
            
        Returns:
            str: 工具执行结果
            
        Raises:
            ValueError: 工具不存在
            Exception: 工具执行失败
        """
        if not self.tools:
            raise ValueError("ToolRegistry not initialized")
        
        logger.debug(f"执行工具: {tool_name}")
        
        try:
            result = await self.tools.execute(tool_name, arguments, auto_record=False)
            return result
            
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {e}")
            raise
    """
    非web场景直接调用，会调用process_message()
        Web 聊天：process_message()，边生成边返回
        CLI / 定时任务：process_direct()，收集完整结果后返回
    """
    async def process_direct(
        self,
        content: str,
        session_id: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        account_id: Optional[str] = None,
    ) -> str:
        """
        直接处理消息（用于 CLI 或 cron 使用）
        
        Args:
            content: 消息内容
            session_id: 会话标识符
            channel: 来源渠道（用于上下文）
            chat_id: 来源聊天 ID（用于上下文）
            account_id: 当前机器人账号 ID（多机器人渠道）
        
        Returns:
            Agent 的响应
        """
        response_parts = []
        
        # 传入空的 context 列表
        async for chunk in self.process_message(
            message=content,
            session_id=session_id,
            context=[],  
            channel=channel,
            chat_id=chat_id,
            account_id=account_id,
        ):
            response_parts.append(chunk)
        
        return "".join(response_parts)
