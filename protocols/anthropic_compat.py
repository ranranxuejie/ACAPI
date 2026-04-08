import json
import uuid
from typing import Any, Optional

from fastapi.responses import JSONResponse, StreamingResponse


def _anthropic_system_to_text(system_value: Any) -> str:
    if isinstance(system_value, str):
        return system_value
    if isinstance(system_value, dict):
        if system_value.get("type") == "text":
            return str(system_value.get("text", ""))
        return ""
    if isinstance(system_value, list):
        parts: list[str] = []
        for item in system_value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join([p for p in parts if p])
    return ""


def _normalize_anthropic_messages(messages: list, extract_text_and_files):
    normalized: list = []
    files: list[dict] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        text, msg_files = extract_text_and_files(msg.get("content", ""))
        files.extend(msg_files)
        normalized.append({
            "role": msg.get("role", "user"),
            "content": text,
        })

    return normalized, files


def _anthropic_usage_from_openai(usage: Optional[dict]) -> dict:
    usage = usage or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
    }


def _anthropic_sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def handle_messages(
    body: dict,
    default_model: str,
    build_user_text,
    extract_text_and_files,
    proxy_chat_events,
):
    model = body.get("model", default_model)
    is_stream = body.get("stream", False)

    messages_raw = body.get("messages", [])
    messages, files = _normalize_anthropic_messages(messages_raw, extract_text_and_files)

    system_prompt = _anthropic_system_to_text(body.get("system", ""))
    if messages and messages[0].get("role") == "system":
        extra_system = messages[0].get("content", "")
        if extra_system:
            system_prompt = f"{system_prompt}\n\n{extra_system}".strip() if system_prompt else extra_system
        messages = messages[1:]

    user_text = build_user_text(messages, system_prompt)
    if not isinstance(user_text, str) or not user_text:
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "无法解析用户消息"},
            },
        )

    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    if is_stream:

        async def anthropic_stream_generator():
            final_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

            yield _anthropic_sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                },
            )
            yield _anthropic_sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            )

            async for event in proxy_chat_events(
                model=model,
                system_prompt=system_prompt,
                user_text=user_text,
                files=files,
            ):
                if event["type"] == "content":
                    yield _anthropic_sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": event["data"]},
                        },
                    )
                elif event["type"] == "reasoning":
                    yield _anthropic_sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": "\n[thinking] " + event["data"]},
                        },
                    )
                elif event["type"] == "usage":
                    final_usage = event["data"]
                elif event["type"] == "error":
                    yield _anthropic_sse(
                        "error",
                        {
                            "type": "error",
                            "error": {"type": "api_error", "message": event["data"]},
                        },
                    )
                    return

            yield _anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": 0})
            yield _anthropic_sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": _anthropic_usage_from_openai(final_usage),
                },
            )
            yield _anthropic_sse("message_stop", {"type": "message_stop"})

        return StreamingResponse(anthropic_stream_generator(), media_type="text/event-stream")

    full_content = ""
    full_reasoning = ""
    final_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    async for event in proxy_chat_events(
        model=model,
        system_prompt=system_prompt,
        user_text=user_text,
        files=files,
    ):
        if event["type"] == "content":
            full_content += event["data"]
        elif event["type"] == "reasoning":
            full_reasoning += event["data"]
        elif event["type"] == "usage":
            final_usage = event["data"]
        elif event["type"] == "error":
            return JSONResponse(
                status_code=500,
                content={
                    "type": "error",
                    "error": {"type": "api_error", "message": event["data"]},
                },
            )

    if full_reasoning:
        if full_content:
            full_content = f"{full_content}\n\n[thinking]\n{full_reasoning}"
        else:
            full_content = f"[thinking]\n{full_reasoning}"

    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": full_content}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": _anthropic_usage_from_openai(final_usage),
    }
