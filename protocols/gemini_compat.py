import json
from typing import Any, Optional

from fastapi.responses import JSONResponse, StreamingResponse


def _gemini_system_to_text(system_value: Any) -> str:
    if isinstance(system_value, str):
        return system_value
    if not isinstance(system_value, dict):
        return ""

    parts = system_value.get("parts", [])
    if not isinstance(parts, list):
        return ""

    texts: list[str] = []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            texts.append(part.get("text", ""))
    return "\n".join([x for x in texts if x])


def _normalize_gemini_contents(contents: list):
    normalized: list = []
    files: list[dict] = []

    for item in contents:
        if not isinstance(item, dict):
            continue

        role_raw = item.get("role", "user")
        role = "assistant" if role_raw == "model" else role_raw

        parts = item.get("parts", [])
        texts: list[str] = []
        if isinstance(parts, list):
            for part in parts:
                if not isinstance(part, dict):
                    continue
                if isinstance(part.get("text"), str):
                    texts.append(part.get("text", ""))

                inline_data = part.get("inlineData")
                if isinstance(inline_data, dict):
                    data = inline_data.get("data", "")
                    if data:
                        mime_type = inline_data.get("mimeType", "application/octet-stream")
                        suffix = str(mime_type).split("/")[-1] if "/" in str(mime_type) else "bin"
                        files.append({"name": f"inline.{suffix}", "data": data})

        normalized.append({
            "role": role,
            "content": "\n".join(texts),
        })

    return normalized, files


def _gemini_usage_from_openai(usage: Optional[dict]) -> dict:
    usage = usage or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or (prompt_tokens + completion_tokens))
    return {
        "promptTokenCount": prompt_tokens,
        "candidatesTokenCount": completion_tokens,
        "totalTokenCount": total_tokens,
    }


def _gemini_sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _prepare_gemini_proxy_input(body: dict, model_name: str, default_model: str, build_user_text):
    raw_model = body.get("model", model_name)
    if isinstance(raw_model, str) and raw_model.strip():
        model = raw_model.strip()
    else:
        model = model_name or default_model
    if model.startswith("models/"):
        model = model.split("/", 1)[1]

    contents_raw = body.get("contents", [])
    if not isinstance(contents_raw, list):
        contents_raw = []
    messages, files = _normalize_gemini_contents(contents_raw)

    system_raw = body.get("systemInstruction", body.get("system_instruction", ""))
    system_prompt = _gemini_system_to_text(system_raw)

    if messages and messages[0].get("role") == "system":
        extra_system = messages[0].get("content", "")
        if extra_system:
            system_prompt = f"{system_prompt}\n\n{extra_system}".strip() if system_prompt else extra_system
        messages = messages[1:]

    user_text = build_user_text(messages, system_prompt)
    if not isinstance(user_text, str) or not user_text:
        return model, "", "", files, JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": 400,
                    "message": "Invalid Gemini request: empty contents",
                    "status": "INVALID_ARGUMENT",
                }
            },
        )

    return model, user_text, system_prompt, files, None


async def handle_generate_content(
    model_name: str,
    body: dict,
    default_model: str,
    build_user_text,
    proxy_chat_events,
):
    model, user_text, system_prompt, files, error_response = _prepare_gemini_proxy_input(
        body,
        model_name,
        default_model,
        build_user_text,
    )
    if error_response is not None:
        return error_response

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
                    "error": {
                        "code": 500,
                        "message": event["data"],
                        "status": "INTERNAL",
                    }
                },
            )

    if full_reasoning:
        if full_content:
            full_content = f"{full_content}\n\n[thinking]\n{full_reasoning}"
        else:
            full_content = f"[thinking]\n{full_reasoning}"

    return {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": full_content}],
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": _gemini_usage_from_openai(final_usage),
    }


async def handle_stream_generate_content(
    model_name: str,
    body: dict,
    default_model: str,
    build_user_text,
    proxy_chat_events,
):
    model, user_text, system_prompt, files, error_response = _prepare_gemini_proxy_input(
        body,
        model_name,
        default_model,
        build_user_text,
    )
    if error_response is not None:
        return error_response

    async def gemini_stream_generator():
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
                yield _gemini_sse(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": event["data"]}],
                                },
                                "index": 0,
                            }
                        ]
                    }
                )
            elif event["type"] == "reasoning":
                yield _gemini_sse(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [{"text": "\n[thinking] " + event["data"]}],
                                },
                                "index": 0,
                            }
                        ]
                    }
                )
            elif event["type"] == "usage":
                final_usage = event["data"]
            elif event["type"] == "error":
                yield _gemini_sse(
                    {
                        "error": {
                            "code": 500,
                            "message": event["data"],
                            "status": "INTERNAL",
                        }
                    }
                )
                return

        yield _gemini_sse(
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": ""}],
                        },
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "usageMetadata": _gemini_usage_from_openai(final_usage),
            }
        )

    return StreamingResponse(gemini_stream_generator(), media_type="text/event-stream")
