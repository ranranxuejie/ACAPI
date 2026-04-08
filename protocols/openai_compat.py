import json
import time
import uuid

from fastapi.responses import JSONResponse, StreamingResponse


def _normalize_openai_messages(messages: list, extract_text_and_files):
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


async def handle_chat_completions(
    body: dict,
    default_model: str,
    build_user_text,
    extract_text_and_files,
    proxy_chat_events,
    wrap_chunk,
):
    model = body.get("model", default_model)
    messages_raw = body.get("messages", [])
    is_stream = body.get("stream", False)

    messages, files = _normalize_openai_messages(messages_raw, extract_text_and_files)

    system_prompt = ""
    if messages and messages[0].get("role") == "system":
        system_prompt = messages[0].get("content", "")
        messages = messages[1:]

    user_text = build_user_text(messages, system_prompt)
    if not isinstance(user_text, str) or not user_text:
        return {"error": {"message": "无法解析用户消息"}}

    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())

    if is_stream:

        async def stream_generator():
            async for event in proxy_chat_events(
                model=model,
                system_prompt=system_prompt,
                user_text=user_text,
                files=files,
            ):
                if event["type"] == "content":
                    yield wrap_chunk(model, chunk_id, content=event["data"])
                elif event["type"] == "reasoning":
                    yield wrap_chunk(model, chunk_id, reasoning=event["data"])
                elif event["type"] == "usage":
                    yield wrap_chunk(model, chunk_id, usage=event["data"])
                elif event["type"] == "error":
                    yield f"data: {json.dumps({'error': {'message': event['data']}})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

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
            return JSONResponse(status_code=500, content={"error": {"message": event["data"]}})

    message = {"role": "assistant", "content": full_content}
    if full_reasoning:
        message["reasoning_content"] = full_reasoning

    return {
        "id": chunk_id,
        "object": "chat.completion",
        "created": created_time,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "stop",
        }],
        "usage": final_usage,
    }
