"""
ACAI Proxy API - OpenAI 兼容代理
将上游 ACAI 站点转换为 OpenAI 格式的 API 接口
"""
import json
import asyncio
import os
import time
import logging
from typing import Optional, Any, AsyncGenerator

import httpx
from fastapi import FastAPI, Request

from protocols.openai_compat import handle_chat_completions
from protocols.anthropic_compat import handle_messages as handle_anthropic_messages
from protocols.gemini_compat import (
    handle_generate_content,
    handle_stream_generate_content,
)

# ============================================================
#  配置区
# ============================================================
BASE_URL = "https://acai.krxxl.cn/api"
UPSTREAM_ORIGIN = "https://acai.krxxl.cn"
AUTH_TOKEN = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJ1c2VySWQiOjEwMTI4LCJzaWduIjoiNGEwMTE3MmI3MjU3NzIxNDk3ODZiMTQ3N2Q2MmQ4ZjQiLCJyb2xlIjoidXNlciIsImV4cCI6MTc3ODI5NDA3NywibmJmIjoxNzc1NjE1Njc3LCJpYXQiOjE3NzU2MTU2Nzd9"
    ".ZSfbbPd5b0X72Wn6yuWt915Q4EU0u-ZX9nk5kwQEymU"
)
SESSION_IDS_RAW = os.getenv("ACAI_SESSION_IDS", "399226")
SESSION_IDS = [int(x.strip()) for x in SESSION_IDS_RAW.split(",") if x.strip().isdigit()]
if not SESSION_IDS:
    SESSION_IDS = [399226]
USER_ID = 10128
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
CHAT_TIMEOUT = 120.0
GENERAL_TIMEOUT = 10.0
HTTP_PROXY = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
ACAI_COOKIE = os.getenv("ACAI_COOKIE", "").strip()

# ============================================================
#  日志 & 应用初始化
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("Proxy")

app = FastAPI(title="ACAI Proxy", version="2.0")

# Session 池 (异步安全)
session_pool: asyncio.Queue[int] = asyncio.Queue(maxsize=len(SESSION_IDS))
for _sid in SESSION_IDS:
    session_pool.put_nowait(_sid)


# ============================================================
#  工具函数
# ============================================================
def _headers(accept: str = "text/event-stream") -> dict:
    """构建上游请求头"""
    headers = {
        "Accept": accept,
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Authorization": AUTH_TOKEN,
        "Connection": "keep-alive",
        "Content-Type": "application/json",
        "Origin": UPSTREAM_ORIGIN,
        "Referer": f"{UPSTREAM_ORIGIN}/chat",
        "Sec-CH-UA": '"Chromium";v="135", "Not-A.Brand";v="8"',
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": '"Windows"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
        "X-APP-VERSION": "2.14.0",
    }
    if ACAI_COOKIE:
        headers["Cookie"] = ACAI_COOKIE if "=" in ACAI_COOKIE else f"https_waf_cookie={ACAI_COOKIE}"
    return headers


def _client_kwargs(timeout: float, follow_redirects: bool = False) -> dict:
    kwargs = {
        "verify": False,
        "timeout": timeout,
        "follow_redirects": follow_redirects,
    }
    if HTTP_PROXY:
        kwargs["proxy"] = HTTP_PROXY
    return kwargs


def _wrap_chunk(
    model: str,
    chunk_id: str,
    content: str = "",
    reasoning: str = "",
    usage: Optional[dict] = None,
) -> str:
    """将内容包装为 OpenAI SSE chunk 格式"""
    delta: dict = {}
    if reasoning:
        delta["reasoning_content"] = reasoning
    if content:
        delta["content"] = content
    if usage:
        delta["usage"] = usage
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        "model": model,
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _build_user_text(messages: list,system_prompt) -> str:
    """
    将 OpenAI 格式的 messages 打包为单条用户文本。
    最后一条作为 current_text，前面的作为 context。
    """
    if not messages:
        return system_prompt
    context = str(messages[:-1]) if len(messages) > 1 else ""
    current_text = messages[-1].get("content", "")
    if not current_text.strip() and not context.strip():
        return system_prompt
    if context:
        return f"<context>{context}</context>\n\n{current_text}"
    return current_text


def _extract_text_and_files(content: Any) -> tuple[str, list[dict]]:
    """提取文本与附件，兼容 OpenAI/Anthropic 的 content 结构。"""
    texts: list[str] = []
    files: list[dict] = []

    if isinstance(content, str):
        return content, files

    if isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                texts.append(item)
                continue
            if not isinstance(item, dict):
                continue

            item_type = item.get("type")
            if item_type == "text":
                text_value = item.get("text", "")
                if isinstance(text_value, str):
                    texts.append(text_value)
            elif item_type == "file" and isinstance(item.get("file"), dict):
                file_info = item["file"]
                files.append({
                    "name": file_info.get("filename", "unknown"),
                    "data": file_info.get("file_data", ""),
                })
            elif item_type in ("image", "document") and isinstance(item.get("source"), dict):
                source = item["source"]
                data = source.get("data", "")
                if data:
                    media_type = source.get("media_type", "application/octet-stream")
                    suffix = str(media_type).split("/")[-1] if "/" in str(media_type) else "bin"
                    files.append({"name": f"{item_type}.{suffix}", "data": data})

    return "\n".join(texts), files


async def _proxy_chat_events(
    model: str,
    system_prompt: str,
    user_text: str,
    files: list[dict],
) -> AsyncGenerator[dict, None]:
    """统一的上游事件流，向下游输出 content/reasoning/usage/error 事件。"""
    selected_session_id = None
    try:
        selected_session_id = await session_pool.get()
        logger.info("分配 session_id=%d", selected_session_id)

        t0 = time.time()
        session_result = await change_session(
            target_session_id=selected_session_id,
            model=model,
            system_prompt=system_prompt,
        )
        if session_result.get("code") != 0:
            yield {
                "type": "error",
                "data": f"upstream session update failed (sessionId={selected_session_id}): {session_result.get('msg', 'unknown')}",
            }
            return

        t1 = time.time()
        logger.info("修改上游会话配置耗时: %.3f 秒", t1 - t0)
        await asyncio.sleep(0.1)

        payload = {"sessionId": selected_session_id, "text": user_text, "files": files}
        t2 = time.time()
        async with httpx.AsyncClient(**_client_kwargs(CHAT_TIMEOUT)) as client:
            async with client.stream("POST", f"{BASE_URL}/chat/completions", headers=_headers(), json=payload) as resp:
                if resp.status_code != 200:
                    upstream_body = (await resp.aread()).decode("utf-8", errors="ignore")
                    raise RuntimeError(f"upstream completions http {resp.status_code}: {upstream_body[:300]}")

                logger.info("上游接通耗时 (TTFB 之前/建连等): %.3f 秒", time.time() - t2)
                buffer = ""
                is_thinking = False
                first_chunk = True

                try:
                    async for raw_bytes in resp.aiter_bytes():
                        if first_chunk:
                            logger.info("首字节到达耗时 (TTFB): %.3f 秒", time.time() - t2)
                            first_chunk = False
                        buffer += raw_bytes.decode("utf-8", errors="ignore")
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            if not line.startswith("data:"):
                                continue
                            raw_str = line[5:].strip()
                            if raw_str == "[DONE]":
                                break

                            try:
                                obj = json.loads(raw_str)
                            except json.JSONDecodeError:
                                continue

                            if obj.get("err"):
                                err_msg = obj.get("err")
                                yield {
                                    "type": "content",
                                    "data": "\n[上游由于配置或模型限制报错: " + err_msg + "]\n",
                                }
                                continue

                            if obj.get("type") == "string":
                                text = obj.get("data", "")
                            elif obj.get("type") in ("object", "stats"):
                                d = obj.get("data", {})
                                usage = {
                                    "prompt_tokens": d.get("promptTokens", 0),
                                    "completion_tokens": d.get("completionTokens", 0),
                                    "total_tokens": d.get("promptTokens", 0) + d.get("completionTokens", 0),
                                }
                                yield {"type": "usage", "data": usage}
                                continue
                            else:
                                continue

                            if not text:
                                continue

                            if "<think>" in text:
                                is_thinking = True
                                text = text.replace("<think>", "")
                            if "</think>" in text:
                                parts = text.split("</think>")
                                if parts[0]:
                                    yield {"type": "reasoning", "data": parts[0]}
                                is_thinking = False
                                text = parts[1] if len(parts) > 1 else ""

                            if text:
                                if is_thinking:
                                    yield {"type": "reasoning", "data": text}
                                else:
                                    yield {"type": "content", "data": text}
                except httpx.RemoteProtocolError:
                    logger.warning("上游连接提前关闭 (incomplete chunked read)，已收集的数据仍有效")

    except Exception as e:
        logger.error("生成异常: %r", e)
        yield {"type": "error", "data": str(e) or repr(e)}
    finally:
        if selected_session_id is not None:
            session_pool.put_nowait(selected_session_id)
            logger.info("归还 session_id=%d", selected_session_id)


# ============================================================
#  上游交互
# ============================================================
async def fetch_remote_models() -> list[dict]:
    """从上游获取可用模型列表"""
    url = f"{BASE_URL}/chat/tmpl"
    async with httpx.AsyncClient(**_client_kwargs(GENERAL_TIMEOUT, follow_redirects=True)) as client:
        try:
            resp = await client.get(url, headers=_headers("application/json, text/plain, */*"))
            if resp.status_code != 200:
                logger.warning("获取模型列表失败: HTTP %d", resp.status_code)
                return []
            data = resp.json()
            if data.get("code") != 0:
                logger.warning("获取模型列表失败: code=%s", data.get("code"))
                return []

            inner = data.get("data", {})
            models_raw = inner.get("models") if isinstance(inner, dict) else inner
            model_list = []
            if isinstance(models_raw, list):
                for m in models_raw:
                    if isinstance(m, dict) and m.get("value"):
                        model_list.append({"id": m["value"], "object": "model", "owned_by": "upstream"})
            elif isinstance(models_raw, dict):
                for m_id in models_raw:
                    model_list.append({"id": m_id, "object": "model", "owned_by": "upstream"})

            logger.info("已同步 %d 个模型", len(model_list))
            return model_list
        except Exception as e:
            logger.error("获取模型列表异常: %s", e)
            return []


async def change_session(
    target_session_id: int,
    model: str = DEFAULT_MODEL,
    system_prompt: str = "",
    plugins: Optional[list] = None,
    mcp: Optional[list] = None,
) -> dict:
    """修改上游会话配置（模型/提示词/插件/MCP）"""
    url = f"{BASE_URL}/chat/session/{target_session_id}"
    payload = {
        "id": target_session_id,
        "created": "2026-03-10 10:14:53",
        "updated": "2026-03-10 10:49:03",
        "uid": USER_ID,
        "name": "调用api专用",
        "model": model,
        "maxToken": 0,
        "contextCount": 0,
        "temperature": 0,
        "presencePenalty": 0,
        "frequencyPenalty": 0,
        "prompt": system_prompt,
        "topSort": 0,
        "icon": "",
        "plugins": plugins or [],
        "mcp": mcp or [],
        "localPlugins": None,
        "useAppId": 0,
    }
    async with httpx.AsyncClient(**_client_kwargs(GENERAL_TIMEOUT, follow_redirects=True)) as client:
        try:
            resp = await client.put(url, headers=_headers("application/json, text/plain, */*"), json=payload)
            try:
                resp_json = resp.json()
            except json.JSONDecodeError:
                logger.warning("修改会话返回非 JSON: HTTP %d, body=%s", resp.status_code, resp.text[:300])
                return {"code": 1, "msg": "upstream non-json response"}

            if resp.status_code == 200 and resp_json.get("code") == 0:
                logger.info(
                    "会话 %d 配置已更新 | model=%s | plugins=%s | mcp=%s",
                    target_session_id, model, plugins or [], mcp or [],
                )
                return resp_json
            logger.warning("修改会话失败: HTTP %d, body=%s", resp.status_code, resp.text[:300])
            return {"code": 1, "msg": f"upstream session http {resp.status_code}"}
        except Exception as e:
            logger.error("修改会话异常: %r", e)
            return {"code": 1, "msg": repr(e)}


# ============================================================
#  路由
# ============================================================
@app.get("/v1/models")
async def list_models():
    """OpenAI 兼容的模型列表接口"""
    models = await fetch_remote_models()
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    return await handle_chat_completions(
        body=body,
        default_model=DEFAULT_MODEL,
        build_user_text=_build_user_text,
        extract_text_and_files=_extract_text_and_files,
        proxy_chat_events=_proxy_chat_events,
        wrap_chunk=_wrap_chunk,
    )


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    body = await request.json()
    return await handle_anthropic_messages(
        body=body,
        default_model=DEFAULT_MODEL,
        build_user_text=_build_user_text,
        extract_text_and_files=_extract_text_and_files,
        proxy_chat_events=_proxy_chat_events,
    )


@app.post("/v1beta/models/{model_name}:generateContent")
@app.post("/v1/models/{model_name}:generateContent")
async def gemini_generate_content(model_name: str, request: Request):
    body = await request.json()
    return await handle_generate_content(
        model_name=model_name,
        body=body,
        default_model=DEFAULT_MODEL,
        build_user_text=_build_user_text,
        proxy_chat_events=_proxy_chat_events,
    )


@app.post("/v1beta/models/{model_name}:streamGenerateContent")
@app.post("/v1/models/{model_name}:streamGenerateContent")
async def gemini_stream_generate_content(model_name: str, request: Request):
    body = await request.json()
    return await handle_stream_generate_content(
        model_name=model_name,
        body=body,
        default_model=DEFAULT_MODEL,
        build_user_text=_build_user_text,
        proxy_chat_events=_proxy_chat_events,
    )


# ============================================================
#  启动入口
# ============================================================
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("API_PORT", "10000")))
    uvicorn.run(app, host=host, port=port)
