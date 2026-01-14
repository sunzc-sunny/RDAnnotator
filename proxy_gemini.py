
# google_proxy.py
import os
from flask import Flask, request, jsonify
import requests
import json

app = Flask(__name__)

TARGET_BASE = "https://runway.devops.rednote.life/openai/google"

# 从环境变量读取 API Key
API_KEY = os.getenv("GOOGLE_API_KEY", "b243abe1f06f4e24a10cb473c9c49f87")  # 🔴 替换为你的实际 key

def convert_openai_to_google(openai_data):
    """将 OpenAI 格式转换为 Google 格式"""
    google_data = {
        "contents": [],
        "generationConfig": {}
    }
    
    # 转换 messages
    if "messages" in openai_data:
        for msg in openai_data["messages"]:
            role = msg["role"]
            content = msg["content"]
            
            # 处理 system 消息
            if role == "system":
                google_data["systemInstruction"] = {
                    "parts": [{"text": content}]
                }
            # 处理 user 和 assistant 消息
            else:
                parts = []
                
                # 处理不同类型的 content
                if isinstance(content, str):
                    # 简单文本消息
                    parts.append({"text": content})
                elif isinstance(content, list):
                    # 多模态消息（文本 + 图片）
                    for item in content:
                        if item.get("type") == "text":
                            parts.append({"text": item["text"]})
                        elif item.get("type") == "image_url":
                            # 提取 base64 图片数据
                            image_url = item["image_url"]["url"]
                            if image_url.startswith("data:"):
                                try:
                                    header, base64_data = image_url.split(",", 1)
                                    mime = header.split(";")[0].split(":")[1]
                                    
                                    parts.append({
                                        "inline_data": {
                                            "mime_type": mime,
                                            "data": base64_data
                                        }
                                    })
                                except Exception as e:
                                    print(f"Error parsing image URL: {e}")
                
                if parts:
                    google_data["contents"].append({
                        "role": "user" if role == "user" else "model",
                        "parts": parts
                    })
    
    # 转换生成配置
    if "temperature" in openai_data:
        google_data["generationConfig"]["temperature"] = openai_data["temperature"]
    
    # 🔴 使用 OpenAI 传入的 max_tokens
    if "max_tokens" in openai_data:
        google_data["generationConfig"]["maxOutputTokens"] = openai_data["max_tokens"]
    else:
        google_data["generationConfig"]["maxOutputTokens"] = 65535
    
    if "top_p" in openai_data:
        google_data["generationConfig"]["topP"] = openai_data["top_p"]
    else:
        google_data["generationConfig"]["topP"] = 1
        
    google_data["generationConfig"]["seed"] = 0
    
    return google_data


def convert_google_to_openai(google_response):
    """将 Google 响应格式转换为 OpenAI 格式（提取 thinking 和答案）"""
    try:
        text = ""
        thinking = ""
        finish_reason = "stop"
        
        if "candidates" in google_response and len(google_response["candidates"]) > 0:
            candidate = google_response["candidates"][0]
            
            # 提取 finish reason
            if "finishReason" in candidate:
                reason = candidate["finishReason"]
                if reason == "MAX_TOKENS":
                    finish_reason = "length"
                    print("⚠️  警告：响应因达到 max_tokens 限制被截断")
                elif reason == "STOP":
                    finish_reason = "stop"
                else:
                    finish_reason = "stop"
            
            # 🔴 提取内容（可能包含 thinking 和普通文本）
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                for part in parts:
                    if "text" in part:
                        text += part["text"]
            
            # 🔴 提取 thoughts（如果有单独的 thoughts 字段）
            if "thoughts" in candidate:
                thinking = "\n".join([t.get("text", "") for t in candidate["thoughts"] if "text" in t])
        
        # 🔴 如果没有文本但有思考内容，记录警告
        if not text and not thinking:
            print("❌ 警告：响应中没有文本内容")
        
        # 构建 OpenAI 格式响应
        openai_response = {
            "id": "chatcmpl-" + str(hash(text + thinking))[:16],
            "object": "chat.completion",
            "created": 1234567890,
            "model": "google-gemini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text,
                        "thinking": thinking  # 🔴 添加 thinking 字段
                    },
                    "finish_reason": finish_reason
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        
        # 提取使用统计
        if "usageMetadata" in google_response:
            metadata = google_response["usageMetadata"]
            thinking_tokens = metadata.get("thoughtsTokenCount", 0)
            
            openai_response["usage"] = {
                "prompt_tokens": metadata.get("promptTokenCount", 0),
                "completion_tokens": metadata.get("candidatesTokenCount", 0),
                "total_tokens": metadata.get("totalTokenCount", 0),
                "thinking_tokens": thinking_tokens  # 🔴 添加思考 token 统计
            }
            
            if thinking_tokens > 0:
                print(f"💭 思考过程使用了 {thinking_tokens} tokens")
        
        return openai_response
        
    except Exception as e:
        print(f"转换响应时出错: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": {
                "message": f"转换响应失败: {str(e)}",
                "type": "conversion_error"
            }
        }


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """处理 OpenAI 格式的聊天补全请求"""
    print(f"\n{'='*60}")
    print(f"收到请求: POST /v1/chat/completions")
    
    openai_data = request.get_json()
    request_size = len(json.dumps(openai_data))
    print(f"请求大小: {request_size / 1024:.2f} KB")
    print(f"OpenAI 请求数据: {json.dumps(openai_data, ensure_ascii=False)[:500]}...")
    
    google_data = convert_openai_to_google(openai_data)
    print(f"Google 请求数据: {json.dumps(google_data, ensure_ascii=False)[:500]}...")
    
    url = f"{TARGET_BASE}/v1:generateContent"
    print(f"转发到: {url}")
    
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        is_stream = openai_data.get("stream", False)
        
        if is_stream:
            return jsonify({"error": "Stream mode not supported yet"}), 400
        
        # 发送请求
        import time
        start_time = time.time()
        print("⏳ 正在等待 API 响应...")
        
        resp = requests.post(
            url=url,
            headers=headers,
            json=google_data,
            timeout=300  # 🔴 5分钟超时
        )
        
        elapsed = time.time() - start_time
        print(f"✅ 响应接收完成 (耗时: {elapsed:.2f}s)")
        print(f"响应状态码: {resp.status_code}")
        
        if resp.status_code == 200:
            google_response = resp.json()
            print(f"Google 响应: {json.dumps(google_response, ensure_ascii=False)[:500]}...")
            
            openai_response = convert_google_to_openai(google_response)
            print(f"OpenAI 响应: {json.dumps(openai_response, ensure_ascii=False)[:500]}...")
            
            return jsonify(openai_response), 200
        else:
            error_text = resp.text
            print(f"错误响应: {error_text}")
            return jsonify({
                "error": {
                    "message": error_text,
                    "type": "api_error",
                    "code": resp.status_code
                }
            }), resp.status_code
            
    except requests.exceptions.Timeout:
        print(f"❌ 请求超时 (超过 300 秒)")
        return jsonify({
            "error": {
                "message": "Request timeout after 300 seconds",
                "type": "timeout"
            }
        }), 504
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": {
                "message": str(e),
                "type": "internal_error"
            }
        }), 500


@app.route('/v1/models', methods=['GET'])
def list_models():
    """返回可用模型列表"""
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": "google-gemini",
                "object": "model",
                "created": 1234567890,
                "owned_by": "google"
            }
        ]
    })


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Google Gemini to OpenAI 代理服务器启动")
    print("=" * 60)
    print(f"API Key 状态: {'✅ 已设置' if API_KEY and API_KEY != 'your_actual_api_key_here' else '❌ 未设置'}")
    print("监听地址: http://localhost:8008/v1")
    print("\n功能特性:")
    print("  ✅ 支持提取 thinking 内容")
    print("  ✅ 支持大 token 限制 (最高 65535)")
    print("  ✅ 5分钟超时设置")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=8008, debug=True, use_reloader=False, threaded=True)
