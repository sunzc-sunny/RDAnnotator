# openai_proxy.py
from flask import Flask, request, jsonify
import requests
import json

app = Flask(__name__)

TARGET_BASE = "https://runway.devops.xiaohongshu.com/openai"
API_KEY = "58b624e27bf04ec288c25de9d793bbc1"
API_VERSION = "2024-02-01"

@app.route('/v1/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy(path):
    # 构建目标URL
    url = f"{TARGET_BASE}/{path}?api-version={API_VERSION}"
    
    print(f"\n{'='*60}")
    print(f"收到请求: {request.method} /v1/{path}")
    print(f"转发到: {url}")
    
    # 准备headers
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json"
    }
    
    # 获取请求数据
    data = request.get_json() if request.is_json else None
    print(f"请求数据: {json.dumps(data, ensure_ascii=False)[:200]}...")
    
    try:
        # 转发请求 - 关键：不使用stream
        resp = requests.request(
            method=request.method,
            url=url,
            headers=headers,
            json=data,
            timeout=60,
            stream=False  # 重要：不使用流式传输
        )
        
        print(f"响应状态码: {resp.status_code}")
        print(f"响应头: {dict(resp.headers)}")
        
        # 直接返回JSON - 这是关键修复
        if resp.status_code == 200:
            response_json = resp.json()
            print(f"响应内容: {json.dumps(response_json, ensure_ascii=False)[:200]}...")
            return jsonify(response_json), 200
        else:
            print(f"错误响应: {resp.text}")
            return jsonify({"error": resp.text}), resp.status_code
            
    except requests.exceptions.Timeout:
        print("❌ 请求超时")
        return jsonify({"error": "Request timeout"}), 504
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求错误: {e}")
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 OpenAI API 代理服务器启动")
    print("=" * 60)
    print("监听地址: http://localhost:8000")
    print("\n在你的代码中使用:")
    print('  base_url="http://localhost:8000/v1"')
    print('  api_key="any-string"')
    print("=" * 60)
    
    # 关闭Flask的重载器，使用单进程模式
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False, threaded=True)
