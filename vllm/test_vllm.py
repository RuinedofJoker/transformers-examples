"""vLLM Qwen3-VL-8B 测试脚本"""

"""
hf download Qwen/Qwen3-VL-8B-Instruct

vllm serve \
--model /root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
--served-model-name qwen3-vl \
--host 0.0.0.0 \
--port 8000 \
--tensor-parallel-size 2 \
--max-model-len 16384 \
--dtype bfloat16
"""

from openai import OpenAI

# 连接 vLLM 服务
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="empty",  # vLLM 不需要真实 key
)

MODEL = "qwen3-vl"


def test_text():
    """纯文本对话测试"""
    print("=" * 50)
    print("测试1: 纯文本对话")
    print("=" * 50)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "你好，请用一句话介绍你自己"}],
        max_tokens=256,
    )
    print(resp.choices[0].message.content)
    print()


def test_image():
    """图片理解测试"""
    print("=" * 50)
    print("测试2: 图片理解")
    print("=" * 50)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://picx.zhimg.com/v2-d6f44389971daab7e688e5b37046e4e4_720w.jpg?source=172ae18b"},
                    },
                    {"type": "text", "text": "描述这张图片的内容"},
                ],
            }],
            max_tokens=512,
        )
        if resp.choices:
            print(resp.choices[0].message.content)
        else:
            print("模型返回为空，可能是图片无法访问。请检查服务器网络或换用本地图片。")
    except Exception as e:
        print(f"图片测试失败: {e}")
    print()


def test_stream():
    """流式输出测试"""
    print("=" * 50)
    print("测试3: 流式输出")
    print("=" * 50)
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "用3句话解释什么是大语言模型"}],
        max_tokens=256,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    test_text()
    test_image()
    test_stream()
