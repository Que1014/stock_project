import requests

def send_ntfy(title, message, topic="deepseek-structure-20260929"):
    url = f"https://ntfy.sh/{topic}"

    headers = {
        "Title": title,
        "Priority": "high",
        "Markdown": "yes",
    }

    response = requests.post(
        url,
        data=message.encode("utf-8"),
        headers=headers,
        timeout=10
    )

    response.raise_for_status()
    return response.text

if __name__ == "__main__":
    # 测试发送消息
    title = "Test Message"
    message = "这是一条测试消息，来自 DeepSeek 分析报告。"
    topic = "deepseek-structure-20260929"

    try:
        response = send_ntfy(title, message, topic)
        print("消息发送成功:", response)
    except Exception as e:
        print("消息发送失败:", str(e))