import os
from random import uniform
from time import sleep
import requests
import re

def sc_send(sendkey, title, desp='', options=None):
    if options is None:
        options = {}
    # 判断 sendkey 是否以 'sctp' 开头，并提取数字构造 URL
    if sendkey.startswith('sctp'):
        match = re.match(r'sctp(\d+)t', sendkey)
        if match:
            num = match.group(1)
            url = f'https://{num}.push.ft07.com/send/{sendkey}.send'
        else:
            raise ValueError('Invalid sendkey format for sctp')
    else:
        url = f'https://sctapi.ftqq.com/{sendkey}.send'
    params = {
        'title': title,
        'desp': desp,
        **options
    }
    headers = {
        'Content-Type': 'application/json;charset=utf-8'
    }
    response = requests.post(url, json=params, headers=headers)
    result = response.json()
    return result


key = os.getenv('SENDKEY', 'SCT302449Tp6JIl7z3NvmEn2eBT3eNMaxu')

def send_with_retry(title, desp='', options=None, max_retries=10):
    if options is None:
        options = {}
    for attempt in range(max_retries):
        try:
            ret = sc_send(key, title, desp, options)
            return ret
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            sleep(uniform(0, 3))
    raise Exception("All attempts to send message failed.")