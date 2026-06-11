import os
import time
import requests
from functools import lru_cache
from typing import Optional, Iterator


class BaseLLMAgent:
    def __init__(self, model_name: str, api_url: str, **kwargs):
        self.model_name = model_name
        self.api_url = api_url
        self.timeout = kwargs.get("timeout", 30)
        self.max_retries = kwargs.get("max_retries", 3)
        self.backoff_factor = kwargs.get("backoff_factor", 0.5)

    def _request(self, payload: dict) -> dict:
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(self.api_url, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()
                elif 500 <= resp.status_code < 600:
                    sleep = self.backoff_factor * (2 ** (attempt - 1))
                    time.sleep(sleep)
                    continue
                else:
                    resp.raise_for_status()
            except Exception as e:
                if attempt == self.max_retries:
                    raise
                time.sleep(self.backoff_factor * (2 ** (attempt - 1)))
        raise RuntimeError("Unreachable retry logic")

    def get_response(self, prompt: str) -> Optional[str]:
        raise NotImplementedError

    def stream_response(self, prompt: str) -> Iterator[str]:
        raise NotImplementedError


class LocalChatAgent(BaseLLMAgent):
    def __init__(self, api_url=None, model=None):
        api_url = api_url or os.getenv("LOCAL_LLM_API", "http://127.0.0.1:11434/api/generate")
        model = model or os.getenv("LOCAL_LLM_MODEL", "llama3")
        super().__init__(model_name=model, api_url=api_url)

    @lru_cache(maxsize=128)
    def get_response(self, prompt: str) -> Optional[str]:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        result = self._request(payload)
        return result.get("response")

    def stream_response(self, prompt: str) -> Iterator[str]:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True
        }
        # 假设后端以 Server-Sent Events (SSE) 格式分片推送
        with requests.post(self.api_url, json=payload, stream=True) as resp:
            for chunk in resp.iter_lines():
                if chunk:
                    data = chunk.decode()
                    yield data  # 或者 json.loads(data).get("token")


# 用法示例
if __name__ == "__main__":
    agent = LocalChatAgent()
    reply = agent.get_response("Hello, UGV 置信度感知控制系统介绍一下？")
    print("Reply:", reply)
