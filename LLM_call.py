import json
import time
import requests
from openai import OpenAI
from zhipuai import ZhipuAI


class LLMInterface:
    def __init__(self, model_provider="zhipu", api_key=None, model_name="glm-4", temperature=0.7):
        self.model_provider = model_provider
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature  # Extracted as a global parameter to increase experiment reproducibility

        # Configure platform-specific clients (compatible with the latest SDK)
        if model_provider == "zhipu":
            self.client = ZhipuAI(api_key=api_key)
        elif model_provider == "openai":
            # Uses the official OpenAI API endpoint by default (https://api.openai.com/v1)
            self.client = OpenAI(api_key=api_key)
        # DeepSeek and Qwen use native requests, no client initialization required

    def call_llm(self, prompt, max_retries=4):
        """
        Unified call entry, including prompt safety truncation and exponential backoff retry mechanism
        """
        # [Core Optimization 1] Prompt length safety valve (prevents token overflow infinite loop)
        # Assuming 1 token ≈ 2 Chinese characters / 4 English characters. Limit max characters to 12000 (~6000 tokens)
        MAX_CHAR_LIMIT = 12000
        if len(prompt) > MAX_CHAR_LIMIT:
            print(f"[Warning] Prompt length ({len(prompt)}) exceeds safety threshold, performing safety truncation!")
            # Keep header (persona) and tail (JSON format requirement), truncate middle history messages
            head_len = 4000
            tail_len = 4000
            prompt = prompt[:head_len] + "\n\n...[System intervention: Due to exceeding cognitive load, middle memories are forgotten]...\n\n" + prompt[-tail_len:]

        for attempt in range(max_retries):
            try:
                if self.model_provider == "zhipu":
                    return self._call_zhipu(prompt)
                elif self.model_provider == "openai":
                    return self._call_openai(prompt)
                elif self.model_provider == "deepseek":
                    return self._call_deepseek(prompt)
                elif self.model_provider == "qwen":
                    return self._call_qwen(prompt)
                else:
                    return self._call_custom_api(prompt)
            except Exception as e:
                # [Core Optimization 2] Exponential backoff algorithm to handle high concurrency rate limiting (HTTP 429)
                wait_time = 2 ** attempt  # Wait time: 1s, 2s, 4s, 8s...
                error_msg = f"API call failed (Attempt {attempt + 1}/{max_retries}): {str(e)}. Waiting {wait_time} seconds before retrying..."
                print(error_msg)

                # If it's a clear token overflow error, break retry immediately to avoid wasting time
                if "context_length_exceeded" in str(e).lower() or "maximum context length" in str(e).lower():
                    print("[Fatal Error] Model context window overflow, aborting retry.")
                    break

                time.sleep(wait_time)

        # Ultimate fallback to ensure data pipeline doesn't crash
        return {
            "thought_process": "Model call failed multiple times, timed out, or context overflowed, unable to process information",
            "is_believed": False,
            "will_spread": False,
            "new_post": ""
        }

    # ==========================================
    # API Call Methods Area
    # ==========================================

    def _call_qwen(self, prompt):
        """Call Qwen API interface (Upgraded to OpenAI compatible mode)"""
        api_endpoint = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": 0.7,
            "max_tokens": 1024,
            "extra_body": {"enable_thinking": False}  # Disable thinking mode
        }

        response = requests.post(api_endpoint, headers=headers, json=payload, timeout=90)

        if response.status_code != 200:
            raise Exception(f"DashScope API Error (HTTP {response.status_code}): {response.text}")

        response_data = response.json()

        try:
            content = response_data["choices"][0]["message"]["content"]
        except KeyError:
            raise Exception(f"Unable to parse Qwen response format: {response_data}")

        return self._parse_response(content)

    def _call_deepseek(self, prompt):
        """Call DeepSeek API interface"""
        api_endpoint = "https://api.deepseek.com/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": 1024,
            "stream": False
        }

        response = requests.post(api_endpoint, headers=headers, json=payload, timeout=40)
        response.raise_for_status()
        response_data = response.json()
        content = response_data["choices"][0]["message"]["content"]
        return self._parse_response(content)

    def _call_zhipu(self, prompt):
        """
        Call Zhipu AI API (New - OpenAI compatible mode)
        Document Reference: https://docs.bigmodel.cn/
        """
        client = OpenAI(
            api_key=self.api_key,  # Zhipu AI official API Key
            base_url="https://open.bigmodel.cn/api/paas/v4"  # Zhipu AI v4 official endpoint
        )

        response = client.chat.completions.create(
            model=self.model_name,  # Using initialized model name
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            top_p=0.7,
            max_tokens=1024
        )

        # Extract response content
        content = response.choices[0].message.content

        # [Defense Logic]: Check if intercepted or returned empty content
        if not content or not content.strip():
            raise ValueError("Model returned empty content, suspected to have triggered underlying risk control interception.")

        return self._parse_response(content)

    def _call_openai(self, prompt):
        """Call official OpenAI v1.0+ new interface"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=1024
        )
        content = response.choices[0].message.content
        return self._parse_response(content)

    def _call_custom_api(self, prompt):
        """Call custom API interface"""
        api_endpoint = "https://api.example.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": 1024
        }

        response = requests.post(api_endpoint, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return self._parse_response(content)

    # ==========================================
    # Response Parsing and Data Cleaning Area
    # ==========================================

    def _parse_response(self, raw_response):
        # [New Tracking Log]: Forcibly print the raw plain text output of the LLM
        print(f"\n{'=' * 20} [LLM RAW OUTPUT] {'=' * 20}")
        print(raw_response)
        print(f"{'=' * 68}\n")

        if "```json" in raw_response:
            raw_response = raw_response.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_response:
            raw_response = raw_response.split("```")[1].split("```")[0].strip()

        try:
            response_data = json.loads(raw_response)
            return self._validate_response(response_data)
        except json.JSONDecodeError as e:
            # [New Tracking Log]: Prompt standard JSON parsing failure
            print(f"[Warning] Standard JSON parsing failed ({str(e)}), attempting heuristic regex extraction...")
            return self._heuristic_parse(raw_response)

    def _validate_response(self, response_data):
        if "thought_process" not in response_data:
            response_data["thought_process"] = response_data.get("thinking", "No thinking process")

        if "is_believed" not in response_data:
            response_data["is_believed"] = response_data.get("believe", False)

        if "will_spread" not in response_data:
            response_data["will_spread"] = response_data.get("decision", False)

        if "new_post" not in response_data:
            response_data["new_post"] = ""

        if str(response_data.get("new_post", "")).strip() in ["无", "目前没有新鲜事发生", "None", "null", ""]:
            response_data["new_post"] = ""

        return response_data

    def _heuristic_parse(self, text):
        result = {
            "thought_process": "Thought parsing failed, model output format is severely disordered",
            "is_believed": False,
            "will_spread": False,
            "new_post": ""
        }

        try:
            if "{" in text and "}" in text:
                start_idx = text.index("{")
                end_idx = text.rindex("}") + 1
                json_str = text[start_idx:end_idx]
                return self._validate_response(json.loads(json_str))
        except Exception as e:
            print(f"[Warning] Heuristic extraction of braces failed: {str(e)}")

        text_lower = text.lower()
        if '"is_believed": true' in text_lower or '"is_believed":true' in text_lower or '"is_believed":  true' in text_lower:
            result["is_believed"] = True

        if '"will_spread": true' in text_lower or '"will_spread":true' in text_lower or '"will_spread":  true' in text_lower:
            result["will_spread"] = True

        if result["will_spread"]:
            post_markers = ['"new_post":', '"new_post" :']
            for marker in post_markers:
                if marker in text_lower:
                    try:
                        content = text.split(marker, 1)[1].strip()
                        if content.startswith('"'):
                            content = content[1:].split('"\n')[0].split('"}')[0]
                        result["new_post"] = content.strip()
                        break
                    except:
                        pass

        return self._validate_response(result)