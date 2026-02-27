import json
import tiktoken

class CalculateToken:
    def __init__(self):
        try:
            with open("model_token/maxToken.json","r+",encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception as e:
            print(e)

    def getMaxTokens(self, model_name):
        for item in self.data:
            if item["model_name"] == model_name:
                return item["max_context_tokens"]
        return 0  # Return 0 if model not found
    
    def num_tokens_from_messages(messages, model="qwen3-max-2026-01-23"):
        """估算消息列表消耗的 token 数（兼容 OpenAI 格式）"""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # 如果模型名不在 tiktoken 已知列表中，回退到 cl100k_base（通义千问通常也兼容）
            encoding = tiktoken.get_encoding("cl100k_base")

        # 通义千问的计费规则可能与 OpenAI 略有不同，这里使用 OpenAI 的通用估算公式
        tokens_per_message = 3
        tokens_per_name = 1

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # 回复的格式占用
        return num_tokens
    
if __name__ == "__main__":
    calculator = CalculateToken()
    # print(calculator.getMaxTokens("qwen3-max-2026-01-23"))
    print(calculator.num_tokens_from_messages("a"))