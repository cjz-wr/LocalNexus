import asyncio
from openai import AsyncOpenAI
import os
import json
from CalculateToken import CalculateToken
import re
from aiQuery import AiQuery
import uuid
from RefineMemory import RefineMemory

class DialogueManager:
    def __init__(self):
        self.key = "sk-32b922c6ed4c479f964e81b8339e56d2"
        self.model = "qwen3-max-2026-01-23"
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = AsyncOpenAI(api_key=self.key, base_url=self.base_url)
        self.ai_system = '''你是一个有用的助手.'''
        
        self.prompt = [{
            "role": "system",
            "content": self.ai_system
        }]

        # 对token的处理
        self.max_tokens = CalculateToken().getMaxTokens(self.model)
        if self.max_tokens == 0:
            while True:
                print("没有找到模型信息，请手动输入模型的最大上下文窗口大小。")
                self.max_tokens = int(input("请输入模型的最大上下文窗口大小（单位：token）："))
                if self.max_tokens > 0:
                    break

        self.limit_tokens = self.max_tokens * 0.75
        self.del_memory_tokens = self.max_tokens * 0.9

        if not os.path.isdir("ai_memory"):
            os.mkdir("ai_memory")
        if not os.path.isfile("ai_memory/chat_history.json"):
            with open("ai_memory/chat_history.json", "w", encoding="utf-8") as f:
                json.dump(self.prompt, ensure_ascii=False, indent=4, fp=f)
        else:
            with open("ai_memory/chat_history.json", "r", encoding="utf-8") as f:
                self.prompt = json.load(f)

        self.history_tokens = CalculateToken.num_tokens_from_messages(self.prompt, model=self.model)
        print(f"当前历史记忆的 token 数: {self.history_tokens}")
        self.all_tokens = self.history_tokens
        self.have_tokens = self.limit_tokens - self.history_tokens
        self.del_token = self.del_memory_tokens - self.history_tokens
        print(f"还剩 {self.del_token} 个 token 可以使用。")

    async def steamChat(self, message):
        self.prompt.append({"role": "user", "content": message})
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=self.prompt,
            stream=True,
            temperature=0.7
        )
        full_response = ""
        usage = None
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
            if hasattr(chunk, 'usage') and chunk.usage:
                usage = chunk.usage

        self.response_message = {"role": "assistant", "content": full_response}
        self.prompt.append(self.response_message)

        pattern = r'<query>(.*?)</query>'
        match = re.search(pattern, full_response, re.DOTALL)
        if match:
            get = AiQuery(self.key, self.model, self.base_url, full_response, self.prompt)
            self.prompt, respon = await get.returnAiResponse()

        print("-" * 50)

        if usage:
            completion_tokens = usage.completion_tokens
            print(f"\n[Token usage] 回复 tokens: {completion_tokens}")
        else:
            completion_tokens = CalculateToken.num_tokens_from_messages([self.response_message], self.model)
            print(f"\n[手动估算] 回复 tokens: {completion_tokens}")

        self.all_tokens += completion_tokens
        print(f"当前总 token 数: {self.all_tokens}")
        self.have_tokens = self.limit_tokens - self.all_tokens
        self.del_token = self.del_memory_tokens - self.all_tokens
        print(f"还剩 {self.del_token} 个 token 可以使用。")
        print("-" * 50)

        # 优化记忆的入口
        if self.have_tokens <= 0:
            await self._refine_memory_async()

        # 删除记忆的窗口
        if self.del_token <= 0:
            for i in range(1, int(len(self.prompt) * 0.3)):
                self.prompt.pop(i)

        # 保存记忆到文件（放入线程池）
        await asyncio.to_thread(self._save_history)

    def _save_history(self):
        with open("ai_memory/chat_history.json", "w", encoding="utf-8") as f:
            json.dump(self.prompt, ensure_ascii=False, indent=4, fp=f)

    async def _refine_memory_async(self):
        new_prompt = self.prompt[1:]  # 去掉系统提示
        dialog_ids = [str(uuid.uuid4()) for _ in range(len(new_prompt))]
        cleaner = RefineMemory(new_prompt, user_id="localnexus", min_chars=3)
        memories = await cleaner.getFromOpenAI(self.key, self.model, self.base_url, dialog_ids=dialog_ids)
        if memories:
            await asyncio.to_thread(cleaner.sendPromptToAi)

    async def run(self):
        while True:
            user_input = await asyncio.to_thread(input, "\n请输入消息（输入 'exit' 退出）：")
            if user_input.lower() == "exit":
                print("退出对话。")
                break
            await self.steamChat(user_input)

async def main():
    manager = DialogueManager()
    await manager.run()

if __name__ == "__main__":
    asyncio.run(main())