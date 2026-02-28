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
        # 原有初始化代码保持不变
        with open("key.txt", 'r', encoding='utf-8') as f:
            self.key = f.read().strip()


        #一个是聊天模型，一个数据处理模型
        self.model = "qwen3-max-2026-01-23"
        self.model2 = "liquid/lfm2-24b-a2b"
        self.base_url2 = "http://127.0.0.1:1234/v1"
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = AsyncOpenAI(api_key=self.key, base_url=self.base_url2)
        self.ai_system = '''你是一个有用的助手.'''

        self.prompt = [{
            "role": "system",
            "content": self.ai_system
        }]

        self.max_tokens = CalculateToken().getMaxTokens(self.model)
        if self.max_tokens == 0:
            while True:
                print("没有找到模型信息，请手动输入模型的最大上下文窗口大小。")
                self.max_tokens = int(input("请输入模型的最大上下文窗口大小（单位：token）："))
                if self.max_tokens > 0:
                    break

        # self.limit_tokens = self.max_tokens * 0.75
        #阈值通过用户修改为特定值，通多token的累加来判断，是否需要触发精炼
        self.limit_tokens = 1000
        self.st_limit_tokens = 0 #用于检测

        self.del_memory_tokens = self.max_tokens * 0.9
        self._refining = False  # 防止重复提炼的标志

        # 新增：文件操作锁
        self.file_lock = asyncio.Lock()

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
        # self.have_tokens = self.limit_tokens - self.history_tokens
        self.del_token = self.del_memory_tokens - self.history_tokens
        print(f"还剩 {self.del_token} 个 token 可以使用。")

    async def steamChat(self, message):
        self.prompt.append({"role": "user", "content": message})
        stream = await self.client.chat.completions.create(
            model=self.model2,
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
            print("\n[系统] 检测到记忆查询请求，正在处理...")
            get = AiQuery(self.key, self.model, self.base_url, full_response, self.prompt)
            self.prompt, final_reply = await get.returnAiResponse()
            if final_reply:
                print("\n" + final_reply)
            else:
                print("\n[警告] 最终回复为空")
            self.response_message = {"role": "assistant", "content": final_reply}
            completion_tokens = CalculateToken.num_tokens_from_messages([self.response_message], self.model)
            print(f"\n[最终回复 tokens]: {completion_tokens}")
        else:
            if usage:
                completion_tokens = usage.completion_tokens
                print(f"\n[Token usage] 回复 tokens: {completion_tokens}")
            else:
                completion_tokens = CalculateToken.num_tokens_from_messages([self.response_message], self.model)
                print(f"\n[手动估算] 回复 tokens: {completion_tokens}")

        print("-" * 50)

        self.all_tokens += completion_tokens
        self.st_limit_tokens += completion_tokens
        print(f"当前总 token 数: {self.all_tokens}")
        # self.have_tokens = self.limit_tokens - self.all_tokens
        self.del_token = self.del_memory_tokens - self.all_tokens
        print(f"还剩 {self.del_token} 个 token 可以使用。")
        print("-" * 50)

        # 触发记忆提炼（非阻塞）
        # if self.have_tokens <= 0 and not self._refining:
        if self.st_limit_tokens >= self.limit_tokens:
            self._refining = True
            asyncio.create_task(self._refine_memory_async())
            self.st_limit_tokens = 0

        if self.del_token <= 0:
            # 保留 system prompt + 最近 70% 的消息
            keep_ratio = 0.7
            total = len(self.prompt)
            keep_count = int((total - 1) * keep_ratio) + 1  # 包含 system
            self.prompt = [self.prompt[0]] + self.prompt[-keep_count+1:]

            # 重新计算 token 计数
            self.all_tokens = CalculateToken.num_tokens_from_messages(self.prompt, self.model)
            self.del_token = self.del_memory_tokens - self.all_tokens

        # 保存历史（受锁保护）
        await self._save_history()

    async def _save_history(self):
        async with self.file_lock:
            # 将同步文件操作放入线程池，避免阻塞事件循环
            await asyncio.to_thread(self._write_history_sync)

    def _write_history_sync(self):
        with open("ai_memory/chat_history.json", "w", encoding="utf-8") as f:
            json.dump(self.prompt, ensure_ascii=False, indent=4, fp=f)

    async def _refine_memory_async(self):
        """后台精炼任务，不阻塞主对话流程"""
        try:
            # 获取当前历史快照（不含 system）
            new_prompt = self.prompt[1:]
            if not new_prompt:
                return

            dialog_ids = [str(uuid.uuid4()) for _ in range(len(new_prompt))]
            cleaner = RefineMemory(new_prompt, user_id="localnexus", min_chars=3)

            # 调用大模型提炼（异步网络请求，不阻塞）
            memories = await cleaner.getFromOpenAI(
                self.key, self.model, self.base_url,
                dialog_ids=dialog_ids
            )

            # 如果提炼出记忆，执行可能的后续处理（如更新 system prompt）
            if memories:
                # sendPromptToAi 内部可能读写文件，需要用锁保护
                async with self.file_lock:
                    # 如果 sendPromptToAi 中包含同步文件操作，放入线程池
                    await asyncio.to_thread(cleaner.sendPromptToAi)

        except asyncio.CancelledError:
            # 任务被取消时的清理（可根据需要添加）
            print("精炼任务被取消")
            raise
        except Exception as e:
            print(f"❌ 后台精炼任务异常: {e}")
        finally:
            # 无论成功或失败，释放精炼标志，允许下次触发
            self._refining = False

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