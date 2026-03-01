import asyncio
from openai import AsyncOpenAI
import os
import json
from CalculateToken import CalculateToken
import re
from aiQuery import AiQuery
import uuid
from RefineMemory import RefineMemory

# 工具提示词常量（使用双引号 JSON 示例）
TOOL_SYSTEM_PROMPT = '''

当用户的问题需要参考历史记忆才能准确回答时，你必须输出一个 JSON 格式的查询请求。JSON 必须包含以下字段：
- `action`: 固定为 "query_memory"
- `params`: 对象，包含查询条件，可选字段如下：
  - `root_category` (string): 根类别，可选值：Work, Tech, Learning, Health, Finance, Ideas, Life, General, UserInfo
  - `keywords` (list of strings): 关键词列表，用于匹配标题、摘要、关键事实
  - `entities` (list of strings): 实体列表，如人名、技术名等
  - `max_results` (integer): 最多返回几条记忆，默认 3



请将 JSON 包裹在标记 `<query></query>` 中，以便系统识别。例如：
<query>{"action": "query_memory", "params": {"keywords": ["上次", "问题"]}}</query>



注意：如果不需要查询记忆，直接正常回答即可，不要输出查询标记。
'''

class DialogueManager:
    def __init__(self):
        # 原有初始化代码保持不变
        with open("key.txt", 'r', encoding='utf-8') as f:
            self.key = f.read().strip()

        # 一个是聊天模型，一个数据处理模型
        # self.model = "qwen3-max-2026-01-23"
        self.model = "liquid/lfm2-24b-a2b"
        self.model2 = "liquid/lfm2-24b-a2b"
        self.base_url2 = "http://127.0.0.1:1234/v1"
        # self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.base_url = "http://127.0.0.1:1234/v1"
        self.client = AsyncOpenAI(api_key=self.key, base_url=self.base_url2)
        self.ai_system = '''你是一名网友，你正在和另外一名网友聊天,你可以使用后面的要求来回忆起你之前的记忆'''

        self.prompt = [{
            "role": "system",
            "content": self.ai_system
        }]

        self.max_tokens = CalculateToken().getMaxTokens(self.model2)
        if self.max_tokens == 0:
            while True:
                print("没有找到模型信息，请手动输入模型的最大上下文窗口大小。")
                self.max_tokens = int(input("请输入模型的最大上下文窗口大小（单位：token）："))
                if self.max_tokens > 0:
                    break

        # 精炼触发阈值（自上次精炼后累计的新 token 数）
        self.limit_tokens = 3000
        self.st_limit_tokens = 0  # 累计用户+助手新 token

        # 清理阈值：达到 max_tokens 的 90% 时触发清理（但清理目标是降到 80% 以下）
        self.del_memory_tokens = self.max_tokens * 0.9
        self._refining = False  # 防止重复提炼的标志

        # 文件操作锁
        self.file_lock = asyncio.Lock()

        if not os.path.isdir("ai_memory"):
            os.mkdir("ai_memory")
        if not os.path.isfile("ai_memory/chat_history.json"):
            with open("ai_memory/chat_history.json", "w", encoding="utf-8") as f:
                json.dump(self.prompt, ensure_ascii=False, indent=4, fp=f)
        else:
            with open("ai_memory/chat_history.json", "r", encoding="utf-8") as f:
                self.prompt = json.load(f)

        # 确保 system 提示包含工具说明
        if TOOL_SYSTEM_PROMPT not in self.prompt[0]["content"]:
            self.prompt[0]["content"] += "\n\n" + TOOL_SYSTEM_PROMPT
            # 同步保存（初始化时无并发风险）
            with open("ai_memory/chat_history.json", "w", encoding="utf-8") as f:
                json.dump(self.prompt, ensure_ascii=False, indent=4, fp=f)

        # 计算当前历史的总 token 数（包含 system 和已有对话）
        self.all_tokens = CalculateToken.num_tokens_from_messages(self.prompt, model=self.model)
        print(f"当前历史记忆的 token 数: {self.all_tokens}")
        self.del_token = self.del_memory_tokens - self.all_tokens
        print(f"还剩 {self.del_token} 个 token 可以使用。")

    async def steamChat(self, message):
        # ---------- 添加用户消息并累计 token ----------
        user_msg = {"role": "user", "content": message}
        self.prompt.append(user_msg)
        user_tokens = CalculateToken.num_tokens_from_messages([user_msg], self.model)
        self.all_tokens += user_tokens
        self.st_limit_tokens += user_tokens
        # ---------------------------------------------

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

        # 累计助手回复的 token（先记录原始回复的 token，后面可能被替换）
        if usage:
            completion_tokens = usage.completion_tokens
            print(f"\n[Token usage] 回复 tokens: {completion_tokens}")
        else:
            completion_tokens = CalculateToken.num_tokens_from_messages([self.response_message], self.model)
            print(f"\n[手动估算] 回复 tokens: {completion_tokens}")
        self.all_tokens += completion_tokens
        self.st_limit_tokens += completion_tokens

        # ---------- 检测工具调用 ----------
        full_response = full_response.replace("<|tool_call_start|>","<query>").replace("<|tool_call_end|>","</query>")
        pattern = r'<(?:query|\|tool_call_start\|)>(.*?)</(?:query|\|tool_call_end\|)>'
        match = re.search(pattern, full_response, re.DOTALL)
        if match:
            
            print("\n[系统] 检测到记忆查询请求，正在处理...")
            get = AiQuery(self.key, self.model, self.base_url, full_response, self.prompt)
            self.prompt, final_reply = await get.returnAiResponse()
            if final_reply:
                print("\n" + final_reply)
            else:
                print("\n[警告] 最终回复为空")

            # 替换最后一条助手消息为最终回复
            self.response_message = {"role": "assistant", "content": final_reply}
            # 因为之前已经 append 了原始回复，现在用最终回复替换
            self.prompt[-1] = self.response_message

            # 调整 token 计数：减去原回复的 token，加上最终回复的 token
            self.all_tokens -= completion_tokens
            final_tokens = CalculateToken.num_tokens_from_messages([self.response_message], self.model)
            self.all_tokens += final_tokens
            completion_tokens = final_tokens
            print(f"\n[最终回复 tokens]: {completion_tokens}")

        print("-" * 50)
        print(f"当前总 token 数: {self.all_tokens}")
        self.del_token = self.del_memory_tokens - self.all_tokens
        print(f"还剩 {self.del_token} 个 token 可以使用。")
        print("-" * 50)

        # ---------- 触发记忆提炼（非阻塞） ----------
        if self.st_limit_tokens >= self.limit_tokens and not self._refining:
            self._refining = True
            asyncio.create_task(self._refine_memory_async())
            self.st_limit_tokens = 0

        # ---------- 基于 token 数的清理 ----------
        # 当总 token 超过阈值的 90% 时主动清理（避免触及硬上限）
        if self.del_token <= 0:
            print("token 接近上限，正在清理历史记录...")
            await self._trim_history()
            # 清理后重新计算 del_token
            self.del_token = self.del_memory_tokens - self.all_tokens
            print(f"清理后还剩 {self.del_token} 个 token 可以使用。")

        # 保存历史（受锁保护）
        await self._save_history()

    async def _trim_history(self):
        """基于 token 数清理历史，删除最早的非系统消息直到总 token 数低于阈值的 80%"""
        target = self.del_memory_tokens * 0.8  # 保留更多余量，避免频繁清理
        while self.all_tokens > target and len(self.prompt) > 1:
            # 删除索引 1 的消息（最老的非 system 消息）
            removed = self.prompt.pop(1)
            removed_tokens = CalculateToken.num_tokens_from_messages([removed], self.model)
            self.all_tokens -= removed_tokens
            print(f"  删除一条消息，释放 {removed_tokens} tokens")
        print(f"清理后总 token: {self.all_tokens}")

    async def _save_history(self):
        async with self.file_lock:
            await asyncio.to_thread(self._write_history_sync)

    def _write_history_sync(self):
        with open("ai_memory/chat_history.json", "w", encoding="utf-8") as f:
            json.dump(self.prompt, ensure_ascii=False, indent=4, fp=f)

    async def _refine_memory_async(self):
        """后台精炼任务，不阻塞主对话流程"""
        try:
            # 复制当前对话历史（不含 system）
            new_prompt = self.prompt[1:]
            if not new_prompt:
                return

            dialog_ids = [str(uuid.uuid4()) for _ in range(len(new_prompt))]
            cleaner = RefineMemory(new_prompt, user_id="localnexus", min_chars=3)

            memories = await cleaner.getFromOpenAI(
                self.key, self.model, self.base_url,
                dialog_ids=dialog_ids
            )

            if memories:
                async with self.file_lock:
                    # sendPromptToAi 现在为空操作，但保留调用
                    await asyncio.to_thread(cleaner.sendPromptToAi)

        except asyncio.CancelledError:
            print("精炼任务被取消")
            raise
        except Exception as e:
            print(f"❌ 后台精炼任务异常: {e}")
        finally:
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