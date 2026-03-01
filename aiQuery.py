import json
import re
import asyncio
from typing import List, Dict, Any, Optional
from treeMemoryStore import TreeMemoryStore
from openai import AsyncOpenAI

class AiQuery:
    def __init__(self, key: str, model: str, url: str, ai_respon: str, memory: List[Dict[str, Any]]):
        self.model = model
        self.memory = memory
        self.client = AsyncOpenAI(api_key=key, base_url=url)
        self.old_system = self.memory[0]["content"] if self.memory else ""
        self.ai_respon = ai_respon

        self.memory[0]["content"] = (
            "[RESULT]后面是工具的回复,请根据目前记忆进行总结回复,如果输出的记忆不对就自行根据用户内容输出其它"
        )

    async def handle_memory_query(self, ai_response: str, store: TreeMemoryStore) -> Optional[str]:
        ai_response = ai_response.replace("<|tool_call_start|>","<query>").replace("<|tool_call_end|>","</query>")
        pattern = r'<query>(.*?)</query>'
        # pattern = r'<(?:query|\|tool_call_start\|)>(.*?)</(?:query|\|tool_call_end\|)>' #预防一些模型自动输出特定格式
        match = re.search(pattern, ai_response, re.DOTALL)
        # print(match.group(1))
        
        if not match:
            return None

        query_json = match.group(1).strip()
        try:
            if type(json.loads(query_json)) == list:
                query_data = json.loads(query_json)[0]
            else:
                query_data = json.loads(query_json)
        except json.JSONDecodeError:
            print("⚠️ 记忆查询 JSON 解析失败")
            return None

        if query_data.get("action") != "query_memory":
            return None

        params = query_data.get("params", {})
        root_category = params.get("root_category")
        keywords = params.get("keywords", [])
        entities = params.get("entities", [])
        max_results = params.get("max_results", 3)

        query_parts = keywords + entities
        text_query = " ".join(query_parts) if query_parts else None

        memories = await asyncio.to_thread(
            store.query_memories,
            root_category=root_category,
            text_query=text_query,
            top_k=max_results
        )

        if not memories:
            return "【记忆查询结果】\n未找到相关记忆。"

        lines = ["【记忆查询结果】"]
        for i, mem in enumerate(memories, 1):
            title = mem.get("title", "无标题")
            summary = mem.get("summary", "")
            facts = mem.get("key_facts", [])
            facts_str = "；".join(facts) if facts else ""
            lines.append(f"{i}. {title}：{summary} {facts_str}".strip())

        return "\n".join(lines)

    async def returnAiResponse(self):
        if self.memory and self.memory[-1]["role"] == "assistant":
            self.memory.pop()

        search_result = await self.handle_memory_query(
            ai_response=self.ai_respon,
            store=TreeMemoryStore("localnexus")
        )

        if search_result:
            self.memory.append({"role": "user", "content": search_result})

        # 第二次调用大模型生成最终回复
        print("[DEBUG] 第二次调用大模型生成最终回复...")
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=self.memory,
            stream=False,
            temperature=0.7
        )
        assistant_reply = response.choices[0].message.content
        print(f"[DEBUG] 第二次调用回复内容: {assistant_reply}")

        #删除工具返回的结果
        self.memory.pop(len(self.memory) - 1)

        self.memory[0]["content"] = self.old_system
        self.memory.append({"role": "assistant", "content": assistant_reply})

        return self.memory, assistant_reply