import json
import re
from typing import List, Dict, Any, Optional
from treeMemoryStore import TreeMemoryStore
from openai import OpenAI

class AiQuery:
    def __init__(self, key: str, model: str, url: str, ai_respon: str, memory: List[Dict[str, Any]]):
        self.model = model
        self.memory = memory
        self.client = OpenAI(api_key=key, base_url=url)
        self.old_system = self.memory[0]["content"] if self.memory else ""
        self.ai_respon = ai_respon

        # 修改系统提示，告知 AI 接下来会插入工具结果
        self.memory[0]["content"] = (
            "[RESULT]后面是工具的回复,请根据目前记忆进行总结回复,如果输出的记忆不对就自行根据用户内容输出其它"
        )

    def handle_memory_query(self, ai_response: str, store: TreeMemoryStore) -> Optional[str]:
        """
        检查 AI 响应中是否包含记忆查询请求，若有则执行查询并返回格式化后的结果文本。
        否则返回 None。
        """
        # 匹配 <query>...</query> 内的内容
        pattern = r'<query>(.*?)</query>'
        match = re.search(pattern, ai_response, re.DOTALL)
        if not match:
            return None

        query_json = match.group(1).strip()
        try:
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

        # ========== 修正点：将 keywords 和 entities 合并为 text_query，并将 limit 改为 top_k ==========
        # 将关键词和实体组合成查询字符串（空格分隔）
        query_parts = keywords + entities
        text_query = " ".join(query_parts) if query_parts else None

        # 调用记忆库查询（适配现有接口：root_category, text_query, top_k）
        memories = store.query_memories(
            root_category=root_category,
            text_query=text_query,
            top_k=max_results
        )
        # ===================================================================================

        if not memories:
            return "【记忆查询结果】\n未找到相关记忆。"

        # 格式化输出
        lines = ["【记忆查询结果】"]
        for i, mem in enumerate(memories, 1):
            title = mem.get("title", "无标题")
            summary = mem.get("summary", "")
            facts = mem.get("key_facts", [])
            facts_str = "；".join(facts) if facts else ""
            lines.append(f"{i}. {title}：{summary} {facts_str}".strip())

        return "\n".join(lines)

    def returnAiResponse(self):
        """
        处理记忆查询并生成最终回复
        """
        # 移除 AI 上一次输出中的查询标记（避免重复处理）
        # self.memory 中最后一条是 assistant 的原始回复，其中可能包含 <query> 标记
        # 我们将这条消息临时移除，后续会用工具结果替换
        if self.memory and self.memory[-1]["role"] == "assistant":
            self.memory.pop()

        # 执行记忆查询
        search_result = self.handle_memory_query(
            ai_response=self.ai_respon,
            store=TreeMemoryStore("localnexus")  # 注意：user_id 应与对话管理器保持一致，这里使用默认值
        )

        # 将查询结果作为用户消息追加，让 AI 基于此生成最终回复
        if search_result:
            self.memory.append({"role": "user", "content": search_result})

        # ========== 修正点：正确调用 OpenAI 客户端，并使用非流式响应 ==========
        # 原代码误写为 self.client.chat.create，且 stream=True 后未处理流式
        # 这里改为非流式调用，直接获取完整内容
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.memory,
                stream=False,
                temperature=0.7
            )
            assistant_reply = response.choices[0].message.content
        except Exception as e:
            print(f"❌ 调用大模型失败: {e}")
            assistant_reply = "抱歉，我暂时无法回答。"
        # ===================================================================

        # 恢复原始系统提示
        self.memory[0]["content"] = self.old_system

        # 将最终回复追加到记忆
        self.memory.append({"role": "assistant", "content": assistant_reply})

        return self.memory, assistant_reply