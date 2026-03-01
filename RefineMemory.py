import json
import asyncio
import openai
import re
import emoji
from typing import Optional, List, Dict, Any
from openai import AsyncOpenAI
from treeMemoryStore import TreeMemoryStore
import os

class RefineMemory:
    def __init__(self, ai_memory, user_id="localnexus", min_chars: int = 3, preserve_code_blocks: bool = True):
        self.ai_memory = [msg for msg in ai_memory if msg["role"] != "system"]
        self.store = TreeMemoryStore(user_id)
        self.min_chars = min_chars
        self.preserve_code_blocks = preserve_code_blocks
        self.url_pattern = re.compile(r'https?://[^\s]+|www\.[^\s]+')
        self.frist_creat = False
        if os.path.isdir("storage"):
            self.frist_creat = True

        self.ai_prompt = """
            # Role
            You are an expert Memory Architect for a personal AI assistant named "LocalNexus". 
            Your task is to analyze conversation logs, extract key information, and structure them into a hierarchical memory tree.

            # Input Language
            The input conversation is in **Chinese**. You must understand the nuances of Chinese context.

            # Output Language
            - All JSON **Keys** must be in **English** (e.g., "root_category", "summary").
            - All JSON **Values** (content) must be in **Chinese** (e.g., "技术开发", "用户解决了 JWT 问题").

            # Classification Rules
            Classify the conversation into ONE of the following root categories:
            1. **Work** (工作职业): Meetings, projects, colleagues, non-tech tasks.
            2. **Tech** (技术开发): Coding, debugging, architecture, tools, software issues. **(Priority: If it involves code, choose Tech)**.
            3. **Learning** (学习教育): Courses, books, languages, non-tech skills.
            4. **Health** (健康情感): Exercise, medical, emotions, diet, sleep.
            5. **Finance** (财务资产): Investments, bills, taxes, assets.
            6. **Ideas** (创意灵感): Brainstorming, todos, creative writing, startup ideas.
            7. **Life** (生活日常): Travel, shopping, family, hobbies, pets, entertainment.
            8. **General** (其他通用): Chit-chat, simple Q&A without personal context, greetings.
            9. **UserInfo** (用户信息): Personal information actively provided by the user, such as name, age, occupation, contact details. Summaries should briefly describe the information, and key_facts should list the specific facts.

            # Processing Steps
            1. **Analyze**: Read the Chinese conversation.
            2. **Filter**: Ignore greetings, polite fillers, and repeated confirmations.
            3. **Classify**: Select the best root category from the list above.
            4. **Summarize**: Write a concise summary in Chinese (max 100 words).
            5. **Extract Facts**: List 3-5 atomic facts in Chinese.
            6. **Extract Entities**: List key entities (people, technologies, concepts) in Chinese.
            7. **Format**: Output strictly valid JSON as a list of objects. Even if only one memory is extracted, it must be inside a list.

            # Output Format Example (exactly as shown)
            [
            {
                "root_category": "Tech",
                "sub_topic": "FastAPI_JWT_认证",
                "title": "JWT 密钥配置错误修复",
                "summary": "用户在使用 FastAPI 实现 JWT 认证时遇到验证失败，经排查是密钥配置错误，已解决。",
                "key_facts": ["使用 python-jose 库", "算法为 HS256", "问题是密钥错误", "已解决"],
                "entities": ["FastAPI", "JWT", "HS256", "python-jose"]
            }
            ]

            # Constraints
            - Do NOT output any markdown formatting (like ```json).
            - Do NOT output any explanation text outside the JSON.
            - If the conversation is pure chit-chat, set category to "General" and keep summary very short.
            - The JSON must be parseable by Python's json.loads().
            """

    def localClean(self, text: str) -> Optional[str]:
        if not text or not isinstance(text, str):
            return None

        code_blocks = []
        if self.preserve_code_blocks:
            pattern = r'(```(?:\w*)?\n.*?\n```|`[^`]+`)'
            matches = list(re.finditer(pattern, text, re.DOTALL))
            for i, match in enumerate(matches):
                placeholder = f"__CODE_BLOCK_{i}__"
                code_blocks.append(match.group(0))
                text = text[:match.start()] + placeholder + text[match.end():]

        text = re.sub(r'[ \t]+', ' ', text)
        text = text.strip()

        effective_text = re.sub(r'__CODE_BLOCK_\d+__', '', text)
        if len(effective_text.replace(' ', '')) < self.min_chars:
            return None

        text = self.remove_trailing_emoji(text)
        try:
            text = emoji.demojize(text, delimiters=(":", ":"))
        except Exception:
            pass

        text = self.url_pattern.sub('[URL]', text)

        if self.preserve_code_blocks:
            for i, block in enumerate(code_blocks):
                placeholder = f"__CODE_BLOCK_{i}__"
                text = text.replace(placeholder, block, 1)

        return text.strip()

    @staticmethod
    def remove_trailing_emoji(text: str) -> str:
        if not text:
            return text
        temp_text = text.rstrip()
        if not temp_text:
            return ""

        end_idx = len(temp_text)
        for i in range(len(temp_text) - 1, -1, -1):
            if emoji.is_emoji(temp_text[i]):
                end_idx = i
            else:
                break

        if end_idx < len(temp_text):
            return temp_text[:end_idx].rstrip()
        return temp_text

    def startLocalClean(self) -> List[Dict[str, Any]]:
        cleaned_messages = []
        for message in self.ai_memory:
            cleaned_content = self.localClean(message["content"])
            if cleaned_content:
                cleaned_messages.append({"role": message["role"], "content": cleaned_content})
        return cleaned_messages

    def processLocalCleanData(self) -> str:
        cleaned_messages = self.startLocalClean()
        data = ''
        for msg in cleaned_messages:
            data += f"{msg['role']}: {msg['content']}\n"
        return data

    async def getFromOpenAI(self, key, model, url, data=None, dialog_ids: Optional[List[str]] = None):
        if data is None:
            data = self.processLocalCleanData()

        if not data.strip():
            print("⚠️ 清洗后无有效对话内容，跳过提炼。")
            return None

        ai_msg = [
            {"role": "system", "content": self.ai_prompt},
            {"role": "user", "content": data},
        ]

        ai_client = AsyncOpenAI(api_key=key, base_url=url)
        try:
            ai_response = await ai_client.chat.completions.create(
                model=model,
                messages=ai_msg,
                stream=False,
                temperature=0.1
            )
            response = ai_response.choices[0].message.content
        except Exception as e:
            print(f"❌ LLM 调用失败: {e}")
            return None

        try:
            memories = json.loads(response)
            if isinstance(memories, dict):
                memories = [memories]
            elif not isinstance(memories, list):
                print("❌ AI 返回的不是列表或字典，无法解析")
                return None
        except json.JSONDecodeError as e:
            print(f"❌ JSON 解析失败: {e}")
            print(f"AI 返回原始内容: {response}")
            return None

        if self.store and memories:
            dialog_ids_list = [dialog_ids for _ in memories] if dialog_ids else None
            await asyncio.to_thread(self.store.add_memories, memories, dialog_ids_list)
            print(f"✅ 成功存入 {len(memories)} 条记忆到存储库。")
        else:
            print("⚠️ 未提供记忆存储实例，仅返回解析结果。")

        return memories

    # ---------- 修复：移除修改 system 的逻辑，变为空操作 ----------
    def sendPromptToAi(self):
        """原用于向 system 添加工具提示词，现已由 DialogueManager 统一处理，此方法保留为空。"""
        pass

# 测试函数（如需异步调用，可改为 async def）
async def main_local():
    raw_messages = [
        {"role": "user", "content": "你好，我想问一下如何用FastAPI实现JWT认证？"},
        {"role": "assistant", "content": "可以的，你需要安装`python-jose`和`passlib`。\n\n文档在这里：https://fastapi.tiangolo.com/tutorial/security/"},
        {"role": "user", "content": "嗯"},
        {"role": "assistant", "content": "有什么具体问题吗？"},
        {"role": "user", "content": "我按照文档做了，但是验证总是失败😭，能帮我看看代码吗？"},
        {"role": "assistant", "content": "当然，请贴出你的代码片段。"},
        {"role": "user", "content": "代码有点长，我简化一下：\n\n```python\ndef verify_token(token):\n    # 这里出错\n    pass\n```"},
        {"role": "assistant", "content": "可能是算法问题，你用的什么算法？"},
        {"role": "user", "content": "HS256，我看文档默认就是这个。👍"},
        {"role": "assistant", "content": "那你检查一下密钥是否正确。还有，确保token没有过期。"},
        {"role": "user", "content": "解决了😊 原来是密钥写错了，谢谢！"},
        {"role": "assistant", "content": "不客气😊"},
        {"role": "user", "content": "👍👍👍"},
        {"role": "user", "content": "访问 www.google.com 看看"},
    ]

    from treeMemoryStore import TreeMemoryStore
    store = TreeMemoryStore("demo_user")
    print("🚀 存储初始化完成。")
    import uuid
    dialog_ids = [str(uuid.uuid4()) for _ in range(len(raw_messages))]

    cleaner = RefineMemory(raw_messages, min_chars=3)
    memories = await cleaner.getFromOpenAI(
        key="sk-32b922c6ed4c479f964e81b8339e56d2",
        model="qwen3-max-2026-01-23",
        url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        dialog_ids=dialog_ids
    )
    print(f"ai输出: {memories}")

    if memories:
        print("\n📦 提炼出的记忆：")
        print(json.dumps(memories, ensure_ascii=False, indent=2))

    print("\n🔍 从记忆库中查询 Tech 类别的记忆：")
    tech_mems = await asyncio.to_thread(store.query_memories, root_category="Tech", level=2)
    for mem in tech_mems:
        print(f"  - {mem['title']} (关联对话数: {len(mem.get('dialog_ids', []))})")

if __name__ == "__main__":
    asyncio.run(main_local())