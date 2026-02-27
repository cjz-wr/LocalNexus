import asyncio
import os
import shutil
import json
from ai_Dialogue import DialogueManager
from treeMemoryStore import TreeMemoryStore

# 清理之前的测试数据
TEST_STORAGE = "./storage/localnexus"
if os.path.exists(TEST_STORAGE):
    shutil.rmtree(TEST_STORAGE)
    print(f"已清理测试存储目录: {TEST_STORAGE}")

# 清理 ai_memory 以便重新开始（保留系统提示）
if os.path.exists("ai_memory"):
    shutil.rmtree("ai_memory")
    print("已清理 ai_memory 目录")

async def run_test():
    print("=" * 60)
    print("异步对话系统全流程测试开始")
    print("=" * 60)

    # 初始化对话管理器
    manager = DialogueManager()

    # 手动在系统提示中加入记忆查询工具的说明，以便 AI 输出 <query> 标签
    query_tool_prompt = """
## 记忆查询工具
当用户的问题需要参考历史记忆才能准确回答时，你必须输出一个 JSON 格式的查询请求。JSON 必须包含以下字段：
- `action`: 固定为 "query_memory"
- `params`: 对象，包含查询条件，可选字段如下：
  - `root_category` (string): 根类别，可选值：Work, Tech, Learning, Health, Finance, Ideas, Life, General
  - `keywords` (list of strings): 关键词列表，用于匹配标题、摘要、关键事实
  - `entities` (list of strings): 实体列表，如人名、技术名等
  - `max_results` (integer): 最多返回几条记忆，默认 3

输出格式示例：
{
  "action": "query_memory",
  "params": {
    "root_category": "Tech",
    "keywords": ["FastAPI", "JWT"],
    "max_results": 2
  }
}

请将 JSON 包裹在标记 `<query></query>` 中，以便系统识别。例如：
<query>
{ "action": "query_memory", "params": { "keywords": ["上次", "问题"] } }
</query>

系统会在你输出查询请求后，暂停当前回答，执行查询，并将查询结果以如下格式追加到对话历史中：
【记忆查询结果】
- 记忆1：标题 / 摘要 / 关键事实
- 记忆2：...
然后你可以继续完成回答。

注意：如果不需要查询记忆，直接正常回答即可，不要输出查询标记。
"""
    manager.prompt[0]["content"] += query_tool_prompt
    # 保存到文件
    with open("ai_memory/chat_history.json", "w", encoding="utf-8") as f:
        json.dump(manager.prompt, ensure_ascii=False, indent=4, fp=f)

    # 模拟用户输入序列
    test_inputs = [
        "你好，我叫小明",
        "你能记住我的名字吗？",
        "我今天遇到了一个技术问题：用FastAPI做JWT认证时密钥错误",
        "最后发现是密钥末尾多了一个空格，解决了",
        "对了，我上次问你的JWT问题你还记得吗？"   # 可能触发记忆查询
    ]

    print("\n开始模拟对话...\n")
    for i, user_input in enumerate(test_inputs, 1):
        print(f"[第{i}轮] 用户: {user_input}")
        print("AI: ", end="", flush=True)
        await manager.steamChat(user_input)
        print("\n" + "-" * 40)

        # 在第三轮后，强制降低 max_tokens 以触发记忆提炼
        if i == 3:
            print("\n⚠️ 强制降低 max_tokens 以触发记忆提炼...")
            # 设置 max_tokens 为当前 all_tokens + 很小余量
            manager.max_tokens = int(manager.all_tokens + 100)
            manager.limit_tokens = manager.max_tokens * 0.75
            manager.del_memory_tokens = manager.max_tokens * 0.9
            manager.have_tokens = manager.limit_tokens - manager.all_tokens
            manager.del_token = manager.del_memory_tokens - manager.all_tokens
            print(f"新的 max_tokens: {manager.max_tokens}, limit_tokens: {manager.limit_tokens}, have_tokens: {manager.have_tokens}")

    # 检查记忆库中是否已存入提炼的记忆
    store = TreeMemoryStore("localnexus")
    memories = store.query_memories(level=2)
    print(f"\n📦 记忆库中共有 {len(memories)} 条提炼后的记忆：")
    for idx, mem in enumerate(memories, 1):
        print(f"{idx}. [{mem['root_category']}] {mem['title']}")
        print(f"   摘要: {mem['summary'][:80]}...")
        print(f"   关键事实: {', '.join(mem['key_facts'])}")
        print(f"   关联对话数: {len(mem.get('dialog_ids', []))}")

    # 测试记忆查询功能
    print("\n🔍 测试记忆查询：搜索与 'JWT' 相关的记忆")
    results = store.query_memories(text_query="JWT", top_k=3)
    for r in results:
        print(f"  - {r['title']}")

    print("\n" + "=" * 60)
    print("测试完成。")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_test())