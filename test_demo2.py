import asyncio
import os
import shutil
import json
import uuid
from ai_Dialogue import DialogueManager
from treeMemoryStore import TreeMemoryStore

# 清理之前的测试数据
TEST_USER = "advanced_demo"
TEST_STORAGE = f"./storage/{TEST_USER}"
if os.path.exists(TEST_STORAGE):
    shutil.rmtree(TEST_STORAGE)
    print(f"已清理测试存储目录: {TEST_STORAGE}")

if os.path.exists("ai_memory"):
    shutil.rmtree("ai_memory")
    print("已清理 ai_memory 目录")

async def advanced_demo():
    print("=" * 70)
    print("异步对话系统高级演示 - 多主题记忆与查询")
    print("=" * 70)

    # 初始化对话管理器
    manager = DialogueManager()

    # 手动在系统提示中加入记忆查询工具的说明（确保 AI 学会输出 <query>）
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

    # 模拟多主题对话序列
    conversations = [
        # 技术主题
        {
            "role": "user",
            "content": "我今天用 FastAPI 写 JWT 认证，总是报错 'Invalid signature'，查了半天发现是密钥复制时末尾多了个空格"
        },
        {
            "role": "assistant",
            "content": "啊，空格问题很隐蔽！建议以后从环境变量读取时用 .strip() 处理。"
        },
        {
            "role": "user",
            "content": "解决了，谢谢！另外我最近在学 Docker，容器里连不上宿主机 MySQL，是怎么回事？"
        },
        {
            "role": "assistant",
            "content": "可以试试 host 网络模式，或者用宿主机 IP 地址（Linux 下是 172.17.0.1）。"
        },

        # 生活主题
        {
            "role": "user",
            "content": "周末想去吃川菜，推荐一家吧"
        },
        {
            "role": "assistant",
            "content": "推荐尝试水煮鱼或麻辣火锅，如果喜欢辣，可以选一家地道的川菜馆。"
        },
        {
            "role": "user",
            "content": "好，我决定去吃火锅，记得多放辣椒。"
        },

        # 工作主题
        {
            "role": "user",
            "content": "下周要开项目评审会，需要准备 PPT"
        },
        {
            "role": "assistant",
            "content": "可以重点展示项目进展、遇到的问题和解决方案。"
        },

        # 健康主题
        {
            "role": "user",
            "content": "最近跑步膝盖有点疼，怎么办？"
        },
        {
            "role": "assistant",
            "content": "可能是跑步姿势或鞋子问题，建议先休息几天，做靠墙静蹲增强膝盖力量。"
        }
    ]

    print("\n开始模拟多主题对话（共 {} 轮）...\n".format(len(conversations)))
    for idx, msg in enumerate(conversations, 1):
        print(f"[轮次 {idx}] {msg['role']}: {msg['content']}")
        if msg['role'] == 'user':
            print("AI: ", end="", flush=True)
            await manager.steamChat(msg['content'])
            print("\n" + "-" * 50)

    # 手动降低 max_tokens 以触发记忆提炼（当前对话 token 可能未超限）
    print("\n⚠️ 强制降低 max_tokens 以触发记忆提炼...")
    manager.max_tokens = int(manager.all_tokens + 100)
    manager.limit_tokens = manager.max_tokens * 0.75
    manager.del_memory_tokens = manager.max_tokens * 0.9
    manager.have_tokens = manager.limit_tokens - manager.all_tokens
    manager.del_token = manager.del_memory_tokens - manager.all_tokens
    print(f"新的 max_tokens: {manager.max_tokens}, limit_tokens: {manager.limit_tokens}, have_tokens: {manager.have_tokens}")

    # 再添加一轮用户输入，触发提炼
    print("\n[触发提炼] 用户: （无新输入，系统自动提炼历史记忆）")
    await manager._refine_memory_async()  # 直接调用提炼

    # 现在记忆库中应有若干条记忆
    store = TreeMemoryStore(TEST_USER)
    all_memories = store.query_memories(level=2)
    print(f"\n📦 记忆库中共有 {len(all_memories)} 条提炼后的记忆：")
    for i, mem in enumerate(all_memories, 1):
        print(f"{i}. [{mem['root_category']}] {mem['title']}")
        print(f"   摘要: {mem['summary'][:100]}...")
        print(f"   关键事实: {', '.join(mem['key_facts'])}")
        print(f"   关联对话数: {len(mem.get('dialog_ids', []))}")

    # 测试记忆查询工具：提问需要参考之前记忆的问题
    test_queries = [
        "我之前提过跑步膝盖疼，有什么建议吗？",
        "上次说的 Docker 连不上 MySQL 怎么解决的？",
        "我周末想吃什么来着？",
        "下周的工作安排是什么？"
    ]

    print("\n" + "=" * 70)
    print("测试记忆查询工具：向 AI 提问，应触发 <query> 并基于记忆回答")
    print("=" * 70)
    for q in test_queries:
        print(f"\n用户: {q}")
        print("AI: ", end="", flush=True)
        await manager.steamChat(q)
        print("\n" + "-" * 50)

    # 展示树形结构（以 Tech 为例）
    print("\n🌳 Tech 类别树形结构：")
    tech_tree = store.get_branch("Tech")
    if tech_tree:
        print(f"根节点: {tech_tree['title']}")
        for branch in tech_tree.get('children', []):
            print(f"  ├─ 子主题: {branch['sub_topic']}")
            for leaf in branch.get('children', []):
                print(f"  │   ├─ {leaf['title']}")
    else:
        print("未找到 Tech 类别")

    # 可选：查看 SQLite 流水表记录（如果已写入）
    print("\n📋 最近 5 条对话流水记录（从 SQLite）：")
    try:
        dialogs = store.get_dialogs_by_conversation("conv_001")  # 需预先写入 conversation_id
        # 此处仅演示，实际未写入，可提示用户
        print("（未写入对话流水，跳过）")
    except:
        pass

    print("\n" + "=" * 70)
    print("高级演示完成。请观察 AI 回复中是否包含记忆查询结果。")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(advanced_demo())