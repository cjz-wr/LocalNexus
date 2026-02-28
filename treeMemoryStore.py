# ========== 设置 HuggingFace 镜像源 ==========
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = './hf_cache'
# ============================================

import lancedb
import uuid
import time
import sqlite3
import pyarrow as pa
import numpy as np
from typing import List, Dict, Optional, Any
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('BAAI/bge-m3')

class TreeMemoryStore:
    def __init__(self, user_id: str, db_path: str = "./storage"):
        self.user_id = user_id
        # LanceDB 存储路径（记忆树）
        self.db_uri = f"{db_path}/{user_id}"
        self.db = lancedb.connect(self.db_uri)
        self.table_name = "memory_tree"
        self._init_table()

        # SQLite 对话流水存储路径
        self.conv_db_path = f"{db_path}/{user_id}/conversations.db"
        self._init_conversation_db()

    # ---------- 记忆树相关（LanceDB）----------
    def _init_table(self):
        # 定义 schema，增加 dialog_ids 字段
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("parent_id", pa.string()),
            pa.field("level", pa.int32()),
            pa.field("root_category", pa.string()),
            pa.field("sub_topic", pa.string()),
            pa.field("title", pa.string()),
            pa.field("summary", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 1024)),
            pa.field("key_facts", pa.list_(pa.string())),
            pa.field("entities", pa.list_(pa.string())),
            pa.field("dialog_ids", pa.list_(pa.string())),  # 新增字段
            pa.field("created_at", pa.timestamp('s'))
        ])
        self.table = self.db.create_table(self.table_name, schema=schema, exist_ok=True)
        self._init_root_categories()

    def _init_root_categories(self):
        # 新增 UserInfo 类别
        roots = ["Work", "Tech", "Learning", "Health", "Finance", "Ideas", "Life", "General", "UserInfo"]
        for cat in roots:
            exists = self.table.search().where(f"root_category = '{cat}' AND level = 0").limit(1).to_list()
            if not exists:
                self.table.add([{
                    "id": str(uuid.uuid4()),
                    "parent_id": None,
                    "level": 0,
                    "root_category": cat,
                    "sub_topic": cat,
                    "title": cat,
                    "summary": f"Root category for {cat}",
                    "vector": [0.0] * 1024,
                    "key_facts": [],
                    "entities": [],
                    "dialog_ids": [],  # 根节点无对话关联
                    "created_at": int(time.time())
                }])

    def _get_or_create_node(self, category: str, sub_topic: str, parent_id: Optional[str], level: int) -> str:
        if parent_id:
            where_clause = f"parent_id = '{parent_id}' AND sub_topic = '{sub_topic}'"
        else:
            where_clause = f"root_category = '{category}' AND level = 0"
        result = self.table.search().where(where_clause).limit(1).to_list()
        if result:
            return result[0]['id']
        node_id = str(uuid.uuid4())
        summary_text = f"Topic: {sub_topic}"
        vector = embedding_model.encode(summary_text).tolist() if level > 0 else [0.0]*1024
        self.table.add([{
            "id": node_id,
            "parent_id": parent_id,
            "level": level,
            "root_category": category,
            "sub_topic": sub_topic,
            "title": sub_topic,
            "summary": summary_text,
            "vector": vector,
            "key_facts": [],
            "entities": [],
            "dialog_ids": [],  # 非叶子节点无对话关联
            "created_at": int(time.time())
        }])
        return node_id

    def add_memories(self, data_list: List[Dict[str, Any]], dialog_ids_list: Optional[List[List[str]]] = None):
        """
        添加记忆节点（level=2），并自动去重（基于向量相似度）
        :param data_list: 记忆数据列表
        :param dialog_ids_list: 与 data_list 对应的对话ID列表，长度必须一致
        """
        print(f"📥 开始存入 {len(data_list)} 条记忆...")
        if dialog_ids_list is None:
            dialog_ids_list = [[] for _ in data_list]
        assert len(data_list) == len(dialog_ids_list), "data_list 与 dialog_ids_list 长度必须一致"

        # 相似度阈值
        SIMILARITY_THRESHOLD = 0.95

        for item, new_dialog_ids in zip(data_list, dialog_ids_list):
            root_cat = item.get('root_category', 'General')
            sub_topic = item.get('sub_topic', 'General')
            title = item.get('title', 'Untitled')
            summary = item.get('summary', '')
            facts = item.get('key_facts', [])
            entities = item.get('entities', [])

            # 计算新记忆的向量（基于摘要）
            new_vector = embedding_model.encode(summary).tolist()

            # 在相同 root_category 下搜索最相似的记忆（仅 level=2 的叶子节点）
            search_result = self.table.search(new_vector) \
                .where(f"root_category = '{root_cat}' AND level = 2") \
                .limit(1) \
                .to_list()

            if search_result:
                existing = search_result[0]
                # 移除 LanceDB 自动添加的元数据字段（如 _distance）
                existing = {k: v for k, v in existing.items() if not k.startswith('_')}
                similarity = self._cosine_similarity(new_vector, existing['vector'])
                if similarity >= SIMILARITY_THRESHOLD:
                    # 重复记忆：合并 dialog_ids
                    merged_ids = list(set(existing['dialog_ids'] + new_dialog_ids))
                    print(f"  🔁 发现重复记忆 [{root_cat}] {title}，合并 dialog_ids（共 {len(merged_ids)} 条）")
                    # 更新现有记录：删除旧记录，插入新记录
                    self.table.delete(f"id = '{existing['id']}'")
                    updated_record = existing.copy()
                    updated_record['dialog_ids'] = merged_ids
                    self.table.add([updated_record])
                    continue

            # 无重复，正常插入新记忆
            try:
                root_node = self.table.search().where(f"root_category = '{root_cat}' AND level = 0").limit(1).to_list()
                root_id = root_node[0]['id'] if root_node else None
                branch_id = self._get_or_create_node(root_cat, sub_topic, root_id, level=1)
                leaf_id = str(uuid.uuid4())
                self.table.add([{
                    "id": leaf_id,
                    "parent_id": branch_id,
                    "level": 2,
                    "root_category": root_cat,
                    "sub_topic": sub_topic,
                    "title": title,
                    "summary": summary,
                    "vector": new_vector,
                    "key_facts": facts,
                    "entities": entities,
                    "dialog_ids": new_dialog_ids,
                    "created_at": int(time.time())
                }])
                print(f"  ✅ 存入: [{root_cat}] -> {sub_topic} -> {title} (关联 {len(new_dialog_ids)} 条对话)")
            except Exception as e:
                print(f"  ❌ 存入失败: {item.get('title')} - Error: {e}")

    def _cosine_similarity(self, vec_a, vec_b):
        """计算余弦相似度"""
        a = np.array(vec_a)
        b = np.array(vec_b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def query_memories(self,
                       root_category: Optional[str] = None,
                       sub_topic: Optional[str] = None,
                       level: Optional[int] = None,
                       text_query: Optional[str] = None,
                       top_k: int = 10) -> List[Dict]:
        """
        多功能查询接口，返回的记忆节点包含 dialog_ids 字段
        """
        filters = []
        if root_category:
            filters.append(f"root_category = '{root_category}'")
        if sub_topic:
            filters.append(f"sub_topic = '{sub_topic}'")
        if level is not None:
            filters.append(f"level = {level}")

        if text_query:
            query_vector = embedding_model.encode(text_query).tolist()
            query = self.table.search(query_vector)
        else:
            query = self.table.search()

        if filters:
            where_clause = " AND ".join(filters)
            query = query.where(where_clause)

        results = query.limit(top_k).to_list()
        return results

    def get_branch(self, root_category: str) -> Dict:
        root = self.table.search().where(f"root_category = '{root_category}' AND level = 0").to_list()
        if not root:
            return {}
        root_node = root[0]
        branches = self.table.search().where(f"parent_id = '{root_node['id']}'").to_list()
        for branch in branches:
            leaves = self.table.search().where(f"parent_id = '{branch['id']}'").to_list()
            branch['children'] = leaves
        root_node['children'] = branches
        return root_node

    # ---------- 对话流水表相关（SQLite）----------
    def _init_conversation_db(self):
        """初始化 SQLite 对话流水表"""
        os.makedirs(os.path.dirname(self.conv_db_path), exist_ok=True)
        conn = sqlite3.connect(self.conv_db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dialogs (
                dialog_id TEXT PRIMARY KEY,
                conversation_id TEXT,
                turn_index INTEGER,
                role TEXT,
                content TEXT,
                timestamp INTEGER
                -- 可选的 memory_ids 字段暂不添加，保持简单
            )
        ''')
        # 为常用查询创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversation ON dialogs (conversation_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON dialogs (timestamp)')
        conn.commit()
        conn.close()

    def add_dialog(self, conversation_id: str, turn_index: int, role: str, content: str) -> str:
        """
        添加一条对话记录，返回生成的 dialog_id
        """
        dialog_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.conv_db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO dialogs (dialog_id, conversation_id, turn_index, role, content, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (dialog_id, conversation_id, turn_index, role, content, int(time.time())))
        conn.commit()
        conn.close()
        return dialog_id

    def add_dialogs_batch(self, dialogs: List[Dict]) -> List[str]:
        """
        批量添加对话记录，每个字典需包含 conversation_id, turn_index, role, content
        返回生成的 dialog_id 列表
        """
        dialog_ids = []
        conn = sqlite3.connect(self.conv_db_path)
        cursor = conn.cursor()
        timestamp = int(time.time())
        for d in dialogs:
            dialog_id = str(uuid.uuid4())
            dialog_ids.append(dialog_id)
            cursor.execute('''
                INSERT INTO dialogs (dialog_id, conversation_id, turn_index, role, content, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (dialog_id, d['conversation_id'], d['turn_index'], d['role'], d['content'], timestamp))
        conn.commit()
        conn.close()
        return dialog_ids

    def get_dialogs_by_ids(self, dialog_ids: List[str]) -> List[Dict]:
        """
        根据对话ID列表获取原始对话记录，按 conversation_id, turn_index 排序
        """
        if not dialog_ids:
            return []
        conn = sqlite3.connect(self.conv_db_path)
        # 使用 Row 工厂使返回结果为字典样式
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        placeholders = ','.join(['?'] * len(dialog_ids))
        cursor.execute(f'''
            SELECT * FROM dialogs 
            WHERE dialog_id IN ({placeholders})
            ORDER BY conversation_id, turn_index
        ''', dialog_ids)
        rows = cursor.fetchall()
        conn.close()
        # 转换为字典列表
        return [dict(row) for row in rows]

    def get_dialogs_by_conversation(self, conversation_id: str) -> List[Dict]:
        """获取某次会话的全部对话记录"""
        conn = sqlite3.connect(self.conv_db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM dialogs 
            WHERE conversation_id = ?
            ORDER BY turn_index
        ''', (conversation_id,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]


# ==================== 搜索演示 ====================
if __name__ == "__main__":
    store = TreeMemoryStore("demo_user")

    # ---------- 第一步：插入一些对话流水 ----------
    print("📝 添加对话流水记录...")
    # 模拟一次对话会话
    conv_id = "conv_001"
    dialogs = [
        {"conversation_id": conv_id, "turn_index": 0, "role": "user", "content": "我在用FastAPI做JWT认证，但一直报错说密钥无效"},
        {"conversation_id": conv_id, "turn_index": 1, "role": "assistant", "content": "请检查你的JWT密钥配置是否正确，可能是环境变量没加载"},
        {"conversation_id": conv_id, "turn_index": 2, "role": "user", "content": "找到问题了，是密钥字符串末尾多了空格，已修复"},
        {"conversation_id": conv_id, "turn_index": 3, "role": "assistant", "content": "很好，建议以后使用环境变量管理密钥"},
    ]
    dialog_ids = store.add_dialogs_batch(dialogs)
    print(f"  生成了 {len(dialog_ids)} 条对话ID: {dialog_ids}")

    # 另一段对话，关于 Docker
    conv_id2 = "conv_002"
    dialogs2 = [
        {"conversation_id": conv_id2, "turn_index": 0, "role": "user", "content": "我的Docker容器连不上宿主机MySQL"},
        {"conversation_id": conv_id2, "turn_index": 1, "role": "assistant", "content": "试试用host网络模式，或者使用宿主机IP地址"},
    ]
    dialog_ids2 = store.add_dialogs_batch(dialogs2)

    # 第三段对话，关于川菜
    conv_id3 = "conv_003"
    dialogs3 = [
        {"conversation_id": conv_id3, "turn_index": 0, "role": "user", "content": "中午吃什么？想吃辣的"},
        {"conversation_id": conv_id3, "turn_index": 1, "role": "assistant", "content": "川菜怎么样？火锅或者水煮鱼"},
        {"conversation_id": conv_id3, "turn_index": 2, "role": "user", "content": "那就火锅吧，记得多放辣椒"},
    ]
    dialog_ids3 = store.add_dialogs_batch(dialogs3)

    # ---------- 第二步：插入记忆节点，并关联对话ID ----------
    test_data = [
        {
            "root_category": "Tech",
            "sub_topic": "FastAPI_Project",
            "title": "JWT 密钥错误修复",
            "summary": "用户在使用 FastAPI 实现 JWT 认证时遇到验证失败，原因是密钥配置错误（末尾空格），已解决。",
            "key_facts": ["算法 HS256", "密钥错误", "python-jose"],
            "entities": ["FastAPI", "JWT"]
        },
        {
            "root_category": "Tech",
            "sub_topic": "Docker_Deployment",
            "title": "容器内无法连接数据库",
            "summary": "Docker 容器中运行的应用无法连接到宿主机的 MySQL，原因是网络模式配置错误，改用 host 网络后解决。",
            "key_facts": ["Docker", "网络模式", "MySQL"],
            "entities": ["Docker", "MySQL"]
        },
        {
            "root_category": "Life",
            "sub_topic": "Food_Preferences",
            "title": "午餐想吃川菜",
            "summary": "用户午餐想吃川菜，最终决定吃火锅。",
            "key_facts": ["偏好川菜", "决定吃火锅"],
            "entities": ["川菜", "火锅"]
        }
    ]
    # 为每条记忆指定关联的对话ID（第一个记忆关联 conv_001 的所有对话，第二个关联 conv_002，第三个关联 conv_003）
    dialog_ids_list = [dialog_ids, dialog_ids2, dialog_ids3]
    store.add_memories(test_data, dialog_ids_list)

    print("\n" + "="*50)
    print("开始演示各种查询方式")
    print("="*50 + "\n")

    # 演示1：按类别过滤
    print("【演示1】查询 Tech 类别下的所有记忆：")
    tech_mems = store.query_memories(root_category="Tech", level=2)
    for mem in tech_mems:
        print(f"  - {mem['title']} (子主题: {mem['sub_topic']})")
    print()

    # 演示2：按子主题过滤
    print("【演示2】查询 FastAPI_Project 子主题下的记忆：")
    fastapi_mems = store.query_memories(root_category="Tech", sub_topic="FastAPI_Project", level=2)
    for mem in fastapi_mems:
        print(f"  - {mem['title']}: {mem['summary'][:30]}...")
    print()

    # 演示3：向量语义搜索
    print("【演示3】语义搜索：输入“数据库连接问题”")
    semantic_results = store.query_memories(text_query="数据库连接问题", top_k=3)
    for i, mem in enumerate(semantic_results, 1):
        print(f"  {i}. {mem['title']} (类别: {mem['root_category']})")
    print()

    # 演示4：组合过滤 + 向量搜索
    print("【演示4】在 Tech 类别中搜索与“认证”相关的记忆")
    filtered_semantic = store.query_memories(
        root_category="Tech",
        text_query="认证",
        top_k=2
    )
    for mem in filtered_semantic:
        print(f"  - {mem['title']} (摘要: {mem['summary'][:40]}...)")
    print()

    # 演示5：获取完整树分支
    print("【演示5】获取 Tech 类别的树形结构：")
    tech_tree = store.get_branch("Tech")
    print(f"根节点: {tech_tree['title']}")
    for branch in tech_tree['children']:
        print(f"  ├─ 子主题: {branch['sub_topic']}")
        for leaf in branch.get('children', []):
            print(f"  │   ├─ {leaf['title']}")
    print()

    # 演示6：根据记忆节点拉取原始对话
    print("【演示6】从记忆节点拉取原始对话（以第一条记忆为例）")
    first_mem = tech_mems[0]  # 取第一个 Tech 记忆
    print(f"记忆标题: {first_mem['title']}")
    print(f"关联的对话ID: {first_mem['dialog_ids']}")
    dialogs = store.get_dialogs_by_ids(first_mem['dialog_ids'])
    print("原始对话内容:")
    for d in dialogs:
        print(f"  [{d['role']}] {d['content']}")
    print()