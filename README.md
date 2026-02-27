# LocalNexus
A privacy-first AI memory architecture for local deployment. Features a unique "Filter-First" pipeline: local noise removal ➡️ cloud refinement ➡️ file-isolated storage. Connects local data sovereignty with cloud intelligence seamlessly. Built with Python, LanceDB & OpenAI.

# 🧠 LocalNexus (本联智记)

> **Rooted Locally, Connected Intelligently.**  
> 根植本地，智联无限。

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-beta-orange.svg)]()

**LocalNexus** 是一个轻量级、隐私优先的 **AI 记忆管理架构**，专为本地部署设计，同时利用云端大模型的强大智能。

与传统记忆方案不同，LocalNexus 独创了 **"Filter-First" (先滤后云)** 处理流水线：
1. 🛡️ **本地去噪**：在数据离开本地前，通过规则引擎过滤无效信息，保护隐私并节省 Token。
2. ☁️ **云端精炼**：仅将精华数据发送至云端 LLM 进行分类、标签化和摘要。
3. 📂 **物理隔离**：采用 **每用户独立文件 (File-Isolated)** 存储策略，无需复杂数据库服务器。
4. ⚡ **异步整理**：后台非阻塞式记忆维护，确保主对话零延迟。

---

## ✨ 核心特性 (Key Features)

- **🔒 极致隐私 (Privacy First)**  
  原始对话数据永远留在本地。只有经过清洗和脱敏的“精华片段”才会被发送到云端。

- **🗑️ 智能去噪 (Smart Noise Filtering)**  
  内置可配置的本地规则引擎，自动剔除表情、寒暄、重复内容，大幅降低云端 API 成本。

- **🏷️ 动态分类与标签 (Dynamic Tagging)**  
  利用云端 LLM 自动为记忆片段生成结构化标签（如 `coding`, `planning`, `chat`），支持基于标签的精准检索过滤。

- **💾 文件级隔离存储 (File-Isolated Storage)**  
  基于 **LanceDB** 嵌入式引擎，每个用户对应一个独立的文件夹 (`./storage/user_id/`)。备份、迁移、删除用户数据只需操作文件系统。

- **⚡ 异步后台工人 (Async Background Worker)**  
  记忆整理、分类、存储过程在独立线程中异步运行，完全不阻塞用户的主对话流程。

- **🎯 意图驱动检索 (Intent-Driven Retrieval)**  
  内置轻量级意图路由器，仅在用户问题涉及历史记忆时触发检索，减少不必要的计算开销。

---

## 🏗️ 架构原理 (Architecture)

LocalNexus 采用 **混合云架构 (Hybrid Cloud Architecture)**：

```mermaid
graph LR
    User[用户对话] --> Engine{本地引擎}
    Engine -- 需要记忆? --> Router[意图路由]
    Router -- 是 --> Search[本地 LanceDB 检索]
    Search --> Context[注入上下文]
    Context --> CloudLLM[云端 LLM 生成回答]
    
    CloudLLM --> Queue[后台整理队列]
    Queue --> Worker[后台工人线程]
    Worker --> Filter[本地去噪]
    Filter --> CloudRefine[云端分类/精炼]
    CloudRefine --> Store[(独立文件存储)]
    
    style User fill:#f9f,stroke:#333
    style Engine fill:#bbf,stroke:#333
    style CloudLLM fill:#ff9,stroke:#333
    style Store fill:#9f9,stroke:#333