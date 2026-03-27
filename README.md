# Amnesia｜失忆图书馆

一个基于“Memory-as-Retrieval”的AI情感记忆系统原型，将用户的情绪表达转化为可检索的语义单元，并用于生成个性化共情回复。
- 将用户输入的“记忆片段”转化为可归档、可检索的情绪卡片（cards），并在检索到相似片段后生成边界清晰的共情式回复。
- 图书馆的失忆是对情感词汇的失忆，恢复方式是通过检索用户自身的表达。
- 每个人都有属于自己的图书馆，存放着零碎的记忆，输入的内容也会改变图书馆的外观。
- 【用户输入的内容】会成为构建用户本身情绪存储系统，从而实现个性化。
- 目标用户：喜欢写随笔记录心情，不想让人看到，但是又渴望共鸣的人。
- 类似产品有icity和to me，豆瓣等等文字内容居多的app（短视频时代前的微博也算）。市面上有很多日记本app，存档结构化做得很简洁。但是大多都没有回应功能。太过社交化的app的self reflect感觉就没那么强，没有那种沉浸感小世界的感觉。
- 场景：在下雨天的出租车上，用户一个人刚刚坐定，听着音乐，随手写下一些心情。或许ta写完了就关上了app，或许留下来等待回应。app收到文字之后荡漾一下，将纸条折起来收纳到书架上。书架上可能放着很多书，用户也可以用检索系统去找，或者还有去年今日的那种推送功能。
- 本项目重点不在于复杂工程实现，而在于探索如何将生成式AI能力转化为“可持续的个体记忆结构”，从而支持长期的人机情感交互。代码可供实现参考。

在这个原型中，针对三项任务，我对几个大语言模型（LLM）进行了评估（结构化情感解析、思维链推理、检索增强生成能力）

 - GPT-4o 在 RAG 任务中展现出最强的边界控制能力（能够避免将检索到的记忆与用户输入混淆）
 - Claude 生成的共情式回应最为连贯且富有沉浸感，但也倾向于过度解读
 - DeepSeek 擅长结构化信息提取，但可能存在过度泛化的倾向
 - 经过情感微调的小型模型能够生成强烈的情感色彩，但在稳定性和可复现性方面有所欠缺

具体可见course_final_work.pdf。

以下为具体信息

## 目录

- [项目概述](#项目概述)
- [核心能力](#核心能力)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
  - [1) 环境准备](#1-环境准备)
  - [2) 安装依赖](#2-安装依赖)
  - [3) 配置 .env](#3-配置-env)
- [使用方法](#使用方法)
  - [A. 生成记忆卡片（结构化文本 + 情绪标签）](#a-生成记忆卡片结构化文本--情绪标签)
  - [B. 启动 RAG 交互界面（Gradio）](#b-启动-rag-交互界面gradio)
- [数据与格式](#数据与格式)
  - [cards.jsonl 的 JSON Schema](#cardsjsonl-的-json-schema)
  - [情绪编码表（emotion_schema）](#情绪编码表emotion_schema)
- [模型支持与实验设置](#模型支持与实验设置)
- [其他](#其他)



## 项目概述

Amnesia（失忆图书馆）是一个面向情感表达与信息组织的交互系统原型：AI 扮演“记忆管理者/图书管理员”，将用户提交的自然语言记忆片段进行摘要、关键词提取、情绪光谱标注（valence/arousal）、隐喻域选择与共情式回复生成，并将结构化结果写入 `data/cards.jsonl` 以供后续检索与再生成。

系统评测与设计动机可参考课程报告：对比了多模型在**结构化生成、情感 CoT、RAG** 三类场景的表现差异，并讨论了 RAG 边界控制与召回策略优化方向。



## 核心能力

1. **结构化情绪卡片生成**
   - 将输入文本映射到统一的 JSON 结构：summary / keywords / 情绪光谱 / thinking / metaphor / reply。
     
2. **情感思维链（CoT）呈现**
   - 在 `thinking` 字段中给出情绪判断依据，便于解释与对比

3. **检索增强生成（RAG）**
   - 将 `cards.jsonl`、`README.md`、`logs.md` 等材料向量化入库，检索相似片段后生成回应；提供 Gradio 网页交互。



## 项目结构

项目当前目录结构如下（以仓库为准）：

```bash
AMNESIA/
├── config/
│ ├── model_config.py # 模型调用、推理参数、路径等配置
├── data/
│ └── cards.jsonl # 结构化情绪卡片存档
├── prompts/
│ └── system_librarian.txt # 系统提示词
├── rag/
│ └── rag_ui.py # RAG 交互界面
├── scripts/
│ └── chat_to_cards.py # 生成 cards.jsonl 的交互脚本
├── emotion_schema.csv
├── emotion_schema.json
├── raw.csv
├── logs.md
├── .env # 环境变量
├── course_final_work.pdf # 结课实验报告
└── README.md

```



## 快速开始

### 1) 环境准备

- Python：建议与项目一致使用 **Python 3.11.x**。
- （可选）本地模型：若使用 LM Studio 承载本地模型，请先安装并配置好可用的本地推理服务。
### 2) 安装依赖

> 以仓库实际依赖清单为准（`requirements.txt` ）。

常见流程示例：

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

### 3) 配置 .env

项目通过 .env 统一管理模型 API 与路由配置。 请自行创建文件，并依照以下内容输入（以openrouter为例）：

```bash
MODEL_TARGET=…… #在此处切换模型

OPENROUTER
OPENROUTER_MODEL_NAME=  #填入模型名称
OPENROUTER_API_BASE=https://openrouter.ai/api/v1    #填入API BASE
OPENROUTER_API_KEY=    #填入API key


模型2
模型2_MODEL_NAME=
模型2_API_BASE=
模型2_API_KEY=
```



## 使用方法

#### A. 生成记忆卡片（结构化文本 + 情绪标签）

该流程将用户输入写入 data/cards.jsonl，作为后续检索与对比实验的数据基础。

```bash
python scripts/chat_to_cards.py
```

你将获得包含 draft 等字段的结构化输出（示例见下文 Schema）。

#### B. 启动 RAG 交互界面（Gradio）

RAG 模块将 cards.jsonl 与项目文档向量化后进行检索，再生成共情式回应；提供网页界面便于阅读。 

```bash
python rag/rag_ui.py
```

*RAG 脚本在设计上对 .jsonl 与 .md 只读不写，避免实验过程中反复测试污染记忆存档文件。

## 数据与格式
#### cards.jsonl 的 JSON Schema

每一行是一条卡片记录，核心结构如下（与系统提示词约束一致）：

```bash
{
  "draft": {
    "summary": "string（对用户输入的精炼、中立总结）",
    "keywords": ["string", "..."],
    "spectrum": {
      "valence": "float（-1.0 ~ 1.0）",
      "arousal": "float（0.0 ~ 1.0）",
      "tones": ["string", "..."]
    },
    "thinking": "展示联想与分析过程，并说明 valence/arousal 取值依据",
    "metaphor_domain": "string（参考 emotion_schema.json 的 domains 生成意象）",
    "metaphor_seed": "int",
    "reply": "string（给用户的隐喻化回复）"
  }
}
```


#### 情绪编码表（emotion_schema）

- emotion_schema.csv：初始版本编码表

- emotion_schema.json：结构化版本，便于在 system prompt 中让模型“按表生成”。


## 模型支持与实验设置

项目在课程对比实验中使用/对接了以下模型（API 或本地推理）： 

- DeepSeek-V3.2-Chat（DeepSeek 官方 API）

- ChatGPT-4o（OpenAI 官方 API）

- Claude 3.7 Sonnet（OpenRouter API）

- OmniDimen-4B-Emotion（HuggingFace上的开源模型，通过 LM Studio 本地运行）


## 参数（以报告复现实验为参考）： 

- chat_to_cards.py：temperature=0.7，timeout=300s

- rag_ui.py：temperature=0.7，top_k=3，max_tokens=512


## 其他

- 这是各种意义上的第一次尝试，无论是代码还是思路，都还有很多不完美的地方，随时欢迎指教，我的邮箱：elokuu_1028@qq.com。

- 感谢的话说给ChatGPT Claude Gemini他们吧！特别是Gemini，最开始的想法都是说给Gemini听的。还有Aven，没有他就没有这个尝试的开头，没有他的话我一定是懒得行动的。

- 在兴奋地构想的过程中，很多东西都成为了我的灵感。在10月，我完美通关了《宇宙之轮姐妹会》，感觉人的记忆可以做成美丽的卡片，卡片可以具有经过精心设计的标签可供检索。

- 作为一个持续在关注的用户，希望我和他们都越来越好。


