"""
LM Studio RAG系统 - 情感数据检索与问答
依赖安装：pip install chromadb sentence-transformers openai
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from config.model_config import get_model_config
from scripts.openai_client import call_chat_completion


class EmotionRAG:
    def __init__(
        self,
        jsonl_path: str | None = None,
        project_paths: list[str] | str | Path | None = None,
        lm_studio_url: str = "http://localhost:1234/v1",
        embedding_model: str = "moka-ai/m3e-base",
        chunk_size: int = 800,
        chunk_overlap: int = 200,
    ):
        """
        初始化RAG系统

        Args:
            jsonl_path: JSONL数据文件路径
            project_paths: 项目文件或目录列表
            lm_studio_url: LM Studio API地址
            embedding_model: 中文向量化模型
        chunk_size: 文本分块大小
        chunk_overlap: 块间重叠字符数
        """
        self.model_cfg = get_model_config()

        print("加载向量模型...")
        self.embedder = SentenceTransformer(embedding_model)

        print("初始化向量数据库...")
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(
            name="emotion_data", metadata={"hnsw:space": "cosine"}
        )

        if jsonl_path:
            print("加载并向量化 JSONL 数据...")
            self.load_data(jsonl_path)
        if project_paths:
            print("加载并向量化项目文件...")
            self.load_project_files(project_paths, chunk_size, chunk_overlap)
        print(f"✅系统初始化完成！共加载 {self.collection.count()} 条数据")

    def load_data(self, jsonl_path: str):
        """加载JSONL数据并建立索引"""
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                last_brace = line.rfind("}")
                json_str = line[: last_brace + 1] if last_brace != -1 else line

                try:
                    item = json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"警告：跳过第{line_num}行，JSON解析失败: {e}")
                    continue

                search_text = self._build_search_text(item)
                embedding = self.embedder.encode(search_text).tolist()

                spectrum = item.get("spectrum", {})
                self.collection.add(
                    embeddings=[embedding],
                    documents=[search_text],
                    metadatas=[
                        {
                            "source": "jsonl",
                            "raw_text": item.get("raw_text", ""),
                            "summary": item.get("summary", ""),
                            "keywords": json.dumps(
                                item.get("keywords", []), ensure_ascii=False
                            ),
                            "valence": spectrum.get("valence", 0.0),
                            "arousal": spectrum.get("arousal", 0.0),
                            "tones": json.dumps(
                                spectrum.get("tones", []), ensure_ascii=False
                            ),
                            "metaphor_domain": item.get("metaphor_domain", ""),
                        }
                    ],
                    ids=[item["id"]],
                )

    def load_project_files(
        self,
        paths: list[str] | str | Path,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        exts: tuple[str, ...] = (".md", ".txt", ".log", ".rst"),
    ):
        """遍历项目文件并分块索引"""
        if isinstance(paths, (str, Path)):
            paths = [paths]

        for p in paths:
            p = Path(p)
            files = [p] if p.is_file() else p.rglob("*")
            for file in files:
                if not file.is_file() or file.suffix.lower() not in exts:
                    continue
                text = file.read_text(encoding="utf-8", errors="ignore")
                for i, chunk in enumerate(
                    self._split_text(text, chunk_size, chunk_overlap)
                ):
                    embedding = self.embedder.encode(chunk).tolist()
                    self.collection.add(
                        embeddings=[embedding],
                        documents=[chunk],
                        metadatas=[{"source": "project", "path": str(file), "chunk": i}],
                        ids=[f"{file}-{i}"],
                    )

    def _split_text(self, text: str, chunk_size: int, chunk_overlap: int):
        """简单分块，避免上下文过长"""
        start = 0
        n = len(text)
        while start < n:
            end = min(start + chunk_size, n)
            yield text[start:end]
            if end == n:
                break
            start = end - chunk_overlap if end - chunk_overlap > start else end

    def _build_search_text(self, item):
        """构造用于检索的文本"""
        spectrum = item.get("spectrum", {})
        keywords = item.get("keywords", [])
        tones = spectrum.get("tones", [])
        valence = spectrum.get("valence", 0.0)
        arousal = spectrum.get("arousal", 0.0)

        return (
            f"原文：{item.get('raw_text', '')}\n"
            f"摘要：{item.get('summary', '')}\n"
            f"关键词：{', '.join(keywords) if keywords else ''}\n"
            f"情感维度：效价{valence}, 唤醒度{arousal}\n"
            f"情感色调：{', '.join(tones) if tones else ''}\n"
            f"隐喻域：{item.get('metaphor_domain', '')}"
        )

    def chat_completion(self, messages, temperature=0.7, max_tokens=None):
        """调用 .env 中配置的 OpenAI-Compatible API 获取回复"""
        cfg = self.model_cfg
        resp = call_chat_completion(
            cfg["base_url"],
            cfg["api_key"],
            cfg["name"],
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp["choices"][0]["message"]["content"]

    def search(self, query, top_k=3, valence_filter=None):
        """
        语义检索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            valence_filter: 情感效价过滤 (min, max)
        """
        query_embedding = self.embedder.encode(query).tolist()

        where_filter = None
        if valence_filter:
            where_filter = {
                "$and": [
                    {"valence": {"$gte": valence_filter[0]}},
                    {"valence": {"$lte": valence_filter[1]}},
                ]
            }

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
        )

        return results

    def query(self, question, top_k=3, temperature=0.7):
        """
        RAG问答

        Args:
            question: 用户问题
            top_k: 检索文档数量
            temperature: LLM生成温度
        """
        print("\n🔍 检索相关数据...")
        results = self.search(question, top_k=top_k)

        context_parts = []
        for i, (doc, metadata) in enumerate(
            zip(results["documents"][0], results["metadatas"][0]), 1
        ):
            context_parts.append(
                f"\n【数据{i}】\n来源: {metadata.get('source', 'unknown')} {metadata.get('path', '')}\n{doc}\n"
            )

        context = "\n".join(context_parts)

        prompt = (
            "你是一个情感分析专家。基于以下情感数据库中的内容，回答用户的问题。\n\n"
            f"数据库内容：\n{context}\n\n"
            f"用户问题：{question}\n\n"
            "请结合数据中的原文、情感维度（效价/唤醒度）、关键词、情感色调和隐喻域进行深入分析，"
            "用简洁要点回答。"
        )

        print("🤖 生成回复...")
        content = self.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的情感分析助手，擅长理解和分析人类情感表达，回答要简洁。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )

        print("\n💬 回复：")
        print(content)
        print("\n")
        return content

    def analyze_emotion_pattern(self, emotion_type):
        """分析特定情感模式"""
        query_map = {
            "消极": "找出最消极悲伤的情感表达",
            "积极": "找出最积极快乐的情感表达",
            "激烈": "找出情绪最强烈激动的表达",
            "平静": "找出情绪最平静淡定的表达",
        }

        if emotion_type in query_map:
            return self.query(query_map[emotion_type], top_k=5)
        else:
            return "不支持的情感类型，请选择：消极、积极、激烈、平静"


def main():
    """使用示例"""
    print("=" * 60)
    print("情感数据RAG系统 - 基于LM Studio")
    print("=" * 60)

    # 初始化系统（将路径改为你的实际路径）
    rag = EmotionRAG(
        jsonl_path="data/cards.jsonl",
        project_paths=["readme.md", "logs", "docs"],
        lm_studio_url="http://localhost:1234/v1",
    )

    # 示例查询
    # print("\n" + "=" * 60)
    # print("示例1：检索特定主题")
    # print("=" * 60)
    # rag.query("项目中关于道德的情感表达有哪些？")

    # print("\n" + "=" * 60)
    # print("示例2：情感分析")
    # print("=" * 60)
    # rag.query("分析数据中最消极的情感特征")

    # print("\n" + "=" * 60)
    # print("示例3：对比分析")
    # print("=" * 60)
    # rag.query("对比分析高唤醒度和低唤醒度的情感表达有什么不同？")

    # 交互模式
    print("\n" + "=" * 60)
    print("进入交互模式（输入'quit'退出）")
    print("=" * 60)

    while True:
        question = input("\n请输入问题：").strip()
        if question.lower() in ["quit", "exit", "退出"]:
            break
        if question:
            rag.query(question)


if __name__ == "__main__":
    main()
