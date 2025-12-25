"""
简单的前端界面：用 Gradio 调用 EmotionRAG，并以 Markdown 显示回答与检索片段。

依赖：pip install gradio
用法示例：
python scripts/rag_ui.py --jsonl data/cards.jsonl --project readme.md --project docs --port 7860
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import gradio as gr

# 确保可以导入 rag.RAG_LM 中的 EmotionRAG
import sys
from pathlib import Path as _Path

ROOT = _Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.RAG_LM import EmotionRAG  # noqa: E402


def discover_md_log_files(root: Path) -> List[str]:
    """Recursively collect .md/.log files under project root for indexing."""
    exts = {".md", ".log"}
    files: List[str] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(str(p))
    return files


def build_rag(args) -> EmotionRAG:
    jsonl_path = args.jsonl
    if not jsonl_path:
        default_jsonl = ROOT / "data" / "cards.jsonl"
        if default_jsonl.exists():
            jsonl_path = str(default_jsonl)
            print(f"ℹ️ 未提供 --jsonl，默认使用 {jsonl_path}")

    # 默认不扫描全项目，按需通过 --auto-projects 开启，再叠加用户指定的 project 参数
    project_paths: List[str] = []
    if args.auto_projects:
        project_paths.extend(discover_md_log_files(ROOT))
    if args.project:
        project_paths.extend(args.project)
    seen = set()
    uniq_projects = []
    for p in project_paths:
        if p not in seen:
            uniq_projects.append(p)
            seen.add(p)

    return EmotionRAG(
        jsonl_path=jsonl_path,
        project_paths=uniq_projects if uniq_projects else None,
        lm_studio_url=args.lm_url,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


def format_context(documents, metadatas) -> str:
    parts = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
        source = meta.get("source", "unknown")
        path = meta.get("path", "")
        title = f"数据 {i}（{source}）"
        if path:
            title += f" - {path}"
        parts.append(f"### {title}\n\n```\n{doc}\n```")
    return "\n\n".join(parts) if parts else "_未检索到内容_"


def make_answer_fn(rag: EmotionRAG, args):
    system_prompt = (
        "你是一个专业的情感分析助手，擅长理解和分析人类情感表达，回答要简洁。"
    )

    def answer(question: str, top_k: int, temperature: float, max_tokens: int):
        if not question.strip():
            return "请输入问题。", "_无上下文_"

        results = rag.search(question, top_k=top_k)
        context_md = format_context(
            results["documents"][0], results["metadatas"][0]
        )

        prompt = (
            "你是一个情感分析专家。基于以下项目资料（代码仓库中的 README/日志/文档及 JSONL 数据）回答用户的问题。\n\n"
            f"项目资料片段：\n{context_md}\n\n"
            f"用户问题：{question}\n\n"
            "请结合片段中的内容，进行分析，"
            "回答用户的问题。"
        )

        content = rag.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return content, context_md

    return answer


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio UI for EmotionRAG")
    parser.add_argument("--jsonl", type=str, default=None, help="JSONL 数据路径")
    parser.add_argument(
        "--project",
        action="append",
        default=[],
        help="额外要索引的文件或目录，可重复",
    )
    parser.add_argument(
        "--auto-projects",
        action="store_true",
        help="自动索引项目内全部 .md/.log（文件多时启动会变慢）",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="moka-ai/m3e-base",
        help="SentenceTransformer 模型名称",
    )
    parser.add_argument(
        "--lm-url",
        type=str,
        default="http://localhost:1234/v1",
        help="LM Studio API 地址",
    )
    parser.add_argument("--chunk-size", type=int, default=800, help="分块大小")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="分块重叠")
    parser.add_argument("--port", type=int, default=7860, help="Gradio 端口")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument(
        "--share", action="store_true", help="Gradio share 链接（需网络支持）"
    )
    parser.add_argument("--title", type=str, default="Emotion RAG UI", help="页面标题")
    return parser.parse_args()


def main():
    args = parse_args()
    rag = build_rag(args)
    answer_fn = make_answer_fn(rag, args)

    with gr.Blocks(title=args.title) as demo:
        gr.Markdown(f"# {args.title}\n输入问题，获取情感分析回答（Markdown 显示）")
        with gr.Row():
            question = gr.Textbox(label="问题", placeholder="请输入你的问题", lines=3)
        with gr.Row():
            top_k = gr.Slider(1, 10, value=3, step=1, label="Top K")
            temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
            max_tokens = gr.Slider(64, 1024, value=512, step=32, label="Max tokens")
        run_btn = gr.Button("检索并回答", variant="primary")
        answer_md = gr.Markdown(label="回答")
        ctx_md = gr.Markdown(label="检索上下文")

        run_btn.click(
            answer_fn,
            inputs=[question, top_k, temperature, max_tokens],
            outputs=[answer_md, ctx_md],
        )

    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
