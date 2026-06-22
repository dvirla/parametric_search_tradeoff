"""
Visualize SFT JSONL examples (from create_musique_sft_data.py or
generate_mock_sft_example.py) as a standalone HTML page.

Features:
  - Color-coded roles: system / user / assistant / tool
  - Collapsible thinking blocks (<think>...</think>)
  - Search queries highlighted in yellow
  - Collapsible tool results
  - Red splice-point marker at patch_start_idx

Works with both:
  - Mock data:  data/sft/mock/mock_examples.jsonl
  - Real data:  data/sft/musique/backup/*.jsonl

Usage:
    uv run python scripts/visualize_sft_examples.py \
        --input data/sft/mock/mock_examples.jsonl \
        --output mock_vis.html

    uv run python scripts/visualize_sft_examples.py \
        --input "data/sft/musique/backup/*.jsonl" \
        --output real_vis.html
"""

import argparse
import glob
import html
import json
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Thinking-block parser
# --------------------------------------------------------------------------

THINK_RE = re.compile(r"(<think>.*?</think>)", re.DOTALL)


def split_content(content: str) -> list[tuple[str, str]]:
    """Split a content string into ('think', text) and ('text', text) segments."""
    parts = []
    for seg in THINK_RE.split(content):
        if seg.startswith("<think>") and seg.endswith("</think>"):
            inner = seg[7:-8].strip()
            if inner:
                parts.append(("think", inner))
        elif seg.strip():
            parts.append(("text", seg))
    return parts


# --------------------------------------------------------------------------
# Per-message HTML renderer
# --------------------------------------------------------------------------

ROLE_CLASSES = {
    "system": "msg-system",
    "user": "msg-user",
    "assistant": "msg-assistant",
    "tool": "msg-tool",
}


def _extract_query(tool_call: dict) -> str:
    try:
        args = json.loads(tool_call["function"]["arguments"])
        return args.get("query", "")
    except Exception:
        return ""


def render_message(msg: dict, idx: int, patch_start_idx: int | None) -> str:
    role = msg.get("role", "unknown")
    raw_content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []
    tool_call_id = msg.get("tool_call_id", "")

    is_splice = patch_start_idx is not None and idx == patch_start_idx
    role_class = ROLE_CLASSES.get(role, "msg-unknown")
    if role == "assistant" and tool_calls:
        role_class = "msg-assistant-tool"

    parts_html = []

    # Content segments (thinking + plain text)
    if raw_content and role != "tool":
        segments = split_content(raw_content)
        for kind, text in segments:
            escaped = html.escape(text)
            if kind == "think":
                parts_html.append(
                    f'<details class="think-block">'
                    f'<summary>&#x1F4AD; Thinking <span class="hint">(click to expand)</span></summary>'
                    f'<pre class="think-body">{escaped}</pre>'
                    f"</details>"
                )
            else:
                parts_html.append(f'<pre class="text-body">{escaped}</pre>')

    # Tool calls (search queries)
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = html.escape(fn.get("name", ""))
        query = html.escape(_extract_query(tc))
        raw_args = html.escape(fn.get("arguments", ""))
        call_id = html.escape(tc.get("id", ""))
        query_html = (
            f'<span class="query-highlight">{query}</span>' if query
            else f'<code class="raw-args">{raw_args}</code>'
        )
        parts_html.append(
            f'<div class="tool-call-block">'
            f'<span class="tool-call-label">&#x1F50D; {name}</span> '
            f'{query_html}'
            f'<span class="call-id"> id={call_id}</span>'
            f"</div>"
        )

    # Tool return content
    if role == "tool":
        escaped = html.escape(raw_content)
        char_count = len(raw_content)
        tid = html.escape(tool_call_id)
        parts_html.append(
            f'<details class="tool-result-block">'
            f'<summary>&#x1F4E6; Tool result ({char_count} chars)'
            f'<span class="call-id"> tool_call_id={tid}</span></summary>'
            f'<pre class="tool-body">{escaped}</pre>'
            f"</details>"
        )

    splice_html = ""
    if is_splice:
        splice_html = (
            f'<div class="splice-marker">'
            f"&#x26A1; SPLICE START &mdash; patch_start_idx = {idx}"
            f"</div>"
        )

    inner = "\n".join(parts_html) if parts_html else '<span class="empty">(empty)</span>'

    return (
        f"{splice_html}"
        f'<div class="message {role_class}{" splice-border" if is_splice else ""}" '
        f'id="msg-{idx}">'
        f'<div class="role-label">[{idx}] {role.upper()}'
        + (f' &rarr; {html.escape(tool_call_id)}' if role == "tool" and tool_call_id else "")
        + f"</div>"
        f'<div class="msg-body">{inner}</div>'
        f"</div>"
    )


# --------------------------------------------------------------------------
# Per-example HTML renderer
# --------------------------------------------------------------------------

def render_example(example: dict, ex_idx: int) -> str:
    msgs = example.get("messages", [])
    patch_start_idx = example.get("patch_start_idx")
    question = example.get("question") or (
        msgs[1]["content"][:120] if len(msgs) > 1 else "?"
    )
    procedure = example.get("procedure", "?")
    hop_count = example.get("hop_count", "?")
    gold = example.get("gold_answer", "")
    description = example.get("description", "")
    canonical_query = example.get("canonical_query", "")

    meta_rows = []
    if gold:
        meta_rows.append(f"<b>Gold answer:</b> {html.escape(gold)}")
    if procedure != "?":
        meta_rows.append(f"<b>Procedure:</b> {procedure}")
    if hop_count != "?":
        meta_rows.append(f"<b>Hops:</b> {hop_count}")
    if patch_start_idx is not None:
        meta_rows.append(f'<b>patch_start_idx:</b> <span class="patch-idx">{patch_start_idx}</span>')
    if canonical_query:
        meta_rows.append(f"<b>Canonical query (spliced):</b> {html.escape(canonical_query)}")
    if description:
        meta_rows.append(f"<i>{html.escape(description)}</i>")

    meta_html = " &nbsp;|&nbsp; ".join(meta_rows)
    msg_htmls = [render_message(m, i, patch_start_idx) for i, m in enumerate(msgs)]

    return (
        f'<div class="example" id="ex-{ex_idx}">'
        f'<h2>Example {ex_idx + 1}</h2>'
        f'<div class="question-box">{html.escape(question)}</div>'
        f'<div class="meta">{meta_html}</div>'
        f'<div class="messages">{"".join(msg_htmls)}</div>'
        f"</div>"
    )


# --------------------------------------------------------------------------
# Full page builder
# --------------------------------------------------------------------------

CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 0; padding: 16px 24px; background: #FAFAFA; color: #212121; }
h1 { margin-bottom: 8px; }
h2 { margin: 24px 0 8px; font-size: 1.1rem; color: #424242; }

/* Table of contents */
.toc { border-collapse: collapse; width: 100%; margin-bottom: 32px;
       font-size: 0.88rem; }
.toc th, .toc td { border: 1px solid #DDD; padding: 6px 10px; text-align: left; }
.toc th { background: #EEEEEE; }
.toc tr:nth-child(even) { background: #F5F5F5; }
.toc a { color: #1565C0; text-decoration: none; }
.toc a:hover { text-decoration: underline; }
.patch-idx { color: #C62828; font-weight: bold; }

/* Example container */
.example { background: white; border: 1px solid #E0E0E0; border-radius: 8px;
           padding: 16px; margin-bottom: 32px; }
.question-box { font-size: 1rem; font-weight: 600; color: #1A237E;
                background: #E8EAF6; border-left: 4px solid #3949AB;
                padding: 10px 14px; border-radius: 4px; margin-bottom: 10px; }
.meta { font-size: 0.82rem; color: #555; margin-bottom: 14px; line-height: 1.6; }

/* Messages */
.messages { display: flex; flex-direction: column; gap: 6px; }
.message { border-radius: 6px; overflow: hidden; border: 1px solid transparent; }
.role-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.05em;
              padding: 4px 12px; }
.msg-body { padding: 8px 12px; }

/* Role colors */
.msg-system  { background: #F5F5F5; border-color: #9E9E9E; }
.msg-system .role-label { background: #9E9E9E; color: white; }

.msg-user    { background: #E0F2F1; border-color: #00897B; }
.msg-user .role-label { background: #00897B; color: white; }

.msg-assistant { background: #E8EAF6; border-color: #5C6BC0; }
.msg-assistant .role-label { background: #5C6BC0; color: white; }

.msg-assistant-tool { background: #F3E5F5; border-color: #8E24AA; }
.msg-assistant-tool .role-label { background: #8E24AA; color: white; }

.msg-tool    { background: #FFF3E0; border-color: #FB8C00; }
.msg-tool .role-label { background: #FB8C00; color: white; }

/* Splice marker */
.splice-marker { background: #C62828; color: white; font-weight: 700;
                 font-size: 0.85rem; padding: 6px 14px; border-radius: 4px 4px 0 0;
                 letter-spacing: 0.04em; }
.splice-border { border-color: #C62828 !important; border-top: none;
                 border-radius: 0 0 6px 6px; }

/* Thinking blocks */
.think-block { margin: 4px 0; border: 1px solid #CE93D8; border-radius: 4px;
               background: #FCE4EC; }
.think-block summary { padding: 5px 10px; cursor: pointer; font-size: 0.82rem;
                        font-weight: 600; color: #6A1B9A; user-select: none; }
.think-block summary:hover { background: #F8BBD0; }
.think-body { margin: 0; padding: 8px 12px; font-size: 0.82rem; white-space: pre-wrap;
              word-break: break-word; color: #4A148C; border-top: 1px solid #CE93D8; }
.hint { font-weight: normal; color: #AB47BC; font-size: 0.78rem; }

/* Text body */
.text-body { margin: 4px 0; font-size: 0.88rem; white-space: pre-wrap;
             word-break: break-word; background: transparent; }

/* Tool calls */
.tool-call-block { margin: 4px 0; padding: 6px 10px; background: #EDE7F6;
                   border-radius: 4px; font-size: 0.85rem; }
.tool-call-label { font-weight: 700; color: #6A1B9A; margin-right: 6px; }
.query-highlight { background: #FFF176; font-weight: 700; padding: 2px 8px;
                   border-radius: 3px; color: #212121; }
.raw-args { font-size: 0.8rem; color: #555; }
.call-id { font-size: 0.75rem; color: #888; margin-left: 8px; }

/* Tool results */
.tool-result-block { margin: 4px 0; border: 1px solid #FFCC80; border-radius: 4px; }
.tool-result-block summary { padding: 5px 10px; cursor: pointer; font-size: 0.82rem;
                              font-weight: 600; color: #E65100; }
.tool-result-block summary:hover { background: #FFE0B2; }
.tool-body { margin: 0; padding: 8px 12px; font-size: 0.8rem; white-space: pre-wrap;
             word-break: break-word; color: #BF360C; border-top: 1px solid #FFCC80; }

.empty { color: #BDBDBD; font-style: italic; font-size: 0.82rem; }
hr { border: none; border-top: 2px solid #EEE; margin: 32px 0; }
"""


def build_toc(examples: list[dict]) -> str:
    rows = []
    for i, ex in enumerate(examples):
        msgs = ex.get("messages", [])
        q = ex.get("question") or (msgs[1]["content"][:80] if len(msgs) > 1 else "?")
        proc = ex.get("procedure", "?")
        hops = ex.get("hop_count", "?")
        splice = ex.get("patch_start_idx")
        splice_cell = (
            f'<span class="patch-idx">{splice}</span>' if splice is not None else "&mdash;"
        )
        rows.append(
            f"<tr>"
            f'<td><a href="#ex-{i}">{i + 1}</a></td>'
            f"<td>{html.escape(q[:80])}</td>"
            f"<td>{proc}</td>"
            f"<td>{hops}</td>"
            f"<td>{len(msgs)}</td>"
            f"<td>{splice_cell}</td>"
            f"</tr>"
        )
    header = (
        "<tr><th>#</th><th>Question</th><th>Procedure</th>"
        "<th>Hops</th><th>Messages</th><th>patch_start_idx</th></tr>"
    )
    return f'<table class="toc">{header}{"".join(rows)}</table>'


def build_page(examples: list[dict]) -> str:
    toc = build_toc(examples)
    body_parts = []
    for i, ex in enumerate(examples):
        body_parts.append(render_example(ex, i))
        if i < len(examples) - 1:
            body_parts.append("<hr>")
    body = "\n".join(body_parts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SFT Examples Viewer</title>
<style>
{CSS}
</style>
</head>
<body>
<h1>SFT Examples Viewer</h1>
<p style="color:#555;font-size:0.88rem">{len(examples)} example(s) loaded</p>
{toc}
{body}
</body>
</html>"""


# --------------------------------------------------------------------------
# JSONL loader
# --------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize SFT JSONL examples as a standalone HTML page."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="JSONL file(s) or glob patterns to visualize",
    )
    parser.add_argument(
        "--output",
        default="sft_visualization.html",
        help="Output HTML file path (default: sft_visualization.html)",
    )
    args = parser.parse_args()

    examples = []
    for pattern in args.input:
        paths = sorted(glob.glob(pattern))
        if not paths:
            print(f"Warning: no files matched '{pattern}'", file=sys.stderr)
        for path in paths:
            loaded = load_jsonl(path)
            print(f"  Loaded {len(loaded)} examples from {path}")
            examples.extend(loaded)

    if not examples:
        print("No examples found — check your --input patterns.", file=sys.stderr)
        sys.exit(1)

    page = build_page(examples)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")

    print(f"\nWrote {len(examples)} example(s) to {out_path}")
    print(f"Open in browser: file://{out_path.resolve()}")


if __name__ == "__main__":
    main()
