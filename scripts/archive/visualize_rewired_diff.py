"""Side-by-side diff of two parallel SFT JSONL files (pre/post rewire).

Renders one HTML page where every example shows two columns: ORIGINAL on the
left, REWIRED on the right. Aligned by line index. Equal message spans appear
side-by-side; replaced spans show the removed search + tool result + post-search
assistant turn on the left, and the merged assistant turn with the bridge
<think> on the right. The bridge thinking block is flagged green.

Usage:
  uv run python scripts/visualize_rewired_diff.py \\
      --original data/sft/musique_onpolicy/procedure1_onpolicy_sft_curated_llm.jsonl \\
      --rewired  data/sft/musique_onpolicy/procedure1_onpolicy_sft_rewired.jsonl \\
      --output   rewired_diff.html
"""

import argparse
import difflib
import html
import json
import re
import sys
from pathlib import Path

THINK_RE = re.compile(r"(<think>.*?</think>)", re.DOTALL)


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def msg_key(m):
    role = m.get("role", "")
    content = (m.get("content") or "").strip()
    tc_parts = []
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function") or {}
        tc_parts.append(f"{fn.get('name','')}::{fn.get('arguments','')}")
    return f"{role}||{content}||{'|'.join(tc_parts)}"


def diff_opcodes(orig, rew):
    a = [msg_key(m) for m in orig]
    b = [msg_key(m) for m in rew]
    return difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()


def split_content(content):
    parts = []
    for seg in THINK_RE.split(content):
        if seg.startswith("<think>") and seg.endswith("</think>"):
            inner = seg[7:-8].strip()
            if inner:
                parts.append(("think", inner))
        elif seg.strip():
            parts.append(("text", seg))
    return parts


def _extract_query(tc):
    try:
        args = json.loads(tc["function"]["arguments"])
        return args.get("query", "") if isinstance(args, dict) else ""
    except Exception:
        return ""


ROLE_CLASSES = {
    "system": "msg-system",
    "user": "msg-user",
    "assistant": "msg-assistant",
    "tool": "msg-tool",
}


def render_message(msg, idx, state):
    """state: 'normal' | 'removed' | 'rewired'."""
    role = msg.get("role", "unknown")
    raw_content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []
    tool_call_id = msg.get("tool_call_id", "")

    role_class = ROLE_CLASSES.get(role, "msg-unknown")
    if role == "assistant" and tool_calls:
        role_class = "msg-assistant-tool"

    state_class = {"removed": "msg-state-removed", "rewired": "msg-state-rewired"}.get(state, "")
    parts_html = []

    if raw_content and role != "tool":
        for kind, text in split_content(raw_content):
            escaped = html.escape(text)
            if kind == "think":
                think_cls = "think-block"
                label = "Thinking"
                if state == "rewired":
                    think_cls += " think-bridge"
                    label = "REWIRED BRIDGE - verbalized recall"
                parts_html.append(
                    f'<details class="{think_cls}" open>'
                    f'<summary>{label}<span class="hint"> (click to collapse)</span></summary>'
                    f'<pre class="think-body">{escaped}</pre>'
                    f'</details>'
                )
            else:
                parts_html.append(f'<pre class="text-body">{escaped}</pre>')

    for tc in tool_calls:
        fn = tc.get("function", {})
        query = html.escape(_extract_query(tc))
        raw_args = html.escape(fn.get("arguments", ""))
        call_id = html.escape(tc.get("id", ""))
        query_html = (
            f'<span class="query-highlight">{query}</span>' if query
            else f'<code class="raw-args">{raw_args}</code>'
        )
        tc_cls = "tool-call-block"
        if state == "removed":
            tc_cls += " tool-call-removed"
        parts_html.append(
            f'<div class="{tc_cls}">'
            f'<span class="tool-call-label">search</span> '
            f'{query_html}'
            f'<span class="call-id"> id={call_id}</span>'
            f'</div>'
        )

    if role == "tool":
        escaped = html.escape(raw_content)
        tid = html.escape(tool_call_id)
        parts_html.append(
            f'<details class="tool-result-block">'
            f'<summary>Tool result ({len(raw_content)} chars)'
            f'<span class="call-id"> tool_call_id={tid}</span></summary>'
            f'<pre class="tool-body">{escaped}</pre>'
            f'</details>'
        )

    badge = ""
    if state == "removed":
        badge = '<span class="state-badge badge-removed">REMOVED</span>'
    elif state == "rewired":
        badge = '<span class="state-badge badge-rewired">REWIRED</span>'

    inner = "\n".join(parts_html) if parts_html else '<span class="empty">(empty)</span>'
    arrow = f' -&gt; {html.escape(tool_call_id)}' if role == "tool" and tool_call_id else ""

    return (
        f'<div class="message {role_class} {state_class}" id="msg-{idx}">'
        f'<div class="role-label">[{idx}] {role.upper()}{arrow}{badge}</div>'
        f'<div class="msg-body">{inner}</div>'
        f'</div>'
    )


def render_example(orig, rew, ex_idx):
    orig_msgs = orig["messages"]
    rew_msgs = rew["messages"]
    opcodes = diff_opcodes(orig_msgs, rew_msgs)

    user_text = ""
    for m in orig_msgs:
        if m.get("role") == "user":
            user_text = (m.get("content") or "").strip()
            break

    rows_html = []
    n_removed_searches = 0
    n_replace_blocks = 0

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for k, l in zip(range(i1, i2), range(j1, j2)):
                left = render_message(orig_msgs[k], k, "normal")
                right = render_message(rew_msgs[l], l, "normal")
                rows_html.append(
                    f'<div class="diff-row row-equal">'
                    f'<div class="diff-cell">{left}</div>'
                    f'<div class="diff-cell">{right}</div>'
                    f'</div>'
                )
        else:
            n_replace_blocks += 1
            for k in range(i1, i2):
                for tc in (orig_msgs[k].get("tool_calls") or []):
                    if (tc.get("function") or {}).get("name") == "search":
                        n_removed_searches += 1
            left_inner = "".join(
                render_message(orig_msgs[k], k, "removed") for k in range(i1, i2)
            ) or '<div class="empty-side">(no original messages)</div>'
            right_inner = "".join(
                render_message(rew_msgs[k], k, "rewired") for k in range(j1, j2)
            ) or '<div class="empty-side">(no rewired messages)</div>'
            rows_html.append(
                f'<div class="diff-row row-replace">'
                f'<div class="diff-cell">{left_inner}</div>'
                f'<div class="diff-cell">{right_inner}</div>'
                f'</div>'
            )

    stats = {
        "n_removed_searches": n_removed_searches,
        "n_replace_blocks": n_replace_blocks,
        "msg_delta": len(orig_msgs) - len(rew_msgs),
        "user_text": user_text,
    }

    snippet = html.escape(user_text[:400])
    if len(user_text) > 400:
        snippet += "..."

    header = (
        f'<h2>Example #{ex_idx + 1} '
        f'<span class="example-stats">'
        f'searches removed: <b>{n_removed_searches}</b> | '
        f'rewired blocks: <b>{n_replace_blocks}</b> | '
        f'messages: {len(orig_msgs)} -&gt; {len(rew_msgs)}'
        f'</span></h2>'
    )
    column_headers = (
        '<div class="diff-row column-headers">'
        '<div class="diff-cell col-header col-original">ORIGINAL</div>'
        '<div class="diff-cell col-header col-rewired">REWIRED</div>'
        '</div>'
    )

    block = (
        f'<div class="example" id="ex-{ex_idx}">'
        f'{header}'
        f'<div class="question-box">{snippet}</div>'
        f'<div class="diff-grid">{column_headers}{"".join(rows_html)}</div>'
        f'</div>'
    )
    return block, stats


CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 0; padding: 16px 24px; background: #FAFAFA; color: #212121; }
h1 { margin-bottom: 8px; }
h2 { margin: 24px 0 8px; font-size: 1.05rem; color: #424242; }
.example-stats { font-size: 0.78rem; color: #666; font-weight: normal; margin-left: 12px; }
.toolbar { padding: 8px 0; margin-bottom: 12px; font-size: 0.85rem; color: #444; }
.toolbar code { background: #ECEFF1; padding: 1px 6px; border-radius: 3px; }
.toc { border-collapse: collapse; width: 100%; margin-bottom: 24px; font-size: 0.85rem; }
.toc th, .toc td { border: 1px solid #DDD; padding: 5px 9px; text-align: left; }
.toc th { background: #ECEFF1; }
.toc tr:nth-child(even) { background: #F5F5F5; }
.toc tr.changed { background: #FFF8E1 !important; }
.toc a { color: #1565C0; text-decoration: none; }
.toc a:hover { text-decoration: underline; }
.example { background: white; border: 1px solid #E0E0E0; border-radius: 8px;
           padding: 16px; margin-bottom: 28px; }
.question-box { font-size: 0.95rem; font-weight: 600; color: #1A237E;
                background: #E8EAF6; border-left: 4px solid #3949AB;
                padding: 8px 12px; border-radius: 4px; margin-bottom: 12px; }
.diff-grid { display: flex; flex-direction: column; gap: 8px; }
.diff-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.diff-cell { display: flex; flex-direction: column; gap: 6px; }
.col-header { padding: 6px 10px; background: #ECEFF1; font-weight: 700;
              letter-spacing: 0.06em; text-align: center; font-size: 0.78rem;
              border-radius: 4px; }
.col-original { color: #424242; }
.col-rewired  { color: #1B5E20; background: #E8F5E9; }
.row-replace { background: #FFF8E1; border-left: 4px solid #F9A825;
               padding: 8px; border-radius: 4px; }
.row-equal { padding: 2px 0; }
.empty-side { color: #BDBDBD; font-style: italic; font-size: 0.8rem;
              padding: 8px; border: 1px dashed #CFD8DC; border-radius: 4px; }
.message { border-radius: 6px; overflow: hidden; border: 1px solid transparent; }
.role-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.05em;
              padding: 4px 12px; display: flex; align-items: center; gap: 8px; }
.msg-body { padding: 8px 12px; }
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
.msg-state-removed { box-shadow: 0 0 0 2px #C62828; opacity: 0.92; }
.msg-state-rewired { box-shadow: 0 0 0 2px #2E7D32; }
.state-badge { font-size: 0.65rem; padding: 2px 7px; border-radius: 3px;
               margin-left: auto; letter-spacing: 0.05em; font-weight: 800; }
.badge-removed { background: #C62828; color: white; }
.badge-rewired { background: #2E7D32; color: white; }
.think-block { margin: 4px 0; border: 1px solid #CE93D8; border-radius: 4px;
               background: #FCE4EC; }
.think-block summary { padding: 5px 10px; cursor: pointer; font-size: 0.82rem;
                        font-weight: 600; color: #6A1B9A; user-select: none; }
.think-block summary:hover { background: #F8BBD0; }
.think-body { margin: 0; padding: 8px 12px; font-size: 0.82rem; white-space: pre-wrap;
              word-break: break-word; color: #4A148C; border-top: 1px solid #CE93D8; }
.think-bridge { border-color: #2E7D32; background: #E8F5E9; }
.think-bridge summary { color: #1B5E20; }
.think-bridge .think-body { color: #1B5E20; border-top-color: #81C784; }
.hint { font-weight: normal; color: #777; font-size: 0.74rem; }
.text-body { margin: 4px 0; font-size: 0.88rem; white-space: pre-wrap;
             word-break: break-word; background: transparent; }
.tool-call-block { margin: 4px 0; padding: 6px 10px; background: #EDE7F6;
                   border-radius: 4px; font-size: 0.85rem; }
.tool-call-label { font-weight: 700; color: #6A1B9A; margin-right: 6px; }
.query-highlight { background: #FFF176; font-weight: 700; padding: 2px 8px;
                   border-radius: 3px; color: #212121; }
.tool-call-removed .query-highlight { background: #FFCDD2; text-decoration: line-through;
                                       color: #B71C1C; }
.tool-call-removed { background: #FFEBEE; }
.raw-args { font-size: 0.8rem; color: #555; }
.call-id { font-size: 0.72rem; color: #888; margin-left: 8px; }
.tool-result-block { margin: 4px 0; border: 1px solid #FFCC80; border-radius: 4px; }
.tool-result-block summary { padding: 5px 10px; cursor: pointer; font-size: 0.82rem;
                              font-weight: 600; color: #E65100; }
.tool-result-block summary:hover { background: #FFE0B2; }
.tool-body { margin: 0; padding: 8px 12px; font-size: 0.8rem; white-space: pre-wrap;
             word-break: break-word; color: #BF360C; border-top: 1px solid #FFCC80; }
.empty { color: #BDBDBD; font-style: italic; font-size: 0.82rem; }
hr { border: none; border-top: 2px solid #EEE; margin: 28px 0; }
"""


def build_toc(rendered_meta):
    rows = []
    for ex_idx, stats in rendered_meta:
        cls = "changed" if stats["n_removed_searches"] > 0 else ""
        snippet = html.escape((stats.get("user_text") or "")[:90])
        rows.append(
            f'<tr class="{cls}">'
            f'<td><a href="#ex-{ex_idx}">{ex_idx + 1}</a></td>'
            f'<td>{snippet}</td>'
            f'<td>{stats["n_removed_searches"]}</td>'
            f'<td>{stats["n_replace_blocks"]}</td>'
            f'<td>{stats["msg_delta"]}</td>'
            f'</tr>'
        )
    header = (
        "<tr><th>#</th><th>Question</th>"
        "<th>Searches removed</th><th>Replace blocks</th><th>Msg delta</th></tr>"
    )
    return f'<table class="toc">{header}{"".join(rows)}</table>'


def build_page(blocks, n_total, changed_only, orig_path, rew_path):
    rendered_meta = [(i, stats) for i, (_h, stats) in enumerate(blocks)]
    toc = build_toc(rendered_meta)

    body_parts = []
    for i, (html_block, _stats) in enumerate(blocks):
        body_parts.append(html_block)
        if i < len(blocks) - 1:
            body_parts.append("<hr>")
    body = "\n".join(body_parts)

    n_changed = sum(1 for _, s in rendered_meta if s["n_removed_searches"] > 0)
    total_removed = sum(s["n_removed_searches"] for _, s in rendered_meta)
    mode = "changed only" if changed_only else "all"

    toolbar = (
        f'<div class="toolbar">'
        f'<b>Original:</b> <code>{html.escape(orig_path)}</code> | '
        f'<b>Rewired:</b> <code>{html.escape(rew_path)}</code><br>'
        f'Showing {len(blocks)}/{n_total} example(s) ({mode}) | '
        f'rewired examples: <b>{n_changed}</b> | '
        f'total searches removed: <b>{total_removed}</b>'
        f'</div>'
    )

    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<title>Rewired SFT Diff</title>\n'
        f'<style>\n{CSS}\n</style>\n'
        '</head>\n<body>\n'
        '<h1>SFT Rewire Diff Viewer</h1>\n'
        f'{toolbar}\n{toc}\n{body}\n'
        '</body>\n</html>'
    )


def main():
    p = argparse.ArgumentParser(description="Side-by-side diff of two parallel SFT JSONL files.")
    p.add_argument("--original", required=True)
    p.add_argument("--rewired", required=True)
    p.add_argument("--output", default="rewired_diff.html")
    p.add_argument("--all", action="store_true",
                   help="Include unchanged examples (default: changed only).")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    orig = load_jsonl(args.original)
    rew = load_jsonl(args.rewired)

    if len(orig) != len(rew):
        print(f"WARNING: line counts differ ({len(orig)} vs {len(rew)}). "
              f"Aligning by min length.", file=sys.stderr)
    n = min(len(orig), len(rew))

    blocks = []
    for i in range(n):
        html_block, stats = render_example(orig[i], rew[i], i)
        if not args.all and stats["n_removed_searches"] == 0:
            continue
        blocks.append((html_block, stats))
        if args.limit is not None and len(blocks) >= args.limit:
            break

    if not blocks:
        print("No examples to render (try --all to include unchanged).", file=sys.stderr)
        sys.exit(1)

    page = build_page(blocks, n_total=n, changed_only=not args.all,
                      orig_path=args.original, rew_path=args.rewired)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")

    print(f"Wrote {len(blocks)} example(s) to {out_path}")
    print(f"Open in browser: file://{out_path.resolve()}")


if __name__ == "__main__":
    main()
