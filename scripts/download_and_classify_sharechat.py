import json
import os
import argparse
import sys
from typing import Literal
from pydantic import BaseModel
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.base_agent import BaseAgent
from dotenv import load_dotenv

load_dotenv()

CLASSIFICATION_SYSTEM_PROMPT = """You are classifying user queries sent to an AI assistant.

Classify the following message as INFO_SEEKING if it requests factual information in any form:
- Factual lookups (who, what, when, where)
- How-to / step-by-step instructions for real-world tasks
- Explanations of concepts, facts, or phenomena
- Comparisons between factual entities
- Current events or news questions

Classify as NOT_INFO_SEEKING if the message is:
- Creative writing / storytelling / roleplay
- Casual chitchat or social greetings
- Pure code generation with no factual question
- Personal opinion / advice requests with no factual basis
- Task execution (translate this, summarize this text, etc.)
- Mathematical or logical calculations (e.g. probability, statistics, arithmetic word problems)
- Time-relative questions that depend on the current date to be answered (e.g. "Is Black Friday the day after tomorrow?", "What happened yesterday?", "How many days until Christmas?")

Also classify the message as answerable_in_paragraph: true if an AI assistant could fully address it in a short paragraph or less (1-4 sentences). Set to false for questions that require long lists, multi-step tutorials, code, tables, or extensive detail to answer properly."""

CLASSIFICATION_USER_TEMPLATE = "Message: {message}"


class InfoSeekingOutput(BaseModel):
    is_info_seeking: bool
    category: Literal["factual_lookup", "how_to", "explanation", "comparison", "current_events", "not_info_seeking"]
    answerable_in_paragraph: bool
    reasoning: str


def load_subset(subset_name: str, limit: int = None) -> list[dict]:
    from datasets import load_dataset
    print(f"Loading subset: {subset_name}")
    ds = load_dataset("tucnguyen/ShareChat", subset_name, split="train")
    total = len(ds)
    print(f"  Total rows: {total}")

    filtered = [
        {"text": row["plain_text"], "subset": subset_name}
        for row in ds
        if row.get("detected_language_final") == "English"
        and row.get("role") == "user"
        and row.get("message_index") == 1
    ]
    print(f"  After filter (en + user + index==1): {len(filtered)}")

    if limit is not None:
        filtered = filtered[:limit]
        print(f"  After --limit {limit}: {len(filtered)}")

    return filtered


def load_already_processed(output_path: str) -> set[str]:
    if not os.path.exists(output_path):
        return set()
    processed = set()
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                processed.add(obj["text"])
            except (json.JSONDecodeError, KeyError):
                pass
    return processed


def main():
    parser = argparse.ArgumentParser(description="Download and classify ShareChat info-seeking queries.")
    parser.add_argument("--output", default="data/sharechat_info_seeking.jsonl", help="Output JSONL file path.")
    parser.add_argument("--limit", type=int, default=None, help="Limit messages per subset (for testing).")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    # Load and filter all subsets
    all_messages = []
    for subset in ("claude", "gemini", "perplexity"):
        all_messages.extend(load_subset(subset, limit=args.limit))

    # Dedup by text (keep first occurrence across subsets)
    seen_texts: set[str] = set()
    deduped: list[dict] = []
    for msg in all_messages:
        if msg["text"] not in seen_texts:
            seen_texts.add(msg["text"])
            deduped.append(msg)
    print(f"\nAfter dedup: {len(deduped)} (removed {len(all_messages) - len(deduped)} duplicates)")

    # Filter out messages containing <REDACTED> tags
    clean = [m for m in deduped if "<REDACTED>" not in m["text"] and "<URL>" not in m["text"] and "<DATE_TIME>" not in m["text"]]
    print(f"After removing <REDACTED> & <URL> & <DATE_TIME> messages: {len(clean)} (removed {len(deduped) - len(clean)})")

    print(f"Total messages going into classification: {len(clean)}")

    # Resume support
    already_processed = load_already_processed(args.output)
    if already_processed:
        print(f"Resuming: skipping {len(already_processed)} already-classified messages")

    to_classify = [m for m in clean if m["text"] not in already_processed]
    print(f"Messages to classify: {len(to_classify)}")

    if not to_classify:
        print("Nothing to classify.")
        return

    # Initialize classifier
    classifier = BaseAgent(
        provider_name="Google",
        model_name="gemini-3-flash-preview",
        output_type=InfoSeekingOutput,
        agent_name="InfoSeekingClassifier",
        use_thinking=False,
        temperature=0,
        system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
    )

    info_seeking_count = 0
    not_info_seeking_count = 0

    with open(args.output, "a", encoding="utf-8") as f:
        for msg in tqdm(to_classify, desc="Classifying"):
            try:
                result = classifier.run(CLASSIFICATION_USER_TEMPLATE.format(message=msg["text"]))
                output = result.output
                record = {
                    "text": msg["text"],
                    "subset": msg["subset"],
                    "is_info_seeking": output.is_info_seeking,
                    "category": output.category,
                    "answerable_in_paragraph": output.answerable_in_paragraph,
                    "reasoning": output.reasoning,
                }
                if output.is_info_seeking:
                    info_seeking_count += 1
                else:
                    not_info_seeking_count += 1
            except Exception as e:
                print(f"\nError classifying message: {e}")
                record = {
                    "text": msg["text"],
                    "subset": msg["subset"],
                    "is_info_seeking": None,
                    "category": "not_info_seeking",
                    "answerable_in_paragraph": None,
                    "reasoning": f"Classification error: {e}",
                }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    print(f"\nClassification complete.")
    print(f"  Info-seeking:     {info_seeking_count}")
    print(f"  Not info-seeking: {not_info_seeking_count}")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
