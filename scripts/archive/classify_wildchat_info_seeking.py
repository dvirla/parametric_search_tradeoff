import csv
import json
import os
import sys
import argparse
from typing import Literal

from pydantic import BaseModel
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.base_agent import BaseAgent
from dotenv import load_dotenv

load_dotenv()

CLASSIFICATION_SYSTEM_PROMPT = """You are classifying user queries sent to an AI assistant.

Classify the following message as INFO_SEEKING only if it seeks substantive factual knowledge about the world:
- Factual lookups about real-world entities, events, places, or people (e.g. "Who invented the telephone?", "What is the capital of France?")
- How-to / step-by-step instructions for real-world tasks (e.g. "How do I change a tire?")
- Explanations of concepts, facts, or phenomena (e.g. "Why is the sky blue?", "How does HTTPS work?")
- Comparisons between factual entities (e.g. "What is the difference between TCP and UDP?")
- Current events or news questions about the world

Classify as NOT_INFO_SEEKING if the message is:
- Questions about the AI assistant itself: its identity, nature, capabilities, or limitations (e.g. "What are you?", "Who are you?", "What can you do?", "Are you an AI?", "What model are you?")
- Prompting or jailbreak attempts directed at the assistant
- Creative writing / storytelling / roleplay
- Casual chitchat, social greetings, or small talk
- Code generation, debugging, or any programming task — even if phrased as a question
- Personal opinion / advice requests with no factual basis
- Task execution (translate this, summarize this text, fix my grammar, etc.)
- Mathematical or logical calculations (e.g. probability, statistics, arithmetic word problems)
- Time-relative questions that depend on the current date to be answered
- Trivial or nonsensical messages with no substantive informational content

Respond with is_info_seeking: true for INFO_SEEKING and false for NOT_INFO_SEEKING.
Also provide a brief reasoning for your classification."""

CLASSIFICATION_USER_TEMPLATE = "Message: {message}"


class InfoSeekingOutput(BaseModel):
    is_info_seeking: bool
    category: Literal["factual_lookup", "how_to", "explanation", "comparison", "current_events", "not_info_seeking"]
    reasoning: str


def load_wildchat(limit: int = None) -> list[dict]:
    from datasets import load_dataset
    print("Loading allenai/WildChat dataset...")
    ds = load_dataset("allenai/WildChat", split="train")
    print(f"Total rows: {len(ds)}")

    filtered = []
    for row in ds:
        if row.get("model") != "gpt-4":
            continue
        if row.get("language") != "English":
            continue
        conversation = row.get("conversation", [])
        if not conversation:
            continue
        first_message = conversation[0]
        content = first_message.get("content", "").strip()
        if not content:
            continue
        filtered.append({
            "conversation_id": row.get("conversation_id", ""),
            "text": content,
        })

    print(f"After filter (model=gpt-4 + language=English + has content): {len(filtered)}")

    if limit is not None:
        filtered = filtered[:limit]
        print(f"After --limit {limit}: {len(filtered)}")

    return filtered


CSV_FIELDS = ["conversation_id", "text", "is_info_seeking", "category", "reasoning"]


def load_already_processed(output_path: str) -> set[str]:
    processed = set()

    # Load from legacy JSONL if it exists alongside the CSV
    jsonl_path = output_path[:-4] + ".jsonl" if output_path.endswith(".csv") else output_path
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    processed.add(json.loads(line)["conversation_id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    # Load from existing CSV output
    if output_path.endswith(".csv") and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("conversation_id"):
                    processed.add(row["conversation_id"])

    return processed


def main():
    parser = argparse.ArgumentParser(description="Classify WildChat (gpt-4 / English) first messages as info-seeking or not.")
    parser.add_argument("--output", default="data/wildchat_info_seeking.csv", help="Output CSV file path.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to classify (for testing).")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    rows = load_wildchat(limit=args.limit)

    already_processed = load_already_processed(args.output)
    if already_processed:
        print(f"Resuming: skipping {len(already_processed)} already-classified messages")

    to_classify = [r for r in rows if r["conversation_id"] not in already_processed]
    print(f"Messages to classify: {len(to_classify)}")

    if not to_classify:
        print("Nothing to classify.")
        return

    classifier = BaseAgent(
        provider_name="ollama",
        model_name="gemma4:31b",
        output_type=InfoSeekingOutput,
        agent_name="WildChatInfoSeekingClassifier",
        system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
    )

    info_seeking_count = 0
    not_info_seeking_count = 0

    write_header = not os.path.exists(args.output)
    with open(args.output, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for row in tqdm(to_classify, desc="Classifying"):
            try:
                result = classifier.run(CLASSIFICATION_USER_TEMPLATE.format(message=row["text"]))
                output = result.output
                record = {
                    "conversation_id": row["conversation_id"],
                    "text": row["text"],
                    "is_info_seeking": output.is_info_seeking,
                    "category": output.category,
                    "reasoning": output.reasoning,
                }
                if output.is_info_seeking:
                    info_seeking_count += 1
                else:
                    not_info_seeking_count += 1
            except Exception as e:
                print(f"\nError classifying conversation {row['conversation_id']}: {e}")
                record = {
                    "conversation_id": row["conversation_id"],
                    "text": row["text"],
                    "is_info_seeking": None,
                    "category": "not_info_seeking",
                    "reasoning": f"Classification error: {e}",
                }

            writer.writerow(record)
            f.flush()

    print(f"\nClassification complete.")
    print(f"  Info-seeking:     {info_seeking_count}")
    print(f"  Not info-seeking: {not_info_seeking_count}")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
