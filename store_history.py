import json
from pathlib import Path


HISTORY_FILE = Path("chat history/conversation_history.json")


def load_conversation(system_message: dict[str, str]) -> list[dict[str, str]]:
    if not HISTORY_FILE.exists():
        return [system_message]

    with open(HISTORY_FILE, "r", encoding="utf-8") as file:
        return json.load(file)




def save_conversation(messages: list[dict]) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    saved_messages = [
        {
            "role": message["role"],
            "content": message["content"],
        }
        for message in messages
    ]

    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(saved_messages, file, indent=4)
