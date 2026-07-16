import ollama
from collections.abc import Iterator


MODEL_NAME = "gemma4:e4b"


def ask_gemma(messages: list[dict[str, str]]) -> Iterator[tuple[str, str]]:
    """Send a message to the local Gemma model."""

   
    response = ollama.chat(
        model=MODEL_NAME,
        messages = messages,
        stream=True,
        think=True,
    )


    for chunk in response:
        if chunk.message.thinking:
            yield "thinking", chunk.message.thinking


        if chunk.message.content:
            yield "content", chunk.message.content

    
    