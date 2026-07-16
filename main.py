from llm import ask_gemma
from store_history import load_conversation, save_conversation
from semantic_memory import (
    add_memory,
    extract_memories,
    search_memories,
)


system_message = {
    "role": "system",
    "content": (
        "Your name is Gemma. You are a friendly AI tutor. "
        "Keep your answers concise and easy to understand. "
        "Use a casual tone. Avoid technical jargon or complex language. "
        "Explain concepts using simple language."
    ),
}

messages = load_conversation(system_message)


print("=" * 50)
print("Beginner Gemma Chat")
print("Enter 'exit' to stop the program.")
print("=" * 50)


while True:
    user_input = input("\nYou: ").strip()

    if user_input.lower() in {"exit", "quit"}:
        print("Chat ended.")
        break

    if not user_input:
        print("Please enter a question.")
        continue

    # Search long-term memory for information related
    # to the current user message.
    relevant_memories = search_memories(user_input)

    # Add the user's actual message to conversation history.
    messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    # Create a temporary copy for the current model request.
    request_messages = messages.copy()

    # Give relevant long-term memories to Gemma.
    if relevant_memories:
        memory_text = "\n".join(
            f"- {memory}"
            for memory in relevant_memories
        )

        memory_message = {
            "role": "system",
            "content": (
                "Here is relevant information you remember about the user:\n"
                f"{memory_text}\n\n"
                "Use this only when it is relevant to the current question."
            ),
        }

        # Insert memory before the latest user message.
        request_messages.insert(
            len(request_messages) - 1,
            memory_message,
        )

    thinking_response = ""
    assistant_response = ""

    thinking_started = False
    content_started = False

    # Use request_messages instead of messages.
    for response_type, chunk in ask_gemma(request_messages):
        if response_type == "thinking":
            if not thinking_started:
                print("\nThinking:\n=========")
                thinking_started = True

            print(chunk, end="", flush=True)
            thinking_response += chunk

        elif response_type == "content":
            if not content_started:
                print("\n\nGemma: ", end="", flush=True)
                content_started = True

            print(chunk, end="", flush=True)
            assistant_response += chunk

        elif response_type == "error":
            print(f"\nError: {chunk}")

    print()

    # Do not add or save a blank assistant message if generation failed.
    if not assistant_response.strip():
        print("Gemma did not return an answer. Please try again.")
        messages.pop()
        continue

    # Keep thinking in the in-memory conversation context.
    messages.append(
        {
            "role": "assistant",
            "thinking": thinking_response,
            "content": assistant_response,
        }
    )

    # Your save function removes the thinking field
    # and stores only role and content.
    save_conversation(messages)

    # Extract useful long-term facts from the user's message.
    new_memories = extract_memories(user_input)

    # Embed and save each extracted fact.
    for memory in new_memories:
        add_memory(memory)
