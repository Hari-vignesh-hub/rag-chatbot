import ollama

messages = [
    {
        "role": "system",
        "content": "You are a helpful programming tutor. Explain concepts simply."
    }
]

print("AI Chatbot")
print("Type 'bye' to exit")

while True:

    user_input = input("You: ")

    if user_input.lower() == "bye":
        print("Bot: Goodbye!")
        break

    messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    response = ollama.chat(
        model="llama3.2",
        messages=messages
    )

    bot_response = response["message"]["content"]

    print("Bot:", bot_response)

    messages.append(
        {
            "role": "assistant",
            "content": bot_response
        }
    )