"""Minimal REPL for interacting with StataAgent."""
from pydantic_ai.messages import ModelMessage
from agent import agent, StataContext


def main():
    ctx = StataContext()
    history: list[ModelMessage] = []
    print("StataAgent ready. Type your question, or 'quit' to exit.\n")

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue

        # run_sync blocks until the agent finishes. For a CLI this is fine.
        # For a UI later, use agent.run() (async) or agent.run_stream().
        result = agent.run_sync(query, deps=ctx, message_history=history)
        history = result.all_messages()
        print(f"\n{result.data}\n")


if __name__ == "__main__":
    main()