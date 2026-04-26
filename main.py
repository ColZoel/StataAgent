"""Minimal REPL for interacting with StataAgent (Anthropic/Claude backend)."""
import argparse
from pydantic_ai.messages import ModelMessage
from agent import agent, StataContext


def main() -> None:
    parser = argparse.ArgumentParser(
        description="StataAgent — Anthropic/Claude edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python main.py\n"
            "  python main.py --reset   # re-run Stata installation discovery"
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the saved Stata installation config and re-run discovery",
    )
    args = parser.parse_args()

    if args.reset:
        from config import reset_saved_config
        reset_saved_config()

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