import argparse
import asyncio

from src.commands import ask


def parse_args():
    parser = argparse.ArgumentParser(prog="main.py", description="AI-Fun CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="Ask a question to the LLM")
    ask_parser.add_argument("question", type=str, help="The question to ask")
    ask_parser.add_argument(
        "--model", "-m", type=str, default="qwen3:8b", help="Model to use"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "ask":
        asyncio.run(ask(args.question, args.model))
    else:
        print(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
