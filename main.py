import argparse
import asyncio

from src.commands import ask, chat_cli


def parse_args():
    parser = argparse.ArgumentParser(prog="main.py", description="AI-Fun CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="Ask a question to the LLM")
    ask_parser.add_argument("question", type=str, help="The question to ask")
    ask_parser.add_argument("-m", "--model", type=str, default="qwen3:8b", help="Model to use")
    ask_parser.add_argument(
        "-s", "--stream", dest="stream", action="store_true", help="Stream the response"
    )
    ask_parser.add_argument(
        "-ns", "--no-stream", dest="stream", action="store_false", help="Disable streaming"
    )
    ask_parser.add_argument(
        "-t", "--think", dest="think", action="store_true", help="Enable thinking"
    )
    ask_parser.add_argument(
        "-nt", "--no-think", dest="think", action="store_false", help="Disable thinking"
    )

    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat")
    chat_parser.add_argument("-m", "--model", type=str, default="qwen3:8b", help="Model to use")
    chat_parser.add_argument(
        "-sys",
        "--system",
        type=str,
        nargs="?",
        const="",
        default=None,
        help="System prompt (optional)",
    )
    chat_parser.add_argument(
        "-s", "--stream", dest="stream", action="store_true", help="Stream the response"
    )
    chat_parser.add_argument(
        "-ns", "--no-stream", dest="stream", action="store_false", help="Disable streaming"
    )
    chat_parser.add_argument(
        "-t", "--think", dest="think", action="store_true", help="Enable thinking"
    )
    chat_parser.add_argument(
        "-nt", "--no-think", dest="think", action="store_false", help="Disable thinking"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "ask":
        asyncio.run(ask(args.question, args.model, args.stream, args.think))
    elif args.command == "chat":
        asyncio.run(chat_cli(args.model, args.system, args.stream, args.think))
    else:
        print(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
