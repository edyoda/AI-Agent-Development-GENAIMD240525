import argparse
import sys

from src.downstream.agent import run_downstream
from src.upstream.graph import run_upstream


def handle_upstream(args):
    run_upstream(args.doc_path)


def handle_downstream(args):
    response = run_downstream(args.query)
    print(f"\nResponse:\n{response}")


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # upstream subcommand
    up_parser = subparsers.add_parser("upstream", help="Process and upload a PDF document to the vector database")
    up_parser.add_argument("doc_path", help="Path to the PDF document")
    up_parser.set_defaults(func=handle_upstream)

    # downstream subcommand
    down_parser = subparsers.add_parser("downstream", help="Query the RAG system")
    down_parser.add_argument("query", help="User query to search documents")
    down_parser.set_defaults(func=handle_downstream)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
