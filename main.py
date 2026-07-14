import argparse
import json
import logging
import sys

from pipeline import run_pipeline


def build_parser():
    p = argparse.ArgumentParser(
        prog="legal-contract-intelligence",
        description="LLM-powered legal contract clause extraction and summarization (CUAD dataset)",
    )
    p.add_argument("--n", type=int, default=50, metavar="N",
                   help="Number of contracts to process (default: 50)")
    p.add_argument("--pdf-dir", metavar="PATH",
                   help="Path to local PDF directory (default: use HuggingFace CUAD)")
    p.add_argument("--output-dir", default="output", metavar="PATH",
                   help="Output directory for results (default: output/)")
    p.add_argument("--no-search", action="store_true",
                   help="Skip semantic search index building")
    p.add_argument("--search", metavar="QUERY",
                   help="Run semantic search after pipeline")
    p.add_argument("--clause-type",
                   choices=["termination_clause", "confidentiality_clause", "liability_clause"],
                   help="Filter search results by clause type")
    p.add_argument("--top-k", type=int, default=5, metavar="K",
                   help="Number of search results to return (default: 5)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity (default: INFO)")
    return p


def main():
    args = build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    enable_search = not args.no_search

    results = run_pipeline(
        n_contracts=args.n,
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
        enable_search=enable_search,
    )

    print(f"\nProcessed {len(results)} contracts")
    print(f"Results: {args.output_dir}/results.csv")
    print(f"         {args.output_dir}/results.json")

    if args.search and enable_search:
        print(f'\nSearching: "{args.search}"')
        try:
            from search import ClauseSearchEngine
            engine = ClauseSearchEngine()
            engine.index(results)
            hits = engine.search(query=args.search, top_k=args.top_k,
                                 clause_type=args.clause_type)
            if hits:
                print(json.dumps(hits, indent=2, ensure_ascii=False))
            else:
                print("No results found.")
        except Exception as exc:
            print(f"Search error: {exc}")
    elif args.search and not enable_search:
        print("Note: --search requires semantic search (remove --no-search)")


if __name__ == "__main__":
    main()
