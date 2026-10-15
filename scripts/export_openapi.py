"""Export the API schema offline without credentials, models or services."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_demo.production.api import create_app
from rag_demo.production.config import ProductionSettings


def build_schema() -> dict:
    settings = ProductionSettings(
        _env_file=None,
        RAG_ENV="test", RAG_DATABASE_URL="sqlite+pysqlite:///:memory:",
        RAG_AUTH_MODE="dev_hs256", RAG_DEV_JWT_SECRET="offline-schema-only-never-used-for-serving",
        RAG_REDIS_URL=None, RAG_MODEL="ollama:qwen2.5:7b", RAG_ALLOWED_MODELS="",
        RAG_SERVICE_VERSION="0.2.0",
    )
    app = create_app(settings=settings, repository=object(), pipeline=object())
    schema = app.openapi()
    schema["servers"] = [{"url": "https://{domain}", "variables": {"domain": {"default": "rag.example.com"}}}]
    return schema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "openapi.json")
    parser.add_argument("--check", action="store_true", help="Fail if the committed schema differs")
    args = parser.parse_args()
    content = json.dumps(build_schema(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != content:
            raise SystemExit("OpenAPI schema is stale; run python scripts/export_openapi.py")
    else:
        args.output.write_text(content, encoding="utf-8")
        print(f"Exported {args.output.name}")


if __name__ == "__main__":
    main()
