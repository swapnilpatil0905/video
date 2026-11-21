"""Worker entry point: python -m video_platform.workers <name> request.json response.json

Workers run inside per-model virtualenvs. They must import only the standard library,
their own model dependencies, and ``video_platform.workers.common`` (no pydantic).
"""

from __future__ import annotations

import importlib
import json
import sys
import traceback


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: python -m video_platform.workers <worker> <request.json> <response.json>", file=sys.stderr)
        return 2
    name, request_path, response_path = sys.argv[1:]
    module = importlib.import_module(f"video_platform.workers.{name}")
    with open(request_path, encoding="utf-8") as handle:
        request = json.load(handle)
    try:
        response = module.run(request)
    except Exception:
        traceback.print_exc()
        return 1
    with open(response_path, "w", encoding="utf-8") as handle:
        json.dump(response, handle, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
