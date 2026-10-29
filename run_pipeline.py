"""Download the novels, retrieve evidence, train, and report test results."""

import sys

from backstory_checker.train import run


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run()


if __name__ == "__main__":
    main()
