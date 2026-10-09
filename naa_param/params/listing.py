"""``naa-param list-params`` — the imported libraries that carry parameters.

Kept as its own entry point for the command name; the listing itself is
``naa-param list-libraries``, narrowed to parameter sets.
"""

from __future__ import annotations

from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    from ..residues.listing import main_params

    return main_params(argv)
