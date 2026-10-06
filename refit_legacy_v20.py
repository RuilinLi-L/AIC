"""Refit a V19 winner with V20's frozen joint selection in a separate process."""
from __future__ import annotations

import sys


def main():
    import select_v19
    import train_v19
    from select_v20 import load_selection

    # This process-local dispatch changes no V19 files or existing processes.
    path = sys.argv[sys.argv.index('--selection-json') + 1]
    selected = load_selection(path)
    if selected['source_format_version'] != 19 or selected['recipe'] not in ('rank32_dropout', 'resolution384_rank32'):
        raise ValueError('legacy V20 refit requires an accepted V19 winner')
    original = select_v19.load_selection
    try:
        select_v19.load_selection = load_selection
        train_v19.main()
    finally:
        select_v19.load_selection = original


if __name__ == '__main__':
    main()
