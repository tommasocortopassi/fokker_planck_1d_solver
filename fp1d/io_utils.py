"""Output helpers: a fresh timestamped run directory, plus JSON saving."""
from pathlib import Path
from datetime import datetime
import json


def create_run_directory(base='output'):
    """Create and return a fresh, uniquely-named directory for one run's
    outputs (plots, animation, raw data, parameters). The microsecond-
    resolution timestamp keeps directories unique even for runs started
    in quick succession.
    """
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_dir = Path(base) / f'run_{stamp}'
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(data, filename):
    """Write `data` as pretty-printed JSON, creating parent directories
    as needed."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
