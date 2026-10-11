"""Metadata-only preflight. No training or phenotype access is performed."""
import argparse
import json
from pathlib import Path
from .io import load_config, validate_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['validate'])
    parser.add_argument('--config', type=Path, default=Path('configs/funcprior.toml'))
    arguments = parser.parse_args()
    config, root = load_config(arguments.config)
    result = validate_inputs(config, root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'INPUTS_PRESENT_UNVERIFIED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
