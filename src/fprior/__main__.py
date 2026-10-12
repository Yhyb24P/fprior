"""Metadata preflight and guarded development diagnostics/evaluation."""
import argparse
import json
from pathlib import Path
from .io import load_config, validate_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['validate', 'diagnose', 'profile', 'evaluate'])
    parser.add_argument('--config', type=Path, default=Path('configs/funcprior.toml'))
    parser.add_argument('--resume', type=Path, help='Continue an existing evaluate directory after strict identity checks')
    parser.add_argument('--reuse-profile', type=Path, help='Reuse one complete real-protein profile fold after strict checks')
    arguments = parser.parse_args()
    if (arguments.resume or arguments.reuse_profile) and arguments.command != 'evaluate':
        parser.error('--resume and --reuse-profile require evaluate')
    config, root = load_config(arguments.config)
    if not isinstance(config.get('primary_trait_id'), str) or not config['primary_trait_id'].strip():
        print(json.dumps(validate_inputs(config, root), ensure_ascii=False, indent=2))
        return 2
    if arguments.command != 'validate':
        from .runner import run
        try:
            result = run(arguments.command, config, root, resume=arguments.resume,
                         reuse_profile=arguments.reuse_profile)
        except ValueError as exc:
            print(json.dumps({'status': 'BLOCKED_OR_FAILED', 'error': str(exc)}, ensure_ascii=False))
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    result = validate_inputs(config, root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'INPUTS_PRESENT_UNVERIFIED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
