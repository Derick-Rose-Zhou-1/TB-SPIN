import argparse
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--retraining', type=Path, required=True)
p.add_argument('--historical-source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
env = os.environ.copy()
env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'model' / 'result3') + os.pathsep + env.get('PYTHONPATH', '')
for seed in (2026091801, 2026091802, 2026091803):
    for variant in ('FULL', 'NO_ORDINAL_SINGLE_CELL_SUPERVISION'):
        subprocess.run([sys.executable, '-u', '-m', 'v126_dnra.run_outer_endpoint',
                        '--historical-source', str(a.historical_source),
                        '--compat-root', str(a.retraining / '98_ENDPOINT_ENGINE_COMPAT' / variant / f'seed_{seed}'),
                        '--inner-run', str(a.retraining / '05_ENDPOINT_INNER' / 'FULL' / f'seed_{seed}'),
                        '--output', str(a.output / f'R3_{variant}_seed{seed}')],
                       check=True, env=env)
