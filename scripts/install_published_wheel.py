"""Install the exact release from one index, allowing for index propagation."""

import argparse
from pathlib import Path
import subprocess
import sys
import time
import tomllib


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index', choices=('testpypi', 'pypi'))
    args = parser.parse_args()
    project = tomllib.loads(Path('pyproject.toml').read_text())['project']
    host = {'testpypi': 'test.pypi.org', 'pypi': 'pypi.org'}[args.index]
    command = [sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--no-deps',
               '--only-binary=:all:', '--index-url', f'https://{host}/simple/',
               f'{project["name"]}=={project["version"]}']
    for attempt in range(6):
        result = subprocess.run(command)
        if result.returncode == 0:
            break
        if attempt == 5:
            raise SystemExit(result.returncode)
        time.sleep(10)
