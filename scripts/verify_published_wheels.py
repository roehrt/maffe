"""Verify that an index serves the exact wheel artifacts built by this run."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import tomllib
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def verify(index, wheel_dir):
    project = tomllib.loads(Path('pyproject.toml').read_text())['project']
    wheels = sorted(Path(wheel_dir).glob('*.whl'))
    if not wheels:
        raise ValueError('No wheel artifacts found')
    expected = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in wheels}
    host = {'pypi': 'pypi.org', 'testpypi': 'test.pypi.org'}[index]
    url = f"https://{host}/pypi/{project['name']}/{project['version']}/json"
    for attempt in range(6):
        try:
            with urlopen(url, timeout=20) as response:
                published = json.load(response)
        except HTTPError as error:
            if error.code != 404:
                raise
        except (URLError, TimeoutError):
            if attempt == 5:
                raise
        else:
            files = {f['filename']: f['digests']['sha256'] for f in published['urls']}
            for name, digest in expected.items():
                if name in files and files[name] != digest:
                    raise ValueError(f'{index}: published artifact differs from this build: {name}')
            if expected.keys() <= files.keys():
                for name in expected:
                    print(f'{index}: SHA256 verified: {name}')
                return
        if attempt < 5:
            time.sleep(10)
    raise RuntimeError(f'{index}: release artifacts were not available after six attempts')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index', choices=('testpypi', 'pypi'))
    parser.add_argument('wheel_dir', type=Path)
    args = parser.parse_args()
    verify(args.index, args.wheel_dir)
