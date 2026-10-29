"""Select wheel builds and index tests for branch, tag and manual CI runs."""

import json
import os
import re
from pathlib import Path
import tomllib


PLATFORMS = {
    'linux': {'name': 'linux-x86_64', 'runner': 'ubuntu-24.04', 'identifier': 'manylinux_x86_64'},
    'windows': {'name': 'windows-x86_64', 'runner': 'windows-2022', 'identifier': 'win_amd64'},
}
PYTHONS = ['3.11', '3.12', '3.13', '3.14']


def plan(event_name, ref_type, ref_name, event, version):
    scope = 'all'
    release = event_name == 'push' and ref_type == 'tag'
    inputs = event.get('inputs', {}) if event_name == 'workflow_dispatch' else {}
    if release:
        choices = {f'v{version}': 'all', **{f'v{version}-{key}': key for key in PLATFORMS}}
        # A fresh retry tag keeps previous release tags immutable.
        release_name = re.sub(r'-r[1-9][0-9]*$', '', ref_name)
        if release_name not in choices:
            raise ValueError(f'Release tag {ref_name!r} does not match version {version}')
        scope = choices[release_name]
    elif event_name == 'workflow_dispatch':
        scope = inputs.get('platforms', 'all')
    elif event_name == 'push' and '[windows-only]' in (event.get('head_commit') or {}).get('message', ''):
        scope = 'windows'
    if scope not in ('all', *PLATFORMS):
        raise ValueError(f'Unknown platform selection: {scope}')
    selected = list(PLATFORMS.values()) if scope == 'all' else [PLATFORMS[scope]]
    publish_pypi = release or str(inputs.get('publish_pypi', 'false')).lower() == 'true'
    publish_testpypi = publish_pypi or str(inputs.get('publish_testpypi', 'false')).lower() == 'true'
    return {
        'build_matrix': {'python': ['cp' + v.replace('.', '') for v in PYTHONS], 'platform': selected},
        'test_matrix': {'include': [dict(platform=p['name'], runner=p['runner'], python=v) for p in selected for v in PYTHONS]},
        'experimental': scope == 'all',
        'publish_testpypi': publish_testpypi,
        'publish_pypi': publish_pypi,
    }


if __name__ == '__main__':
    event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
    version = tomllib.loads(Path('pyproject.toml').read_text())['project']['version']
    result = plan(os.environ['GITHUB_EVENT_NAME'], os.environ.get('GITHUB_REF_TYPE', ''),
                  os.environ.get('GITHUB_REF_NAME', ''), event, version)
    with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
        for key, value in result.items():
            output.write(f'{key}={json.dumps(value, separators=(",", ":"))}\n')
    print(json.dumps(result, indent=2))
