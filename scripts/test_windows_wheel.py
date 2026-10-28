"""Test the installed wheel without compiler DLLs or source files on search paths."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    if sys.platform != 'win32':
        raise RuntimeError('This check requires Windows')
    test_file = Path(sys.argv[1]).resolve(strict=True)
    env = os.environ.copy()
    # Python itself is located by its absolute path, and its own DLLs are next to it.
    # Leave only Windows system paths, excluding MinGW, MSYS2 and other toolchains.
    windows = Path(env['SystemRoot'])
    env['PATH'] = os.pathsep.join(map(str, (windows / 'System32', windows)))
    with tempfile.TemporaryDirectory() as test_dir:
        subprocess.run(
            [sys.executable, '-I', str(test_file)],
            cwd=test_dir,
            env=env,
            check=True,
        )


if __name__ == '__main__':
    main()
