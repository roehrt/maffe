"""Bundle Windows dependencies and reject DLLs without reviewed notices."""

from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile


def check(wheel):
    with ZipFile(wheel) as archive:
        dlls = [Path(name).name.lower() for name in archive.namelist() if name.lower().endswith('.dll')]
        unreviewed = [name for name in dlls if not name.startswith('libwinpthread-1-')]
        if unreviewed:
            raise ValueError(f'Unreviewed DLL dependencies: {unreviewed}')
        notice_file, = [name for name in archive.namelist() if name.endswith('.dist-info/licenses/THIRD_PARTY_NOTICES.txt')]
        notices = archive.read(notice_file).decode()
        if dlls and ('Copyright (c) 2011 mingw-w64 project' not in notices or '(C) 2010 Lockless Inc.' not in notices):
            raise ValueError('Missing winpthreads license notices')
        print(f'Reviewed bundled DLLs: {dlls}')


if __name__ == '__main__':
    wheel, dest = map(Path, sys.argv[1:])
    subprocess.run([sys.executable, '-m', 'delvewheel', 'repair', '-w', str(dest), str(wheel)], check=True)
    check(dest / wheel.name)
