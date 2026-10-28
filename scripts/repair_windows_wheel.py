"""Bundle Windows dependencies and reject DLLs without reviewed notices."""

from pathlib import Path
import base64
import csv
import hashlib
import io
import re
import subprocess
import sys
from zipfile import ZipFile


RUNTIME_LICENSES = 'ZPL-2.1 AND BSD-2-Clause AND ISC AND HPND AND SunPro'


def add_runtime_metadata(wheel):
    # Repair tools add runtime code, so the resulting wheel's license expression
    # must describe that code as well. Linux wheels keep their existing metadata.
    with ZipFile(wheel) as archive:
        entries = [(item, archive.read(item.filename)) for item in archive.infolist()]
    metadata_name, = [item.filename for item, _ in entries if item.filename.endswith('.dist-info/METADATA')]
    record_name = metadata_name.removesuffix('METADATA') + 'RECORD'
    updated = []
    records = []
    for item, data in entries:
        if item.filename == record_name:
            continue
        if item.filename == metadata_name:
            data, count = re.subn(
                rb'(?m)^(License-Expression: [^\r\n]*(?:\r?\n[ \t]+[^\r\n]*)*)',
                lambda match: match[1] + b' AND ' + RUNTIME_LICENSES.encode(), data, count=1,
            )
            if count != 1:
                raise ValueError('Missing license expression in wheel metadata')
        updated.append((item, data))
        if not item.is_dir():
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
            records.append((item.filename, 'sha256=' + digest, len(data)))
    records.append((record_name, '', ''))
    record = io.StringIO(newline='')
    csv.writer(record).writerows(records)
    replacement = wheel.with_suffix('.tmp')
    with ZipFile(replacement, 'w') as archive:
        for item, data in updated:
            archive.writestr(item, data)
        archive.writestr(record_name, record.getvalue())
    replacement.replace(wheel)


def check(wheel):
    with ZipFile(wheel) as archive:
        dlls = [Path(name).name.lower() for name in archive.namelist() if name.lower().endswith('.dll')]
        unreviewed = [name for name in dlls if not name.startswith('libwinpthread-1-')]
        if unreviewed:
            raise ValueError(f'Unreviewed DLL dependencies: {unreviewed}')
        notice_file, = [name for name in archive.namelist() if name.endswith('.dist-info/licenses/THIRD_PARTY_NOTICES.txt')]
        notices = archive.read(notice_file).decode()
        if 'COPYING.MinGW-w64-runtime.txt' not in notices or 'Lucent Technologies' not in notices:
            raise ValueError('Missing MinGW C-runtime notices')
        if dlls and ('Copyright (c) 2011 mingw-w64 project' not in notices or '(C) 2010 Lockless Inc.' not in notices):
            raise ValueError('Missing winpthreads license notices')
        print(f'Reviewed bundled DLLs: {dlls}')


if __name__ == '__main__':
    wheel, dest = map(Path, sys.argv[1:])
    subprocess.run([sys.executable, '-m', 'delvewheel', 'repair', '-w', str(dest), str(wheel)], check=True)
    repaired = dest / wheel.name
    add_runtime_metadata(repaired)
    check(repaired)
