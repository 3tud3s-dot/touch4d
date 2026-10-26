"""Narrow P2 source-layout migration; never ignore an arbitrary source mismatch.

Only the frozen pre-refactor source and the reviewed post-move file inventory
are equivalent for fixed400 resume. New runs hash every experiment implementation
as well. Future edits invalidate this bridge and need their own explicit review.
"""

import hashlib
import json
from pathlib import Path


BASE_COMMIT = 'fdd4013c7f166daae12379f4ee020b4da5724965'
LEGACY_SOURCE_HASHES = {'f06b08c33b7dbf418589492e3794cb284c6bcd3670953072bc2b27d33ceb9b57': '05f90c6ce0aa89d4c3573b6287dead9ed9534f56', '6cfda5dafab664d67ab9b9b3162c03d7ad7b528ec091eaa377ead4dab02e4fda': 'fdd4013c7f166daae12379f4ee020b4da5724965'}
MIGRATED_FILES_SHA256 = '38e4ce38b974d68e87ad7d50a99e7e80eb36ca9fd119b349cc1449265755c4ec'
SELF_PATH = 'scripts/experiments/fixed_window/provenance.py'
DIRECTORIES = ('training', 'dataloader', 'data_process', 'experiments',
               'experiments/fixed_window', 'experiments/oracle')


def source_files(project):
    return {path.relative_to(project).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in DIRECTORIES
            for path in sorted((Path(project)/'scripts'/directory).glob('*.py'))
            if not path.name.startswith('test_')}


def normalized_inventory(project):
    return {path.relative_to(project).as_posix(): hashlib.sha256(
                path.read_text(encoding='utf-8').encode()).hexdigest()
            for directory in DIRECTORIES
            for path in sorted((Path(project)/'scripts'/directory).glob('*.py'))
            if not path.name.startswith('test_') and path.relative_to(project).as_posix() != SELF_PATH}


def inventory_digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def resume_config_matches(saved, current, project):
    if saved == current:
        return True
    if {k: v for k, v in saved.items() if k != 'source_sha256'} != {
            k: v for k, v in current.items() if k != 'source_sha256'}:
        return False
    return (saved.get('source_sha256') in LEGACY_SOURCE_HASHES
            and inventory_digest(normalized_inventory(project)) == MIGRATED_FILES_SHA256)
