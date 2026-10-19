"""Verify exact m5 source bytes after cloning the dataset runtime branch."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
m=json.loads((root/'M5_SOURCE_MANIFEST.json').read_text())
for row in m['files']:
 path=root/row['path']
 assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],row['path']
print(json.dumps(dict(status='M5_RUNTIME_SOURCE_HASHES_VERIFIED',files=len(m['files']),original_source_commit=m['original_git_commit'],large_assets_verified=False,new_machine_simulation_verified=False)))
