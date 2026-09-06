"""Fetch the unchanged canonical evaluator without embedding it in kernel source."""

import hashlib
import json
import os
import tarfile
import urllib.request
from pathlib import Path

manifest = json.loads(Path("evaluation-source.json").read_text())
archive = Path("source.tar.gz")
# The signed private URL rides the untracked .env, never a logged command.
urllib.request.urlretrieve(os.environ["JHTR_SOURCE_URL"], archive)
assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["sha256"]
with tarfile.open(archive) as source:
    source.extractall(Path.cwd(), filter="data")
print("Canonical source:", manifest["source_commit"], manifest["sha256"], flush=True)
