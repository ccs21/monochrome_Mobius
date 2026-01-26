#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import subprocess

# Wrapper for convenience:
# python dump_patch_hashes.py "D:\PatchFolder" [expect]
# Outputs hashes_patch.* in current directory.

def main():
    patch_dir = Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 else (Path(".").resolve() / "patch")
    expect = sys.argv[2] if len(sys.argv) >= 3 else None
    cmd = [sys.executable, str(Path(__file__).with_name("hash_dump.py")), "--mode", "patch", "--patch", str(patch_dir)]
    if expect:
        cmd += ["--expect", str(int(expect))]
    subprocess.check_call(cmd)

if __name__ == "__main__":
    main()
