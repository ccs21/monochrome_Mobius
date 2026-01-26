#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import subprocess

# Wrapper for convenience:
# python dump_original_lang_en_hashes.py "D:\GameRoot" [expect]
# Outputs hashes_original.* in current directory.

def main():
    root = Path(sys.argv[1]).resolve() if len(sys.argv) >= 2 else Path(".").resolve()
    expect = sys.argv[2] if len(sys.argv) >= 3 else None
    cmd = [sys.executable, str(Path(__file__).with_name("hash_dump.py")), "--mode", "original", "--root", str(root)]
    if expect:
        cmd += ["--expect", str(int(expect))]
    subprocess.check_call(cmd)

if __name__ == "__main__":
    main()
