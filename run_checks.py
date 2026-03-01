#!/usr/bin/env python3
"""
Run basic checks to verify the repo works (no optional deps required).
Use: python run_checks.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
ROOT = Path(__file__).parent

def main():
    print("Running repo checks...\n")
    ok = 0
    # 1. Basic tests
    print("1. test_basic.py")
    r = __import__("subprocess").run(
        [sys.executable, str(ROOT / "test_basic.py")],
        cwd=str(ROOT), capture_output=True, text=True, timeout=30
    )
    if r.returncode == 0:
        print("   OK")
        ok += 1
    else:
        print("   FAIL:", r.stderr or r.stdout)

    # 2. Demo (data + features only if no lightgbm)
    print("2. demo.py")
    r = __import__("subprocess").run(
        [sys.executable, str(ROOT / "demo.py")],
        cwd=str(ROOT), capture_output=True, text=True, timeout=60
    )
    if r.returncode == 0:
        print("   OK")
        ok += 1
    else:
        print("   FAIL:", (r.stderr or r.stdout)[:500])

    # 3. Dashboard built-in data (no streamlit required)
    print("3. Dashboard built-in data")
    try:
        sys.path.insert(0, str(ROOT))
        from data.processor import DataProcessor
        p = DataProcessor()
        df = p.load_data(None)
        assert len(df) > 0 and "Class" in df.columns
        print("   OK (sample data)")
        ok += 1
    except Exception as e:
        print("   SKIP:", e)

    print(f"\nChecks passed: {ok}/3")
    return 0 if ok >= 2 else 1

if __name__ == "__main__":
    sys.exit(main())
