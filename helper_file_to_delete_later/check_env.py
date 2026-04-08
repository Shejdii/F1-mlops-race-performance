import sys

print("Python exe:", sys.executable)
print("sys.path:")
for p in sys.path:
    print("  ", p)

try:
    import mylib
    import utils
    print("\n[OK] mylib & utils import work")
except Exception as e:
    print("\n[ERROR] Import failed:", e)
