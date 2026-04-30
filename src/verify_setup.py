"""
Quick verification script to check if everything is set up correctly
Run this before running experiments
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

print("="*60)
print("MC-SHAP Setup Verification")
print("="*60)
print()

# Check files
print("1. Checking project structure...")
required_files = [
    ('models/model.pkl', 'Model file'),
    ('data/market_features.csv', 'Data file'),
    ('src/mc_shap_python.py', 'Python baseline'),
    ('src/ext_openmp/mc_shap_openmp.cpp', 'OpenMP source'),
    ('src/ext_cuda/mc_shap_cuda.cu', 'CUDA source'),
]

all_files_ok = True
for path, desc in required_files:
    full_path = PROJECT_ROOT / path
    if full_path.exists():
        print(f"  ✓ {desc}: {path}")
    else:
        print(f"  ✗ {desc}: {path} MISSING")
        all_files_ok = False

print()

# Check builds
print("2. Checking compiled extensions...")
openmp_build = SCRIPT_DIR / "ext_openmp" / "build"
cuda_build = SCRIPT_DIR / "ext_cuda" / "build"

openmp_ok = False
cuda_ok = False

if openmp_build.exists():
    so_files = list(openmp_build.glob("*.so"))
    if so_files:
        print(f"  ✓ OpenMP: found {len(so_files)} .so file(s)")
        for f in so_files:
            print(f"      - {f.name}")
        openmp_ok = True
    else:
        print(f"  ✗ OpenMP: build directory exists but no .so files")
else:
    print(f"  ✗ OpenMP: build directory does not exist")

if cuda_build.exists():
    so_files = list(cuda_build.glob("*.so"))
    if so_files:
        print(f"  ✓ CUDA: found {len(so_files)} .so file(s)")
        for f in so_files:
            print(f"      - {f.name}")
        cuda_ok = True
    else:
        print(f"  ✗ CUDA: build directory exists but no .so files")
else:
    print(f"  ✗ CUDA: build directory does not exist")

print()

# Try importing
print("3. Testing imports...")

# Python baseline
try:
    sys.path.insert(0, str(SCRIPT_DIR))
    from mc_shap_python import mc_shap_batch
    print("  ✓ Python MC-SHAP imports successfully")
except ImportError as e:
    print(f"  ✗ Python MC-SHAP import failed: {e}")

# OpenMP
if openmp_ok:
    try:
        sys.path.insert(0, str(openmp_build))
        import mc_shap_openmp
        print(f"  ✓ OpenMP extension imports successfully")
        print(f"      Module: {mc_shap_openmp.__file__}")
    except ImportError as e:
        print(f"  ✗ OpenMP extension import failed: {e}")
else:
    print("  ⊘ OpenMP: skipped (not built)")

# CUDA
if cuda_ok:
    try:
        sys.path.insert(0, str(cuda_build))
        from mc_shap_cuda import mc_shap_cuda_linear
        print(f"  ✓ CUDA extension imports successfully")
    except ImportError as e:
        print(f"  ✗ CUDA extension import failed: {e}")
else:
    print("  ⊘ CUDA: skipped (not built)")

print()

# Environment
print("4. Checking environment...")
import os
print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'NOT SET')}")
print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
print(f"  Hostname: {os.uname().nodename}")

print()

# Summary
print("="*60)
print("SUMMARY")
print("="*60)

if all_files_ok and (openmp_ok or cuda_ok):
    print("✓ Setup looks good! Ready to run experiments.")
    print()
    print("Next steps:")
    print("  1. Run quick test:     python src/mc_shap_eval.py")
    print("  2. Run all experiments: python src/run_experiments.py")
else:
    print("⚠ Setup incomplete. Issues found:")
    if not all_files_ok:
        print("  - Some required files are missing")
    if not openmp_ok:
        print("  - OpenMP extension not built")
    if not cuda_ok:
        print("  - CUDA extension not built")
    print()
    print("Build extensions with:")
    print("  cd src/ext_openmp && ./build.sh")
    print("  cd src/ext_cuda && ./build.sh")

print()