import time, joblib, numpy as np, pandas as pd, sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL = PROJECT_ROOT / "models" / "model.pkl"
DATA = PROJECT_ROOT / "data" / "market_features.csv"

print(f"Project root: {PROJECT_ROOT}")
print(f"Script directory: {SCRIPT_DIR}")

def ensure_model():
    try:
        obj = joblib.load(MODEL)
    except FileNotFoundError:
        print("Model artifact missing, training a fresh model via train_model.py")
        from train_model import main as train_model_main
        train_model_main()
        obj = joblib.load(MODEL)
    except ValueError as e:
        if "node array from the pickle has an incompatible dtype" in str(e):
            print("Existing model pickle incompatible with current scikit-learn; retraining.")
            from train_model import main as train_model_main
            train_model_main()
            obj = joblib.load(MODEL)
        else:
            raise
    return obj

def main():
    print("="*60)
    print("MC-SHAP Evaluation")
    print("="*60)
    
    obj = ensure_model()
    model = obj['model']
    df = pd.read_csv(DATA)
    print(f"Data shape: {df.shape}")
    
    X = df.drop(columns=['target']).values
    y = df['target'].values if 'target' in df.columns else None
    X_bg = X if len(X) < 512 else X[:512]
    X_test = X

    print(f"Test samples: {X_test.shape[0]}, Features: {X_test.shape[1]}")
    print(f"Background samples: {X_bg.shape[0]}")
    print()

    def f_model(Xin):
        """Model function that handles both single and batch predictions"""
        return model.predict_proba(Xin)[:, 1]

    # Python Baseline
    print("="*60)
    print("Python MC-SHAP (Baseline)")
    print("="*60)
    from mc_shap_python import mc_shap_batch
    
    t0 = time.time()
    phi_py = mc_shap_batch(f_model, X_test, X_bg, P=64, seed=0)
    t1 = time.time()
    time_py = t1 - t0
    print(f"Time: {time_py:.4f}s")
    print(f"Shape: {phi_py.shape}")
    print(f"Mean |SHAP|: {np.abs(phi_py).mean():.6f}")
    print()

    # OpenMP Extension
    print("="*60)
    print("OpenMP MC-SHAP")
    print("="*60)
    
    openmp_build = SCRIPT_DIR / "ext_openmp" / "build"
    print(f"Looking in: {openmp_build}")
    
    if openmp_build.exists():
        so_files = list(openmp_build.glob("*.so"))
        print(f"Found .so files: {[f.name for f in so_files]}")
        
        if so_files:
            sys.path.insert(0, str(openmp_build))
            try:
                import mc_shap_openmp
                print(f"✓ Successfully imported mc_shap_openmp")
                
                # Check if fast version exists
                has_fast = hasattr(mc_shap_openmp, 'mc_shap_openmp_fast')
                
                if has_fast:
                    print("  Using FAST version (batch predictions)")
                    t0 = time.time()
                    phi_omp = mc_shap_openmp.mc_shap_openmp_fast(f_model, X_test, X_bg, 64, 0)
                    t1 = time.time()
                else:
                    print("  ⚠ Using SLOW version (you should rebuild with the optimized code)")
                    print("  This will be VERY slow due to GIL contention")
                    print("  Expected time: ~300-500s (slower than Python!)")
                    print("  Skipping OpenMP to save time...")
                    print()
                    print("  To fix: Replace mc_shap_openmp.cpp with the optimized version")
                    print("  Then run: cd src/ext_openmp && ./build.sh")
                    phi_omp = None
                    time_omp = None
                
                if phi_omp is not None:
                    time_omp = t1 - t0
                    print(f"Time: {time_omp:.4f}s")
                    print(f"Shape: {phi_omp.shape}")
                    print(f"Mean |SHAP|: {np.abs(phi_omp).mean():.6f}")
                    print(f"Speedup: {time_py/time_omp:.2f}x")
                    
                    corr = float(np.corrcoef(phi_py.ravel(), phi_omp.ravel())[0,1])
                    print(f"Correlation with Python: {corr:.6f}")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("✗ No .so files found")
    else:
        print(f"✗ Build directory does not exist")
    print()

    # CUDA Extension
    print("="*60)
    print("CUDA MC-SHAP")
    print("="*60)
    
    cuda_build = SCRIPT_DIR / "ext_cuda" / "build"
    print(f"Looking in: {cuda_build}")
    
    if cuda_build.exists():
        so_files = list(cuda_build.glob("*.so"))
        print(f"Found .so files: {[f.name for f in so_files]}")
        
        if so_files and y is not None:
            sys.path.insert(0, str(cuda_build))
            try:
                from mc_shap_cuda import mc_shap_cuda_linear
                print(f"✓ Successfully imported mc_shap_cuda")
                
                from sklearn.linear_model import LogisticRegression
                print("Training logistic regression baseline...")
                lr = LogisticRegression(max_iter=1000).fit(X, y)
                W = lr.coef_.ravel().astype(np.float64)
                b = float(lr.intercept_[0])
                baseline = X_bg.mean(axis=0).astype(np.float64)
                
                t0 = time.time()
                phi_cuda = mc_shap_cuda_linear(
                    X_test.astype(np.float64), 
                    baseline, 
                    W, b, 
                    64, 128, 0
                )
                t1 = time.time()
                time_cuda = t1 - t0
                
                print(f"Time: {time_cuda:.4f}s")
                print(f"Shape: {phi_cuda.shape}")
                print(f"Mean |SHAP|: {np.abs(phi_cuda).mean():.6f}")
                print(f"Speedup: {time_py/time_cuda:.2f}x")
                print("(Note: Uses linear model, not tree model)")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                import traceback
                traceback.print_exc()
        elif y is None:
            print("✗ Cannot run: no target variable")
        else:
            print("✗ No .so files found")
    else:
        print(f"✗ Build directory does not exist")
    print()
    
    print("="*60)
    print("Evaluation Complete")
    print("="*60)

if __name__ == "__main__":
    main()