"""
Comprehensive experiment runner for MC-SHAP implementations
Tests different configurations and saves results to CSV
"""
import time, joblib, numpy as np, pandas as pd, sys, os
from pathlib import Path
from datetime import datetime

# Get absolute paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL = PROJECT_ROOT / "models" / "model.pkl"
DATA = PROJECT_ROOT / "data" / "market_features.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)

def load_extensions():
    """Load all available extensions"""
    extensions = {
        'python': None,
        'openmp': None,
        'cuda': None
    }
    
    # Python baseline
    try:
        from mc_shap_python import mc_shap_batch
        extensions['python'] = mc_shap_batch
        print("✓ Python MC-SHAP loaded")
    except ImportError as e:
        print(f"✗ Python MC-SHAP failed: {e}")
    
    # OpenMP
    openmp_build = SCRIPT_DIR / "ext_openmp" / "build"
    if openmp_build.exists():
        sys.path.insert(0, str(openmp_build))
        try:
            import mc_shap_openmp
            extensions['openmp'] = mc_shap_openmp.mc_shap_openmp
            print(f"✓ OpenMP MC-SHAP loaded from {openmp_build}")
        except ImportError as e:
            print(f"✗ OpenMP MC-SHAP failed: {e}")
    
    # CUDA
    cuda_build = SCRIPT_DIR / "ext_cuda" / "build"
    if cuda_build.exists():
        sys.path.insert(0, str(cuda_build))
        try:
            from mc_shap_cuda import mc_shap_cuda_linear
            extensions['cuda'] = mc_shap_cuda_linear
            print(f"✓ CUDA MC-SHAP loaded from {cuda_build}")
        except ImportError as e:
            print(f"✗ CUDA MC-SHAP failed: {e}")
    
    return extensions

def run_experiment(impl_name, impl_func, f_model, X_test, X_bg, P, seed, 
                   lr_weights=None, n_blocks=128):
    """Run a single experiment"""
    try:
        if impl_name == 'python':
            t0 = time.time()
            phi = impl_func(f_model, X_test, X_bg, P=P, seed=seed)
            t1 = time.time()
            
        elif impl_name == 'openmp':
            t0 = time.time()
            phi = impl_func(f_model, X_test, X_bg, P, seed)
            t1 = time.time()
            
        elif impl_name == 'cuda':
            if lr_weights is None:
                return None
            W, b, baseline = lr_weights
            t0 = time.time()
            phi = impl_func(X_test.astype(np.float64), baseline, W, b, P, n_blocks, seed)
            t1 = time.time()
        
        elapsed = t1 - t0
        mean_abs = float(np.abs(phi).mean())
        
        return {
            'success': True,
            'time': elapsed,
            'phi': phi,
            'mean_abs_shap': mean_abs
        }
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None

def main():
    print("="*80)
    print("MC-SHAP Comprehensive Experiments")
    print("="*80)
    print(f"Hostname: {os.uname().nodename}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
    print()
    
    # Load model and data
    obj = joblib.load(MODEL)
    model = obj['model']
    df = pd.read_csv(DATA)
    print(f"Data shape: {df.shape}")
    
    X = df.drop(columns=['target']).values
    y = df['target'].values if 'target' in df.columns else None
    
    # Load extensions
    print("\nLoading extensions...")
    extensions = load_extensions()
    print()
    
    # Prepare linear model for CUDA
    lr_weights = None
    if extensions['cuda'] is not None and y is not None:
        from sklearn.linear_model import LogisticRegression
        print("Training logistic regression for CUDA...")
        lr = LogisticRegression(max_iter=1000).fit(X, y)
        W = lr.coef_.ravel().astype(np.float64)
        b = float(lr.intercept_[0])
        baseline = X[:512].mean(axis=0).astype(np.float64)
        lr_weights = (W, b, baseline)
        print("✓ Linear model ready\n")
    
    def f_model(Xin):
        return model.predict_proba(Xin)[:, 1]
    
    # Experiment configurations
    configs = [
        {'n_samples': 10, 'P': 64},
        {'n_samples': 50, 'P': 64},
        {'n_samples': 100, 'P': 64},
        {'n_samples': 500, 'P': 64},
        {'n_samples': 10, 'P': 128},
        {'n_samples': 10, 'P': 256},
    ]
    
    results = []
    
    for config in configs:
        n_samples = config['n_samples']
        P = config['P']
        
        print("="*80)
        print(f"Configuration: n_samples={n_samples}, P={P}")
        print("="*80)
        
        # Prepare data for this config
        X_test = X[:n_samples]
        X_bg = X[:512]
        
        print(f"Test: {X_test.shape}, Background: {X_bg.shape}\n")
        
        # Run each implementation
        for impl_name in ['python', 'openmp', 'cuda']:
            if extensions[impl_name] is None:
                print(f"{impl_name:8s}: SKIPPED (not loaded)")
                continue
            
            print(f"{impl_name:8s}: Running...", end=' ', flush=True)
            
            result = run_experiment(
                impl_name, 
                extensions[impl_name],
                f_model,
                X_test,
                X_bg,
                P,
                seed=0,
                lr_weights=lr_weights,
                n_blocks=128
            )
            
            if result is not None:
                print(f"{result['time']:.4f}s (mean |SHAP|: {result['mean_abs_shap']:.6f})")
                
                results.append({
                    'implementation': impl_name,
                    'n_samples': n_samples,
                    'n_features': X_test.shape[1],
                    'P': P,
                    'time_seconds': result['time'],
                    'mean_abs_shap': result['mean_abs_shap'],
                    'omp_threads': os.environ.get('OMP_NUM_THREADS', 'NA')
                })
                
                # Store phi for correlation later
                if impl_name == 'python':
                    python_phi = result['phi']
                elif impl_name == 'openmp' and 'python_phi' in locals():
                    corr = np.corrcoef(python_phi.ravel(), result['phi'].ravel())[0,1]
                    results[-1]['correlation_with_python'] = float(corr)
            else:
                print("FAILED")
        
        print()
    
    # Create results DataFrame
    df_results = pd.DataFrame(results)
    
    # Calculate speedups
    for config in df_results[['n_samples', 'P']].drop_duplicates().values:
        mask = (df_results['n_samples'] == config[0]) & (df_results['P'] == config[1])
        python_time = df_results[mask & (df_results['implementation'] == 'python')]['time_seconds']
        
        if len(python_time) > 0:
            baseline = python_time.values[0]
            for impl in ['openmp', 'cuda']:
                impl_time = df_results[mask & (df_results['implementation'] == impl)]['time_seconds']
                if len(impl_time) > 0:
                    speedup = baseline / impl_time.values[0]
                    df_results.loc[mask & (df_results['implementation'] == impl), 'speedup'] = speedup
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"experiments_{timestamp}.csv"
    df_results.to_csv(results_file, index=False)
    
    print("="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(df_results.to_string(index=False))
    print()
    print(f"✓ Results saved to: {results_file}")
    print()
    
    # Print speedup summary
    if 'speedup' in df_results.columns:
        print("SPEEDUP SUMMARY")
        print("-"*80)
        speedup_summary = df_results[df_results['speedup'].notna()].groupby('implementation')['speedup'].agg(['mean', 'min', 'max'])
        print(speedup_summary)
        print()

if __name__ == "__main__":
    main()