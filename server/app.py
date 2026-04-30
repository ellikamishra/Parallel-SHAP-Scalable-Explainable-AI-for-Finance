# server/app.py
from fastapi import FastAPI, BackgroundTasks, HTTPException, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlmodel import SQLModel, Session, create_engine, select
from .models import Run
from .benchmark import run_benchmark, hardware_info
from pathlib import Path
import logging
import traceback
import threading

# Serialize benchmark runs so overlapping submissions don't fight for CPU
# and inflate each other's runtimes. Rows are written with runtime_sec=0
# until the worker actually starts; the UI reads that as "queued".
_BENCH_LOCK = threading.Lock()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Parallel-SHAP Bench")
ENGINE = create_engine(f"sqlite:///{Path(__file__).with_name('db.sqlite3')}")
SQLModel.metadata.create_all(ENGINE)

static_dir = Path(__file__).with_name("static")
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(static_dir / "index.html")

def _parse_optional_int(value: str | None) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(value)

@app.post("/benchmark")
def benchmark(
    bg: BackgroundTasks,
    label: str = Form(...),
    backend: str = Form(...),
    P_raw: str = Form("64"),
    threads_raw: str | None = Form(None),
    N_raw: str | None = Form(None),
    dataset_name: str = Form("market_features.csv"),
    notes: str | None = Form(None),
):
    P = int(P_raw)
    threads = _parse_optional_int(threads_raw)
    N = _parse_optional_int(N_raw)
    
    logger.info(f"New benchmark request: backend={backend}, P={P}, threads={threads}, N={N}")
    
    # Create placeholder row
    with Session(ENGINE) as s:
        run = Run(
            label=label, 
            backend=backend, 
            dataset_name=dataset_name,
            params={"P": P, "threads": threads, "N": N},
            hardware=hardware_info(), 
            runtime_sec=0.0
        )
        s.add(run)
        s.commit()
        s.refresh(run)
        run_id = run.id
    
    logger.info(f"Created run ID {run_id}, starting background task...")

    def worker():
        try:
            logger.info(f"[Run {run_id}] Waiting for bench lock...")
            with _BENCH_LOCK:
                logger.info(f"[Run {run_id}] Starting benchmark execution...")
                rt, speedup, corr = run_benchmark(backend, P, N, threads, dataset_name)
            logger.info(f"[Run {run_id}] Completed: runtime={rt:.4f}s, speedup={speedup:.2f}x, corr={corr:.4f}")
            
            with Session(ENGINE) as s:
                r = s.get(Run, run_id)
                if r:  # Safety check
                    r.runtime_sec = float(rt)
                    r.speedup_vs_baseline = float(speedup)
                    r.fidelity_corr = float(corr) if corr is not None else None
                    r.notes = notes
                    s.add(r)
                    s.commit()
                    logger.info(f"[Run {run_id}] Results saved to database")
                else:
                    logger.error(f"[Run {run_id}] Run not found in database!")
                    
        except Exception as e:
            error_msg = f"ERROR: {str(e)}"
            logger.error(f"[Run {run_id}] {error_msg}")
            logger.error(f"[Run {run_id}] Traceback: {traceback.format_exc()}")
            
            with Session(ENGINE) as s:
                r = s.get(Run, run_id)
                if r:
                    r.notes = error_msg
                    r.runtime_sec = 0.0  # Mark as failed
                    s.add(r)
                    s.commit()

    bg.add_task(worker)
    return {"run_id": run_id, "status": "queued"}

@app.get("/runs")
def list_runs():
    with Session(ENGINE) as s:
        rows = s.exec(select(Run).order_by(Run.id.desc())).all()
        return rows

@app.get("/runs/{run_id}")
def get_run(run_id: int):
    with Session(ENGINE) as s:
        row = s.get(Run, run_id)
        if not row:
            raise HTTPException(404, "Run not found")
        return row

@app.delete("/runs/{run_id}")
def delete_run(run_id: int):
    """Delete a run"""
    with Session(ENGINE) as s:
        row = s.get(Run, run_id)
        if not row:
            raise HTTPException(404, "Run not found")
        s.delete(row)
        s.commit()
    return {"status": "deleted"}

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "ok", "hardware": hardware_info()}