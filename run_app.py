"""
Q-Diagnose Unified Application Launcher
========================================
Smart India Hackathon 2025 | Problem Statement ID: 26139
Theme: MedTech / BioTech / HealthTech | Organization: Egreen Quanta

Starts the FastAPI server (serving both the REST API and the React production dashboard)
and automatically launches your web browser.

Usage:
    python run_app.py
"""

import sys
import time
import webbrowser
import threading
from pathlib import Path

# Ensure root directory is on Python path
ROOT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

import database as db
from qdiagnose.settings import settings

BANNER = r"""
================================================================================
   ____            ____  _                                    
  / __ \          |  _ \(_)                                   
 | |  | |______   | | | |_  __ _  __ _ _ __   ___  ___  ___  
 | |  | |______|  | | | | |/ _` |/ _` | '_ \ / _ \/ __|/ _ \ 
 | |__| |         | |_| | | (_| | (_| | | | | (_) \__ \  __/ 
  \___\_\         |____/|_|\__,_|\__, |_| |_|\___/|___/\___| 
                                  __/ |                       
                                 |___/                        
  Hybrid Quantum-Classical AI Platform for Early Disease Risk Detection
  Smart India Hackathon 2025 -- Problem Statement ID: 26139
================================================================================
"""

DISCLAIMER = (
    "DISCLAIMER:\n"
    "Q-Diagnose is an investigational research and decision-support prototype.\n"
    "It is NOT a clinically validated medical diagnostic device and must NOT\n"
    "be used for primary clinical diagnosis or direct patient management.\n"
)


def open_browser(url: str, delay: float = 1.5):
    """Wait for server startup and open browser."""
    time.sleep(delay)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main():
    print(BANNER)
    print(DISCLAIMER)
    print("=" * 80)

    # 1. Initialize SQLite Database
    print("[1/3] Initializing local experimental database...")
    db.init_db()

    # 2. Check Static Frontend
    frontend_dist = ROOT_DIR / "frontend" / "dist"
    if frontend_dist.exists() and (frontend_dist / "index.html").exists():
        print(f"[2/3] Frontend build found at: {frontend_dist}")
    else:
        print(f"[2/3] WARNING: Frontend build not found at {frontend_dist}.")
        print("      To build the frontend, run: cd frontend && npm run build")

    # 3. Check registered datasets
    datasets = db.list_datasets()
    print(f"[3/3] Registered datasets: {len(datasets)} found.")

    host = settings.host
    port = settings.port
    url = f"http://{host}:{port}"

    print("-" * 80)
    print(f"[*] Interactive Dashboard : {url}/")
    print(f"[*] Health / Status API   : {url}/health")
    print("-" * 80)
    print("[*] Launching browser...")

    # Spawn thread to open browser
    threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    # Launch Uvicorn
    import uvicorn
    uvicorn.run("main:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
