import subprocess
import os
import sys
import time

def start_backend():
    
    print("Starting backend...")
    return subprocess.Popen([
        sys.executable, "-m", "uvicorn", "backend.app:app", "--reload", "--host", "127.0.0.1", "--port", "8000"
    ])

def start_frontend():
    
    print("Starting frontend...")
    frontend_dir = os.path.join(os.getcwd(), "frontend")
    return subprocess.Popen([sys.executable, "-m", "http.server", "5500"], cwd=frontend_dir)

if __name__ == "__main__":
    try:
        # Step 1: Start backend
        backend_process = start_backend()
        time.sleep(2)  # Give the backend some time to initialize

        # Step 2: Start frontend
        frontend_process = start_frontend()

        print("Both backend and frontend are running. Press Ctrl+C to stop.")

        # Step 3: Wait for processes to terminate
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("Shutting down...")
        backend_process.terminate()
        frontend_process.terminate()
