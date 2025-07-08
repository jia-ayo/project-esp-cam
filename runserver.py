# ✅ CLI launcher for pipx using subprocess
def main():
    import subprocess
    subprocess.run([
        "gunicorn", "server:app", "--bind", "0.0.0.0:8000", "--workers", "2"
    ])