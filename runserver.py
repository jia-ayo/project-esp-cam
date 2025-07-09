# ✅ CLI launcher using Gunicorn's internal Python API
def main():
    from gunicorn.app.wsgiapp import run
    run()
