def main():
    import sys
    from gunicorn.app.wsgiapp import WSGIApplication

    sys.argv = [
        "gunicorn",
        "server:app",                     # ✅ Your Flask app object
        "--config", "gunicorn_config.py" # ✅ Load your custom config file
    ]

    WSGIApplication("%(prog)s [OPTIONS] [APP_MODULE]").run()
