import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', 5000)}"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'sync'
worker_connections = 1000
timeout = 30
keepalive = 2
max_requests = 1000
max_requests_jitter = 50

# Restart workers after this many requests, to help prevent memory leaks
preload_app = True

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Process naming
proc_name = 'espcam_server'

# Server mechanics
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None

# Custom startup hook to display server info
def on_starting(server):
    import socket
    def get_server_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"
    
    server_ip = get_server_ip()
    port = server.cfg.bind.split(':')[1]
    print("=" * 60)
    print(f"🚀 ESP-CAM Rice Detection Server Started with Gunicorn!")
    print(f"📡 Server IP: {server_ip}")
    print(f"🔌 Port: {port}")
    print(f"🌐 Full URL: http://{server_ip}:{port}")
    print(f"📋 API Endpoint: http://{server_ip}:{port}/predict")
    print(f"💻 Local URL: http://localhost:{port}/predict")
    print(f"👥 Workers: {server.cfg.workers}")
    print("=" * 60)
