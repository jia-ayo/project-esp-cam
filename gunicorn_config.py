import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', 5000)}"
backlog = 2048

# Worker processes - Reduced for memory efficiency
workers = 1  # Single worker to avoid memory issues with YOLO model
worker_class = 'sync'
worker_connections = 1000
timeout = 120  # Increased timeout for model inference
keepalive = 2
max_requests = 100  # Reduced to prevent memory leaks
max_requests_jitter = 10

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
    # Handle both string and list format for bind
    bind_address = server.cfg.bind
    if isinstance(bind_address, list):
        bind_address = bind_address[0]
    port = bind_address.split(':')[1]
    
    print("=" * 60)
    print(f"🚀 ESP-CAM Rice Detection Server Started with Gunicorn!")
    print(f"📡 Server IP: {server_ip}")
    print(f"🔌 Port: {port}")
    print(f"🌐 Full URL: http://{server_ip}:{port}")
    print(f"📋 API Endpoint: http://{server_ip}:{port}/predict")
    print(f"💻 Local URL: http://localhost:{port}/predict")
    print(f"👥 Workers: {server.cfg.workers}")
    print("=" * 60)
