import psutil

print(f"{'PID':<8} | {'Memory (MB)':<12} | {'CWD':<45} | {'Command Line'}")
print("-" * 100)

for p in psutil.process_iter(['pid', 'name', 'memory_info', 'cmdline']):
    try:
        if 'python' in p.info['name'].lower():
            cwd = p.cwd() if hasattr(p, 'cwd') else 'N/A'
            mem = p.info['memory_info'].rss / 1024 / 1024
            cmd = " ".join(p.info['cmdline'] or [])
            print(f"{p.info['pid']:<8} | {mem:<12.1f} | {cwd:<45} | {cmd[:60]}")
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
