"""Windows offline fixed-phrase recognition for the studio GUI."""
from pathlib import Path
import subprocess
import threading


class VoiceControl(threading.Thread):
    def __init__(self, events, script):
        super().__init__(daemon=True)
        self.events = events
        self.script = Path(script)
        self.process = None
        self.stopping = threading.Event()

    def run(self):
        try:
            with subprocess.Popen(
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(self.script)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ) as process:
                self.process = process
                if self.stopping.is_set():
                    process.terminate()
                    return
                for line in process.stdout:
                    if self.stopping.is_set():
                        return
                    command = line.strip()
                    if command == "__READY__":
                        self.events.put(("voice_ready", None))
                    elif command:
                        self.events.put(("voice_command", command))
                error = process.stderr.read().strip()
                if process.wait() and not self.stopping.is_set():
                    self.events.put(("voice_error", error.splitlines()[-1] if error else "語音辨識程序已中止"))
        except Exception as exc:
            if not self.stopping.is_set():
                self.events.put(("voice_error", str(exc)))

    def close(self):
        self.stopping.set()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        if self.is_alive():
            self.join(2)
