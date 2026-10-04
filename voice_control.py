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

    def run(self):
        try:
            self.process = subprocess.Popen(
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(self.script)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            for line in self.process.stdout:
                command = line.strip()
                if command == "__READY__":
                    self.events.put(("voice_ready", None))
                elif command:
                    self.events.put(("voice_command", command))
            error = self.process.stderr.read().strip()
            if self.process.poll() and error:
                self.events.put(("voice_error", error.splitlines()[-1]))
        except Exception as exc:
            self.events.put(("voice_error", str(exc)))

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        if self.is_alive():
            self.join(2)
