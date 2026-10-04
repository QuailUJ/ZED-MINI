"""Chinese desktop capture, conversion and switchable skeleton playback."""
import bisect
import ctypes
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
import tkinter as tk
import winsound
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
from PIL import Image, ImageTk

from capture_session import CaptureSession
from playback_source import PlaybackSource
from motion_data import unique_directory
from recording_io import draw_skeleton, save_json
from voice_control import VoiceControl

RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
PROJECT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else RESOURCE_DIR


def play_cue(kind):
    notes = {
        "accepted": ((660, 90), (830, 90), (990, 120)),
        "recording": ((880, 100), (1175, 150)),
        "stopped": ((990, 100), (660, 160)),
        "rejected": ((330, 180),),
    }[kind]
    threading.Thread(target=lambda: [winsound.Beep(frequency, duration) for frequency, duration in notes], daemon=True).start()


def named_session_directory(parent, name):
    name = name.strip()
    if not name or name in (".", "..") or any(char in name for char in '<>:"/\\|?*') or name.endswith((" ", ".")):
        raise ValueError("名稱不可空白，且不能包含 Windows 檔名禁用字元或以空白、句點結尾")
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / name
    path.mkdir(exist_ok=True)
    system = path / ".studio"
    if system.exists():
        raise FileExistsError("工作階段目前正在使用")
    system.mkdir()
    # Older versions placed technical files beside user outputs. Move only
    # known app-owned entries; never touch friendly output folders/files.
    for child in list(path.iterdir()):
        if child == system:
            continue
        if child.name == "session.json" or child.name.startswith(("static_", "motion_", "conversion_")):
            child.replace(system / child.name)
    if os.name == "nt":
        attributes = ctypes.windll.kernel32.GetFileAttributesW(str(system))
        if attributes != 0xFFFFFFFF:
            ctypes.windll.kernel32.SetFileAttributesW(str(system), attributes | 2)
    return path


def session_metadata_path(session):
    return Path(session) / ".studio" / "session.json"


def saved_calibration_path(session):
    return Path(session) / ".calibration.json"


def hide_on_windows(path):
    if os.name == "nt":
        attributes = ctypes.windll.kernel32.GetFileAttributesW(str(path))
        if attributes != 0xFFFFFFFF:
            ctypes.windll.kernel32.SetFileAttributesW(str(path), attributes | 2)


def persist_calibration(session, result):
    session = Path(session)
    model = session / f"{session.name}.osim"
    publish_file(result["model"], model)
    calibration = json.loads(Path(result["calibration"]).read_text(encoding="utf-8"))
    calibration["model"] = model.name
    path = saved_calibration_path(session)
    save_json(path, calibration)
    hide_on_windows(path)
    return path


def next_recording_paths(session, kind):
    session = Path(session)
    system = session / ".studio"
    system.mkdir(exist_ok=True)
    for number in range(1, 1000):
        internal = system / f"{kind}_{number:02d}"
        public = session / f"action_{number:02d}" if kind == "motion" else None
        if not internal.exists() and (public is None or not public.exists()):
            internal.mkdir()
            return internal, public
    raise FileExistsError("錄製編號已用完")


def publish_file(source, destination):
    """Expose one friendly file without duplicating large recording data."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    os.link(source, temporary)
    os.replace(temporary, destination)


def cleanup_session_data(session):
    system = Path(session) / ".studio"
    if system.is_dir():
        shutil.rmtree(system)


def append_error_log(session, label, source, summary):
    session = Path(session)
    log = session / f"{session.name}-errors.log"
    detail = Path(source).read_text(encoding="utf-8", errors="replace") if Path(source).is_file() else str(summary)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {label}\n{detail.rstrip()}\n\n")
    return log


def write_quality_note(directory, result):
    directory = Path(directory)
    quality = result["quality"]
    errors = result.get("marker_errors", {})
    filled = quality.get("filled_marker_samples", 0)
    lines = [
        "動作品質備註", "",
        "此資料夾內的 TRC 是實際送入 OpenSim IK 的處理版本。",
        "原始錄影與可用的原始骨架點沒有被改寫；缺失的內部區段使用前後有效三維座標做線性補點。",
        f"補點標記樣本數：{filled}",
        f"最長連續補點時間：{quality.get('max_interpolated_gap_seconds', 0):.3f} 秒",
        f"裁除開頭幀數：{quality.get('trimmed_leading_frames', 0)}",
        f"裁除結尾幀數：{quality.get('trimmed_trailing_frames', 0)}",
    ]
    if errors:
        lines.extend([f"IK 平均 RMS 誤差：{errors['mean_rms_m'] * 1000:.1f} mm",
                      f"IK 最大標記誤差：{errors['max_error_m'] * 1000:.1f} mm"])
    if result.get("pelvis_orientation_reference"):
        lines.append("骨盆傾角以左右髖及骨盆中心三點的逐幀方向作為柔性參考；腰椎未鎖定。此方向仍受 ZED 骨盆點估計誤差影響。")
    if result.get("sagittal_correction"):
        lines.append("已套用正面錄影方向修正，並以腰椎角度抵消骨盆傾斜，使軀幹維持直立；IK 誤差為修正前的標記擬合報告。")
    gaps = quality.get("interpolated_gaps", [])
    if gaps:
        lines.extend(["", "補點區段："])
        lines.extend(f"- {gap['marker']}：{gap['start']:.3f}–{gap['end']:.3f} 秒，補 {gap['seconds']:.3f} 秒（{gap['samples']} 個樣本）" for gap in gaps)
    elif filled:
        lines.extend(["", "本次轉換程式版本未保存逐段明細，以上總數與最長時間仍有效。"])
    else:
        lines.extend(["", "本段動作沒有使用線性補點。"])
    note = directory / f"{directory.name}-notes.txt"
    note.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return note


def photo(image, width, height):
    height_in, width_in = image.shape[:2]
    ratio = min(max(1, width) / width_in, max(1, height) / height_in, 1)
    if ratio < 1:
        image = cv2.resize(image, (max(1, int(width_in * ratio)), max(1, int(height_in * ratio))), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = Image.fromarray(rgb)
    result.thumbnail((max(1, width), max(1, height)))
    return ImageTk.PhotoImage(result)


class FirstRunGuide(tk.Toplevel):
    STEPS = [
        ("歡迎｜先認識操作順序", "整個流程分成三步：\n\n① 先錄一段靜態站姿，建立這個人的模型。\n② 再錄動作，按結束後自動計算動作結果。\n③ 回放影片，隨時切換是否顯示骨架。\n\n沒有相機也能閱讀教學、開啟既有錄製回放。現在不需要連接相機。"),
        ("1｜準備相機與工作階段", "程式啟動後會自動連接相機；請新增或開啟工作階段。\n\n• 固定相機，保持水平，避免錄到一半移動。\n• 受測者面向相機，全身、雙腳都要入鏡。\n• 開始錄製時，畫面中只留一名受測者。\n\n沒有偵測到相機時，程式會顯示提示，可按「連接相機」重試；教學及回放仍可使用。"),
        ("2｜先做靜態校正", "按「開始靜態校正錄製」後走到鏡頭前。\n程式會等到單一受測者的全身標記連續完整，才正式開始錄影。保持靜止數秒，再按「結束錄製並轉換」；走回電腦造成的尾段缺失會在分析資料中裁掉。\n\n等到顯示「靜態校正：已完成」，才能錄動作。工作階段外層會產生同名 OSIM。\n\n同一個人、相機位置不變時可以共用這次校正；換人或移動相機後請新增工作階段並重新校正。"),
        ("3｜錄製動作與保存結果", "按「開始動作錄製」，完成動作後按「結束錄製並轉換」。\n由你手動決定錄多久，不會在 10 秒時自動停止。\n\n每段動作各自保存同名 MP4、TRC、MOT。動作缺失一秒內會補點，超過 0.2 秒時完成訊息會顯示品質警告；超過一秒才停止轉換。"),
        ("4｜回放與遇到問題時怎麼做", "本次程式開啟期間，可選取清單中的錄製並按「回放選取錄製」，用播放／暫停、時間軸與「顯示骨架」檢查。\n\n程式正常關閉後只保留乾淨成果：OSIM，以及每段動作的 MP4、TRC、MOT；支援回放與重新轉換的暫存資料會刪除。轉換失敗時，詳細內容會追加到 OSIM 旁的同一份錯誤日誌。\n\n之後可隨時按主畫面「使用教學」再看一次。"),
    ]

    def __init__(self, parent):
        super().__init__(parent)
        self.title("第一次使用｜ZED MINI 使用教學")
        self.geometry("720x530")
        self.minsize(620, 480)
        self.index = 0
        self.parent = parent
        self.next_time = tk.BooleanVar(value=False)
        self.title_label = ttk.Label(self, font=("Microsoft JhengHei", 16, "bold"), padding=(24, 20))
        self.title_label.pack(anchor="w")
        body_frame = ttk.Frame(self)
        body_frame.pack(fill="both", expand=True)
        self.content = tk.Text(body_frame, wrap="word", font=("Microsoft JhengHei", 11), relief="flat", padx=24, pady=8, height=12, cursor="arrow", background="#f4f6f8")
        scrollbar = ttk.Scrollbar(body_frame, command=self.content.yview)
        self.content.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.content.pack(side="left", fill="both", expand=True)
        self.counter = ttk.Label(self, padding=(24, 8))
        self.counter.pack(anchor="w")
        ttk.Checkbutton(self, text="下次啟動仍顯示教學", variable=self.next_time).pack(anchor="w", padx=24)
        buttons = ttk.Frame(self, padding=20)
        buttons.pack(fill="x")
        self.previous_button = ttk.Button(buttons, text="上一步", command=lambda: self.move(-1))
        self.previous_button.pack(side="left")
        ttk.Button(buttons, text="先看主畫面", command=self.finish).pack(side="left", padx=10)
        self.next_button = ttk.Button(buttons, command=lambda: self.move(1))
        self.next_button.pack(side="right")
        self.draw_step()

    def draw_step(self):
        title, body = self.STEPS[self.index]
        self.title_label.configure(text=title)
        self.content.configure(state="normal")
        self.content.delete("1.0", "end")
        self.content.insert("1.0", body)
        self.content.configure(state="disabled")
        self.content.yview_moveto(0)
        self.counter.configure(text=f"{self.index + 1} / {len(self.STEPS)}")
        self.previous_button.configure(state="disabled" if self.index == 0 else "normal")
        self.next_button.configure(text="開始使用" if self.index == len(self.STEPS) - 1 else "下一步")

    def move(self, delta):
        if self.index + delta >= len(self.STEPS):
            self.finish()
            return
        self.index = max(0, self.index + delta)
        self.draw_step()

    def finish(self):
        try:
            save_json(self.parent.settings_path, {"show_guide": self.next_time.get()})
        except OSError as exc:
            self.parent.status.set(f"教學偏好無法保存：{exc}；仍可繼續使用。")
        self.destroy()


class Playback(tk.Toplevel):
    def __init__(self, parent, directory):
        super().__init__(parent)
        self.title("錄製回放 — " + directory.name)
        self.geometry("1000x720")
        self.after_id = None
        self.closed = False
        self.ready = False
        self.duration = 0.0
        self.source = PlaybackSource(directory)
        self.source.start()
        self.playing = False
        self.position = 0.0
        self.last_index = -1
        self.last_frame = None
        self.show_bones = tk.BooleanVar(value=True)
        self.image_label = ttk.Label(self, text="正在載入影片與骨架，可隨時關閉…", anchor="center")
        self.image_label.pack(fill="both", expand=True)
        self.slider_value = tk.DoubleVar(value=0)
        self.slider = ttk.Scale(self, from_=0, to=max(self.duration, .001), variable=self.slider_value, command=self.seek, state="disabled")
        self.slider.pack(fill="x", padx=12)
        row = ttk.Frame(self)
        row.pack(fill="x", padx=12, pady=12)
        self.play_button = ttk.Button(row, text="播放", command=self.toggle, state="disabled")
        self.play_button.pack(side="left")
        ttk.Checkbutton(row, text="顯示骨架", variable=self.show_bones, command=self.render).pack(side="left", padx=16)
        self.clock = ttk.Label(row)
        self.clock.pack(side="right")
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Configure>", lambda event: self.render() if event.widget is self else None)
        self.tick()

    def seek(self, value):
        if not self.ready:
            return
        self.position = float(value)
        self.started = time.monotonic() - self.position
        self.source.request(min(self.count - 1, int(self.position * self.fps + 1e-6)))

    def toggle(self):
        if not self.ready:
            return
        if self.position >= self.duration:
            self.position = 0
        self.playing = not self.playing
        self.started = time.monotonic() - self.position
        self.play_button.configure(text="暫停" if self.playing else "播放")

    def render(self):
        if self.closed or self.last_frame is None:
            return
        frame = self.last_frame.copy()
        sample_index = bisect.bisect_right(self.times, self.last_index / self.fps + 1e-7) - 1
        if self.show_bones.get() and sample_index >= 0:
            sample = self.samples[sample_index]
            if sample["body_id"] == self.meta["body_id"] and self.last_index / self.fps - sample["time"] <= .2:
                draw_skeleton(frame, sample["points"], self.meta["bones"])
        self.image = photo(frame, self.winfo_width() - 24, self.winfo_height() - 110)
        self.image_label.configure(image=self.image, text="")
        self.clock.configure(text=f"{self.last_index / self.fps:.2f} / {self.duration:.2f} 秒")

    def tick(self):
        if self.closed:
            return
        try:
            while True:
                event, value = self.source.events.get_nowait()
                if event == "ready":
                    self.meta, self.samples, self.fps, self.count = value
                    self.times = [sample["time"] for sample in self.samples]
                    self.duration = (self.count - 1) / self.fps
                    self.ready = True
                    self.slider.configure(to=max(self.duration, .001), state="normal")
                    self.play_button.configure(state="normal")
                    self.source.request(0)
                elif event == "error":
                    self.ready = self.playing = False
                    self.last_frame = None
                    self.play_button.configure(text="播放", state="disabled")
                    self.slider.configure(state="disabled")
                    self.clock.configure(text="回放失敗")
                    self.image_label.configure(image="", text=f"無法回放：{value}\n原始錄製檔案未修改。")
        except queue.Empty:
            pass
        if self.ready and self.playing:
            self.position = min(self.duration, time.monotonic() - self.started)
            # Variable updates don't invoke seek or register new Tcl callbacks per frame.
            self.slider_value.set(self.position)
            self.source.request(min(self.count - 1, int(self.position * self.fps + 1e-6)))
            if self.position >= self.duration:
                self.playing = False
                self.play_button.configure(text="播放")
        frame = self.source.take_frame()
        if frame is not None and self.ready:
            self.last_index, self.last_frame = frame
            self.render()
        self.after_id = self.after(20, self.tick)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.after_id is not None:
            self.after_cancel(self.after_id)
        self.source.close()
        self.source.join(3)
        self.destroy()


class Studio(tk.Tk):
    def __init__(self, *, settings_path=None, show_guide=None, enable_voice=True, auto_connect=True):
        super().__init__()
        self.title("ZED MINI 動作錄製工作室")
        # At this size the inner preview is 1136x639 (16:9), so the full
        # camera frame and every control fit without preview letterboxing.
        self.geometry("1160x1060")
        self.minsize(900, 680)
        self.events = queue.Queue()
        self.camera = None
        self.camera_ready = False
        self.state = "idle"
        self.process = None
        self.process_log = None
        self.closing = False
        self.destroyed = False
        self.guide = None
        self.guide_after_id = None
        self.voice = None
        self.voice_enabled = tk.BooleanVar(value=enable_voice)
        self.voice_status = tk.StringVar(value="語音控制：正在啟動" if enable_voice else "語音控制：已關閉")
        self.settings_path = Path(settings_path) if settings_path else PROJECT / ".studio_settings.json"
        self.cancelled = False
        self.last_preview_key = None
        self.last_preview_snapshot = None
        self.session = None
        self.calibration = None
        self.live_static_directory = None
        self.current = None
        self.publish_targets = {}
        self.job = None
        self.live_bones = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="請建立工作階段並連接 ZED 相機。")
        self.tracking = tk.StringVar(value="相機尚未連接")
        self.session_label = tk.StringVar(value="尚未建立工作階段")
        self.calibration_label = tk.StringVar(value="靜態校正：尚未完成")
        toolbar = ttk.Frame(self, padding=10)
        toolbar.pack(fill="x")
        self.new_button = ttk.Button(toolbar, text="新增工作階段", command=self.new_session)
        self.new_button.pack(side="left")
        self.open_session_button = ttk.Button(toolbar, text="開啟既有工作階段", command=self.open_session)
        self.open_session_button.pack(side="left", padx=(8, 0))
        self.connect_button = ttk.Button(toolbar, text="連接相機", command=self.connect)
        self.connect_button.pack(side="left", padx=8)
        ttk.Button(toolbar, text="使用教學", command=self.show_guide).pack(side="left", padx=8)
        ttk.Checkbutton(toolbar, text="預覽顯示骨架", variable=self.live_bones).pack(side="right")
        ttk.Checkbutton(toolbar, text="啟用語音控制", variable=self.voice_enabled, command=self.toggle_voice).pack(side="right", padx=12)
        ttk.Label(self, textvariable=self.voice_status, padding=(12, 0)).pack(anchor="w")
        ttk.Label(self, textvariable=self.session_label, padding=(12, 0)).pack(anchor="w")
        ttk.Label(self, textvariable=self.calibration_label, padding=(12, 4)).pack(anchor="w")
        self.preview_frame = ttk.Frame(self, width=640, height=360)
        self.preview_frame.pack(fill="both", expand=True, padx=12, pady=6)
        self.preview_frame.pack_propagate(False)
        self.preview = ttk.Label(self.preview_frame, text="尚未連接相機\n\n第一次使用可先按「使用教學」", anchor="center", justify="center", background="#20252b", foreground="white")
        self.preview.pack(fill="both", expand=True)
        ttk.Label(self, textvariable=self.tracking, padding=(12, 3)).pack(anchor="w")
        controls = ttk.Frame(self, padding=(12, 6))
        controls.pack(fill="x")
        self.static_button = ttk.Button(controls, text="開始靜態校正錄製", command=lambda: self.start("static"))
        self.static_button.pack(side="left")
        self.motion_button = ttk.Button(controls, text="開始動作錄製", command=lambda: self.start("motion"))
        self.motion_button.pack(side="left", padx=8)
        self.stop_button = ttk.Button(controls, text="結束錄製並轉換", command=self.stop)
        self.stop_button.pack(side="left")
        self.cancel_button = ttk.Button(controls, text="取消轉換", command=self.cancel_conversion)
        self.cancel_button.pack(side="left", padx=8)
        self.busy_indicator = ttk.Progressbar(controls, mode="indeterminate", length=110)
        self.busy_indicator.pack(side="right")
        self.was_busy = False
        ttk.Label(self, text="靜態校正：面向相機、全身及雙腳入鏡，站穩後再開始；人員或相機位置改變時請新增工作階段。", padding=(12, 4)).pack(anchor="w")
        self.listing = ttk.Treeview(self, columns=("kind", "status", "folder"), show="headings", height=5)
        for column, title, width in [("kind", "類型", 90), ("status", "狀態", 180), ("folder", "錄製資料夾", 700)]:
            self.listing.heading(column, text=title)
            self.listing.column(column, width=width)
        self.listing.pack(fill="x", padx=12, pady=4)
        actions = ttk.Frame(self, padding=(12, 4))
        actions.pack(fill="x")
        ttk.Button(actions, text="回放選取錄製", command=self.playback).pack(side="left")
        self.retry_button = ttk.Button(actions, text="重新轉換選取錄製", command=self.retry)
        self.retry_button.pack(side="left", padx=8)
        ttk.Button(actions, text="開啟資料夾", command=self.show_folder).pack(side="left")
        ttk.Label(self, textvariable=self.status, wraplength=1080, padding=12).pack(fill="x")
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.update_controls()
        self.poll_after_id = self.after(50, self.poll)
        if show_guide is None:
            try:
                show_guide = json.loads(self.settings_path.read_text(encoding="utf-8")).get("show_guide", True)
            except (OSError, ValueError, AttributeError):
                show_guide = True
        if show_guide:
            self.guide_after_id = self.after(250, self.show_guide)
        if enable_voice:
            self.after(50, self.start_voice)
        if auto_connect:
            self.after(150, self.connect)

    def start_voice(self):
        if self.closing or not self.voice_enabled.get() or (self.voice is not None and self.voice.is_alive()):
            return
        self.voice_status.set("語音控制：正在啟動")
        self.voice = VoiceControl(self.events, RESOURCE_DIR / "voice_commands.ps1")
        self.voice.start()

    def toggle_voice(self):
        if self.voice_enabled.get():
            self.start_voice()
        else:
            if self.voice is not None:
                self.voice.close()
                self.voice = None
            self.voice_status.set("語音控制：已關閉")

    def handle_voice_command(self, command):
        def start_current_stage():
            self.start("motion" if self.calibration is not None else "static")

        actions = {
            "開始錄製": (lambda: str((self.motion_button if self.calibration is not None else self.static_button)["state"]) == "normal", start_current_stage),
            "開始靜態錄製": (lambda: str(self.static_button["state"]) == "normal", lambda: self.start("static")),
            "結束錄製": (lambda: self.state == "recording", self.stop),
        }
        if command not in actions or not actions[command][0]():
            play_cue("rejected")
            self.status.set(f"已聽到「{command}」，但目前狀態不能執行。")
            return
        play_cue("stopped" if command == "結束錄製" else "accepted")
        actions[command][1]()

    def show_guide(self):
        if self.closing:
            return
        if self.guide is not None and self.guide.winfo_exists():
            self.guide.lift()
        else:
            self.guide = FirstRunGuide(self)

    def report_callback_exception(self, exc_type, exc_value, exc_tb):
        self.status.set(f"操作未完成：{exc_value}。介面仍可繼續使用。")
        try:
            with (PROJECT / "studio_errors.log").open("a", encoding="utf-8") as stream:
                traceback.print_exception(exc_type, exc_value, exc_tb, file=stream)
        except OSError:
            pass

    def update_controls(self):
        idle = self.state == "idle" and self.process is None and not self.closing
        for button in (self.new_button, self.open_session_button, self.retry_button):
            button.configure(state="normal" if idle else "disabled")
        self.connect_button.configure(state="normal" if idle and not self.camera_ready and (self.camera is None or not self.camera.is_alive()) else "disabled")
        can_record = idle and self.camera_ready and self.session is not None
        self.static_button.configure(state="normal" if can_record else "disabled")
        self.motion_button.configure(state="normal" if can_record and self.calibration is not None else "disabled")
        self.stop_button.configure(state="normal" if self.state in ("starting", "recording") and not self.closing else "disabled")
        self.cancel_button.configure(state="normal" if self.process is not None and not self.cancelled else "disabled")
        busy = self.process is not None or self.state in ("starting", "stopping") or (self.camera is not None and self.camera.is_alive() and not self.camera_ready)
        if busy != self.was_busy:
            self.busy_indicator.start(15) if busy else self.busy_indicator.stop()
            self.was_busy = busy

    def new_session(self):
        name = simpledialog.askstring("新增工作階段", "請輸入受測者或工作階段名稱：", parent=self)
        if name is None:
            return
        try:
            self.session = named_session_directory(PROJECT / "recordings", name)
        except (ValueError, FileExistsError) as exc:
            messagebox.showerror("無法建立工作階段", "此名稱已存在，請換一個名稱。" if isinstance(exc, FileExistsError) else str(exc), parent=self)
            return
        self.calibration = None
        self.live_static_directory = None
        self.session_label.set(f"工作階段：{self.session.name}")
        self.calibration_label.set("靜態校正：尚未完成")
        save_json(session_metadata_path(self.session), {"version": 1, "name": self.session.name, "calibration": None})
        self.status.set("請先錄製靜態站姿；相機需保持水平及固定。")
        self.update_controls()

    def open_session(self):
        path = filedialog.askdirectory(title="選擇既有工作階段", initialdir=PROJECT / "recordings", parent=self)
        if not path:
            return
        session = Path(path).resolve()
        if session.parent != (PROJECT / "recordings").resolve():
            messagebox.showerror("無法開啟工作階段", "請選擇 recordings 內的工作階段資料夾。", parent=self)
            return
        system = session / ".studio"
        system.mkdir(exist_ok=True)
        hide_on_windows(system)
        self.session = session
        self.live_static_directory = None
        calibration = saved_calibration_path(session)
        self.calibration = calibration if calibration.is_file() and (session / f"{session.name}.osim").is_file() else None
        self.session_label.set(f"工作階段：{session.name}")
        if self.calibration is not None:
            self.calibration_label.set("靜態校正：已載入，可繼續錄製動作")
            self.status.set("既有工作階段已開啟；受測者與相機位置未改變時可繼續錄製動作。")
        else:
            self.calibration_label.set("靜態校正：尚未完成，請重新錄製")
            self.status.set("既有工作階段已開啟，但沒有可沿用的校正；請先錄製靜態站姿。")
        save_json(session_metadata_path(session), {"version": 1, "name": session.name,
                                                   "calibration": str(self.calibration) if self.calibration else None})
        self.update_controls()

    def connect(self):
        if self.closing or (self.camera is not None and self.camera.is_alive()):
            return
        # A persisted calibration is intentionally reusable when the subject and
        # physical camera setup have not changed. The user chooses a new session
        # when either one changes.
        self.live_static_directory = None
        if self.calibration is None:
            self.calibration_label.set("靜態校正：尚未完成")
            if self.session is not None:
                save_json(session_metadata_path(self.session), {"version": 1, "name": self.session.name, "calibration": None})
        self.camera = CaptureSession(self.events)
        self.camera.start()
        self.status.set("正在初始化 ZED 與人體追蹤…")
        self.update_controls()

    def start(self, kind):
        if self.session is None or not self.camera_ready or self.state != "idle":
            return
        self.current, published = next_recording_paths(self.session, kind)
        self.publish_targets[str(self.current.resolve())] = published
        self.state = "starting"
        if kind == "static":
            self.calibration = None
            self.live_static_directory = self.current
            self.calibration_label.set("靜態校正：等待重新錄製")
            save_json(session_metadata_path(self.session), {"version": 1, "name": self.session.name, "calibration": None})
        self.camera.commands.put(("start", (self.current, kind)))
        self.status.set("等待單一受測者的全身標記連續完整；就位後才會正式開始錄製…")
        self.update_controls()

    def stop(self):
        if self.state not in ("starting", "recording"):
            return
        was_starting = self.state == "starting"
        self.state = "stopping"
        self.camera.commands.put(("stop", None))
        self.status.set("正在取消等待…" if was_starting else "正在保存原始資料與影片…")
        self.update_controls()

    def add_recording(self, directory):
        directory = Path(directory).resolve()
        meta = json.loads((directory / "recording.json").read_text(encoding="utf-8"))
        key = str(directory)
        labels = {"recorded": "已保存", "interrupted": "錄製中斷", "recording": "錄製中"}
        shown = meta.get("published_dir", key)
        values = ("靜態" if meta["kind"] == "static" else "動作", labels.get(meta["status"], meta["status"]), shown)
        if self.listing.exists(key):
            self.listing.item(key, values=values)
        else:
            self.listing.insert("", "end", iid=key, values=values)
        self.listing.selection_set(key)
        self.listing.see(key)
        return meta

    def selected(self):
        selection = self.listing.selection()
        return Path(selection[0]) if selection else None

    def open_recording(self):
        path = filedialog.askdirectory(title="選擇含 recording.json 的錄製資料夾", initialdir=PROJECT / "recordings")
        if path:
            try:
                self.add_recording(path)
            except Exception as exc:
                messagebox.showerror("無法開啟", str(exc))

    def playback(self):
        directory = self.selected()
        if directory is not None:
            try:
                Playback(self, directory)
            except Exception as exc:
                messagebox.showerror("回放失敗", str(exc))

    def show_folder(self):
        directory = self.selected() or self.session
        if directory:
            target = directory
            if directory.is_dir() and (directory / "recording.json").is_file():
                meta = json.loads((directory / "recording.json").read_text(encoding="utf-8"))
                target = Path(meta.get("published_dir", self.session or directory))
            os.startfile(str(target))

    def retry(self):
        directory = self.selected()
        if directory is None or self.state != "idle" or self.process is not None:
            return
        try:
            meta = json.loads((directory / "recording.json").read_text(encoding="utf-8"))
            calibration = meta.get("calibration")
            if meta["kind"] == "motion" and (not calibration or not Path(calibration).is_file()):
                calibration = filedialog.askopenfilename(title="選擇該次錄製對應的 calibration.json", filetypes=[("校正設定", "*.json")])
                if not calibration:
                    return
            self.convert(directory, calibration, activate=directory == self.live_static_directory)
        except Exception as exc:
            self.status.set(f"重新轉換失敗：{exc}")

    def convert(self, directory, calibration=None, activate=False):
        meta = json.loads((directory / "recording.json").read_text(encoding="utf-8"))
        mode = "scale" if meta["kind"] == "static" else "ik"
        if mode == "ik" and calibration is None:
            raise ValueError("缺少本次錄製的靜態校正")
        output = unique_directory(directory, "conversion")
        if getattr(sys, "frozen", False):
            command = [sys.executable, "--pipeline-worker", "--mode", mode,
                       "--source", str(directory / "raw.trc"), "--output-dir", str(output)]
        else:
            command = [sys.executable, "-u", str(RESOURCE_DIR / "run_pipeline.py"), "--mode", mode,
                       "--source", str(directory / "raw.trc"), "--output-dir", str(output)]
        if calibration:
            command.extend(["--calibration", str(calibration)])
            meta["calibration"] = str(calibration)
        meta["last_conversion"] = str(output)
        save_json(directory / "recording.json", meta)
        # Worker requires an empty output directory; process stdout lives beside it.
        self.process_log = (directory / f"{output.name}.log").open("w", encoding="utf-8")
        try:
            self.process = subprocess.Popen(command, cwd=PROJECT, stdout=self.process_log, stderr=subprocess.STDOUT,
                                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        except Exception:
            self.process_log.close()
            self.process_log = None
            raise
        self.job = (directory, output, mode, activate)
        self.cancelled = False
        self.listing.set(str(directory), "status", "縮放模型中" if mode == "scale" else "IK 轉換中")
        self.status.set("正在縮放模型…" if mode == "scale" else "正在補點、對齊座標並計算 IK…")
        self.update_controls()

    def cancel_conversion(self):
        if self.process is None or self.process.poll() is not None:
            return
        try:
            self.process.terminate()
        except OSError as exc:
            self.status.set(f"無法取消轉換：{exc}")
            return
        self.cancelled = True
        self.status.set("正在取消轉換；原始影片與 TRC 仍會保留。")
        self.update_controls()

    def complete_job(self):
        directory, output, mode, activate = self.job
        code = self.process.returncode
        self.process_log.close()
        self.process = self.process_log = self.job = None
        try:
            if self.cancelled:
                save_json(output / "result.json", {"status": "cancelled", "mode": mode, "source": str(directory / "raw.trc")})
                self.listing.set(str(directory), "status", "已取消轉換")
                self.status.set("已取消本次轉換，原始資料保留，可重新轉換。")
                return
            result = json.loads((output / "result.json").read_text(encoding="utf-8"))
            if code != 0 or result["status"] != "complete":
                raise ValueError(result.get("error", "轉換程序失敗"))
            if mode == "scale" and activate and self.camera_ready:
                self.calibration = persist_calibration(self.session, result)
                self.calibration_label.set("靜態校正：已完成，可開始動作錄製")
                save_json(session_metadata_path(self.session), {"version": 1, "name": self.session.name, "calibration": str(self.calibration)})
            elif mode == "ik":
                meta = json.loads((directory / "recording.json").read_text(encoding="utf-8"))
                published = Path(meta["published_dir"])
                publish_file(output / "processed.trc", published / f"{published.name}.trc")
                publish_file(result["motion"], published / f"{published.name}.mot")
                write_quality_note(published, result)
            self.listing.set(str(directory), "status", "轉換完成")
            detail = f"補點 {result['quality']['filled_marker_samples']} 個標記樣本。"
            if mode == "scale":
                duration = result["quality"]["selected_static_end"] - result["quality"]["selected_static_start"]
                detail += f" 已使用 {duration:.2f} 秒完整站姿。"
            if result["quality"].get("quality_warning"):
                detail += f" 品質警告：最長補點 {result['quality']['max_interpolated_gap_seconds']:.2f} 秒。"
            if mode == "ik":
                error = result["marker_errors"]
                detail += f" IK 平均 RMS {error['mean_rms_m'] * 1000:.1f} mm，最大誤差 {error['max_error_m'] * 1000:.1f} mm。"
            self.status.set(f"完成：{output.name}。{detail}")
        except Exception as exc:
            self.listing.set(str(directory), "status", "轉換失敗／需檢查")
            try:
                meta = json.loads((directory / "recording.json").read_text(encoding="utf-8"))
                source = output / "error.txt"
                if not source.is_file():
                    source = directory / f"{output.name}.log"
                label = f"{'動作' if meta['kind'] == 'motion' else '靜態'}轉換失敗：{Path(meta.get('published_dir', directory)).name}"
                append_error_log(self.session, label, source, exc)
            except Exception:
                pass
            self.status.set(f"轉換未完成：{exc}。原始資料已保留；詳細日誌位於錄製資料夾。")
        finally:
            self.cancelled = False
            self.update_controls()

    def handle_event(self, event, value):
        if event == "camera_ready":
            self.camera_ready = True
            self.status.set("相機已就緒，請建立工作階段並先錄製靜態站姿。")
        elif event == "recording":
            self.state = "recording"
            play_cue("recording")
            self.status.set("錄製中；按「結束錄製並轉換」完成。")
        elif event == "voice_ready":
            self.voice_status.set("語音控制：可用（開始錄製／開始靜態錄製／結束錄製）")
        elif event == "voice_command":
            self.handle_voice_command(value)
        elif event == "voice_error":
            self.voice_enabled.set(False)
            self.voice_status.set(f"語音控制：無法啟動（{value}）")
        elif event == "start_failed":
            self.state = "idle"
            self.status.set(value)
        elif event in ("recorded", "interrupted"):
            self.state = "idle"
            directory = Path(value)
            meta = self.add_recording(directory)
            if meta["kind"] == "motion" and self.calibration is not None:
                meta["calibration"] = str(self.calibration)
                published = self.publish_targets.get(str(directory.resolve()))
                if published is not None:
                    published.mkdir(parents=True, exist_ok=True)
                    meta["published_dir"] = str(published)
                    publish_file(directory / "video.mp4", published / f"{published.name}.mp4")
                    publish_file(directory / "raw.trc", published / f"{published.name}.trc")
                save_json(directory / "recording.json", meta)
            if event == "recorded" and not self.closing:
                self.convert(directory, self.calibration, activate=True)
            else:
                self.status.set(f"錄製已中止，資料保留於 {directory}")
        elif event in ("camera_error", "error"):
            self.status.set(value)
            if event == "camera_error":
                self.camera_ready = False
                self.calibration = None
                self.live_static_directory = None
                self.calibration_label.set("靜態校正：相機中斷後需重錄")
                self.state = "idle"
                self.tracking.set("相機未連接；仍可查看使用教學")
                self.preview.configure(image="", text="相機未連接\n\n可查看「使用教學」")
                self.last_preview_key = None
                self.last_preview_snapshot = None
        elif event == "camera_closed":
            self.camera_ready = False
        self.update_controls()

    def poll(self):
        if self.destroyed:
            return
        try:
            self.poll_once()
        except Exception:
            self.report_callback_exception(*sys.exc_info())
        finally:
            if not self.destroyed:
                self.poll_after_id = self.after(50, self.poll)

    def poll_once(self):
        try:
            while True:
                event, value = self.events.get_nowait()
                try:
                    self.handle_event(event, value)
                except Exception as exc:
                    self.state = "idle"
                    self.status.set(f"操作失敗，資料保留：{exc}")
                    self.update_controls()
        except queue.Empty:
            pass
        if self.process is not None and self.process.poll() is not None:
            self.complete_job()
        if self.camera_ready and self.camera is not None:
            snapshot = self.camera.snapshot()
            key = (id(snapshot), self.live_bones.get(), self.preview.winfo_width(), self.preview.winfo_height())
            if snapshot is not None and key != self.last_preview_key:
                self.last_preview_key = key
                self.last_preview_snapshot = snapshot
                frame, points, bones, count, body_id, elapsed = snapshot
                frame = frame.copy()
                if self.live_bones.get():
                    draw_skeleton(frame, points, bones)
                self.image = photo(frame, self.preview.winfo_width(), self.preview.winfo_height())
                self.preview.configure(image=self.image, text="")
                tracking = f"有效人數：{count}　追蹤 ID：{body_id if body_id is not None else '缺失'}"
                if elapsed is not None:
                    tracking += f"　錄製：{elapsed:.1f} 秒"
                self.tracking.set(tracking)
        if self.closing:
            if self.process is None and (self.camera is None or not self.camera.is_alive()):
                for child in self.winfo_children():
                    if isinstance(child, Playback):
                        child.close()
                if self.session is not None:
                    try:
                        cleanup_session_data(self.session)
                    except OSError as exc:
                        (self.session / f"{self.session.name}-清理錯誤.log").write_text(str(exc), encoding="utf-8")
                self.destroyed = True
                self.destroy()
                return
            self.status.set("正在安全關閉：保存錄製並等待轉換完成；也可按「取消轉換」停止本次計算。")
        self.update_controls()

    def close(self):
        self.closing = True
        if self.guide_after_id is not None:
            self.after_cancel(self.guide_after_id)
        if self.camera is not None:
            self.camera.stop_event.set()
        if self.voice is not None:
            self.voice.close()
            self.voice = None
        self.update_controls()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--pipeline-worker":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        from run_pipeline import main as pipeline_main
        pipeline_main()
    else:
        Studio().mainloop()
