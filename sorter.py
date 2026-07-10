import os
import time
import shutil
import threading
import queue
import numpy as np
import cv2
import customtkinter as ctk
from tkinter import filedialog

APP_NAME = "RAJ VIDEO SORTER"
VERSION = "v1.2"

# ---------- THEME ----------
BG = "#07090D"
PANEL = "#0E1420"
PANEL2 = "#121A29"
LINE = "#1E2A3D"
TEXT = "#E8EEF7"
DIM = "#6B7C93"
CYAN = "#2DD9F5"
CYAN_D = "#08B8D4"
AMBER = "#FFB020"
GREEN = "#3DDC84"
RED = "#FF5470"

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v",
              ".flv", ".wmv", ".mpg", ".mpeg", ".ts", ".mts", ".3gp"}

# Sensitivity: HIGH = ek letter ka chhota tukda bhi pakde
SENS_THRESH = {"HIGH": 0.0008, "MEDIUM": 0.0025, "LOW": 0.006}
MIN_BLOB = {"HIGH": 15, "MEDIUM": 30, "LOW": 60}
FRAMES_TO_SCAN = 16
PERSISTENCE = 0.70   # pura video static
HALF_PERSIST = 0.78  # sirf aadhe video mein aane wala logo


# ---------- DETECTION ENGINE ----------
def sample_frames(path, n=FRAMES_TO_SCAN):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total > n * 2:
            idxs = np.linspace(total * 0.05, total * 0.95, n).astype(int)
            for ix in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(ix))
                ok, fr = cap.read()
                if ok and fr is not None:
                    frames.append(fr)
        if len(frames) < 6:
            cap.release()
            cap = cv2.VideoCapture(path)
            cnt = 0
            while cnt < n:
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append(fr)
                cnt += 1
    finally:
        cap.release()
    return frames


def detect_logo(path, sens="HIGH"):
    """Bottom-right zone deep scan v1.2.
    - Chhota sa bhi sharp static tukda (letter/logo piece) = LOGO
    - Aadhe video mein aane wala logo bhi pakda jata hai (half-split check)
    - Black-bar border lines ignore
    - Pura blur smudge ya khali corner = NO LOGO
    Video file ko sirf PADHA jata hai — kabhi modify nahi hota."""
    frames = sample_frames(path)
    if len(frames) < 3:
        raise RuntimeError("read failed")
    maps = []
    for fr in frames:
        h, w = fr.shape[:2]
        roi = fr[int(h * 0.62):h, int(w * 0.50):w]
        if roi.size == 0:
            continue
        scale = 480.0 / roi.shape[1]
        roi = cv2.resize(roi, (480, max(2, int(roi.shape[0] * scale))),
                         interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 130)
        maps.append((edges > 0).astype(np.float32))
    if not maps:
        raise RuntimeError("roi failed")
    hmin = min(m.shape[0] for m in maps)
    maps = [m[:hmin, :] for m in maps]
    arr = np.stack(maps, axis=0)
    half = len(maps) // 2
    p_full = arr.mean(axis=0)
    p_a = arr[:half].mean(axis=0) if half >= 3 else p_full
    p_b = arr[half:].mean(axis=0) if (len(maps) - half) >= 3 else p_full
    static = ((p_full >= PERSISTENCE) | (p_a >= HALF_PERSIST)
              | (p_b >= HALF_PERSIST)).astype(np.uint8)
    static = cv2.morphologyEx(static, cv2.MORPH_CLOSE,
                              np.ones((3, 3), np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(static, 8)
    H, W = static.shape
    good_area = 0
    max_blob = 0
    for i in range(1, num):
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        ar = stats[i, cv2.CC_STAT_AREA]
        # black-bar / border lines ignore
        if bw > 0.85 * W and bh <= 6:
            continue
        if bh > 0.85 * H and bw <= 6:
            continue
        good_area += ar
        if ar > max_blob:
            max_blob = ar
    ratio = float(good_area) / float(H * W)
    return (ratio >= SENS_THRESH.get(sens, 0.0008)
            or max_blob >= MIN_BLOB.get(sens, 15))


# ---------- APP ----------
class SorterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        self.title(f"{APP_NAME}  {VERSION}")
        self.geometry("560x760")
        self.minsize(520, 700)
        self.configure(fg_color=BG)

        self.input_dir = ""
        self.output_dir = ""
        self.worker_thread = None
        self.q = queue.Queue()
        self.stop_flag = threading.Event()
        self.pause_flag = threading.Event()

        self._build_ui()
        self.after(100, self._poll)

    # ---------- UI ----------
    def _build_ui(self):
        pad = {"padx": 14, "pady": (10, 0)}

        # Title
        top = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0)
        top.pack(fill="x")
        ctk.CTkLabel(top, text="RAJ ", font=("Segoe UI", 18, "bold"),
                     text_color=TEXT).pack(side="left", padx=(14, 0), pady=10)
        ctk.CTkLabel(top, text="VIDEO SORTER", font=("Segoe UI", 18, "bold"),
                     text_color=CYAN).pack(side="left", pady=10)
        ctk.CTkLabel(top, text=VERSION, font=("Segoe UI", 11),
                     text_color=DIM).pack(side="right", padx=14)

        # Folders
        for kind in ("INPUT", "OUTPUT"):
            row = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=9)
            row.pack(fill="x", **pad)
            ctk.CTkLabel(row, text=kind, width=62, font=("Segoe UI", 10, "bold"),
                         text_color=DIM).pack(side="left", padx=(10, 4), pady=10)
            lbl = ctk.CTkLabel(row, text="folder select karo...", anchor="w",
                               font=("Consolas", 11), text_color=TEXT)
            lbl.pack(side="left", fill="x", expand=True, padx=4)
            btn = ctk.CTkButton(row, text="Browse", width=76, height=28,
                                fg_color=PANEL2, hover_color=LINE,
                                text_color=CYAN, font=("Segoe UI", 11, "bold"),
                                command=lambda k=kind, l=lbl: self._pick(k, l))
            btn.pack(side="right", padx=10, pady=8)

        # Info + sensitivity
        info = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=9)
        info.pack(fill="x", **pad)
        ctk.CTkLabel(info, text="Detection Zone: Bottom-Right  |  16-frame deep scan\n"
                                "Logo/text ka ek tukda bhi = LOGO  •  Pura blur/khali = NO_LOGO\n"
                                "COPY MODE — originals 100% safe, zero quality change",
                     font=("Segoe UI", 11), text_color=DIM,
                     justify="left").pack(side="left", padx=10, pady=8)
        self.sens = ctk.CTkOptionMenu(info, values=["HIGH", "MEDIUM", "LOW"],
                                      width=92, height=28, fg_color=PANEL2,
                                      button_color=LINE, button_hover_color=LINE,
                                      text_color=CYAN, font=("Segoe UI", 11, "bold"))
        self.sens.set("HIGH")
        self.sens.pack(side="right", padx=10)

        # Start
        self.start_btn = ctk.CTkButton(self, text="▶   START SCAN", height=46,
                                       corner_radius=10, fg_color=CYAN_D,
                                       hover_color=CYAN, text_color="#03222B",
                                       font=("Segoe UI", 15, "bold"),
                                       command=self._start)
        self.start_btn.pack(fill="x", **pad)

        # Progress
        prog = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=10)
        prog.pack(fill="x", **pad)
        toprow = ctk.CTkFrame(prog, fg_color="transparent")
        toprow.pack(fill="x", padx=12, pady=(10, 0))
        self.pct_lbl = ctk.CTkLabel(toprow, text="0%", font=("Consolas", 24, "bold"),
                                    text_color=CYAN)
        self.pct_lbl.pack(side="left")
        self.frac_lbl = ctk.CTkLabel(toprow, text="0 / 0 videos",
                                     font=("Consolas", 12), text_color=DIM)
        self.frac_lbl.pack(side="right")
        self.bar = ctk.CTkProgressBar(prog, height=10, corner_radius=99,
                                      fg_color=PANEL2, progress_color=CYAN)
        self.bar.set(0)
        self.bar.pack(fill="x", padx=12, pady=8)
        self.cur_lbl = ctk.CTkLabel(prog, text="Ready.", anchor="w",
                                    font=("Consolas", 11), text_color=DIM)
        self.cur_lbl.pack(fill="x", padx=12)

        stats = ctk.CTkFrame(prog, fg_color="transparent")
        stats.pack(fill="x", padx=8, pady=(4, 10))
        self.stat_done = self._stat(stats, "DONE")
        self.stat_left = self._stat(stats, "BAKI")
        self.stat_eta = self._stat(stats, "ETA")
        self.stat_speed = self._stat(stats, "/VIDEO")

        # Result counters
        res = ctk.CTkFrame(self, fg_color="transparent")
        res.pack(fill="x", **pad)
        self.logo_card = self._counter(res, "LOGO", AMBER)
        self.clean_card = self._counter(res, "NO LOGO", GREEN)

        # Controls
        ctrl = ctk.CTkFrame(self, fg_color="transparent")
        ctrl.pack(fill="x", **pad)
        self.pause_btn = ctk.CTkButton(ctrl, text="⏸  PAUSE", height=38,
                                       fg_color="transparent", border_width=1,
                                       border_color=LINE, text_color=TEXT,
                                       hover_color=PANEL2,
                                       font=("Segoe UI", 12, "bold"),
                                       command=self._pause, state="disabled")
        self.pause_btn.pack(side="left", expand=True, fill="x", padx=(0, 5))
        self.stop_btn = ctk.CTkButton(ctrl, text="⏹  STOP", height=38,
                                      fg_color="transparent", border_width=1,
                                      border_color=RED, text_color=RED,
                                      hover_color=PANEL2,
                                      font=("Segoe UI", 12, "bold"),
                                      command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", expand=True, fill="x", padx=(5, 0))

        # Log
        self.log = ctk.CTkTextbox(self, fg_color="#060A12", text_color=DIM,
                                  font=("Consolas", 10), corner_radius=9,
                                  border_width=1, border_color=LINE)
        self.log.pack(fill="both", expand=True, padx=14, pady=(10, 6))
        self.log.configure(state="disabled")

        ctk.CTkLabel(self, text="CREATED BY RAJ — CONTENT WORLD",
                     font=("Segoe UI", 9), text_color="#3A4A61").pack(pady=(0, 8))

    def _stat(self, parent, label):
        f = ctk.CTkFrame(parent, fg_color=PANEL2, corner_radius=7)
        f.pack(side="left", expand=True, fill="x", padx=4)
        v = ctk.CTkLabel(f, text="—", font=("Consolas", 14, "bold"), text_color=TEXT)
        v.pack(pady=(6, 0))
        ctk.CTkLabel(f, text=label, font=("Segoe UI", 9), text_color=DIM).pack(pady=(0, 6))
        return v

    def _counter(self, parent, label, color):
        f = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=10,
                         border_width=1, border_color=color)
        f.pack(side="left", expand=True, fill="x", padx=4)
        v = ctk.CTkLabel(f, text="0", font=("Consolas", 26, "bold"), text_color=color)
        v.pack(pady=(10, 0))
        ctk.CTkLabel(f, text=label, font=("Segoe UI", 10, "bold"),
                     text_color=DIM).pack(pady=(0, 10))
        return v

    # ---------- ACTIONS ----------
    def _pick(self, kind, lbl):
        d = filedialog.askdirectory()
        if not d:
            return
        if kind == "INPUT":
            self.input_dir = d
        else:
            self.output_dir = d
        lbl.configure(text=d)

    def _logline(self, txt):
        self.log.configure(state="normal")
        self.log.insert("end", txt + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _start(self):
        if self.worker_thread and self.worker_thread.is_alive():
            return
        if not self.input_dir or not self.output_dir:
            self._logline("[!] Pehle Input aur Output folder select karo.")
            return
        files = sorted(
            os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir)
            if os.path.splitext(f)[1].lower() in VIDEO_EXTS
        )
        if not files:
            self._logline("[!] Input folder mein koi video nahi mili.")
            return
        self.stop_flag.clear()
        self.pause_flag.clear()
        self.start_btn.configure(state="disabled")
        self.pause_btn.configure(state="normal", text="⏸  PAUSE")
        self.stop_btn.configure(state="normal")
        self._logline(f"[SCAN] {len(files)} videos mili. Shuru...")
        self.worker_thread = threading.Thread(
            target=self._worker, args=(files, self.sens.get()), daemon=True)
        self.worker_thread.start()

    def _pause(self):
        if self.pause_flag.is_set():
            self.pause_flag.clear()
            self.pause_btn.configure(text="⏸  PAUSE")
        else:
            self.pause_flag.set()
            self.pause_btn.configure(text="▶  RESUME")

    def _stop(self):
        self.stop_flag.set()
        self.pause_flag.clear()

    # ---------- WORKER ----------
    def _worker(self, files, sens):
        logo_dir = os.path.join(self.output_dir, "LOGO")
        clean_dir = os.path.join(self.output_dir, "NO_LOGO")
        os.makedirs(logo_dir, exist_ok=True)
        os.makedirs(clean_dir, exist_ok=True)
        total = len(files)
        done = logo_n = clean_n = err_n = 0
        times = []
        for path in files:
            if self.stop_flag.is_set():
                break
            while self.pause_flag.is_set() and not self.stop_flag.is_set():
                time.sleep(0.2)
            name = os.path.basename(path)
            self.q.put({"cur": name})
            t0 = time.time()
            try:
                has_logo = detect_logo(path, sens)
                dest = logo_dir if has_logo else clean_dir
                shutil.copy2(path, os.path.join(dest, name))
                if has_logo:
                    logo_n += 1
                    self.q.put({"log": f"[LOGO]  {name} -> LOGO ✓"})
                else:
                    clean_n += 1
                    self.q.put({"log": f"[CLEAN] {name} -> NO_LOGO ✓"})
            except Exception as e:
                err_n += 1
                self.q.put({"log": f"[ERROR] {name} skip ({e})"})
            done += 1
            times.append(time.time() - t0)
            avg = sum(times) / len(times)
            eta = int(avg * (total - done))
            self.q.put({"done": done, "total": total, "logo": logo_n,
                        "clean": clean_n, "eta": eta, "avg": avg})
        stopped = self.stop_flag.is_set()
        self.q.put({"log": f"[DONE] {'STOPPED' if stopped else 'COMPLETE'} — "
                           f"Logo: {logo_n} | No Logo: {clean_n} | Errors: {err_n}"})
        self.q.put({"finished": True})

    # ---------- UI POLL ----------
    def _poll(self):
        try:
            while True:
                m = self.q.get_nowait()
                if "cur" in m:
                    self.cur_lbl.configure(text=f"Scanning: {m['cur']}")
                if "log" in m:
                    self._logline(m["log"])
                if "done" in m:
                    d, t = m["done"], m["total"]
                    pct = int(d * 100 / t) if t else 0
                    self.pct_lbl.configure(text=f"{pct}%")
                    self.frac_lbl.configure(text=f"{d} / {t} videos")
                    self.bar.set(d / t if t else 0)
                    self.stat_done.configure(text=str(d))
                    self.stat_left.configure(text=str(t - d))
                    mm, ss = divmod(m["eta"], 60)
                    self.stat_eta.configure(text=f"{mm:02d}:{ss:02d}")
                    self.stat_speed.configure(text=f"{m['avg']:.1f}s")
                    self.logo_card.configure(text=str(m["logo"]))
                    self.clean_card.configure(text=str(m["clean"]))
                if m.get("finished"):
                    self.start_btn.configure(state="normal")
                    self.pause_btn.configure(state="disabled", text="⏸  PAUSE")
                    self.stop_btn.configure(state="disabled")
                    self.cur_lbl.configure(text="Ready.")
        except queue.Empty:
            pass
        self.after(100, self._poll)


if __name__ == "__main__":
    app = SorterApp()
    app.mainloop()
