#!/usr/bin/env python3

import os
import time
import struct
import threading
import shutil
import subprocess
from datetime import datetime

import numpy as np
import serial
from serial.tools import list_ports

import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


VID = 0x0483
PID = 0x5740


def getport():
    for device in list_ports.comports():
        if device.vid == VID and device.pid == PID:
            return device.device
    raise OSError("tinySA not found")


class TinySA:
    def __init__(self, port):
        self.ser = serial.Serial(port, 115200, timeout=1)
        self.lock = threading.Lock()

    def close(self):
        with self.lock:
            try:
                if self.ser and self.ser.is_open:
                    self.ser.close()
            except Exception:
                pass

    def _read_prompt(self):
        data = self.ser.read_until(b"ch> ")
        if not data.endswith(b"ch> "):
            raise RuntimeError("tinySA prompt not received")

    def _flush_buffers(self):
        try:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
        except Exception:
            pass

        end_time = time.time() + 0.3
        while time.time() < end_time:
            try:
                waiting = self.ser.in_waiting
            except Exception:
                waiting = 0

            if waiting > 0:
                try:
                    self.ser.read(waiting)
                except Exception:
                    break
                time.sleep(0.02)
            else:
                break

    def scan(self, f_low, f_high, points, rbw, stop_event=None):
        with self.lock:
            if not self.ser or not self.ser.is_open:
                raise RuntimeError("Serial port is closed")

            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("Scan stopped")

            self._flush_buffers()

            if rbw == 0:
                rbw_k = (f_high - f_low) * 7e-6
            else:
                rbw_k = rbw / 1e3

            rbw_k = max(3, min(600, rbw_k))

            self.ser.write(f"rbw {int(rbw_k)}\r".encode())
            self._read_prompt()

            timeout = ((f_high - f_low) / 20e3) / (rbw_k ** 2) + points / 500 + 1
            self.ser.timeout = max(2, timeout * 2)

            if stop_event is not None and stop_event.is_set():
                raise RuntimeError("Scan stopped")

            self.ser.write(f"scanraw {int(f_low)} {int(f_high)} {int(points)}\r".encode())

            head = self.ser.read_until(b"{")
            if not head.endswith(b"{"):
                raise RuntimeError("tinySA did not start scan payload")

            raw = self.ser.read_until(b"}ch> ")
            if not raw.endswith(b"}ch> "):
                raise RuntimeError("tinySA returned incomplete scan payload")

            payload = raw[:-5]

            expected_len = points * 3
            if len(payload) != expected_len:
                raise RuntimeError(
                    f"Unexpected payload size: got {len(payload)}, expected {expected_len}"
                )

            try:
                data = struct.unpack("<" + "xH" * points, payload)
            except struct.error as e:
                raise RuntimeError(f"Failed to decode scan payload: {e}")

            arr = np.array(data, dtype=np.uint16)

            SCALE = 128
            power = arr / 32 - SCALE

            try:
                self.ser.timeout = 1
                self.ser.write(b"rbw auto\r")
                self._read_prompt()
            except Exception:
                pass

            return power


class RTPlotter:
    def __init__(self, root):
        self.root = root
        self.root.title("tinySA RT Plotter")

        self.running = False
        self.worker = None
        self.stop_event = threading.Event()
        self.tiny = None

        self.data_lock = threading.Lock()
        self.maxhold = None
        self.freq = None
        self.power = None
        self.scan_count = 0

        self.f_low = None
        self.f_high = None
        self.num_points = None
        self.rbw_value = None

        self.build_gui()
        self.build_plot()

    def build_gui(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.X)

        ttk.Label(frame, text="Start Hz").grid(row=0, column=0, padx=3, pady=3)
        self.start = tk.StringVar(value="2300000000")
        ttk.Entry(frame, textvariable=self.start, width=15).grid(row=0, column=1, padx=3, pady=3)

        ttk.Label(frame, text="End Hz").grid(row=0, column=2, padx=3, pady=3)
        self.end = tk.StringVar(value="2500000000")
        ttk.Entry(frame, textvariable=self.end, width=15).grid(row=0, column=3, padx=3, pady=3)

        ttk.Label(frame, text="Points").grid(row=0, column=4, padx=3, pady=3)
        self.points_var = tk.StringVar(value="401")
        ttk.Entry(frame, textvariable=self.points_var, width=8).grid(row=0, column=5, padx=3, pady=3)

        ttk.Label(frame, text="RBW").grid(row=0, column=6, padx=3, pady=3)
        self.rbw = tk.StringVar(value="0")
        ttk.Entry(frame, textvariable=self.rbw, width=8).grid(row=0, column=7, padx=3, pady=3)

        ttk.Label(frame, text="Mode").grid(row=1, column=0, padx=3, pady=3)
        self.mode = tk.StringVar(value="Raw")
        ttk.Combobox(
            frame,
            textvariable=self.mode,
            values=["Raw", "Max Hold"],
            state="readonly",
            width=10
        ).grid(row=1, column=1, padx=3, pady=3)

        self.port = tk.StringVar()

        ttk.Button(frame, text="Auto Detect", command=self.detect).grid(row=1, column=2, padx=3, pady=3)

        self.start_btn = ttk.Button(frame, text="Start", command=self.start_scan)
        self.start_btn.grid(row=1, column=3, padx=3, pady=3)

        self.stop_btn = ttk.Button(frame, text="Stop", command=self.stop_scan, state=tk.DISABLED)
        self.stop_btn.grid(row=1, column=4, padx=3, pady=3)

        ttk.Button(frame, text="Reset MaxHold", command=self.reset_max).grid(row=1, column=5, padx=3, pady=3)
        ttk.Button(frame, text="Snapshot", command=self.take_snapshot).grid(row=1, column=6, padx=3, pady=3)

        self.status = tk.StringVar(value="Idle")
        ttk.Label(self.root, textvariable=self.status, relief=tk.SUNKEN, anchor="w").pack(fill=tk.X)

    def build_plot(self):
        self.fig = Figure(figsize=(10, 5))
        self.ax = self.fig.add_subplot(111)

        self.raw_line, = self.ax.plot([], [], label="Raw")
        self.max_line, = self.ax.plot([], [], label="Max Hold")

        self.ax.set_xlabel("Frequency")
        self.ax.set_ylabel("dBm")
        self.ax.grid(True)
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def detect(self):
        try:
            p = getport()
            self.port.set(p)
            self.status.set(f"tinySA detected on {p}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def reset_max(self):
        with self.data_lock:
            self.maxhold = None

    def clear_plot_data(self):
        with self.data_lock:
            self.maxhold = None
            self.freq = None
            self.power = None
            self.scan_count = 0

        self.raw_line.set_data([], [])
        self.max_line.set_data([], [])
        self.canvas.draw_idle()

    def parse_inputs(self):
        port = self.port.get().strip() or getport()
        f_low = float(self.start.get())
        f_high = float(self.end.get())
        num_points = int(self.points_var.get())
        rbw_value = float(self.rbw.get())

        if f_low >= f_high:
            raise ValueError("Start frequency must be less than end frequency")
        if num_points < 2:
            raise ValueError("Points must be at least 2")
        if rbw_value < 0:
            raise ValueError("RBW must be 0 or positive")

        return port, f_low, f_high, num_points, rbw_value

    def start_scan(self):
        if self.running:
            self.stop_scan()

        try:
            port, self.f_low, self.f_high, self.num_points, self.rbw_value = self.parse_inputs()
        except Exception as e:
            messagebox.showerror("Input Error", str(e))
            return

        try:
            self.tiny = TinySA(port)
        except Exception as e:
            messagebox.showerror("Serial Error", str(e))
            return

        self.clear_plot_data()

        with self.data_lock:
            self.freq = np.linspace(self.f_low, self.f_high, self.num_points)

        self.stop_event.clear()
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status.set("Scanning...")

        self.worker = threading.Thread(target=self.loop, daemon=True)
        self.worker.start()

    def stop_scan(self):
        self.running = False
        self.stop_event.set()

        tiny = self.tiny
        self.tiny = None

        if tiny is not None:
            try:
                tiny.close()
            except Exception:
                pass

        worker = self.worker
        if worker and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=2)

        self.worker = None

        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status.set("Stopped")

    def loop(self):
        while not self.stop_event.is_set():
            try:
                tiny = self.tiny
                if tiny is None:
                    break

                t0 = time.time()

                power = tiny.scan(
                    self.f_low,
                    self.f_high,
                    self.num_points,
                    self.rbw_value,
                    stop_event=self.stop_event
                )

                if self.stop_event.is_set():
                    break

                dt = time.time() - t0

                with self.data_lock:
                    self.power = power

                    if self.maxhold is None:
                        self.maxhold = power.copy()
                    else:
                        self.maxhold = np.maximum(self.maxhold, power)

                    self.scan_count += 1
                    count = self.scan_count

                self.root.after(0, lambda c=count, d=dt: self.status.set(f"scan {c}  {d:.2f}s"))

            except Exception as e:
                if self.stop_event.is_set():
                    break

                err = str(e)

                def handle_error():
                    self.status.set(f"Error: {err}")
                    self.stop_scan()
                    messagebox.showerror("Scan Error", err)

                self.root.after(0, handle_error)
                break

    def update_plot(self):
        with self.data_lock:
            freq = None if self.freq is None else self.freq.copy()
            power = None if self.power is None else self.power.copy()
            maxhold = None if self.maxhold is None else self.maxhold.copy()

        if freq is not None and power is not None:
            if self.mode.get() == "Raw":
                self.raw_line.set_data(freq, power)
                self.raw_line.set_visible(True)
                self.max_line.set_visible(False)
                y = power
            else:
                if maxhold is not None:
                    self.max_line.set_data(freq, maxhold)
                    self.raw_line.set_visible(False)
                    self.max_line.set_visible(True)
                    y = maxhold
                else:
                    self.raw_line.set_data(freq, power)
                    self.raw_line.set_visible(True)
                    self.max_line.set_visible(False)
                    y = power

            self.ax.set_xlim(freq[0], freq[-1])

            ymin = float(np.min(y)) - 5
            ymax = float(np.max(y)) + 5

            if ymin == ymax:
                ymin -= 1
                ymax += 1

            self.ax.set_ylim(ymin, ymax)
            self.canvas.draw_idle()

        self.root.after(100, self.update_plot)

    def _run_capture_command(self, cmd):
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "capture command failed")

    def take_snapshot(self):
        try:
            self.root.update_idletasks()
            self.root.update()

            x = int(self.root.winfo_rootx())
            y = int(self.root.winfo_rooty())
            w = int(self.root.winfo_width())
            h = int(self.root.winfo_height())

            if w <= 1 or h <= 1:
                raise RuntimeError("Invalid window size")

            script_dir = os.path.dirname(os.path.abspath(__file__))
            filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(script_dir, filename)

            errors = []

            if shutil.which("import"):
                try:
                    self._run_capture_command([
                        "import",
                        "-window", "root",
                        "-crop", f"{w}x{h}+{x}+{y}",
                        filepath
                    ])
                    self.status.set(f"Snapshot saved: {filepath}")
                    return
                except Exception as e:
                    errors.append(f"import: {e}")

            if shutil.which("gnome-screenshot"):
                try:
                    self._run_capture_command([
                        "gnome-screenshot",
                        "-f",
                        filepath
                    ])
                    self.status.set(
                        f"Snapshot saved full screen to {filepath} "
                        f"(gnome-screenshot cannot reliably crop this window in all setups)"
                    )
                    return
                except Exception as e:
                    errors.append(f"gnome-screenshot: {e}")

            if shutil.which("grim"):
                try:
                    self._run_capture_command([
                        "grim",
                        "-g",
                        f"{x},{y} {w}x{h}",
                        filepath
                    ])
                    self.status.set(f"Snapshot saved: {filepath}")
                    return
                except Exception as e:
                    errors.append(f"grim: {e}")

            if shutil.which("spectacle"):
                try:
                    self._run_capture_command([
                        "spectacle",
                        "-b",
                        "-n",
                        "-o",
                        filepath
                    ])
                    self.status.set(
                        f"Snapshot saved full screen to {filepath} "
                        f"(spectacle fallback)"
                    )
                    return
                except Exception as e:
                    errors.append(f"spectacle: {e}")

            raise RuntimeError(
                "No working screenshot tool found.\n\n"
                "Install one of these system tools:\n"
                "sudo apt install imagemagick\n"
                "or\n"
                "sudo apt install gnome-screenshot\n"
                "or on Wayland:\n"
                "sudo apt install grim\n\n"
                "Details:\n" + "\n".join(errors)
            )

        except Exception as e:
            messagebox.showerror("Snapshot Error", str(e))

    def on_close(self):
        self.stop_scan()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = RTPlotter(root)
    root.after(100, app.update_plot)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
