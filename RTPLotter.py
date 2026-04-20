#!/usr/bin/env python3

# tinySA command-line sweep tool:
# runs automatic sweeps by default, saves PNG/CSV/PDF outputs, and never uses GUI or XLSX.

import os
import csv
import time
import struct
import signal
import argparse
import threading
from datetime import datetime

import numpy as np
import serial
from serial.tools import list_ports

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages


VID = 0x0483
PID = 0x5740

DEFAULT_START_HZ = 2300000000
DEFAULT_END_HZ = 2500000000
DEFAULT_POINTS = 401
DEFAULT_RBW = 0
DEFAULT_MODE = "auto"
DEFAULT_SWEEPS = 30
DEFAULT_WAIT_SECONDS = 1.0


STOP_EVENT = threading.Event()


def getport():
    for device in list_ports.comports():
        if device.vid == VID and device.pid == PID:
            return device.device
    raise OSError("tinySA not found")


def format_dt(dt_obj):
    return dt_obj.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_plot_png(filepath, freq, power, title):
    fig = Figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.plot(freq, power, label="Sweep")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("dBm")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)


def save_sweep_csv(filepath, freq, power, export_time, scan_count, sweep_index=None):
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frequency_Hz", "Power_dBm", "Export_Timestamp", "Scan_Count", "Sweep_Index"])
        for i in range(len(freq)):
            writer.writerow([
                float(freq[i]),
                float(power[i]),
                export_time,
                int(scan_count),
                "" if sweep_index is None else int(sweep_index),
            ])


def save_summary_csv(filepath, records):
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Sweep_Index", "Start_Time", "End_Time", "Duration_s", "Image_File", "CSV_File"])
        for record in records:
            writer.writerow([
                int(record["index"]),
                record["start"],
                record["end"],
                float(record["duration"]),
                os.path.basename(record["image_path"]),
                os.path.basename(record["csv_path"]),
            ])


def save_summary_pdf(pdf_path, records, run_started_at, run_finished_at, freq):
    with PdfPages(pdf_path) as pdf:
        summary_fig = Figure(figsize=(11.69, 8.27))
        ax = summary_fig.add_subplot(111)
        ax.axis("off")

        title_lines = [
            "Automatic Sweep Report",
            f"Run folder: {os.path.basename(os.path.dirname(pdf_path))}",
            f"Run started: {format_dt(run_started_at)}",
            f"Run finished: {format_dt(run_finished_at)}",
            f"Frequency range: {freq[0]:.0f} Hz to {freq[-1]:.0f} Hz",
            f"Total sweeps saved: {len(records)}",
        ]

        y = 0.95
        for line in title_lines:
            ax.text(0.03, y, line, fontsize=12, va="top")
            y -= 0.055

        y -= 0.02
        ax.text(0.03, y, "Sweep Summary", fontsize=12, va="top")
        y -= 0.05

        header = f"{'Sweep':<8}{'Start Time':<24}{'End Time':<24}{'Duration (s)':<14}"
        ax.text(0.03, y, header, family="monospace", fontsize=10, va="top")
        y -= 0.035

        for record in records:
            line = (
                f"{record['index']:<8}"
                f"{record['start']:<24}"
                f"{record['end']:<24}"
                f"{record['duration']:<14.2f}"
            )
            ax.text(0.03, y, line, family="monospace", fontsize=9, va="top")
            y -= 0.03

            if y < 0.06:
                pdf.savefig(summary_fig)
                summary_fig = Figure(figsize=(11.69, 8.27))
                ax = summary_fig.add_subplot(111)
                ax.axis("off")
                y = 0.95

        summary_fig.tight_layout()
        pdf.savefig(summary_fig)

        for record in records:
            fig = Figure(figsize=(11.69, 8.27))
            ax = fig.add_subplot(111)

            ax.plot(record["freq"], record["power"], label="Sweep")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("dBm")
            ax.set_title(
                f"Sweep {record['index']} | Start: {record['start']} | End: {record['end']}"
            )
            ax.grid(True)
            ax.legend()
            fig.tight_layout()
            pdf.savefig(fig)


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
            power = arr / 32 - 128

            try:
                self.ser.timeout = 1
                self.ser.write(b"rbw auto\r")
                self._read_prompt()
            except Exception:
                pass

            return power


def parse_args():
    parser = argparse.ArgumentParser(
        description="tinySA command-line sweep tool (default: automatic mode, 30 sweeps, 1 second wait)."
    )
    parser.add_argument("--port", default=None, help="Serial port. Default: auto-detect tinySA.")
    parser.add_argument("--start-hz", type=float, default=DEFAULT_START_HZ, help=f"Start frequency in Hz. Default: {DEFAULT_START_HZ}")
    parser.add_argument("--end-hz", type=float, default=DEFAULT_END_HZ, help=f"End frequency in Hz. Default: {DEFAULT_END_HZ}")
    parser.add_argument("--points", type=int, default=DEFAULT_POINTS, help=f"Number of sweep points. Default: {DEFAULT_POINTS}")
    parser.add_argument("--rbw", type=float, default=DEFAULT_RBW, help=f"RBW in Hz, or 0 for auto. Default: {DEFAULT_RBW}")
    parser.add_argument("--mode", choices=["auto", "live"], default=DEFAULT_MODE, help=f"Run mode. Default: {DEFAULT_MODE}")
    parser.add_argument("--sweeps", type=int, default=DEFAULT_SWEEPS, help=f"Number of sweeps in automatic mode. Default: {DEFAULT_SWEEPS}")
    parser.add_argument("--wait", type=float, default=DEFAULT_WAIT_SECONDS, help=f"Wait time between sweeps in automatic mode. Default: {DEFAULT_WAIT_SECONDS}")
    parser.add_argument("--output-dir", default=None, help="Output directory. Default: script directory.")
    return parser.parse_args()


def validate_args(args):
    if args.start_hz >= args.end_hz:
        raise ValueError("Start frequency must be less than end frequency")
    if args.points < 2:
        raise ValueError("Points must be at least 2")
    if args.rbw < 0:
        raise ValueError("RBW must be 0 or positive")
    if args.sweeps < 1:
        raise ValueError("Sweeps must be at least 1")
    if args.wait < 0:
        raise ValueError("Wait time must be 0 or positive")


def create_run_folder(base_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(base_dir, timestamp)
    ensure_dir(folder)
    return folder, timestamp


def handle_signal(signum, frame):
    STOP_EVENT.set()


def run_auto_mode(tiny, args, freq, base_dir):
    folder, timestamp = create_run_folder(base_dir)
    run_started_dt = datetime.now()

    summary_csv_path = os.path.join(folder, f"{timestamp}_sweeps_summary.csv")
    report_pdf_path = os.path.join(folder, f"{timestamp}_sweeps_report.pdf")
    records = []
    completed_sweep_durations = []

    print(f"[INFO] Automatic sweep mode started")
    print(f"[INFO] Output folder: {folder}")
    print(f"[INFO] Sweeps: {args.sweeps}")
    print(f"[INFO] Wait between sweeps: {args.wait:.2f} s")

    for sweep_idx in range(1, args.sweeps + 1):
        if STOP_EVENT.is_set():
            print("[INFO] Stop requested")
            break

        sweep_start_dt = datetime.now()
        t0 = time.time()

        power = tiny.scan(
            args.start_hz,
            args.end_hz,
            args.points,
            args.rbw,
            stop_event=STOP_EVENT
        )

        if STOP_EVENT.is_set():
            print("[INFO] Stop requested")
            break

        duration_seconds = time.time() - t0
        sweep_end_dt = datetime.now()
        completed_sweep_durations.append(duration_seconds)

        image_name = f"sweep_{sweep_idx:03d}_{sweep_start_dt.strftime('%Y%m%d_%H%M%S')}.png"
        csv_name = f"sweep_{sweep_idx:03d}_{sweep_start_dt.strftime('%Y%m%d_%H%M%S')}.csv"

        image_path = os.path.join(folder, image_name)
        csv_path = os.path.join(folder, csv_name)

        save_plot_png(
            filepath=image_path,
            freq=freq,
            power=power,
            title=f"Sweep {sweep_idx} | Start: {format_dt(sweep_start_dt)} | End: {format_dt(sweep_end_dt)}"
        )

        save_sweep_csv(
            filepath=csv_path,
            freq=freq,
            power=power,
            export_time=format_dt(sweep_end_dt),
            scan_count=sweep_idx,
            sweep_index=sweep_idx
        )

        records.append({
            "index": sweep_idx,
            "start": format_dt(sweep_start_dt),
            "end": format_dt(sweep_end_dt),
            "duration": duration_seconds,
            "freq": freq.copy(),
            "power": power.copy(),
            "image_path": image_path,
            "csv_path": csv_path,
        })

        save_summary_csv(summary_csv_path, records)
        save_summary_pdf(report_pdf_path, records, run_started_dt, sweep_end_dt, freq)

        avg_duration = sum(completed_sweep_durations) / len(completed_sweep_durations)
        remaining_sweeps = args.sweeps - sweep_idx
        estimated_remaining_seconds = remaining_sweeps * avg_duration + remaining_sweeps * args.wait

        print(
            f"[INFO] Sweep {sweep_idx}/{args.sweeps} finished in {duration_seconds:.2f}s | "
            f"avg {avg_duration:.2f}s | remaining ~{estimated_remaining_seconds:.2f}s"
        )

        if sweep_idx < args.sweeps:
            wait_start = time.time()
            while True:
                if STOP_EVENT.is_set():
                    print("[INFO] Stop requested during wait")
                    break

                elapsed = time.time() - wait_start
                remaining = args.wait - elapsed
                if remaining <= 0:
                    break

                time.sleep(min(0.1, remaining))

            if STOP_EVENT.is_set():
                break

    run_finished_dt = datetime.now()

    if records:
        save_summary_csv(summary_csv_path, records)
        save_summary_pdf(report_pdf_path, records, run_started_dt, run_finished_dt, freq)
        print("[INFO] Automatic sweep mode finished")
        print(f"[INFO] Summary CSV: {summary_csv_path}")
        print(f"[INFO] PDF report:  {report_pdf_path}")
    else:
        print("[INFO] No sweeps were saved")


def run_live_mode(tiny, args, freq, base_dir):
    folder, timestamp = create_run_folder(base_dir)

    print(f"[INFO] Live mode started")
    print(f"[INFO] Output folder: {folder}")
    print("[INFO] Press Ctrl+C to stop")

    scan_count = 0
    completed_sweep_durations = []

    while not STOP_EVENT.is_set():
        t0 = time.time()

        power = tiny.scan(
            args.start_hz,
            args.end_hz,
            args.points,
            args.rbw,
            stop_event=STOP_EVENT
        )

        if STOP_EVENT.is_set():
            break

        scan_count += 1
        duration_seconds = time.time() - t0
        completed_sweep_durations.append(duration_seconds)

        now_dt = datetime.now()
        stamp = now_dt.strftime("%Y%m%d_%H%M%S")

        image_path = os.path.join(folder, f"live_sweep_{scan_count:06d}_{stamp}.png")
        csv_path = os.path.join(folder, f"live_sweep_{scan_count:06d}_{stamp}.csv")

        save_plot_png(
            filepath=image_path,
            freq=freq,
            power=power,
            title=f"Live Sweep {scan_count} | Time: {format_dt(now_dt)}"
        )

        save_sweep_csv(
            filepath=csv_path,
            freq=freq,
            power=power,
            export_time=format_dt(now_dt),
            scan_count=scan_count,
            sweep_index=scan_count
        )

        avg_duration = sum(completed_sweep_durations) / len(completed_sweep_durations)

        print(
            f"[INFO] Live sweep {scan_count} finished in {duration_seconds:.2f}s | "
            f"avg {avg_duration:.2f}s | CSV: {os.path.basename(csv_path)} | PNG: {os.path.basename(image_path)}"
        )

    print("[INFO] Live mode stopped")


def main():
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    args = parse_args()
    validate_args(args)

    port = args.port if args.port else getport()
    base_dir = args.output_dir if args.output_dir else os.path.dirname(os.path.abspath(__file__))
    base_dir = ensure_dir(base_dir)

    freq = np.linspace(args.start_hz, args.end_hz, args.points)

    tiny = None
    try:
        print(f"[INFO] Using port: {port}")
        print(f"[INFO] Frequency range: {args.start_hz:.0f} Hz to {args.end_hz:.0f} Hz")
        print(f"[INFO] Points: {args.points}")
        print(f"[INFO] RBW: {args.rbw}")

        tiny = TinySA(port)

        if args.mode == "auto":
            run_auto_mode(tiny, args, freq, base_dir)
        else:
            run_live_mode(tiny, args, freq, base_dir)

    finally:
        if tiny is not None:
            tiny.close()


if __name__ == "__main__":
    main()
