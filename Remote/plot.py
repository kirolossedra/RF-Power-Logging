#!/usr/bin/env python3

# tinySA command-line sweep tool:
# runs automatic sweeps by default, keeps auto-mode sweep data in memory first,
# and exports PNG/CSV/PDF only after acquisition stops or finishes.

import os
import csv
import time
import struct
import signal
import argparse
import threading
from datetime import datetime, timedelta

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
DEFAULT_SWEEPS = 200
DEFAULT_PERIOD_SECONDS = 1.0


STOP_EVENT = threading.Event()


def getport():
    for device in list_ports.comports():
        if device.vid == VID and device.pid == PID:
            return device.device
    raise OSError("tinySA not found")


def format_dt(dt_obj):
    return dt_obj.strftime("%Y-%m-%d %H:%M:%S")


def format_dt_ms(dt_obj):
    return dt_obj.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


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
        writer.writerow([
            "Frequency_Hz",
            "Power_dBm",
            "Export_Timestamp",
            "Scan_Count",
            "Sweep_Index",
        ])
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
        writer.writerow([
            "Sweep_Index",
            "Scheduled_Start_Time",
            "Actual_Start_Time",
            "End_Time",
            "Duration_s",
            "Start_Offset_s",
            "Automatic_Buffer_s",
            "Image_File",
            "CSV_File",
        ])
        for record in records:
            writer.writerow([
                int(record["index"]),
                record["scheduled_start"],
                record["actual_start"],
                record["end"],
                float(record["duration"]),
                float(record["start_offset"]),
                float(record["auto_buffer"]),
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
            f"Run started: {format_dt_ms(run_started_at)}",
            f"Run finished: {format_dt_ms(run_finished_at)}",
            f"Frequency range: {freq[0]:.0f} Hz to {freq[-1]:.0f} Hz",
            f"Total sweeps saved: {len(records)}",
            "Scheduling: synchronized to target start times",
            "Export policy: saved once after acquisition ends",
        ]

        y = 0.95
        for line in title_lines:
            ax.text(0.03, y, line, fontsize=12, va="top")
            y -= 0.05

        y -= 0.01
        ax.text(0.03, y, "Sweep Summary", fontsize=12, va="top")
        y -= 0.045

        header = (
            f"{'Sweep':<8}"
            f"{'Scheduled Start':<26}"
            f"{'Actual Start':<26}"
            f"{'End':<26}"
            f"{'Dur(s)':<10}"
            f"{'Offset(s)':<10}"
            f"{'Buffer(s)':<10}"
        )
        ax.text(0.03, y, header, family="monospace", fontsize=8.0, va="top")
        y -= 0.03

        for record in records:
            line = (
                f"{record['index']:<8}"
                f"{record['scheduled_start']:<26}"
                f"{record['actual_start']:<26}"
                f"{record['end']:<26}"
                f"{record['duration']:<10.3f}"
                f"{record['start_offset']:<10.3f}"
                f"{record['auto_buffer']:<10.3f}"
            )
            ax.text(0.03, y, line, family="monospace", fontsize=7.2, va="top")
            y -= 0.025

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
                f"Sweep {record['index']} | "
                f"Scheduled: {record['scheduled_start']} | "
                f"Actual: {record['actual_start']} | "
                f"End: {record['end']}"
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

            return power.astype(np.float32, copy=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="tinySA command-line sweep tool (automatic mode stores sweeps in memory first, then exports after acquisition ends)."
    )
    parser.add_argument("--port", default=None, help="Serial port. Default: auto-detect tinySA.")
    parser.add_argument("--start-hz", type=float, default=DEFAULT_START_HZ, help=f"Start frequency in Hz. Default: {DEFAULT_START_HZ}")
    parser.add_argument("--end-hz", type=float, default=DEFAULT_END_HZ, help=f"End frequency in Hz. Default: {DEFAULT_END_HZ}")
    parser.add_argument("--points", type=int, default=DEFAULT_POINTS, help=f"Number of sweep points. Default: {DEFAULT_POINTS}")
    parser.add_argument("--rbw", type=float, default=DEFAULT_RBW, help=f"RBW in Hz, or 0 for auto. Default: {DEFAULT_RBW}")
    parser.add_argument("--mode", choices=["auto", "live"], default=DEFAULT_MODE, help=f"Run mode. Default: {DEFAULT_MODE}")
    parser.add_argument("--sweeps", type=int, default=DEFAULT_SWEEPS, help=f"Number of sweeps in automatic mode. Default: {DEFAULT_SWEEPS}")
    parser.add_argument(
        "--period",
        type=float,
        default=DEFAULT_PERIOD_SECONDS,
        help=f"Fixed interval between scheduled sweep START times in automatic mode. Default: {DEFAULT_PERIOD_SECONDS}"
    )
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
    if args.period <= 0:
        raise ValueError("Period must be positive")


def create_run_folder(base_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(base_dir, timestamp)
    ensure_dir(folder)
    return folder, timestamp


def handle_signal(signum, frame):
    STOP_EVENT.set()


def sleep_until(target_monotonic):
    while True:
        if STOP_EVENT.is_set():
            return False

        remaining = target_monotonic - time.monotonic()
        if remaining <= 0:
            return True

        time.sleep(min(0.01, remaining))


def export_auto_results(
    folder,
    timestamp,
    freq,
    run_started_dt,
    run_finished_dt,
    completed_count,
    scheduled_start_dt_arr,
    actual_start_dt_arr,
    end_dt_arr,
    duration_arr,
    start_offset_arr,
    auto_buffer_arr,
    power_arr,
):
    summary_csv_path = os.path.join(folder, f"{timestamp}_sweeps_summary.csv")
    report_pdf_path = os.path.join(folder, f"{timestamp}_sweeps_report.pdf")

    print(f"[INFO] Exporting {completed_count} completed sweeps")

    records = []

    for i in range(completed_count):
        sweep_idx = i + 1

        scheduled_start_dt = scheduled_start_dt_arr[i]
        actual_start_dt = actual_start_dt_arr[i]
        sweep_end_dt = end_dt_arr[i]
        power = power_arr[i].copy()

        image_name = f"sweep_{sweep_idx:03d}_{actual_start_dt.strftime('%Y%m%d_%H%M%S_%f')[:-3]}.png"
        csv_name = f"sweep_{sweep_idx:03d}_{actual_start_dt.strftime('%Y%m%d_%H%M%S_%f')[:-3]}.csv"

        image_path = os.path.join(folder, image_name)
        csv_path = os.path.join(folder, csv_name)

        save_plot_png(
            filepath=image_path,
            freq=freq,
            power=power,
            title=(
                f"Sweep {sweep_idx} | "
                f"Scheduled: {format_dt_ms(scheduled_start_dt)} | "
                f"Actual: {format_dt_ms(actual_start_dt)} | "
                f"End: {format_dt_ms(sweep_end_dt)}"
            )
        )

        save_sweep_csv(
            filepath=csv_path,
            freq=freq,
            power=power,
            export_time=format_dt_ms(sweep_end_dt),
            scan_count=sweep_idx,
            sweep_index=sweep_idx
        )

        records.append({
            "index": sweep_idx,
            "scheduled_start": format_dt_ms(scheduled_start_dt),
            "actual_start": format_dt_ms(actual_start_dt),
            "end": format_dt_ms(sweep_end_dt),
            "duration": float(duration_arr[i]),
            "start_offset": float(start_offset_arr[i]),
            "auto_buffer": float(auto_buffer_arr[i]),
            "freq": freq.copy(),
            "power": power,
            "image_path": image_path,
            "csv_path": csv_path,
        })

        if sweep_idx % 10 == 0 or sweep_idx == completed_count:
            print(f"[INFO] Exported {sweep_idx}/{completed_count} sweeps")

    save_summary_csv(summary_csv_path, records)
    save_summary_pdf(report_pdf_path, records, run_started_dt, run_finished_dt, freq)

    print("[INFO] Export finished")
    print(f"[INFO] Summary CSV: {summary_csv_path}")
    print(f"[INFO] PDF report:  {report_pdf_path}")


def run_auto_mode(tiny, args, freq, base_dir):
    folder, timestamp = create_run_folder(base_dir)
    run_started_dt = datetime.now()
    run_started_monotonic = time.monotonic()

    all_power = np.empty((args.sweeps, args.points), dtype=np.float32)
    scheduled_start_dt_arr = [None] * args.sweeps
    actual_start_dt_arr = [None] * args.sweeps
    end_dt_arr = [None] * args.sweeps
    duration_arr = np.empty(args.sweeps, dtype=np.float64)
    start_offset_arr = np.empty(args.sweeps, dtype=np.float64)
    auto_buffer_arr = np.empty(args.sweeps, dtype=np.float64)

    completed_count = 0
    completed_sweep_durations = []

    print(f"[INFO] Automatic sweep mode started")
    print(f"[INFO] Output folder: {folder}")
    print(f"[INFO] Sweeps: {args.sweeps}")
    print(f"[INFO] Fixed start period: {args.period:.3f} s")
    print("[INFO] Auto mode stores sweeps in memory first; export happens after acquisition ends")

    for sweep_idx in range(1, args.sweeps + 1):
        if STOP_EVENT.is_set():
            print("[INFO] Stop requested")
            break

        scheduled_start_monotonic = run_started_monotonic + (sweep_idx - 1) * args.period
        scheduled_start_dt = run_started_dt + timedelta(seconds=(sweep_idx - 1) * args.period)

        if sweep_idx > 1:
            reached = sleep_until(scheduled_start_monotonic)
            if not reached:
                print("[INFO] Stop requested during automatic buffer")
                break

        actual_start_monotonic = time.monotonic()
        actual_start_dt = datetime.now()

        start_offset_seconds = actual_start_monotonic - scheduled_start_monotonic
        if start_offset_seconds > 0.001:
            print(f"[WARN] Sweep {sweep_idx} started late by {start_offset_seconds:.3f}s")

        t0 = actual_start_monotonic

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

        duration_seconds = time.monotonic() - t0
        end_dt = datetime.now()
        completed_sweep_durations.append(duration_seconds)

        next_scheduled_start_monotonic = run_started_monotonic + sweep_idx * args.period
        automatic_buffer_seconds = max(0.0, next_scheduled_start_monotonic - time.monotonic())

        row = completed_count
        all_power[row, :] = power
        scheduled_start_dt_arr[row] = scheduled_start_dt
        actual_start_dt_arr[row] = actual_start_dt
        end_dt_arr[row] = end_dt
        duration_arr[row] = duration_seconds
        start_offset_arr[row] = start_offset_seconds
        auto_buffer_arr[row] = automatic_buffer_seconds
        completed_count += 1

        avg_duration = sum(completed_sweep_durations) / len(completed_sweep_durations)
        remaining_sweeps = args.sweeps - sweep_idx
        estimated_remaining_seconds = remaining_sweeps * args.period

        print(
            f"[INFO] Sweep {sweep_idx}/{args.sweeps} | "
            f"scheduled {format_dt_ms(scheduled_start_dt)} | "
            f"actual {format_dt_ms(actual_start_dt)} | "
            f"duration {duration_seconds:.3f}s | "
            f"start offset {start_offset_seconds:.3f}s | "
            f"auto buffer {automatic_buffer_seconds:.3f}s | "
            f"avg scan {avg_duration:.3f}s | "
            f"remaining ~{estimated_remaining_seconds:.3f}s"
        )

    run_finished_dt = datetime.now()

    if completed_count > 0:
        export_auto_results(
            folder=folder,
            timestamp=timestamp,
            freq=freq,
            run_started_dt=run_started_dt,
            run_finished_dt=run_finished_dt,
            completed_count=completed_count,
            scheduled_start_dt_arr=scheduled_start_dt_arr,
            actual_start_dt_arr=actual_start_dt_arr,
            end_dt_arr=end_dt_arr,
            duration_arr=duration_arr,
            start_offset_arr=start_offset_arr,
            auto_buffer_arr=auto_buffer_arr,
            power_arr=all_power,
        )
        print("[INFO] Automatic sweep mode finished")
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
        t0 = time.monotonic()

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
        duration_seconds = time.monotonic() - t0
        completed_sweep_durations.append(duration_seconds)

        now_dt = datetime.now()
        stamp = now_dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]

        image_path = os.path.join(folder, f"live_sweep_{scan_count:06d}_{stamp}.png")
        csv_path = os.path.join(folder, f"live_sweep_{scan_count:06d}_{stamp}.csv")

        save_plot_png(
            filepath=image_path,
            freq=freq,
            power=power,
            title=f"Live Sweep {scan_count} | Time: {format_dt_ms(now_dt)}"
        )

        save_sweep_csv(
            filepath=csv_path,
            freq=freq,
            power=power,
            export_time=format_dt_ms(now_dt),
            scan_count=scan_count,
            sweep_index=scan_count
        )

        avg_duration = sum(completed_sweep_durations) / len(completed_sweep_durations)

        print(
            f"[INFO] Live sweep {scan_count} finished in {duration_seconds:.3f}s | "
            f"avg {avg_duration:.3f}s | CSV: {os.path.basename(csv_path)} | PNG: {os.path.basename(image_path)}"
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
        print(f"[INFO] Auto-mode start period: {args.period:.3f}s")

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
