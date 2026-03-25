#!/usr/bin/env python3

import time
import struct
import threading
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
        self.ser = serial.Serial(port,115200,timeout=1)

    def scan(self,f_low,f_high,points,rbw):

        while self.ser.in_waiting:
            self.ser.read_all()
            time.sleep(0.05)

        if rbw==0:
            rbw_k=(f_high-f_low)*7e-6
        else:
            rbw_k=rbw/1e3

        rbw_k=max(3,min(600,rbw_k))

        self.ser.write(f"rbw {int(rbw_k)}\r".encode())
        self.ser.read_until(b"ch> ")

        timeout=((f_high-f_low)/20e3)/(rbw_k**2)+points/500+1
        self.ser.timeout=timeout*2

        self.ser.write(f"scanraw {int(f_low)} {int(f_high)} {int(points)}\r".encode())

        self.ser.read_until(b"{")
        raw=self.ser.read_until(b"}ch> ")

        self.ser.write(b"rbw auto\r")
        self.ser.read_until(b"ch> ")

        payload=raw[:-5]

        data=struct.unpack("<"+"xH"*points,payload)

        arr=np.array(data,dtype=np.uint16)

        SCALE=128
        power=arr/32-SCALE

        return power


class RTPlotter:

    def __init__(self,root):

        self.root=root
        root.title("tinySA RT Plotter")

        self.running=False
        self.maxhold=None
        self.freq=None
        self.power=None
        self.scan_count=0

        self.build_gui()
        self.build_plot()

    def build_gui(self):

        frame=ttk.Frame(self.root,padding=10)
        frame.pack(fill=tk.X)

        ttk.Label(frame,text="Start Hz").grid(row=0,column=0)
        self.start=tk.StringVar(value="2300000000")
        ttk.Entry(frame,textvariable=self.start,width=15).grid(row=0,column=1)

        ttk.Label(frame,text="End Hz").grid(row=0,column=2)
        self.end=tk.StringVar(value="2500000000")
        ttk.Entry(frame,textvariable=self.end,width=15).grid(row=0,column=3)

        ttk.Label(frame,text="Points").grid(row=0,column=4)
        self.points=tk.StringVar(value="401")
        ttk.Entry(frame,textvariable=self.points,width=6).grid(row=0,column=5)

        ttk.Label(frame,text="RBW").grid(row=0,column=6)
        self.rbw=tk.StringVar(value="0")
        ttk.Entry(frame,textvariable=self.rbw,width=6).grid(row=0,column=7)

        ttk.Label(frame,text="Mode").grid(row=1,column=0)
        self.mode=tk.StringVar(value="Raw")

        ttk.Combobox(frame,
                     textvariable=self.mode,
                     values=["Raw","Max Hold"],
                     state="readonly",
                     width=10).grid(row=1,column=1)

        self.port=tk.StringVar()

        ttk.Button(frame,text="Auto Detect",command=self.detect).grid(row=1,column=2)

        self.start_btn=ttk.Button(frame,text="Start",command=self.start_scan)
        self.start_btn.grid(row=1,column=3)

        self.stop_btn=ttk.Button(frame,text="Stop",command=self.stop_scan,state=tk.DISABLED)
        self.stop_btn.grid(row=1,column=4)

        ttk.Button(frame,text="Reset MaxHold",command=self.reset_max).grid(row=1,column=5)

        self.status=tk.StringVar(value="Idle")
        ttk.Label(self.root,textvariable=self.status,relief=tk.SUNKEN).pack(fill=tk.X)

    def build_plot(self):

        self.fig=Figure(figsize=(10,5))
        self.ax=self.fig.add_subplot(111)

        self.raw_line,=self.ax.plot([],[],label="Raw")
        self.max_line,=self.ax.plot([],[],label="Max Hold")

        self.ax.set_xlabel("Frequency")
        self.ax.set_ylabel("dBm")
        self.ax.grid(True)

        self.ax.legend()

        self.canvas=FigureCanvasTkAgg(self.fig,master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True)

    def detect(self):

        try:
            p=getport()
            self.port.set(p)
            self.status.set(f"tinySA detected on {p}")
        except Exception as e:
            messagebox.showerror("Error",str(e))

    def reset_max(self):

        self.maxhold=None

    def start_scan(self):

        try:

            port=self.port.get() or getport()

            self.tiny=TinySA(port)

            self.f_low=float(self.start.get())
            self.f_high=float(self.end.get())
            self.points=int(self.points.get())
            self.rbw=float(self.rbw.get())

            self.freq=np.linspace(self.f_low,self.f_high,self.points)

        except Exception as e:
            messagebox.showerror("Input Error",str(e))
            return

        self.running=True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        threading.Thread(target=self.loop,daemon=True).start()

    def stop_scan(self):

        self.running=False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def loop(self):

        while self.running:

            t0=time.time()

            power=self.tiny.scan(self.f_low,self.f_high,self.points,self.rbw)

            self.power=power

            if self.maxhold is None:
                self.maxhold=power.copy()
            else:
                self.maxhold=np.maximum(self.maxhold,power)

            dt=time.time()-t0

            self.scan_count+=1

            self.status.set(f"scan {self.scan_count}  {dt:.2f}s")

    def update_plot(self):

        if self.power is not None:

            if self.mode.get()=="Raw":

                self.raw_line.set_data(self.freq,self.power)
                self.raw_line.set_visible(True)
                self.max_line.set_visible(False)

                y=self.power

            else:

                self.max_line.set_data(self.freq,self.maxhold)
                self.raw_line.set_visible(False)
                self.max_line.set_visible(True)

                y=self.maxhold

            self.ax.set_xlim(self.freq[0],self.freq[-1])

            ymin=np.min(y)-5
            ymax=np.max(y)+5

            self.ax.set_ylim(ymin,ymax)

            self.canvas.draw_idle()

        self.root.after(100,self.update_plot)


def main():

    root=tk.Tk()

    app=RTPlotter(root)

    root.after(100,app.update_plot)

    root.mainloop()


if __name__=="__main__":
    main()
