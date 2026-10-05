"""Resource guard — run a long local-LLM job and stop it if the machine overloads.

Local models (Ollama on CPU) can saturate the machine for hours. This wrapper
starts the command, samples CPU, RAM and swap every few seconds, logs a line
each minute, and on a sustained breach stops the command (process tree) and
asks Ollama to unload its models so the memory comes back.

Limits (all configurable):
  * RAM in use >= --max-ram-pct (default 90%) on 2 consecutive samples
  * available RAM < --min-free-gb (default 4 GB)
  * swap growth > --max-swap-growth-gb since start (default 2 GB) — thrashing
  * CPU >= --max-cpu-pct (default 95%) averaged over --cpu-window-s (180 s)

Exit code: the command's own, or 3 when the guard stopped it.

Run:  python -m eval.resource_guard -- python cli.py eval --with-llm
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from collections import deque
from datetime import datetime

import psutil

import config

GUARD_EXIT = 3


def _ollama_unload() -> list[str]:
    """Ask Ollama to drop every loaded model (keep_alive=0)."""
    try:
        with urllib.request.urlopen(f"{config.OLLAMA_HOST}/api/ps", timeout=3) as r:
            loaded = [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return []
    for name in loaded:
        try:
            req = urllib.request.Request(
                f"{config.OLLAMA_HOST}/api/generate",
                data=json.dumps({"model": name, "keep_alive": 0}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=10).read()
        except Exception:
            pass
    return loaded


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-ram-pct", type=float, default=90.0)
    ap.add_argument("--min-free-gb", type=float, default=4.0)
    ap.add_argument("--max-swap-growth-gb", type=float, default=2.0)
    ap.add_argument("--max-cpu-pct", type=float, default=95.0)
    ap.add_argument("--cpu-window-s", type=float, default=180.0)
    ap.add_argument("--interval-s", type=float, default=5.0)
    ap.add_argument("--log-every-s", type=float, default=60.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="command to run, after --")
    args = ap.parse_args(argv)
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        ap.error("no command given (put it after --)")

    def log(msg: str) -> None:
        print(f"[guard {datetime.now():%H:%M:%S}] {msg}", flush=True)

    swap0 = psutil.swap_memory().used
    cpu_hist: deque[float] = deque(maxlen=max(1, int(args.cpu_window_s / args.interval_s)))
    psutil.cpu_percent(None)                         # prime the CPU counter
    log(f"start: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    ram_strikes, last_log, peak = 0, 0.0, {"cpu": 0.0, "ram": 0.0}
    reason = None
    while proc.poll() is None:
        time.sleep(args.interval_s)
        cpu = psutil.cpu_percent(None)
        vm = psutil.virtual_memory()
        swap_growth = (psutil.swap_memory().used - swap0) / 2**30
        cpu_hist.append(cpu)
        peak["cpu"], peak["ram"] = max(peak["cpu"], cpu), max(peak["ram"], vm.percent)
        cpu_avg = sum(cpu_hist) / len(cpu_hist)

        ram_strikes = ram_strikes + 1 if vm.percent >= args.max_ram_pct else 0
        if ram_strikes >= 2:
            reason = f"RAM {vm.percent:.0f}% >= {args.max_ram_pct:.0f}%"
        elif vm.available / 2**30 < args.min_free_gb:
            reason = f"free RAM {vm.available / 2**30:.1f} GB < {args.min_free_gb} GB"
        elif swap_growth > args.max_swap_growth_gb:
            reason = f"swap grew {swap_growth:.1f} GB (> {args.max_swap_growth_gb} GB): thrashing"
        elif len(cpu_hist) == cpu_hist.maxlen and cpu_avg >= args.max_cpu_pct:
            reason = f"CPU {cpu_avg:.0f}% avg over {args.cpu_window_s:.0f}s >= {args.max_cpu_pct:.0f}%"
        if reason:
            log(f"OVERLOAD - {reason}. Stopping the job.")
            _kill_tree(proc)
            unloaded = _ollama_unload()
            log(f"stopped; unloaded Ollama models: {unloaded or 'none loaded'}")
            break
        if time.time() - last_log >= args.log_every_s:
            log(f"CPU {cpu:.0f}% (avg {cpu_avg:.0f}%) | RAM {vm.percent:.0f}% "
                f"({vm.available / 2**30:.1f} GB free) | swap +{swap_growth:.1f} GB")
            last_log = time.time()

    code = GUARD_EXIT if reason else proc.wait()
    log(f"end: exit {code} | peak CPU {peak['cpu']:.0f}% | peak RAM {peak['ram']:.0f}%")
    return code


if __name__ == "__main__":
    sys.exit(main())
