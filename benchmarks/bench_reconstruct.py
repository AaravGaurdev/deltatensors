"""
Reconstruction benchmark: deltatensors 0.2.0 vs the current code.

Builds synthetic Qwen2.5-shaped bfloat16 models (base + fine-tune), compresses
them with int4 into a v1 file (written by the real 0.2.0 code, extracted from
git) and a v2 file (current writer), then measures each reconstruction path in
a fresh subprocess:

  - wall time
  - peak RSS (resource.getrusage on Linux/macOS, PeakWorkingSetSize on Windows)
  - peak tracemalloc (Python + numpy heap; a second run with tracemalloc on)

plus a compute-only comparison (base and payloads already in memory, one CPU
core) for the int4 decode + add itself.

Results are written to benchmarks/results.md. Don't edit the numbers by hand;
re-run instead.

    python benchmarks/bench_reconstruct.py                      # 0.1B and 0.5B
    python benchmarks/bench_reconstruct.py --sizes 0.5B,1.5B --include-7b
    python benchmarks/bench_reconstruct.py --workdir /scratch/dt-bench --keep

Needs: numpy, torch + safetensors (for the 0.2.0 path), git (to extract
0.2.0), optionally psutil (core pinning, RAM checks) and CUDA.
Disk: about 2.7 x the bfloat16 model size at peak (base, fine-tune, deltas;
the fine-tune is deleted once the deltas exist, outputs after each run).
"""

from __future__ import annotations

import argparse
import datetime
import gc
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

OLD_REF = "8c605a9"  # deltatensors 0.2.0
MARK = "BENCH_RESULT "

# Qwen2.5-like shapes. "0.1B" is a small config for machines that can't hold
# the 0.2.0 path's full-model float32 copies at 0.5B.
CONFIGS = {
    "0.1B": dict(hidden=768, layers=8, inter=3072, kv=768, vocab=32000, tie=True),
    "0.5B": dict(hidden=896, layers=24, inter=4864, kv=128, vocab=151936, tie=True),
    "1.5B": dict(hidden=1536, layers=28, inter=8960, kv=256, vocab=151936, tie=True),
    "7B": dict(hidden=3584, layers=28, inter=18944, kv=512, vocab=152064, tie=False),
}


# ---------------------------------------------------------------------------
# synthetic models
# ---------------------------------------------------------------------------

def model_shapes(cfg):
    h, i, kv = cfg["hidden"], cfg["inter"], cfg["kv"]
    shapes = {"model.embed_tokens.weight": (cfg["vocab"], h), "model.norm.weight": (h,)}
    if not cfg["tie"]:
        shapes["lm_head.weight"] = (cfg["vocab"], h)
    for L in range(cfg["layers"]):
        p = f"model.layers.{L}."
        shapes.update({
            p + "self_attn.q_proj.weight": (h, h), p + "self_attn.q_proj.bias": (h,),
            p + "self_attn.k_proj.weight": (kv, h), p + "self_attn.k_proj.bias": (kv,),
            p + "self_attn.v_proj.weight": (kv, h), p + "self_attn.v_proj.bias": (kv,),
            p + "self_attn.o_proj.weight": (h, h),
            p + "mlp.gate_proj.weight": (i, h), p + "mlp.up_proj.weight": (i, h),
            p + "mlp.down_proj.weight": (h, i),
            p + "input_layernorm.weight": (h,), p + "post_attention_layernorm.weight": (h,),
        })
    return shapes


def n_params(cfg):
    return sum(int(np.prod(s)) for s in model_shapes(cfg).values())


def build_model(size, root: Path, log):
    """Write base/ and the two .wdelta files for one size; reuse if present."""
    from deltatensors.safetensors_io import ShardedWriter, f32_to_bf16

    d = root / size
    v1, v2 = d / "delta_v1.wdelta", d / "delta_v2.wdelta"
    if (d / "base").exists() and v1.exists() and v2.exists():
        return d
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    shapes = model_shapes(CONFIGS[size])
    names = sorted(shapes)
    log(f"[{size}] generating {n_params(CONFIGS[size]) / 1e9:.2f}B-parameter synthetic model")
    for which in ("base", "ft"):
        w = ShardedWriter(d / which, [(k, "bfloat16", shapes[k]) for k in names], shard_size="1GB")
        for idx, k in enumerate(names):
            rng = np.random.default_rng(idx)  # same base values for both
            x = rng.standard_normal(shapes[k], dtype=np.float32)
            x *= 0.02
            if which == "ft":
                x += np.random.default_rng(10_000 + idx).standard_normal(shapes[k], dtype=np.float32) * 0.001
            w.write(k, f32_to_bf16(x))
            del x
        w.commit()
    (d / "base" / "config.json").write_text(json.dumps({"model_type": "synthetic", "torch_dtype": "bfloat16"}))

    log(f"[{size}] writing v2 delta (current writer)")
    _quiet(lambda: __import__("deltatensors").save_delta_from_paths(
        v2, d / "ft", d / "base", strategy="int4", outlier_fraction=0.01))
    log(f"[{size}] writing v1 delta (deltatensors 0.2.0)")
    old = import_old(root)
    _quiet(lambda: old.save_delta_from_paths(v1, str(d / "ft"), str(d / "base"), strategy="int4",
                                              outlier_fraction=0.01))
    shutil.rmtree(d / "ft")
    return d


def _quiet(fn):
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


def import_old(root: Path):
    """deltatensors 0.2.0, extracted from git as package 'deltatensors_020'."""
    pkg_root = root / "_old"
    if not (pkg_root / "deltatensors_020" / "__init__.py").exists():
        data = subprocess.run(["git", "-C", str(REPO), "archive", "--format=zip", OLD_REF, "deltatensors"],
                              check=True, capture_output=True).stdout
        tmp = pkg_root / "_x"
        shutil.rmtree(tmp, ignore_errors=True)
        zipfile.ZipFile(io.BytesIO(data)).extractall(tmp)
        shutil.rmtree(pkg_root / "deltatensors_020", ignore_errors=True)
        (tmp / "deltatensors").rename(pkg_root / "deltatensors_020")
        shutil.rmtree(tmp)
    sys.path.insert(0, str(pkg_root))
    import deltatensors_020
    return deltatensors_020


# ---------------------------------------------------------------------------
# measurement (child processes)
# ---------------------------------------------------------------------------

def peak_rss_bytes() -> int:
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r if sys.platform == "darwin" else r * 1024
    except ImportError:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return int(pmc.PeakWorkingSetSize)


def pin_to_core(core: int) -> bool:
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {core})
            return True
        import psutil
        psutil.Process().cpu_affinity([core])
        return True
    except Exception:
        return False


def child(spec: dict) -> dict:
    import tracemalloc
    mode = spec["mode"]
    d = Path(spec["dir"])
    if spec.get("cache_dir"):
        os.environ["DELTATENSORS_CACHE_DIR"] = spec["cache_dir"]

    if mode == "compute":
        return compute_only(spec)

    # import everything the path needs before the baseline is taken
    if mode == "old_load":
        old = import_old(Path(spec["root"]))
    else:
        import deltatensors as dt
    if spec.get("device") == "cuda":
        import torch
        torch.zeros(1, device="cuda")
    gc.collect()
    baseline = peak_rss_bytes()
    if spec.get("tracemalloc"):
        tracemalloc.start()

    t0 = time.perf_counter()
    with _redirect():
        if mode == "old_load":
            out = old.load_delta_from_paths(str(d / "delta_v1.wdelta"), str(d / "base"))
        elif mode == "new_load":
            out = dt.load_delta_from_paths(d / spec["file"], d / "base", verify=spec["verify"],
                                           num_workers=spec["workers"], device=spec["device"])
        elif mode == "reconstruct":
            out = dt.reconstruct_to_safetensors(d / "base", d / spec["file"], spec["out"], verify=spec["verify"],
                                                num_workers=spec["workers"], device=spec["device"])
    wall = time.perf_counter() - t0
    res = {"wall_s": wall, "peak_rss": peak_rss_bytes(), "baseline_rss": baseline}
    if spec.get("tracemalloc"):
        res["tracemalloc_peak"] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    del out
    return res


def _redirect():
    import contextlib
    return contextlib.redirect_stdout(io.StringIO())


def compute_only(spec: dict) -> dict:
    """int4 decode + add with everything in memory, on one pinned core."""
    from deltatensors.compress_int4 import decompress_int4_add_
    from deltatensors.format import WDeltaReader
    from deltatensors.safetensors_io import SafetensorsDir, to_float32
    old = import_old(Path(spec["root"]))
    from deltatensors_020.compress_int4 import decompress_int4 as old_decompress_int4

    pinned = pin_to_core(spec["core"])
    d = Path(spec["dir"])
    budget = spec["elements"]
    with SafetensorsDir(d / "base") as base, WDeltaReader(d / "delta_v1.wdelta") as r1, \
            WDeltaReader(d / "delta_v2.wdelta") as r2:
        names, total = [], 0
        for k in sorted(base.tensors, key=lambda k: -base.tensors[k].numel):
            n = base.tensors[k].numel
            if n <= budget // 4 and total + n <= budget:
                names.append(k)
                total += n
        work = [(to_float32(base.read(k), "bfloat16"), r1.payload(k), r2.payload(k)) for k in names]
    del old

    times = {"old": 0.0, "new_v1": 0.0, "new_v2": 0.0}
    best = {k: float("inf") for k in times}
    for _ in range(spec["repeats"]):
        cur = dict.fromkeys(times, 0.0)
        for b, p1, p2 in work:
            t0 = time.perf_counter()
            out = (b + old_decompress_int4(p1)).astype(p1["dtype"])  # 0.2.0 load_delta_from_paths body
            cur["old"] += time.perf_counter() - t0
            del out
            for key, p in (("new_v1", p1), ("new_v2", p2)):
                buf = b.copy()  # stands in for reading the base into the scratch buffer (I/O)
                t0 = time.perf_counter()
                decompress_int4_add_(p, buf)
                cur[key] += time.perf_counter() - t0
                del buf
        best = {k: min(best[k], cur[k]) for k in best}
    return {"elements": total, "tensors": len(names), "pinned": pinned, **{f"{k}_s": v for k, v in best.items()}}


def run_child(spec: dict, timeout: float) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", json.dumps(spec)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    for line in p.stdout.splitlines():
        if line.startswith(MARK):
            return json.loads(line[len(MARK):])
    raise RuntimeError(f"child failed ({spec['mode']}):\n{p.stdout[-2000:]}\n{p.stderr[-4000:]}")


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def available_ram() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        return float("inf")


def machine_info() -> dict:
    info = {
        "date": datetime.date.today().isoformat(),
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        info["ram_available_gb_at_start"] = round(psutil.virtual_memory().available / 1e9, 1)
    except ImportError:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        info["commit"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                        capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    return info


def gb(x):
    return f"{x / 1e9:.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="0.1B,0.5B", help=f"comma-separated, from {list(CONFIGS)}")
    ap.add_argument("--include-7b", action="store_true", help="also run 7B (needs ~40 GB disk)")
    ap.add_argument("--workers", type=int, default=None, help="N for the multi-worker runs (default min(8, cpus))")
    ap.add_argument("--workdir", default=None, help="where synthetic models go (default: a temp dir)")
    ap.add_argument("--keep", action="store_true", help="keep the synthetic models")
    ap.add_argument("--no-old", action="store_true", help="skip the 0.2.0 paths")
    ap.add_argument("--no-cuda", action="store_true")
    ap.add_argument("--no-tracemalloc", action="store_true")
    ap.add_argument("--ignore-ram-check", default="", metavar="SIZES",
                    help="comma-separated sizes to run even if the RAM estimate says they won't fit")
    ap.add_argument("--core", type=int, default=0, help="CPU core for the compute-only run")
    ap.add_argument("--compute-elements", type=int, default=100_000_000)
    ap.add_argument("--results", default=str(REPO / "benchmarks" / "results.md"))
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--child", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        print(MARK + json.dumps(child(json.loads(args.child))), flush=True)
        return

    from deltatensors.reconstruct import default_num_workers
    workers = args.workers or default_num_workers()
    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    forced = {s.strip() for s in args.ignore_ram_check.split(",") if s.strip()}
    if args.include_7b and "7B" not in sizes:
        sizes.append("7B")
    try:
        import torch
        cuda = torch.cuda.is_available() and not args.no_cuda
    except ImportError:
        cuda = False

    root = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="dt-bench-"))
    root.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)  # noqa: E731
    info = machine_info()
    info["workers_N"] = workers
    results, notes = [], []

    try:
        for size in sizes:
            cfg = CONFIGS[size]
            P = n_params(cfg)
            largest = max(int(np.prod(s)) for s in model_shapes(cfg).values())
            free_disk = shutil.disk_usage(root).free
            if free_disk < 3.0 * 2 * P:
                notes.append(f"{size}: skipped, needs ~{gb(3.0 * 2 * P)} GB free disk in {root}, "
                             f"{gb(free_disk)} GB available.")
                log(notes[-1])
                continue
            d = build_model(size, root, log)
            row = {"size": size, "params": P, "largest": largest,
                   "v1_mb": (d / "delta_v1.wdelta").stat().st_size / 1e6,
                   "v2_mb": (d / "delta_v2.wdelta").stat().st_size / 1e6, "runs": []}
            results.append(row)
            cache = root / f"cache-{size}"
            out = root / f"out-{size}"
            base_spec = {"dir": str(d), "root": str(root), "cache_dir": str(cache)}

            if not args.no_old:
                log(f"[{size}] compute-only, one core")
                row["compute"] = run_child(dict(base_spec, mode="compute", core=args.core, repeats=3,
                                                elements=args.compute_elements), args.timeout)

            runs = []
            # 0.2.0 holds the base and the output as float32 dicts plus the whole file
            old_need = 4 * P * 2 + 2 * P * 0.6 + 0.5e9
            ram_ok = lambda need: size in forced or need <= 0.8 * available_ram()  # noqa: E731
            if args.no_old:
                pass
            elif not ram_ok(old_need):
                notes.append(f"{size}: 0.2.0 load_delta_from_paths skipped, needs ~{gb(old_need)} GB RAM, "
                             f"{gb(available_ram())} GB available.")
                log(notes[-1])
            else:
                runs.append(("0.2.0 load_delta_from_paths (v1 file)", dict(mode="old_load")))
            new_need = 4.4 * P + 0.3e9  # float32 result dict
            if not ram_ok(new_need):
                notes.append(f"{size}: load_delta_from_paths (float32 dict) skipped, the result alone needs "
                             f"~{gb(4 * P)} GB RAM, {gb(available_ram())} GB available.")
                log(notes[-1])
            else:
                runs += [
                    ("load_delta_from_paths, v1 file, 1 worker",
                     dict(mode="new_load", file="delta_v1.wdelta", workers=1, verify="full", device="cpu")),
                    ("load_delta_from_paths, v2 file, 1 worker",
                     dict(mode="new_load", file="delta_v2.wdelta", workers=1, verify="full", device="cpu")),
                    (f"load_delta_from_paths, v2 file, {workers} workers",
                     dict(mode="new_load", file="delta_v2.wdelta", workers=workers, verify="full", device="cpu")),
                ]
            runs += [
                ("reconstruct_to_safetensors, v2, 1 worker",
                 dict(mode="reconstruct", file="delta_v2.wdelta", workers=1, verify="full", device="cpu")),
                (f"reconstruct_to_safetensors, v2, {workers} workers",
                 dict(mode="reconstruct", file="delta_v2.wdelta", workers=workers, verify="full", device="cpu")),
                (f"reconstruct_to_safetensors, v2, {workers} workers, verify=cached (warm)",
                 dict(mode="reconstruct", file="delta_v2.wdelta", workers=workers, verify="cached",
                      device="cpu", warm=True)),
            ]
            if cuda:
                runs.append((f"reconstruct_to_safetensors, v2, {workers} workers, CUDA",
                             dict(mode="reconstruct", file="delta_v2.wdelta", workers=workers, verify="full",
                                  device="cuda")))

            for label, spec in runs:
                spec = dict(base_spec, **spec, out=str(out))
                if spec.pop("warm", False):
                    shutil.rmtree(cache, ignore_errors=True)
                    run_child(dict(spec, verify="cached"), args.timeout)  # populate the cache
                    shutil.rmtree(out, ignore_errors=True)
                log(f"[{size}] {label}")
                r = run_child(spec, args.timeout)
                shutil.rmtree(out, ignore_errors=True)
                if not args.no_tracemalloc:
                    r["tracemalloc_peak"] = run_child(dict(spec, tracemalloc=True), args.timeout)["tracemalloc_peak"]
                    shutil.rmtree(out, ignore_errors=True)
                r["label"] = label
                row["runs"].append(r)
                log(f"    {r['wall_s']:.1f} s, peak RSS {gb(r['peak_rss'])} GB")
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)

    write_results(Path(args.results), info, results, notes)
    log(f"wrote {args.results}")


def write_results(path: Path, info: dict, results: list, notes: list) -> None:
    L = ["# Reconstruction benchmark", "",
         "Generated by `benchmarks/bench_reconstruct.py`; do not edit by hand.", "",
         "## Machine", ""]
    L += [f"- {k}: {v}" for k, v in info.items()]
    L += ["", "Synthetic Qwen2.5-shaped bfloat16 models, int4 deltas (outlier_fraction=0.01). "
          "v1 files are written by deltatensors 0.2.0, v2 files by the current writer. "
          "Every path verifies the base hash (`verify=\"full\"`) unless noted. "
          "Each run is a fresh process; peak RSS includes the interpreter and imports "
          "(the 0.2.0 path imports torch). tracemalloc peaks come from a separate run with tracing on.", ""]
    for row in results:
        L += [f"## {row['size']} ({row['params'] / 1e9:.2f}B parameters)", "",
              f"- largest tensor: {row['largest'] / 1e6:.1f}M elements "
              f"= {gb(4 * row['largest'])} GB as float32, {gb(2 * row['largest'])} GB as bfloat16",
              f"- delta size: v1 {row['v1_mb']:.0f} MB, v2 {row['v2_mb']:.0f} MB", ""]
        c = row.get("compute")
        if c:
            L += ["### int4 decode + add, in memory, one core" + ("" if c["pinned"] else " (not pinned)"), "",
                  f"{c['tensors']} tensors, {c['elements'] / 1e6:.0f}M elements; best of 3.", "",
                  "| path | time (s) | speedup vs 0.2.0 |", "|---|---:|---:|",
                  f"| 0.2.0 `(base + decompress_int4(p)).astype()` | {c['old_s']:.3f} | 1.00x |",
                  f"| `decompress_int4_add_`, v1 payload | {c['new_v1_s']:.3f} | {c['old_s'] / c['new_v1_s']:.2f}x |",
                  f"| `decompress_int4_add_`, v2 payload | {c['new_v2_s']:.3f} | {c['old_s'] / c['new_v2_s']:.2f}x |",
                  ""]
        L += ["### End to end (includes disk I/O)", "",
              "| path | wall (s) | peak RSS (GB) | peak RSS above baseline (GB) | tracemalloc peak (GB) |",
              "|---|---:|---:|---:|---:|"]
        for r in row["runs"]:
            tm = gb(r["tracemalloc_peak"]) if "tracemalloc_peak" in r else "-"
            L.append(f"| {r['label']} | {r['wall_s']:.1f} | {gb(r['peak_rss'])} | "
                     f"{gb(r['peak_rss'] - r['baseline_rss'])} | {tm} |")
        L.append("")
    if notes:
        L += ["## Skipped", ""] + [f"- {n}" for n in notes] + [""]
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
