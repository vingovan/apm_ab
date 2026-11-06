#!/usr/bin/env python3
"""
apm_ab.py - Chạy WebRTC APM (HPF / AEC3 / NS / AGC2) trên file ghi âm tổng đài
và xuất nhiều biến thể để A/B test WER.

Engine: WebRTC Audio Processing Module qua binding `livekit` (pip install livekit).
Phụ thuộc: numpy, scipy, soundfile, livekit (và jiwer nếu dùng lệnh `wer`).

Hai lệnh con:

  1) process  - xử lý file ghi âm, xuất các biến thể + manifest.csv
       python apm_ab.py process recordings/ -o out/ --near-ch 0 --ref-ch 1 --delay-ms auto

     Đầu vào:
       * WAV stereo: 1 kênh là giọng khách (near-end / inbound), 1 kênh là bot/agent
         (far-end / outbound = tín hiệu tham chiếu cho AEC). Mặc định ch0 = khách, ch1 = bot
         (giống record_session stereo của FreeSWITCH: kênh trái = read, kênh phải = write;
         hãy kiểm tra lại với hệ thống của bạn).
       * WAV mono: chỉ có giọng khách -> các biến thể có AEC sẽ bị bỏ qua.

     Đầu ra:
       out/<variant>/<tên_file>.wav   (mono, PCM16)
       out/manifest.csv               (đường dẫn + chỉ số mức âm, clipping, suy giảm vọng)

  2) wer      - tính WER theo biến thể (và theo nhóm, ví dụ SNR bucket)
       python apm_ab.py wer --refs refs.csv --hyps hyps.csv [--group-col snr_bucket]

     refs.csv : file,ref[,snr_bucket,...]     (file = tên file gốc, không có đuôi)
     hyps.csv : file,variant,hyp              (kết quả ASR của bạn chạy trên manifest)

Lưu ý:
  * Binding livekit chỉ cho bật/tắt từng module, không chỉnh được mức NS/AGC.
    Cần chỉnh sâu (mức NS, target dBFS, max gain AGC2) thì dùng C++ API
    webrtc::AudioProcessing::Config hoặc thư viện webrtc-audio-processing.
  * APM xử lý theo frame 10 ms, sample rate hỗ trợ: 8000/16000/32000/48000.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SUPPORTED_RATES = (8000, 16000, 32000, 48000)

# Định nghĩa các biến thể A/B. Có thể sửa/thêm tuỳ ý.
VARIANTS: dict[str, dict[str, bool]] = {
    "raw":          dict(hpf=False, aec=False, ns=False, agc=False),
    "hpf":          dict(hpf=True,  aec=False, ns=False, agc=False),
    "hpf_agc":      dict(hpf=True,  aec=False, ns=False, agc=True),
    "hpf_ns":       dict(hpf=True,  aec=False, ns=True,  agc=False),
    "hpf_aec":      dict(hpf=True,  aec=True,  ns=False, agc=False),
    "hpf_aec_ns":   dict(hpf=True,  aec=True,  ns=True,  agc=False),
    "hpf_aec_agc":  dict(hpf=True,  aec=True,  ns=False, agc=True),
    "full":         dict(hpf=True,  aec=True,  ns=True,  agc=True),
}


# ----------------------------------------------------------------------------- utils
def to_float(x: np.ndarray) -> np.ndarray:
    if x.dtype.kind == "i":
        return x.astype(np.float32) / 32768.0
    return x.astype(np.float32)


def to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def frame_energy_db(x: np.ndarray, n: int) -> np.ndarray:
    """Năng lượng (dBFS) từng frame n mẫu."""
    m = len(x) // n
    if m == 0:
        return np.array([])
    f = x[: m * n].reshape(m, n)
    return 10 * np.log10(np.mean(f ** 2, axis=1) + 1e-12)


def estimate_delay_ms(near: np.ndarray, ref: np.ndarray, sr: int, max_ms: int = 600) -> int:
    """Ước lượng độ trễ vọng (ref -> near) bằng GCC-PHAT trên toàn file."""
    n = len(near) + len(ref)
    nfft = 1 << (n - 1).bit_length()
    X = np.fft.rfft(near, nfft)
    Y = np.fft.rfft(ref, nfft)
    R = X * np.conj(Y)
    R /= np.abs(R) + 1e-12
    cc = np.fft.irfft(R, nfft)
    max_lag = int(sr * max_ms / 1000)
    lags = cc[: max_lag + 1]  # chỉ xét near trễ hơn ref (lag dương)
    lag = int(np.argmax(np.abs(lags)))
    return int(round(lag * 1000 / sr))


# ----------------------------------------------------------------------------- APM
def run_apm(near: np.ndarray, ref: np.ndarray | None, sr: int, cfg: dict[str, bool],
            delay_ms: int) -> np.ndarray:
    """Chạy WebRTC APM theo frame 10 ms. near/ref: float32 mono cùng sr."""
    from livekit import rtc  # import muộn để lệnh `wer` không cần livekit

    use_aec = cfg["aec"] and ref is not None
    apm = rtc.AudioProcessingModule(
        echo_cancellation=use_aec,
        noise_suppression=cfg["ns"],
        high_pass_filter=cfg["hpf"],
        auto_gain_control=cfg["agc"],
    )
    spf = sr // 100  # samples per 10 ms frame
    n_frames = len(near) // spf
    near_i16 = to_int16(near[: n_frames * spf])
    ref_i16 = to_int16(ref[: n_frames * spf]) if use_aec else None
    out = np.empty_like(near_i16)

    for i in range(n_frames):
        s = slice(i * spf, (i + 1) * spf)
        if use_aec:
            rf = rtc.AudioFrame(bytearray(ref_i16[s].tobytes()), sr, 1, spf)
            apm.process_reverse_stream(rf)
            apm.set_stream_delay_ms(delay_ms)
        nf = rtc.AudioFrame(bytearray(near_i16[s].tobytes()), sr, 1, spf)
        apm.process_stream(nf)
        out[s] = np.frombuffer(bytes(nf.data), dtype=np.int16)

    return out.astype(np.float32) / 32768.0


# ----------------------------------------------------------------------------- metrics
def level_metrics(y: np.ndarray, sr: int) -> dict[str, float]:
    """Mức âm phần 'có tiếng' (frame >= -45 dBFS), peak và tỷ lệ clipping."""
    e = frame_energy_db(y, sr // 100)
    active = e[e > -45]
    return dict(
        active_rms_dbfs=round(float(np.mean(active)), 2) if len(active) else float("nan"),
        peak_dbfs=round(float(20 * np.log10(np.max(np.abs(y)) + 1e-12)), 2),
        clip_rate=round(float(np.mean(np.abs(y) >= 0.999)), 6),
    )


def echo_attenuation_db(x_in: np.ndarray, y_out: np.ndarray, ref: np.ndarray | None,
                        sr: int, skip_s: float = 2.0) -> float:
    """
    Suy giảm năng lượng (dB) ở các frame bot đang nói to (ref active), bỏ qua vài giây
    đầu để AEC hội tụ. Đây là ERLE xấp xỉ: nếu khách nói chen (double-talk) trong lúc
    bot nói thì con số này cũng bao gồm phần giọng khách bị nén -> chỉ dùng để so sánh
    tương đối giữa các biến thể.
    """
    if ref is None:
        return float("nan")
    n = sr // 100
    m = min(len(x_in), len(y_out), len(ref)) // n
    er = frame_energy_db(ref[: m * n], n)
    ex = frame_energy_db(x_in[: m * n], n)
    ey = frame_energy_db(y_out[: m * n], n)
    start = int(skip_s * 100)
    mask = np.zeros(m, bool)
    mask[start:] = er[start:] > (np.max(er) - 25)  # frame ref trong 25 dB so với peak
    mask &= ex > -60
    if mask.sum() < 10:
        return float("nan")
    return round(float(np.mean(ex[mask]) - np.mean(ey[mask])), 2)


# ----------------------------------------------------------------------------- process
@dataclass
class Item:
    path: Path
    name: str


def collect_inputs(inputs: list[str]) -> list[Item]:
    items: list[Item] = []
    for p in inputs:
        pp = Path(p)
        if pp.is_dir():
            for f in sorted(pp.rglob("*.wav")):
                items.append(Item(f, f.stem))
        elif pp.suffix.lower() == ".wav":
            items.append(Item(pp, pp.stem))
    return items


def cmd_process(args: argparse.Namespace) -> None:
    items = collect_inputs(args.inputs)
    if not items:
        sys.exit("Không tìm thấy file .wav nào.")
    variants = args.variants.split(",") if args.variants else list(VARIANTS)
    for v in variants:
        if v not in VARIANTS:
            sys.exit(f"Biến thể không hợp lệ: {v}. Có: {', '.join(VARIANTS)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for it in items:
        data, sr_in = sf.read(str(it.path), always_2d=True)
        data = to_float(data)
        near = data[:, args.near_ch]
        ref = data[:, args.ref_ch] if data.shape[1] > max(args.near_ch, args.ref_ch) else None

        sr = args.rate or (sr_in if sr_in in SUPPORTED_RATES else 16000)
        near = resample(near, sr_in, sr)
        ref = resample(ref, sr_in, sr) if ref is not None else None

        if ref is not None and float(np.max(np.abs(ref))) < 1e-4:
            ref = None  # kênh tham chiếu im lặng

        if ref is None:
            delay = 0
        elif args.delay_ms == "auto":
            delay = estimate_delay_ms(near, ref, sr)
        else:
            delay = int(args.delay_ms)

        out_sr = args.out_rate or sr_in
        print(f"[{it.name}] sr={sr_in}->{sr} dur={len(near)/sr:.1f}s "
              f"ref={'yes' if ref is not None else 'no'} delay={delay}ms")

        for v in variants:
            cfg = VARIANTS[v]
            if cfg["aec"] and ref is None:
                continue  # không có kênh tham chiếu -> bỏ biến thể AEC
            y = near[: len(near) // (sr // 100) * (sr // 100)] if v == "raw" else \
                run_apm(near, ref, sr, cfg, delay)
            y_out = resample(y, sr, out_sr)
            vdir = out_dir / v
            vdir.mkdir(exist_ok=True)
            op = vdir / f"{it.name}.wav"
            sf.write(str(op), to_int16(y_out), out_sr, subtype="PCM_16")

            row = dict(file=it.name, variant=v, path=str(op), sample_rate=out_sr,
                       duration_s=round(len(y) / sr, 2), delay_ms=delay, **cfg)
            row.update(level_metrics(y, sr))
            row["echo_atten_db"] = echo_attenuation_db(near, y, ref, sr)
            rows.append(row)

    man = out_dir / "manifest.csv"
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nĐã ghi {len(rows)} file. Manifest: {man}")

    # Tóm tắt theo biến thể
    print(f"\n{'variant':<14}{'n':>4}{'rms_dbfs':>10}{'clip%':>8}{'echo_atten_db':>15}")
    for v in variants:
        r = [x for x in rows if x["variant"] == v]
        if not r:
            continue
        def avg(k):
            vals = [x[k] for x in r if not np.isnan(x[k])]
            return float(np.mean(vals)) if vals else float("nan")
        print(f"{v:<14}{len(r):>4}{avg('active_rms_dbfs'):>10.1f}"
              f"{avg('clip_rate')*100:>8.3f}{avg('echo_atten_db'):>15.1f}")


# ----------------------------------------------------------------------------- wer
def cmd_wer(args: argparse.Namespace) -> None:
    import jiwer

    def norm(s: str) -> str:
        s = s.lower()
        for ch in ".,!?;:\"'()[]{}-–—…":
            s = s.replace(ch, " ")
        return " ".join(s.split())

    refs = {}
    with open(args.refs, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            refs[r["file"]] = r
    hyps = []
    with open(args.hyps, encoding="utf-8") as f:
        hyps = list(csv.DictReader(f))

    groups: dict[tuple[str, str], tuple[list[str], list[str]]] = {}
    for h in hyps:
        r = refs.get(h["file"])
        if r is None:
            continue
        g = r.get(args.group_col, "all") if args.group_col else "all"
        key = (h["variant"], g)
        groups.setdefault(key, ([], []))
        groups[key][0].append(norm(r["ref"]))
        groups[key][1].append(norm(h["hyp"]))

    out_rows = []
    for (v, g), (rr, hh) in sorted(groups.items()):
        o = jiwer.process_words(rr, hh)
        out_rows.append(dict(variant=v, group=g, n=len(rr), wer=round(o.wer * 100, 2),
                             sub=o.substitutions, dele=o.deletions, ins=o.insertions))

    base = {r["group"]: r["wer"] for r in out_rows if r["variant"] == "raw"}
    print(f"{'variant':<14}{'group':<14}{'n':>5}{'WER%':>8}{'Δ vs raw':>10}{'S':>7}{'D':>7}{'I':>7}")
    for r in out_rows:
        d = r["wer"] - base[r["group"]] if r["group"] in base else float("nan")
        r["delta_vs_raw"] = round(d, 2)
        print(f"{r['variant']:<14}{r['group']:<14}{r['n']:>5}{r['wer']:>8.2f}{d:>+10.2f}"
              f"{r['sub']:>7}{r['dele']:>7}{r['ins']:>7}")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nĐã ghi {args.out}")


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("process", help="Xử lý file và xuất biến thể A/B")
    p.add_argument("inputs", nargs="+", help="File .wav hoặc thư mục")
    p.add_argument("-o", "--out", default="apm_out")
    p.add_argument("--near-ch", type=int, default=0, help="Kênh giọng khách (mặc định 0)")
    p.add_argument("--ref-ch", type=int, default=1, help="Kênh bot/agent = tham chiếu AEC (mặc định 1)")
    p.add_argument("--delay-ms", default="auto", help="'auto' (GCC-PHAT) hoặc số ms")
    p.add_argument("--rate", type=int, choices=SUPPORTED_RATES,
                   help="Sample rate xử lý APM (mặc định = rate gốc nếu hỗ trợ, nếu không 16000)")
    p.add_argument("--out-rate", type=int, help="Sample rate file xuất (mặc định = rate gốc)")
    p.add_argument("--variants", help=f"Danh sách, phân tách dấu phẩy. Có: {','.join(VARIANTS)}")
    p.set_defaults(func=cmd_process)

    w = sub.add_parser("wer", help="Tính WER theo biến thể / nhóm")
    w.add_argument("--refs", required=True)
    w.add_argument("--hyps", required=True)
    w.add_argument("--group-col", help="Cột trong refs.csv để chia nhóm, vd snr_bucket")
    w.add_argument("--out", help="Ghi kết quả ra CSV")
    w.set_defaults(func=cmd_wer)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
