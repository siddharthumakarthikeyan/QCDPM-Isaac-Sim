"""Synthesised score for the trailer (no samples, no licensed material): a slow pad in D minor, a pulse that
enters with the physics chapter, low impacts on the chapter changes and a riser into the title.

    python3 tools/trailer_audio.py        (reads renders/trailer_timeline.json, writes renders/trailer_audio.wav)
Swap in any licensed track by passing a different file to compose_trailer.py --audio.
"""

import json
import os
import sys
import wave

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SR = 48000
name = sys.argv[1] if len(sys.argv) > 1 else "trailer"
tl = json.load(open(os.path.join(ROOT, "renders", f"{name}_timeline.json")))
N = int((tl["total"] + 0.5) * SR)
t = np.arange(N) / SR
L, R = np.zeros(N), np.zeros(N)
rng = np.random.default_rng(5)
BPM = 92.0
beat = 60.0 / BPM
bar = 4 * beat


def hz(semi):                       # semitones from D2
    return 73.416 * 2 ** (semi / 12.0)


def add(sig, start, pan=0.0, gain=1.0):
    i = int(start * SR)
    if i >= N:
        return
    sig = sig[: N - i] * gain
    L[i:i + len(sig)] += sig * (1 - pan) / 2 * 1.4
    R[i:i + len(sig)] += sig * (1 + pan) / 2 * 1.4


def env(n, a, r):
    e = np.ones(n)
    na, nr = min(int(a * SR), n), min(int(r * SR), n)
    e[:na] = np.linspace(0, 1, na) ** 2
    e[n - nr:] *= np.linspace(1, 0, nr) ** 2
    return e


def pad(freq, dur, bright=0.5):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    out = np.zeros(n)
    for det in (-0.07, 0.0, 0.06):
        f = freq * 2 ** (det / 12)
        ph = rng.uniform(0, 2 * np.pi)
        for h in range(1, 7):
            out += np.sin(2 * np.pi * f * h * tt + ph * h) * bright ** (h - 1) / h
    return out * env(n, dur * 0.35, dur * 0.4) / 3


def pluck(freq, dur=0.5):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    return (np.sin(2 * np.pi * freq * tt) + 0.35 * np.sin(4 * np.pi * freq * tt)) * np.exp(-tt * 9) * env(n, 0.004, 0.05)


def impact(dur=3.5):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    f = 34 + 46 * np.exp(-tt * 5)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 1.5)
    noise = rng.normal(0, 1, n) * np.exp(-tt * 14)
    k = np.exp(-np.arange(64) / 10.0)
    return body + 0.25 * np.convolve(noise, k / k.sum(), "same")


def riser(dur):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    noise = rng.normal(0, 1, n)
    out = np.zeros(n)
    y = 0.0
    alpha = 0.002 + 0.12 * (tt / dur) ** 3                 # opening low-pass
    for i in range(n):
        y += alpha[i] * (noise[i] - y)
        out[i] = y
    tone = np.sin(2 * np.pi * np.cumsum(hz(12) * 2 ** (tt / dur)) / SR) * 0.15
    return (out * 2.5 + tone) * (tt / dur) ** 2 * env(n, 0.1, 0.05)


# chord roots / tones (semitones from D2): Dm, Bb, F, C
CH = [(0, [12, 15, 19, 24]), (-4, [8, 12, 15, 20]), (3, [15, 19, 22, 27]), (-2, [10, 14, 17, 22])]
marks = tl["chapters"]


def mark(name, default):
    return next((tm for tm, ch in marks if ch == name), default)


t_phys = mark("CONTROL", 20.0)                              # pulse comes in with the controller
t_build = mark("APPLICATION: CONSTRUCTION", 40.0)           # kick through the three applications
t_why = mark("WHY IT MATTERS", tl["total"] - 15.0)
t_end = marks[-1][0] if marks[-1][1] is None and len(marks) > 1 else tl["total"] - 7.0

nb = int(np.ceil(tl["total"] / (2 * bar))) + 1
for b in range(nb):                                         # pad and bass: one chord per two bars
    t0 = b * 2 * bar
    root, tones = CH[b % 4]
    lift = 1.0 if t0 < t_build else 1.25
    for q, s in enumerate(tones):
        add(pad(hz(s), 2 * bar * 1.25, 0.45 if t0 < t_phys else 0.55), t0, pan=(-0.6, 0.2, -0.2, 0.6)[q], gain=0.085 * lift)
    add(pad(hz(root - 12), 2 * bar * 1.2, 0.3), t0, gain=0.22)
    if t_phys <= t0 < t_end:                                # eighth-note pulse on chord tones
        for e in range(16):
            s = tones[(0, 2, 1, 2, 3, 2, 1, 2)[e % 8]] + 12
            g = 0.055 if t0 < t_build else 0.075
            add(pluck(hz(s)), t0 + e * beat / 2, pan=0.5 * np.sin(e * 1.3), gain=g * (1.0 if e % 2 == 0 else 0.6))
    if t_build <= t0 < t_why:                               # soft kick on the beats while it builds
        for k in range(8):
            add(impact(0.5), t0 + k * beat, gain=0.16)
add(riser(min(3.0, max(1.0, 1.4))), max(0.0, 0.9 - 0.0), gain=0.10)
add(impact(), 0.9, gain=0.55)                               # title
for tm, _ in marks[1:]:
    add(riser(1.6), max(0, tm - 1.6), gain=0.12)
    add(impact(), tm, gain=0.42)

x = np.stack([L, R], 1)
d1, d2 = int(0.29 * SR), int(0.41 * SR)                     # two cross-fed echoes for space
for _ in range(3):
    y = x.copy()
    y[d1:, 0] += 0.28 * x[:-d1, 1]
    y[d2:, 1] += 0.28 * x[:-d2, 0]
    x = y
fade = np.ones(N)
fade[: int(0.5 * SR)] = np.linspace(0, 1, int(0.5 * SR))
nf = int(4.0 * SR)
fade[-nf:] = np.linspace(1, 0, nf) ** 1.5
x *= fade[:, None]
x = np.tanh(x / np.abs(x).max() * 1.6) * 0.82
out = os.path.join(ROOT, "renders", f"{name}_audio.wav")
with wave.open(out, "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((x * 32767).astype(np.int16).tobytes())
print(out, f"{N / SR:.1f} s")
