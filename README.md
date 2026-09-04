# Orchestra Rehearsal Pipeline

Turns a three-hour raw recording of an amateur orchestra rehearsal into files the
players will actually listen to — cut into blocks, loudness-matched, mastered,
uploaded, and announced — with one human decision in the middle.

日本語の詳細版は [README.ja.md](README.ja.md)。

---

## The problem

A rehearsal recording is one enormous file. Inside it are two or three ensemble
blocks separated by breaks, each preceded by tuning and surrounded by warm-up
noise. Nobody listens to that.

And when they do, it doesn't work. The dynamic range of an orchestra in a
rehearsal room is around 28 LU. On a phone in a car, the tutti is painful and the
conductor's instructions are inaudible. The quiet parts are where the information
is.

So every week someone has to find the boundaries, cut, level, encode, upload, and
write the message. This does that.

## What it produces

From `260829_001.TAKE` and `260829_002.TAKE` on the recorder's SD card:

```
output/260829/export/
  260829_前半_聴きやすさ調整版_残響あり.mp3     51 min
  260829_中盤_聴きやすさ調整版_残響あり.mp3     78 min
  260829_後半_聴きやすさ調整版_残響あり.mp3     46 min
```

Each block hits −20 LUFS integrated with a −1.0 dBTP ceiling, has ID3 tags, and
is already uploaded to Box (password-protected share link) and Google Drive, with
the announcement text written for you.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env                                        # credentials
cp pipeline_defaults.example.json pipeline_defaults.json    # ensemble name, Box parent folder
cp session_config.example.json session_config.json

.venv/bin/python pipeline.py ingest    --date 260829 --recorder zoom-f3
.venv/bin/python pipeline.py merge     --date 260829
.venv/bin/python pipeline.py propose   --date 260829     # → confirmed.json

#   Listen to output/260829/preview_clips/ and fix confirmed.json if needed.
#   This is the one human step.

.venv/bin/python pipeline.py apply     --date 260829
.venv/bin/python pipeline.py normalize --date 260829
.venv/bin/python pipeline.py mix       --date 260829
.venv/bin/python pipeline.py export    --date 260829
.venv/bin/python pipeline.py box-upload    --date 260829
.venv/bin/python pipeline.py gdrive-upload --date 260829
.venv/bin/python pipeline.py notify        --date 260829
```

Requires `ffmpeg` 6+ and Python 3.9+. Recorder support: ZOOM F3 and M4; other
recorders need a profile in `orchpipe/recorder_profiles/`.

## The three ideas worth stealing

### 1. Find block boundaries by listening for the tuning A

Orchestras tune to A=442 Hz before every block. That is a far stronger signal
than silence detection or energy segmentation, both of which failed here — one
attempt merged two real blocks into a single 94-minute span.

Detection uses **octaves of A only**. Including other pitches produces false
positives on E and C♯, because those are harmonics of A and light up whenever the
orchestra plays anything in A major. The block starts 2 seconds before the first
sustained A; a several-second gap after it is the oboist re-articulating, not the
end of tuning, so the gap is not a boundary.

On the 260829 session this got all three block starts exactly right, and every
boundary was judged correct by ear.

### 2. Parallel compression, not downward compression

A downward compressor only touches loud passages. The quiet ones leave at the
level they were recorded, which is the actual problem.

Parallel compression splits the signal, crushes one copy (threshold −46 dB, ratio
20:1), adds +17 dB of makeup, and sums it back. A downward compressor used to
build an upward effect:

| | Change |
|---|---|
| Quietest 1% | −54.5 → −36.8 dBFS (**+17.7 dB**) |
| Median | +7.7 dB |
| Loudest 1% | +0.6 dB |
| Peak | +0.2 dB |
| Loudness range | 27.9 → 17.8 LU |

The dry path stays untouched throughout, so transients and the shape of the
envelope survive. Listeners do not report it as "squashed" — the reaction was
that the conductor had become audible.

The makeup is capped per block from the measured noise floor, so a noisy
recording does not get its hiss lifted into the mix. Noise floors differed by
9.4 dB between blocks of the same session.

### 3. Convolution reverb from a real hall

Rehearsal rooms are dry, and three hours of dry orchestra is tiring. Mixing in
15% of a measured concert-hall impulse response fixes that without changing the
loudness guarantees — the reverb runs **before** the master chain, so gain
solving and true-peak limiting still hold.

Two things happen to the impulse response first. Everything before the direct
sound is discarded and 30 ms of silence is prepended, because a measured IR
starts at ~1 ms and without a pre-delay the reverb sits on top of the source and
blurs it. And the IR is energy-normalized, so `--reverb-mix` means the same thing
for a 0.9-second hall as for a 2.4-second one.

`ffmpeg`'s `afir` was not usable: with `dry=0` it output silence, and it rejected
24-bit IRs outright. `pedalboard.Convolution` works and is a pip install, with no
plugin host required.

## How loudness is guaranteed

Compression changes loudness, so a single measure-then-apply pass does not land
on target. The gain is solved iteratively: measure integrated loudness, apply a
provisional gain, measure again *through the full chain*, correct, repeat until
within 0.15 LU. Measurement discards to `-f null`, so a pass over 77 minutes
takes about 17 seconds.

After writing, integrated loudness and true peak are verified. Deviation over
1 LU, or any ceiling breach, fails the run rather than shipping.

Order matters in the chain: compressor before limiter. Ridden the other way, the
limiter catches percussion transients alone and changes their timbre. The limiter
is a safety valve, not a sound.

## Repository layout

```
pipeline.py                 all subcommands (ingest … notify)
research.py                 ASR, state classification, digest generation
orchpipe/                   the implementation
  recorder_profiles/        ZOOM F3, ZOOM M4, single-file
  research/                 experimental stages, not part of the daily run
tools/make_fallback_ir.py   generates the bundled impulse response
docs/                       design documents, one per feature
docs/journal/               the build log — every decision, in order
assets/ir/hall.wav          synthetic hall IR (see note below)
```

`docs/journal/build-log.md` is the interesting one if you want to know why
anything is the way it is. It is a running record of what was measured, what
failed, and what was decided, written as the work happened.

## Caveats, honestly

**Every constant was measured on one ensemble in one room.** The −46 dB parallel
threshold, the +17 dB makeup, the 442 Hz tuning reference, the 2-second pre-roll:
all of them come from measuring *these* recordings. Treat them as a starting
point, not a preset.

**The bundled impulse response is synthetic.** Measured hall IRs are licensed
material and cannot be redistributed. `assets/ir/hall.wav` is generated by
`tools/make_fallback_ir.py` — six early reflections plus band-dependent diffuse
decay (2.1 s in the low-mids falling to 0.8 s near 10 kHz, measuring 1.72 s
overall). It is not a model of any real room. Point `--reverb-ir` at a real one
if you have it.

**Box and Google Drive credentials are yours to supply.** See `.env.example` and
`pipeline_defaults.example.json`; neither the real file is tracked. Nothing is
bundled and nothing phones home.

**The interface is Japanese.** Log output, generated announcement text, and the
design documents are all in Japanese. The code and this file are not.

## Licence

MIT. See [LICENSE](LICENSE).
