# Rehearsal Recording Pipeline

Turns a multi-hour raw recording of an amateur orchestra rehearsal into files the
players can actually listen to — cut into blocks, loudness-matched, mastered, and
uploaded — with two decisions from a human and nothing else.

Built for one ensemble's weekly rehearsals. Every parameter in it was chosen by
measuring *that* ensemble's recordings, not by copying a preset. That is the
point, and also the main caveat.

日本語版は [README.md](README.md)。

## The problem

A rehearsal recording is three hours of one file. Inside it there are two or
three ensemble blocks separated by breaks, preceded by tuning, and surrounded by
warm-up noise. Nobody listens to that. And when they do, the quiet passages are
inaudible on a phone while the tutti is painfully loud.

So every week someone has to find the boundaries, cut, level, encode, upload, and
post the links. This does that.

```
ingest → merge → propose → [human checks boundaries] → apply
       → normalize → mix → export → box-upload → gdrive-upload → notify
```

The human touches two things: the block boundaries (`confirmed.json`) and the
session settings (`session_config.json`). Everything else is reproducible.

## What is actually interesting here

### Finding the blocks by listening for the tuning

Scoring windows for "does this sound like an ensemble playing" and running Viterbi
over it *fails* on real rehearsals: chatter during breaks and long sectional work
break it. On one test day it proposed a single 94-minute block spanning two real
blocks and the break between them.

Tuning is a better signal. **It always happens immediately before an ensemble
block, and never inside one.** So `propose` locates the tuning (A = 442 Hz and its
octaves, matched over one-second windows) and starts each block there.

| Block | Confirmed by a human | `propose` | Δ start |
|---|---|---|---|
| 1 | 00:01:17 – 00:42:40 | 00:01:20 – 00:44:21 | 3 s |
| 2 | 00:47:08 – 01:33:15 | 00:47:11 – 01:34:51 | 3 s |
| 3 | 01:42:15 – 02:59:45 | 01:42:18 – 03:00:04 | 3 s |

Ends run long by design: a guard band widens each block outward, because keeping
30 seconds of warm-up costs nothing and cutting off the last chord cannot be
undone. A human listened to all six boundaries and accepted them.

Getting the tuning detector right took a while. Matching *harmonics* of A means
the 3rd harmonic lands on E and the 5th on C♯ — so an A major chord in the middle
of a piece looks exactly like tuning. One session's block 3 started 5 minutes 24
seconds late because of a C♯6 at 1111 Hz. Octaves only, harmonics used solely to
extend the onset backwards.

### Making quiet passages audible without squashing the loud ones

The distributed files used to be peak-normalised, which is the wrong thing: peak
is one sample and has little to do with perceived loudness. On one day it left
**8.6 LU** between blocks. Now each block is normalised to **−20 LUFS** integrated,
which is where classical releases sit (pop masters go to −8; streaming services
normalise anyway, so there is nothing to gain by crushing).

That fixes loudness *between* blocks. It does nothing for the range *within* one.
Measured over one 45-second stretch, one-second RMS ran from −52.7 dBFS in the
quiet passages to −13.1 dBFS at the tutti — **39.5 dB apart**. A phone speaker
does not reproduce the bottom of that, and neither does a car.

A downward compressor does not help: it only touches the loud end. The fix is
**parallel compression**, the standard trick in classical mastering — split the
signal, crush one copy hard, add it back underneath. Measured across a full
77-minute block:

| 1-second RMS | Before | After | Δ |
|---|---|---|---|
| quietest 1% | −54.5 | −36.8 | **+17.7** |
| median | −33.6 | −25.9 | +7.7 |
| loudest 1% | −14.1 | −13.5 | **+0.6** |
| peak | −10.6 | −10.4 | **+0.2** |

Monotonic: the quieter it was, the more it moves; the fortissimo does not. A
percussionist's fortissimo shifts by 0.2 dB. Loudness range drops from 27.9 to
17.8 LU — still far wider than a pop master.

The lift raises the noise floor by the same amount, so the makeup gain is capped
from a measurement of that floor rather than fixed. On one day the floor differed
by **9.4 dB between blocks of the same rehearsal**, so a fixed value would have
been wrong for at least one of them.

### A real hall, not an algorithm

A concert-hall impulse response is convolved in at 15%. Reverb changes no level
statistics, so it sits before the master chain and the loudness guarantees hold.
The output is truncated to the input length, because a block that grows by two
seconds no longer lines up with the confirmed boundaries.

### Getting it out before you get home

The raw material is about 8 GB and cannot go over a mobile connection. So the
phone builds one 320 kbps proxy of the whole rehearsal (a fixed −14 dB of headroom
and nothing else — the recorder writes 32-bit float and peaks above 0 dBFS), and
sends only that. The mastering happens at home on the machine, driven by
boundaries confirmed on the phone during the commute.

Verified against the original-file path on the same material: block gains agreed
to **0.01 dB**, integrated loudness and true peak matched exactly, proposed
boundaries were identical, and the audio differed by −34.6 dB relative to signal
at worst (on percussion transients, where MP3 is weakest). A blind A/B was
"can't tell at all".

## Requirements

- Python 3.9+, ffmpeg 7+
- `numpy`, `scipy`, `mutagen`, `pedalboard`, `requests`, Google API client
- `faster-whisper` for the speech detection used by the digest builder (optional)
- A hall impulse response as a plain WAV at `assets/ir/hall.wav`. **None is
  included** — the one used in development ships with a commercial plugin and is
  not redistributable. Any hall IR works; 1.7–2.0 s decay suits an orchestra.
  `--reverb-mix 0` turns it off.
- Box and Google Drive OAuth apps, if you want the upload stages. Copy
  `.env.example` to `.env` and fill it in.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # fill in credentials
cp session_config.example.json output/<date>/session_config.json
.venv/bin/python pipeline.py ingest --date 260829
```

Recorder support is a small plugin interface (`orchpipe/recorder_profiles/`);
ZOOM M4 and F3 are implemented.

## Honest limitations

- **Tuned to one ensemble, one room, one pair of microphones.** The numbers above
  are measurements of that setup. The shape of the reasoning should transfer; the
  constants may not.
- **The tuning-first boundary detection has been validated on one day.** It is the
  default because it was dramatically better than the alternative on that day, and
  because it degrades to the old method when the detected tuning count disagrees
  with the expected block count. That is not the same as being proven.
- **The parallel compression assumes a low noise floor.** It is capped
  automatically, but on a genuinely noisy recording the cap will do most of the
  work and the effect will be small.
- **The digest builder is research-grade.** It classifies playing / speech /
  silence and keeps only the playing. It works, but the classifier is the least
  principled part of this repository.
- Interface and comments are in Japanese.

## Notes

The commit history is the interesting documentation: most changes carry the
measurement that motivated them and the alternatives that were rejected. The
instruction documents (`orchestra_recording_pipeline_*.md`) are the specifications
each change was written against, updated afterwards with what actually happened —
including the parts that did not work.

MIT licensed. See [LICENSE](LICENSE).

Written with [Claude Code](https://claude.com/claude-code).
