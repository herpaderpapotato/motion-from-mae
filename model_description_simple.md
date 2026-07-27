# How the Model Works — Plain-Language Overview

## What it does

This system watches a video and draws a graph of the motion in it. For every frame of
the video it answers two questions:

1. **Where is the motion right now?** — a position value between 0 (all the way down)
   and 1 (all the way up).
2. **Is anything actually moving?** — a yes/no "activity" signal.

The result is a smooth position curve over time, like a wave that rises and falls with
the rhythm of the movement on screen.

## The two-part design

The system is really two models working together, like a translator and a writer:

### Part 1: The "eyes" — VideoMAE (pre-trained, not modified)

VideoMAE is a large, general-purpose video-understanding model that was trained by
researchers on huge amounts of video. It has already learned to recognize objects,
textures, and movement.

- **What goes in:** short bursts of 16 video frames at a time, cropped to the
  lower-middle part of the picture (where the action is) and shrunk to a standard size.
- **What comes out:** for every pair of frames, a set of 5 number-lists (one for the
  whole picture, plus one for each quarter: top-left, top-right, bottom-left,
  bottom-right). Each list is a compressed "description" of what the model saw there —
  768 numbers that capture appearance and motion.

This part is **frozen**: we never change it. It just converts raw pixels into compact
descriptions, once per video, and those descriptions are saved to disk so we never have
to process the pixels again.

### Part 2: The "brain" — the DNX temporal head (this project's model)

This is the small model that this project actually trains. It reads the sequence of
descriptions produced by the eyes — thousands of them for a long video — and figures
out the position curve.

Two ideas make it work well:

- **It looks at the whole timeline at once.** Using the same "attention" technique
  that powers chatbots like ChatGPT and Claude, every moment in the video can compare
  itself with every other moment. That lets the model notice things like "this frame
  looks like the top of a stroke because the frames around it are moving downward."
- **It explicitly measures rhythm.** A separate small component compares each moment
  to the moments just before and after it (up to about a second in each direction) and
  asks "does the video look the same as it did N frames ago?" Repetitive motion makes
  a striped pattern in these comparisons — the stripes reveal the speed and timing of
  the rhythm, regardless of what the scene actually looks like.

## How it answers "where is the motion?"

Instead of blurting out a single number, the model spreads its guess across 64 small
buckets covering the range 0 to 1 — like placing chips on a roulette table. If it's
confident, almost all the chips go on one bucket; if it's unsure, they spread out. The
final position is the weighted average of the chips. This trick (called HL-Gauss) makes
training more stable and lets the model express uncertainty.

## How it learns

During training, the model watches videos that humans have already labeled with the
correct position curve. It compares its guesses to the labels and adjusts itself to:

- put its "chips" near the correct position,
- get the *shape and timing* of the wave right, not just the average height,
- correctly say when nothing is moving,
- and forgive tiny timing differences — if the human label is a few frames early or
  late, the model isn't punished for disagreeing by a fraction of a second.

## Why this design?

Training a big video model from scratch would need enormous amounts of data and
computing power. By borrowing frozen, pre-trained "eyes" and training only a small
"brain" on top (about 12 million adjustable numbers versus the eyes' 86 million), the
system learns its specialized task from a modest amount of labeled video, trains
quickly, and runs efficiently — the expensive pixel-crunching happens once per video
and is cached.
