# Bank Conflict Lab

An interactive visualizer for **CUDA shared-memory bank conflicts** and the
**XOR swizzle** that GPUs use to avoid them — the same trick behind Hopper's
`wgmma.mma_async` matrix descriptors and TMA (`cp.async.bulk.tensor`) copies.

Shared memory is 32 banks wide. A warp's 32 lanes run at full speed only when
they hit 32 *different* banks. Store a tile row-major and a column read collapses
all 32 lanes onto one bank — a 32× stall. Toggle the XOR swizzle and watch the
lanes scatter back across all 32 banks.

**Live demo:** https://siddharthriyer.github.io/cuda-bank-swizzle-viz/

## What you can do

- Toggle **No swizzle** vs **XOR swizzle** (`physCol = col ⊕ row`).
- Switch the warp between reading a **column** and a **row**.
- **Hover any cell** to open the bit inspector: it breaks the cell's address
  into its column/row bits, shows the `col ⊕ row` XOR, and highlights how the
  low 5 bits of the word address *are* the bank.
- Watch the live bank-traffic histogram and conflict-degree readout.

## Running locally

It's a single self-contained `index.html` — no build, no dependencies. Just open
it in a browser, or serve the folder:

```bash
python3 -m http.server
```

## Caveats

Toy model: 32×32 tile, 4 bytes/word, 128-byte row pitch, one warp = 32 lanes.
Real swizzle atoms (128B/64B/32B) XOR only a few bits at a vector granularity —
a partial version of the full `⊕` shown here — but the mechanism is identical.
