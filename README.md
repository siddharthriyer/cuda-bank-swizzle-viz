# Bank Conflict Lab

An interactive visualizer for CUDA shared-memory bank conflicts and the
XOR swizzle that GPUs use to avoid them, which is the same trick behind Hopper's
`wgmma.mma_async` matrix descriptors and TMA (`cp.async.bulk.tensor`) copies.


**Live demo:** https://siddharthriyer.github.io/cuda-bank-swizzle-viz/

