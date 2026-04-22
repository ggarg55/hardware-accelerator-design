# Data Reuse and Loop Tiling

> **Learning Objectives**
> - Recognize deep neural networks as massively nested for-loops
> - Understand how Loop Reordering changes memory access patterns
> - Define Loop Tiling (Blocking) and why it is critical for on-chip SRAM caches
> - Calculate optimal tile sizes based on local memory constraints

---

## 1. Convolutions as Nested Loops

Underneath the high-level PyTorch abstraction `nn.Conv2d`, a standard convolutional layer is nothing more than a deeply nested 7-dimensional `for` loop.

To compute the output of a convolution layer, a naive CPU implementation looks like this:

```python
# Naive 7D Convolution Loop
for n in range(Batch_Size):
    for co in range(Channels_Out):
        for ci in range(Channels_In):
            for ho in range(Height_Out):
                for wo in range(Width_Out):
                    for kh in range(Kernel_Height):
                        for kw in range(Kernel_Width):
                            # The single MAC operation
                            Output[n, co, ho, wo] += Weight[co, ci, kh, kw] * Input[n, ci, ho*S+kh, wo*S+kw]
```

### The Problem with the Naive Approach
A CPU processes arrays sequentially. If your array is larger than the CPU's small L1 Cache, the CPU will constantly load a value from slow DRAM, use it once, and then evict it from the cache to make room for the next value. 

Because neural network data is reused heavily (the same Weights are used across all outputs `ho, wo`; the same Inputs are used across many `co` filters), executing loops in a naive, unoptimized order results in terrible **Cache Thrashing**.

---

## 2. Loop Reordering

**Loop Reordering** (Loop Permutation) simply swaps the order of the `for` loops to change the data access pattern without changing the mathematical result.

If you bring the `co` (Channels Out) loop to the absolute innermost level:
- You fetch one spatial pixel of `Input` and one spatial grid of `Weight`.
- You hold the `Input` perfectly stationary in the register.
- You cycle through all the weights across the output channels, multiplying them by that one single input.

Loop reordering is the software equivalent of hardware **Dataflow**. By changing the loop order, the compiler decides which of the three data arrays (Inputs, Weights, Outputs) gets accessed most frequently at the innermost core, implicitly defining what stays stationary in the registers.

---

## 3. Loop Tiling (Blocking)

While Loop Reordering helps, it isn't enough if the matrices are massive. If `Channels_In` is 2048, iterating across it will overflow an on-chip SRAM.

**Loop Tiling** (often called Loop Blocking) is the process of breaking a massive matrix multiplication down into smaller sub-matrix blocks (tiles) that fit perfectly inside the local, ultra-fast memory (like a TPU's 24MB local SRAM buffer, or a PE's inner Scratchpad).

### The Math of Tiling
Instead of doing a massive 1000x1000 matrix multiplication in one pass, we break the loop into "Outer Loops" (which pull chunks from DRAM) and "Inner Loops" (which compute exclusively from SRAM).

```python
# Tiled Matrix Multiplication
TILE_SIZE = 64

# Outer loops: Fetching tiles from DRAM to SRAM
for i_outer in range(0, M, TILE_SIZE):
    for j_outer in range(0, N, TILE_SIZE):
        for k_outer in range(0, K, TILE_SIZE):
            
            # Inner loops: Processing the tile purely in SRAM
            for i_inner in range(i_outer, i_outer + TILE_SIZE):
                for j_inner in range(j_outer, j_outer + TILE_SIZE):
                    for k_inner in range(k_outer, k_outer + TILE_SIZE):
                        C[i_inner, j_inner] += A[i_inner, k_inner] * B[k_inner, j_inner]
```

By ensuring that the $64 \times 64$ sub-matrices of A, B, and C all fit simultaneously into the local SRAM, the MAC units speed along at 100% utilization. Once the inner loops finish, the hardware writes the completed output tile back to DRAM, and loads the next tile.

### The Arithmetic of Reuse
The goal of tiling is to maximize the **Reuse Factor**.
In a naive matrix multiplication of two $[N \times N]$ matrices:
- Total Operations: $2N^3$
- Data fetched without tiling: $2N^3$ (each element fetched for every dot product)
- **Reuse Factor**: $1$ (Every byte fetched is used only once)

In a tiled implementation with tile size $B$:
- Every weight fetched from DRAM is kept in SRAM and used for $B$ different activations.
- **Reuse Factor**: $\approx B$
If $B = 64$, you have reduced your DRAM traffic by **$64 \times$**. This is the difference between a chip that is "starving" for data (Memory Bound) and a chip that is "saturated" with math (Compute Bound).

---

## 4. Hardware Optimization Hierarchy

Real accelerators use multi-level tiling and reordering. An advanced neural compiler (like TVM or XLA) analyzes the physical dimensions of the chip to generate the loops.

1. **Outer Tiling (DRAM to Global Buffer):** Sized to fit perfectly inside the multi-megabyte shared on-chip SRAM.
2. **Inner Tiling (Global Buffer to Array):** Sized to match the physical dimensions of the Systolic Array (e.g., $256 \times 256$).
3. **PE Level Reordering:** Ordered carefully to establish Weight Stationary or Output Stationary dataflows depending on which metric minimizes interconnect wiring power.

This massive optimization process is why a poorly optimized script running natively in Python on an accelerator might achieve 2% hardware utilization, while a compiled script hitting the tensor cores hits 95% utilization.

---

## Key Takeaways

- Deep Learning algorithms are mathematically identical to heavily nested 7D `for` loops.
- **Loop Reordering** changes the sequence of memory accesses, acting as the software-defined equivalent of hardware dataflow routing.
- **Loop Tiling (Blocking)** fragments massive data tensors into smaller topological tiles to fit inside finite on-chip SRAM.
- Tiling increases the **Reuse Factor** by approximately the size of the tile, drastically reducing expensive DRAM traffic.
- Understanding tiling is the prerequisite for the **Roofline Model** (Module 4), which predicts whether a specific tile size will cause your chip to hit its maximum speed or its memory wall.

---

## Practice Problems

### Problem 1: Optimal Tile Sizing

> **Context**: You are designing the compiler scheduling for an accelerator. The chip has a single fast SRAM containing exactly 16 Kilobytes (16,384 bytes) of working space. You are multiplying a massive Matrix A by Matrix B, producing Matrix C. All values are INT8 (1 byte). 
> 
> **Tasks**:
> - If you must load a square tile of $T \times T$ for Matrix A, Matrix B, and the accumulating Matrix C simultaneously into the SRAM, what is the theoretical maximum dimension $T$ that will fit without overflowing the memory? [2]

<details>
<summary><b>Solution</b></summary>

- We need to hold 3 tiles in memory simultaneously (Tile A, Tile B, Tile C).
- Therefore, $3 \times (T \times T \text{ bytes}) \le 16,384 \text{ bytes}$
- $3 \times T^2 \le 16384$
- $T^2 \le 5461.33$
- $T \le \sqrt{5461.33} \approx 73.9$
- Since tiles are typically powers of 2 for easy binary shifting and addressing, we would optimally choose a tile size of $T = 64$.
- Space used by $T=64$: $3 \times (64 \times 64) = 12,288$ bytes. It fits perfectly!

</details>

---

[← Previous Chapter: Pruning and Sparsity](02_pruning_and_sparsity.md) | [Next Chapter: Hardware-Software Co-Design →](04_hw_sw_codesign.md)
