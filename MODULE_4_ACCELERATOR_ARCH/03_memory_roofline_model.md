# Memory Hierarchy and the Roofline Model

> **Learning Objectives**
> - Construct and interpret a Roofline Plot for hardware accelerators
> - Differentiate between Compute-Bound and Memory-Bound workloads
> - Understand Arithmetic Intensity and how data reuse algorithms push workloads to the right
> - Calculate the Ridge Point to determine optimal architectural balance

---

## 1. The fundamental limitation: The Memory Wall

In the previous chapters, we looked at how accelerators employ massive arrays of MACs. A 256×256 TPU array has 65,536 multipliers. If that chip clocks at 1 GHz, it can theoretically calculate **131 Tera-MACs per second**.

But there is a catch. Without data perfectly positioned on the fastest internal SRAM registers at all times, the MACs will just sit idle. Off-chip memory (DRAM) accesses are historically much, much slower than MAC speeds. This disparity is known as **The Memory Wall**. 

If a workload needs to fetch fresh data from DRAM for every computation, the hardware performance collapses. It does not matter if you have 1 million MAC units; if your DRAM bus only delivers enough data to feed 1,000 MACs, the other 999,000 will simply burn power waiting.

---

## 2. Arithmetic Intensity

To understand if a network is going to suffer from the memory wall, we calculate its **Arithmetic Intensity**.
Arithmetic Intensity (also known as Operational Intensity) is a ratio metric describing how many operations you perform per byte of memory traffic.

$$ \text{Arithmetic Intensity (FLOPs/Byte)} = \frac{\text{Total Compute Operations (FLOPs or MACs)}}{\text{Total Bytes Sent/Received from Memory}} $$

### Example
Suppose we have a Fully Connected (FC) layer where we multiply a 1x1000 vector with a 1000x1000 matrix. 
- Total MACs: $1000 \times 1000 = 1,000,000$ operations.
- Total Memory required (assuming 8-bit precision): Size of matrix + size of vector = $\approx 1,000,000$ bytes.
- Arithmetic Intensity = 1,000,000 operations / 1,000,000 bytes = **1 Operation / Byte**.

An intensity of 1 is very low. This tells us the layer is extremely memory-heavy.

In contrast, a 2D convolution with a large sliding window uses the exact same filter weights across thousands of pixels. A convolution can easily achieve an Arithmetic Intensity of **30-50 Operations / Byte**, meaning it is highly suited for hardware acceleration.

---

## 3. The Roofline Model

The **Roofline Model** is a visual tool used by hardware architects to understand the bounds of an accelerator's performance. It explicitly charts Arithmetic Intensity against actual Operations Per Second.

The Roofline is formed by two physical, unchangeable ceilings on your chip:
1. **The Compute Roof:** The absolute maximum peak operations per second (determined by the number of MACs and clock speed).
2. **The Memory Roof (Slanted):** Maximum operations you can sustain constrained by the Memory Bandwidth (GB/s).

```mermaid
XYChart
    title "The Roofline Model"
    x-axis "Arithmetic Intensity (Operations/Byte)" 1 --> 100
    y-axis "Performance (Operations/Sec)" 1 --> 100
    line [10, 20, 30, 40, 50, 50, 50, 50, 50]
```
*(Imagine the plot above: Performance rises linearly on a slope due to memory bandwidth, until it hits the horizontal compute roof limit.)*

The equation for the Roofline is:
$$ \text{Attainable Performance} = \min \begin{cases} \text{Peak Compute Performance (TOPS)} \\ \text{Peak Memory Bandwidth (TB/s)} \times \text{Arithmetic Intensity (OPS/Byte)} \end{cases} $$

### 3.1 The Two Regions

The point where the slanted memory roof meets the flat compute roof is called the **Ridge Point**.

1. **Left of the Ridge Point (Memory-Bound Region):**
   - Workloads here have low arithmetic intensity. 
   - Performance is constrained by how fast the DDR memory bus can push bytes.
   - Example workloads: Fully-connected layers with small batches, Attention mechanisms waiting for KV caches.
   - Adding more MAC units to your chip here will do **absolutely nothing** — the data bus is already the bottleneck.

2. **Right of the Ridge Point (Compute-Bound Region):**
   - Workloads here have high arithmetic intensity (e.g. they reuse weights efficiently).
   - Performance is capped by the Peak Compute line (how many MACs physically exist).
   - Example workloads: Convolutional layers with large batch sizes.
   - If your workload falls here, upgrading to a faster memory bus will do **absolutely nothing**.

### 3.2 Architectural Balance: The Ridge Point
The **Ridge Point** is the most important number in an accelerator's datasheet. It tells you the exact point of balance. 
- A **high Ridge Point** means the chip has massive compute but a very weak memory bus. It is only fast for very specific, highly optimized kernels.
- A **low Ridge Point** means the chip has a huge, expensive memory bus relative to its compute power. It is "easy" to program but expensive to build.
Google's TPU v1 was famous for having a high ridge point (~250 Ops/Byte), requiring users to use massive batch sizes and intensive tiling just to "hit the roof."

---

## 4. How to Improve Performance under the Roofline?

If your algorithm is running poorly, the Roofline tells you how to fix it:

A. **If Memory-Bound (Left Side):**
- **Increase Batch Size**: In inference, a larger batch size allows you to use the same weight matrix for multiple inputs simultaneously, increasing reuse.
- **Loop Tiling**: As seen in Module 5, tiling keeps data on-chip longer, increasing arithmetic intensity.
- **Quantization**: Moving from FP32 to INT8 instantly quadruples your arithmetic intensity, sliding your workload to the right. 

B. **If Compute-Bound (Right Side):**
- You have reached the "Peak" of your hardware. To go faster, you must increase clock speed or add more physical MAC units.
- **Sparsity**: Exploiting unstructured or N:M sparsity (Module 5) can "cheat" the roof by performing fewer actual math operations for the same logical result.

### Code Example: Roofline Calculator

```python
def roofline(peak_compute_tops, peak_bw_gbs, arithmetic_intensity):
    """Calculate attainable performance under the Roofline model."""
    peak_compute = peak_compute_tops * 1e12          # Convert TOPS to OPS
    peak_bw = peak_bw_gbs * 1e9                      # Convert GB/s to B/s
    memory_roof = peak_bw * arithmetic_intensity     # OPS from memory limit
    attainable = min(peak_compute, memory_roof)
    
    ridge_point = peak_compute / peak_bw
    bound = "Compute-Bound" if arithmetic_intensity >= ridge_point else "Memory-Bound"
    
    return attainable / 1e12, ridge_point, bound

# NVDLA-like accelerator: 12 TOPS, 48 GB/s
for ai in [10, 100, 250, 500]:
    perf, ridge, bound = roofline(12, 48, ai)
    print(f"AI={ai:4d} OPS/B → {perf:6.2f} TOPS ({bound})")
# AI=  10 OPS/B →   0.48 TOPS (Memory-Bound)
# AI= 100 OPS/B →   4.80 TOPS (Memory-Bound)
# AI= 250 OPS/B →  12.00 TOPS (Compute-Bound)  ← Ridge Point
# AI= 500 OPS/B →  12.00 TOPS (Compute-Bound)
```

---

## 5. Worked Example: Three Layers on NVDLA

Let's evaluate three different layers from a computer vision model running on the **NVDLA** accelerator ($12$ TOPS Peak Compute, $48$ GB/s Peak Bandwidth, Ridge Point = $250$ OPS/B).

| Layer Type | Arithmetic Intensity (AI) | Bound | Attainable Performance | Efficiency ($P_{act} / P_{peak}$) |
|:-----------|:---------------------------|:------|:-----------------------|:---------------------------------|
| **Pointwise (1x1)** | 4 OPS/B | Memory | $4 \times 48 = \mathbf{0.19}$ TOPS | $1.6\%$ (Terrible) |
| **Depthwise (3x3)** | 25 OPS/B | Memory | $25 \times 48 = \mathbf{1.2}$ TOPS | $10\%$ (Poor) |
| **Convolution (3x3)**| 300 OPS/B | Compute| **$12$** TOPS | $100\%$ (Perfect) |

**Conclusion**: The simple 1x1 convolution is starving for data so badly that $98\%$ of the hardware is sitting idle. To fix this, an architect wouldn't add more MACs; they would instead implement **Loop Tiling** to cache the 1x1 weights in SRAM, artificially raising the AI.

---

## Practice Problems

### Problem 1: Calculating the Ridge Point

> **Context**: The NVDLA accelerator has a Peak Compute of 12 TOPS (Tera-Operations Per Second) and its AXI bus has a Peak Memory Bandwidth of 48 GB/s. 
>
> **Tasks**:
> - (a) What is the arithmetic intensity of the Ridge Point? [2]
> - (b) If a neural network layer has an arithmetic intensity of 100 OPS/Byte, is it Compute-Bound or Memory-Bound? [1]
> - (c) What is the attainable performance of that layer? [1]

<details>
<summary><b>Solution</b></summary>

**(a)** Ridge Point Calculation:
- The Ridge Point occurs where the Memory Roof formula equals Peak Compute.
- Peak Compute = Peak Bandwidth × Arithmetic Intensity
- 12 TOPS = 48 GB/s × AI
- AI = (12 × 10¹² ops/s) / (48 × 10⁹ bytes/s) = **250 Operations / Byte**.

**(b)** Since 100 OPS/Byte is less than the Ridge Point of 250 OPS/Byte, the layer is on the slanted incline on the left. It is **Memory-Bound**.

**(c)** Attainable Performance:
- Memory Bandwidth × AI = (48 GB/s) × 100 OPS/Byte
- = 4800 Giga-OPS = **4.8 TOPS**.
- The hardware physically has 12 TOPS capability, but due to the memory bottleneck, it only runs at 4.8 TOPS.

### Problem 2: Moving the Dot

> **Context**: You are hired by StartupX to optimize an LLM running on their custom accelerator. The current Attention kernel has an Arithmetic Intensity of **30 OPS/Byte**. The hardware has a Ridge Point of **100 OPS/Byte**.
> 
> **Tasks**:
> - (a) Is the chip currently compute-bound or memory-bound? [1]
> - (b) If you use **KV Caching** (Module 7), you reduce memory traffic, increasing the AI to **120 OPS/Byte**. By what factor does your throughput (TOPS) increase? [2]

<details>
<summary><b>Solution</b></summary>

**(a)** The AI (30) < Ridge Point (100). The chip is **Memory-Bound**.

**(b)** Throughput Increase:
- **Before**: Performance was limited by Memory Bandwidth. Let's call Peak Bandwidth $B$. Performance = $30 \times B$.
- **After**: The new AI (120) > Ridge Point (100). The chip is now **Compute-Bound**. It is running at its absolute peak (Peak Compute).
- Peak Compute = Ridge Point $\times B = 100 \times B$.
- Speedup = (After / Before) = $(100 \times B) / (30 \times B) = \mathbf{3.33x}$.
- Note: Even though the AI increased by 4x, the performance only increased by 3.33x because we hit the "Compute Roof" along the way!

</details>

---

[← Previous Chapter: Dataflows](02_dataflow_taxonomies.md) | [Next Chapter: Eyeriss Case Study →](04_eyeriss_case_study.md)
