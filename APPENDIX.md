# Appendix: Glossary & Reference Sheets

This appendix contains a quick-reference guide to the key terminology, formulas, and concepts discussed throughout the **Hardware Accelerator Design** curriculum.

---

## 1. Glossary of Terms

### Memory & Hardware Structures
- **ALU (Arithmetic Logic Unit):** The fundamental digital logic circuit that performs integer and bitwise operations (like ADD and MULTIPLY).
- **MAC (Multiply-Accumulate Unit):** A hardware component combining a multiplier and an adder, executing $A = A + (B \times C)$. The fundamental building block of AI chips.
- **SRAM (Static Random-Access Memory):** Fast, expensive, on-chip memory (Cache). Constructed using transistors (no capacitors). Latency is measured in single nanoseconds.
- **DRAM / HBM (Dynamic RAM / High-Bandwidth Memory):** Slower, cheaper, massive off-chip memory used for main storage. Constructed with capacitors that require constant refreshing. Latency is tens to hundreds of nanoseconds.
- **Systolic Array:** A dense 2D grid of MAC units that pass data directly to neighbors asynchronously (like blood through a heart), maximizing compute density and minimizing SRAM reads. Used in Google's TPU.
- **ReRAM (Resistive RAM):** A type of non-volatile memory that stores data by changing the physical resistance of a solid dielectric material. Used in Compute-in-Memory arrays.
- **LPU (Language Processing Unit):** A hardware architecture (pioneered by Groq) that eliminates DRAM/HBM entirely, using massive distributed SRAM and deterministic software scheduling to process LLMs with ultra-low latency.
- **KV Cache:** A memory-optimization technique in LLM inference that stores previously calculated Key and Value vectors in VRAM to avoid redundant $O(N^2)$ recomputation for every new token.

### AI Algorithmic Terms
- **CNN (Convolutional Neural Network):** Highly hardware-efficient network that slides localized filter weights across matrices. High data-reuse, sparse activations.
- **Transformer:** The dominant architecture for sequence modeling (LLMs, ViTs). Utilizes Self-Attention to map global context across sequence elements. 
- **SNN (Spiking Neural Network):** A biology-inspired network that passes discrete Boolean spikes asynchronously over time instead of static continuous FP32 numbers.
- **Dataflow (WS, OS, RS):** The scheduling strategy used to map mathematical operations to physical hardware to minimize expensive memory accesses. (Weight Stationary, Output Stationary, Row Stationary).
- **Flash Attention:** An IO-aware algorithm that tiles the attention computation and uses an "online softmax" trick to keep all intermediate $N \times N$ matrices inside fast SRAM, eliminating HBM bottlenecks.

### Bottlenecks & Phenomena
- **Von Neumann Bottleneck:** The fundamental limitation of modern computers where data processing speed is physically throttled by the rate at which data can be fetched from separate memory storage over a bus.
- **The Memory Wall:** The expanding gap between how fast ALUs can compute and how fast memory can deliver data.
- **Arithmetic Intensity:** The ratio of Total Computations performed (FLOPs) to Total Data Accessed (Bytes). Higher intensity means better utilization of the GPU.

---

## 2. Key Mathematical Formulas

### Matrix Multiplication Complexity
For multiplying a $[M \times K]$ matrix with a $[K \times N]$ matrix:
- **Total MAC Operations:** $M \times K \times N$
- **Total Parameters (Data):** $(M \times K) + (K \times N) + (M \times N)$

### Attention Complexity
For a sequence of length $N$ with hidden dimension $d$:
- **Attention Matrix Size:** $N \times N$ (Quadratic scaling)
- **Standard Inference Complexity:** $O(N^2)$ in both memory and compute.
- **Pre-fill Phase:** Linear with scaling $N$ for weights, but $N^2$ for the attention score matrix.

### The Roofline Model
Calculates the absolute maximum performance of a specific hardware chip.
- **Theoretical Limit:** $\text{Performance} = \min \begin{cases} \text{Peak Compute} \ (\text{FLOPs/s}) \\ \text{Peak Bandwidth} \ (\text{Bytes/s}) \times \text{Arithmetic Intensity} \end{cases}$

### Quantization (Affine)
Mapping floating-point continuous numbers ($r$) to scaled integer grids ($q$).
- **Equation:** $r = S (q - Z)$
  - $S$ = Scale Factor (Step size between intervals)
  - $Z$ = Zero Point (Which integer perfectly maps to physical $0.0$)

### Neuromorphic Compute-In-Memory (Ohm's Law)
Mapping Neural connections directly into physics.
- **Equation:** $I_{total} = \sum (V_{input} \times G_{weight})$
  - $V$ (Voltage) corresponds to Input Activation.
  - $G$ (Conductance) corresponds to stored Matrix Weight.
  - $I$ (Current) corresponds to the MAC Output.

---

## 3. Scale Comparisons table

| Hardware / Paradigm | Latency/Access Speed | Energy / Operation | Dominant Workload | Example System |
| :--- | :--- | :--- | :--- | :--- |
| **SRAM (On-Chip Cache)** | $1 - 5 \text{ ns}$ | ~ $5 \text{ pJ}$ (Picojoules) | Temporary Buffers, Register Files | L1/L2 Cache |
| **HBM / GDDR (Off-Chip)** | $50 - 200 \text{ ns}$ | ~ $100 \text{ pJ}$ ($20\times$ worse) | Large Weight Storage, Embeddings | Nvidia H100 |
| **MAC Operation (Compute)** | $< 1 \text{ ns}$ | ~ $0.1 \text{ pJ}$ | Dense Matrix Multiplication | TPU Systolic Array |
| **Von Neumann Arch.** | High overhead | High (Bus transport) | General-purpose Software | Intel CPUs |
| **Analog CIM (ReRAM)** | Instant | Extremely Low | Vector Dot Products (Edge AI) | Lab Prototypes |
| **Biological Brain** | Milliseconds | Femtojoules | Event-based asynchronous logic | Human |

---

[← Return to Curriculum Root](README.md)
