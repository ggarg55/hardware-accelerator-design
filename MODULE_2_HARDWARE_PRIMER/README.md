# Module 2: Digital Hardware Design Primer

> **How do you actually build a hardware circuit? What are the building blocks, and how do they connect?**

This module introduces the fundamental components and design patterns used in every hardware accelerator. You don't need a background in electrical engineering — we build from the ground up, starting with logic gates and ending with complete system architectures.

---

## Learning Objectives

By the end of this module, you will be able to:
- Identify and explain the six core hardware components: gates, flip-flops, registers, adders, multipliers, and multiplexers
- Decompose any algorithm into its control path (FSM) and data path (arithmetic + storage)
- Understand synthesizability constraints (why `while` loops fail and `for` loops succeed)
- Design hardware for vector and matrix multiplication — the core of all AI accelerators
- Read and understand accelerator system architectures (DDR, AXI, BRAM, FIFO)
- Explain the FPGA vs. ASIC trade-off and the role of High-Level Synthesis

---

## Chapters

| # | Chapter | Key Topics |
|:---|:---|:---|
| 1 | [Hardware Building Blocks](01_hardware_building_blocks.md) | Gates, flip-flops, registers, adders (chain vs tree), multipliers, multiplexers, comparators |
| 2 | [Control and Data Paths](02_control_and_data_paths.md) | Control/data path decomposition; FSM design; GCD worked example; synthesizability rules |
| 3 | [Matrix Multiplication Hardware](03_matrix_multiplication_hardware.md) | Dot products, matrix-vector, matrix-matrix; O(N³) resource scaling; memory bottleneck |
| 4 | [System Architecture & HLS](04_system_architecture_and_hls.md) | DDR, AXI, BRAM, FIFO; FPGA vs ASIC; High-Level Synthesis; power efficiency |

---

## Prerequisites

- **Module 1**: [Foundations of AI Hardware](../MODULE_1_FOUNDATIONS/)
- Basic understanding of binary numbers and Boolean logic (helpful but not required)

---

## What's Next?

After completing this module, proceed to **[Module 3: Neural Network Computations](../MODULE_3_NN_COMPUTATIONS/)** to learn how the building blocks from this module combine to implement CNN layers, number systems, and convolution arithmetic in hardware.
