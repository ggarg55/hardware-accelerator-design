# Module 7: Transformers & LLM Acceleration

Welcome to **Module 7**, the final module of the AI Hardware Accelerator Design curriculum! 

In this module, we transition away from standard Convolutions (CNNs) and focus on the dominant architecture of the modern AI era: **The Transformer**. First introduced in 2017, Transformers have scaled up to billions (and trillions) of parameters, resulting in Large Language Models (LLMs) like GPT-4, Llama, and Claude.

These models present completely new and severe bottlenecks for hardware designers that did not exist in the CNN era. 

## What You Will Learn

This module covers 5 core topics:

1. **[Self-Attention Hardware computations](01_self_attention_hardware.md):** Breaking down the Query, Key, and Value ($Q, K, V$) matrix multiplications, and understanding why the $O(N^2)$ sequence length scaling breaks traditional hardware.
2. **[Flash Attention](02_flash_attention.md):** The revolutionary algorithmic and hardware-aware mapping technique that dramatically reduces SRAM/DRAM traffic during attention operations using tiling.
3. **[KV Cache in LLM Inference](03_kv_cache_inference.md):** Understanding the memory-bound nature of autoregressive text generation, and the engineering behind the Key-Value (KV) cache.
4. **[Vision Transformers (ViT)](04_vision_transformers_vit.md):** How the Transformer architecture was adapted for image processing, and how it differs from CNNs on hardware.
5. **[The LPU & Future of Inference](05_groq_lpu_and_future.md):** A case study on Groq's Language Processing Unit (LPU), overcoming memory bandwidth limits with massive SRAM arrays and deterministic execution.

## Prerequisites

- Understanding of the **Von Neumann Bottleneck** and the **Memory Wall** (Module 1, 4)
- Knowledge of fundamental **Matrix Multiplication Hardware** and **Dataflow Taxonomies** (Module 4)
- Understanding of **SRAM vs. DRAM latency** hierarchies (Module 2)

---

**Let's begin!** Proceed to Chapter 1: [Self-Attention Hardware](01_self_attention_hardware.md).
