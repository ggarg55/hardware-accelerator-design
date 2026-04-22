# Spiking Neural Networks (SNNs)

> **Learning Objectives**
> - Understand the Leaky Integrate-and-Fire (LIF) neuron model versus standard Artificial Neurons
> - Distinguish between Rate Coding and Temporal Coding in SNNs
> - Learn how Time functions as an active variable in neuromorphic computations
> - Describe the mechanism of Spike-Timing-Dependent Plasticity (STDP) for unsupervised learning

---

## 1. The Death of the Floating-Point Matrix

In traditional Artificial Neural Networks (ANNs), information is represented as a static, continuous real number (e.g., $0.783$). The standard mathematical formula for an artificial neuron is a MAC followed by a static activation function:

$$ y = \text{ReLU} \left( \sum W_i X_i + b \right) $$

In biological brains, neurons DO NOT pass numbers to each other. They pass **Action Potentials (Spikes)**. A spike is a binary event: it either happened at time $t$, or it didn't. 

**Spiking Neural Networks (SNNs)** are the third generation of neural networks, designed to operate natively via these discrete time-based events. 
- There are no floating-point MACs. 
- You do not multiply inputs by weights. 
- If an input spike arrives on a synapse, you simply **ADD** the synaptic weight to the neuron's internal voltage.

---

## 2. The Leaky Integrate-and-Fire (LIF) Model

The most common computational model used in neuromorphic hardware (like Intel's Loihi or IBM's TrueNorth) is the **Leaky Integrate-and-Fire (LIF)** neuron.

The LIF neuron holds a chemical "voltage" state, called the Membrane Potential ($V_{mem}$).

### The Three Phases of an LIF Neuron:
1. **Integrate:** When a spike arrives from a preceding neuron, the weight of that synapse is immediately added to $V_{mem}$. 
2. **Leak:** Just like a leaky bucket, if no spikes arrive, the voltage $V_{mem}$ decays over time back to a resting state ($0.0$). This forces the neuron to care about *recent* events, forgetting the distant past.
3. **Fire:** If a rapid burst of incoming spikes causes $V_{mem}$ to cross a specific threshold (e.g., $1.0$), the neuron physically fires a single boolean spike out of its axon to the next layer. It then resets its voltage to $0.0$.

```mermaid
flowchart LR
    subgraph Synapses
        S1["Spike! (+0.4)"] 
        S2["Spike! (+0.5)"]
        S3["Spike! (+0.3)"]
    end
    
    subgraph LIF Neuron
        V["Membrane Potential (V)"]
        L["Leak (decay over time)"]
        T{"V > Threshold?"}
        
        V --> L
        V --> T
    end
    
    S1 --> V
    S2 -.->|"Time delay"| V
    S3 --> V
    
    T -->|"Yes"| OUT["Fire Spike! 💥<br/>(Reset to 0)"]
    
    style OUT fill:#F44336,color:#fff
```

### Hardware Advantage
Instead of dense MAC arrays pulling megawatts to multiply fractions, an LIF hardware accelerator acts merely as a massive memory accumulator array. It performs integer additions only when triggered by asynchronous routing events. 

---

## 3. How do Spikes Encode Information?

If a neuron only sends boolean spikes (`1` or `0`), how can we represent complex values like the brightness of a pixel ($0.78$) or a speech frequency? SNNs encode continuous data into discrete spikes using two primary methods.

### 3.1 Rate Coding
In rate coding, the value is represented by the **frequency of spikes**.
- A bright pixel ($1.0$) causes the input neuron to fire 100 times per second.
- A dim pixel ($0.2$) causes the neuron to fire 20 times per second.
- A black pixel ($0.0$) results in total silence.

*Pros:* Highly fault-tolerant.
*Cons:* Extremely energy-inefficient. To pass one simple value, the hardware has to execute up to 100 separate physical spike events, negating the primary benefit of neuromorphic computing.

### 3.2 Temporal Coding (Time-To-First-Spike)
In temporal coding, the value is represented by **WHEN the spike fires**. 
- A bright pixel ($1.0$) causes the neuron to fire exactly one spike instantly at $t = 1 \text{ ms}$.
- A dim pixel ($0.2$) causes the neuron to fire exactly one spike delayed at $t = 50 \text{ ms}$.

**The Hardware Win:** 
In temporal coding, the **time dimension perfectly replaces the bit-depth**. 
In a digital CPU, representing a value with 32-bit precision requires 32 physical wires and a giant multiplier. In a neuromorphic chip, that same 32-bit precision can be represented by the *exact millisecond* a single wire pulses. We have traded complex physical logic (space) for high-resolution timing (time). 
Every pixel requires only **1 physical spike** to transmit its entire value, resulting in the absolute theoretical minimum energy per inference.

### Code Example: Rate Coding vs. Temporal Coding

```python
import numpy as np

def encode_rate(value, duration=100):
    """Frequency-based coding."""
    # A value of 0.8 fires 80% of the time
    return (np.random.rand(duration) < value).astype(int)

def encode_temporal(value, duration=100):
    """Time-to-first-spike coding."""
    # A high value (1.0) fires at t=1, low value (0.1) fires at t=90
    spike_train = np.zeros(duration)
    fire_time = int((1.0 - value) * (duration - 1))
    spike_train[fire_time] = 1
    return spike_train

val = 0.7
rate_spikes = encode_rate(val)
temp_spikes = encode_temporal(val)

print(f"Value: {val}")
print(f"Rate Coding Spikes:     {np.sum(rate_spikes)} spikes")
print(f"Temporal Coding Spikes: {np.sum(temp_spikes)} spike (Always 1!)")
```

---

## 5. Worked Example: Leaky-Integrate-and-Fire Trace

Let's trace a digital LIF neuron with a decaying membrane potential.

**Parameters**:
- **Threshold**: $100 \text{ mV}$
- **Synaptic Weight**: $+40 \text{ mV}$ per spike
- **Leak**: $10\%$ reduction of current voltage per step

| Time | Event | Voltage (V) Calculation | New Voltage | Fire? |
|:-----|:------|:------------------------|:------------|:------|
| 0    | Start | $0$ | $0$ | No |
| 1    | Spike | $(0 \times 0.9) + 40$ | $40$ | No |
| 2    | Silence| $(40 \times 0.9)$ | $36$ | No |
| 3    | Spike | $(36 \times 0.9) + 40$ | $72.4$ | No |
| 4    | Spike | $(72.4 \times 0.9) + 40$| **$105.16$**| **YES!** |
| 5    | Reset | $0$ | $0$ | No |

**Conclusion**: In an SNN, the "History" matters. If the final spike at $t=4$ had arrived much later (e.g., at $t=10$), the accumulated voltage would have leaked away, and the neuron would not have fired. This inherent temporal filtering is what makes SNNs so powerful for processing speech and video.

---

## 4. Spike-Timing-Dependent Plasticity (STDP)

How do these SNNs learn? Traditional backpropagation requires the chain-rule of calculus, which demands continuous, smooth mathematical derivatives. Binary spikes ($1$ or $0$) are entirely non-differentiable (you cannot do calculus on a stair-step function).

While modern researchers use approximations (Surrogate Gradients) to force backprop onto SNNs, Neuromorphic hardware prefers a biological learning rule called **STDP (Spike-Timing-Dependent Plasticity)**.

STDP is a localized, unsupervised learning rule. It requires no labels, no loss functions, and no central GPU calculating global gradients. 

**The Rule:** The weight of a synapse changes based purely on the physical time delay between the Pre-Synaptic spike (input) and the Post-Synaptic spike (output).

1. **Long-Term Potentiation (LTP):** If the input spike arrives *just before* the neuron fires, it means the input helped cause the firing. **Increase the weight.**
2. **Long-Term Depression (LTD):** If the input spike arrives *shortly after* the neuron already fired, it was useless. **Decrease the weight.**

```mermaid
flowchart TD
    subgraph STDP ["The STDP Learning Rule"]
        direction LR
        PrePost["Pre-spike before Post-spike<br/>(t_post - t_pre > 0)"] -->|Potentiation| UP["Increase Weight<br/>(Delta_w > 0)"]
        PostPre["Post-spike before Pre-spike<br/>(t_post - t_pre < 0)"] -->|Depression| DOWN["Decrease Weight<br/>(Delta_w < 0)"]
    end
    
    style UP fill:#4CAF50,color:#fff
    style DOWN fill:#F44336,color:#fff
```

**Why this is "Opus-level" efficient:**
Traditional AI (backpropagation) requires a global "God's Eye" view of the network to calculate errors and pass them backward. **STDP is strictly local**. A synapse only needs to "know" the timing of the two neurons it is physically connected to. This allows learning to happen **asynchronously and distributedly** across a chip, with no central controller. 

---

## Key Takeaways

- SNNs replace floating-point activations with binary spikes, and replace heavy MAC arrays with simple **Integrate-and-Fire** addition accumulators.
- The **Leaky** aspect of LIF forces neurons to evaluate temporal proximity.
- **Temporal Coding** allows a single spike arriving at a specific millisecond to convey exactly as much information as a 32-bit floating-point number, enabling colossal energy savings.
- **STDP** offers an unsupervised, biologically plausible alternative to backpropagation, allowing physical neuromorphic chips to learn "at the edge" by observing local spike timings without calculating global calculus derivatives.

---

## Practice Problems

### Problem 1: Integrate-and-Fire Dynamics

> **Context**: You are analyzing a digital LIF neuron executing at $1 \text{ ms}$ timesteps. 
> - Resting potential is $0.0$.
> - Firing threshold is $1.0$.
> - Leak rate: The voltage loses $0.1$ every millisecond step that passes.
> - An input spike arrives with a synaptic weight of $0.5$.
> 
> Timeline of incoming input spikes:
> - $t=1$: Spike arrives.
> - $t=2$: Silence.
> - $t=3$: Spike arrives.
> - $t=4$: Spike arrives.
>
> **Tasks**:
> - (a) Trace the internal membrane voltage at the **end** of each timestep $t=1$ through $t=4$. Assumed the leak occurs *before* adding the spike for that step if a spike exists. [3]
> - (b) At what timestep does the neuron fire? [1]

<details>
<summary><b>Solution</b></summary>

**(a) Voltage Tracing:**
- **$t=1$**: Starts at $0$. Leak is $0$. Receives spike ($+0.5$). 
  - $V_{t=1} = \mathbf{0.5}$
- **$t=2$**: No incoming spike. Leak occurs ($-0.1$). 
  - $V_{t=2} = 0.5 - 0.1 = \mathbf{0.4}$
- **$t=3$**: Spike arrives ($+0.5$). Leak occurs ($-0.1$). 
  - $V_{t=3} = 0.4 - 0.1 + 0.5 = \mathbf{0.8}$
- **$t=4$**: Spike arrives ($+0.5$). Leak occurs ($-0.1$). 
  - $V_{temp} = 0.8 - 0.1 + 0.5 = 1.2$
  - Since $1.2 \ge 1.0$, the biological threshold is crossed!
  - It fires and resets to $0$. So ending voltage is $\mathbf{0.0}$.

**(b)** The neuron reaches the threshold and fires at **$t=4$**.

### Problem 2: Energy per Synaptic Event

> **Context**: You are comparing a 32-bit floating point MAC on a GPU with a Synaptic Event on a Loihi neuromorphic chip.
> - **GPU MAC**: $200 \text{ pJ}$ (including DRAM fetch).
> - **Loihi Synaptic Event**: $0.02 \text{ pJ}$ (Addition only, local memory).
>
> **Tasks**:
> - (a) If a neural network requires 1 billion operations, calculate the energy for both. [1]
> - (b) If the network is 90% sparse (meaning only 10% of neurons actually fire), how does the energy of the Loihi chip change? [1]

<details>
<summary><b>Solution</b></summary>

**(a) Total Energy:**
- GPU: $10^9 \times 200 \text{ pJ} = \mathbf{0.2 \text{ Joules}}$.
- Loihi: $10^9 \times 0.02 \text{ pJ} = \mathbf{0.00002 \text{ Joules}}$.

**(b) Sparse Energy Impact:**
- A GPU still performs the 1 billion multiplications even if the inputs are zero (unless it has specialized zero-skip hardware).
- An SNN *only* consumes energy when a spike occurs. If only 10% of spikes happen, the Loihi chip only performs $100$ million operations.
- New Energy: $0.1 \times 0.00002 = \mathbf{0.000002 \text{ Joules}}$.
- **Result**: The combination of simplified math and event-driven sparsity makes the SNN **$100,000\times$** more efficient in this scenario.

</details>

---

[← Previous Chapter: Brain-Inspired Computing](01_brain_inspired_computing.md) | [Next Chapter: Compute-in-Memory with ReRAM →](03_compute_in_memory_reram.md)
