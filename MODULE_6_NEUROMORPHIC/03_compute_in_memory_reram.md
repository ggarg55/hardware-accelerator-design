# Compute-in-Memory with ReRAM

> **Learning Objectives**
> - Define the physical concept of Compute-in-Memory (CIM)
> - Understand how Memristors (ReRAM) function as tunable analog resistors
> - Map the mathematical equation for Matrix Multiplication directly onto Ohm's and Kirchhoff's physical laws
> - Evaluate the overhead challenges of CIM arrays (ADC and DAC conversions)

---

## 1. Escaping the Von Neumann Bottleneck

We've established that pumping data back and forth between memory and MAC units consumes immense power. Spiking neural networks encode data temporally, but they still typically require physical adder logic distinct from the SRAM itself.

What if we didn't just place the compute *next* to the memory? What if **the memory cell itself performed the computation physically?** 

This is the principle of **Compute-in-Memory (CIM)**. 
Instead of requesting $W$ from memory, sending it to an ALU, and feeding it $X$, we simply blast $X$ *into* the memory array as a voltage. The laws of physics will instantly calculate the dot product inside the structure without a single digital logic gate turning on.

### Analog (AIMC) vs. Digital (DIMC)
While the "purest" form of CIM is analog (using resistors), the industry is split into two paradigms:
1. **Analog In-Memory Computing (AIMC)**: Uses physics (currents and voltages) to compute. Extremely efficient but noisy and requires expensive converters.
2. **Digital In-Memory Computing (DIMC)**: Uses logic gates physically collocated within the memory bitcells. More robust to noise and doesn't need ADCs, but takes up more silicon area than a raw resistor grid.

---

## 2. The Memristor (ReRAM)

To achieve CIM for AI, we need a special kind of memory component called a **Memristor** (Memory + Resistor), typically implemented as **ReRAM** (Resistive Random Access Memory).

A typical SRAM cell locks a `1` or `0` using a cage of 6 transistors. It is volatile (loses data when powered off).

A Memristor is physically continuous. It consists of a thin metal oxide layer between two electrodes. By applying specific high-voltage programming pulses across the electrodes, oxygen ions are literally pushed into or pulled out of the oxide layer, which changes the physical **resistance** of the material.
Because this alters the physical chemistry of the wire, the resistance stays locked permanently, even when power is turned off (non-volatile memory). 

### Modeling the Synapse
Because the resistance can be programmed to discrete analog levels (e.g., $1000 \ \Omega$, $2000 \ \Omega$, $3000 \ \Omega$), a single ReRAM cell acts as an analog storage container for our Neural Network Weights ($W$). 
Rather than resistance $R$, electrical engineers prefer to use **Conductance ($G$)**, which is simply the inverse: $G = \frac{1}{R}$.

A highly conductive wire ($G$ is high) represents a strong neural connection (large weight $W$). A highly resistive wire ($G$ is low) represents a weak or pruned connection (small weight). 

---

## 3. Physical Matrix Multiplication 

Once the weights are programmed into a microscopic grid grid of ReRAM cells (a Crossbar Array), we can execute an entire neural network layer at the speed of light.

To understand how, we align the math of Neural Networks with the fundamental laws of electrical physics.

### 3.1 Unpacking the Math

**The Neural Math (One element):**
$$ \text{Output} = \text{Input} \times \text{Weight} $$

**Ohm's Law:**
If we apply a Voltage ($V$) across a component with Conductance ($G$), the resulting physical electric Current ($I$) is:
$$ I = V \times G $$

Notice that it is the exact same equation! 
- If we encode our Input Activations as **Voltages**.
- If we encode our Neural Weights as physical **Conductances**.
- Then the flowing **Electrical Current** is instantly equivalent to the MAC multiplication.

### 3.2 Unpacking the Matrix (Kirchhoff's Law)

**The Neural Math (Dot Product):**
$$ \text{Total Output} = (W_1 X_1) + (W_2 X_2) + (W_3 X_3) $$

**Kirchhoff's Current Law:**
The total current flowing out of a common wire is exactly equal to the sum of all the currents flowing into it.
$$ I_{total} = I_1 + I_2 + I_3 $$

When we pulse all our Voltage inputs simultaneously along horizontal wires, and the currents drop through the ReRAM "weights" into vertical wires, the vertical wires physically sum the total current. 

```mermaid
flowchart LR
    subgraph Crossbar Array Computation
        direction TB
        V1["V_in 1"] -->|Wire| R1["ReRAM G1"]
        V2["V_in 2"] -->|Wire| R2["ReRAM G2"]
        V3["V_in 3"] -->|Wire| R3["ReRAM G3"]
        
        R1 -->|I_1| SUM
        R2 -->|I_2| SUM
        R3 -->|I_3| SUM
        
        SUM["Vertical Wire physically sums current"] --> I_OUT["I_total<br/>(Dot Product)"]
    end
    
    style R1 fill:#FF9800,color:#fff
    style R2 fill:#FF9800,color:#fff
    style R3 fill:#FF9800,color:#fff
    style I_OUT fill:#4CAF50,color:#fff
```

### 3.3 The Crossbar Operation Walkthrough

To compute a layer, the CIM hardware follows these steps:
1. **Programming Phase**: The Weight matrix $W$ is converted into conductance values. A "Write" circuit applies high-voltage pulses to set the resistance of each ReRAM cell at the intersection of rows and columns.
2. **Inference Phase**: 
    - The Input vector $X$ is converted into analog voltages $V_j$ by DACs.
    - All $V_j$ are applied to the horizontal rows **simultaneously**.
    - Current $I_{i,j} = V_j \cdot G_{i,j}$ flows through each cell.
    - According to Kirchhoff's Law, the total current $I_{out, i}$ at the bottom of column $i$ is the sum of all currents in that column.
    - $I_{out, i} = \sum_j V_j \cdot G_{i,j}$.
3. **Conversion Phase**: The ADCs at the bottom of each column sample the current and convert it back into a digital number for the next layer.

**The Power of CIM:** A massively parallel digital TPU needs thousands of discrete logic gates to do this. An analog ReRAM crossbar does it in a single step merely by pulsing electricity through a grid of programmable resistors. The memory *is* the multiplier. The wires *are* the adders.

### Code Example: Simulating an Analog Crossbar

```python
import numpy as np

def simulate_crossbar(V_in, G_matrix, noise_sigma=0.01):
    """Simulate physical matrix multiplication in a ReRAM crossbar."""
    # Add physical noise (variability in programming conductance)
    G_noisy = G_matrix + np.random.normal(0, noise_sigma, G_matrix.shape)
    
    # Kirchhoff's Law: I[channel] = sum(V[row] * G[row, channel])
    I_out = np.dot(V_in, G_noisy)
    
    return I_out

# Weights quantized to conductances (0.0 to 1.0 mS)
weights = np.array([[0.5, 0.1], [0.2, 0.8]])
inputs = np.array([0.9, 0.4]) # Input voltages

ideal_result = np.dot(inputs, weights)
analog_result = simulate_crossbar(inputs, weights)

print(f"Ideal Physical Result: {ideal_result}")
print(f"Noisy Analog Result:  {analog_result}")
print(f"Error: {np.abs(ideal_result - analog_result)}")
```

---

## 4. The Overhead Penalty (ADC and DAC)

Because the rest of the computer (and the software) is digital, we must translate digital values (INT8) into analog pulses (Voltages), and then translate analog results (Currents) back to digital to send them to the next layer.

### The Conversion Bottleneck
1. **DAC (Digital-to-Analog Converter):** Required at the edge of horizontal rows to turn an 8-bit digital pixel into a specific analog voltage.
2. **ADC (Analog-to-Digital Converter):** Required at the bottom of the vertical columns to read the specific amperage of the summed current and snap it back to a discrete digital output.

**The "85/15" Problem:** 
In a real-world AIMC accelerator, **the ADCs and DACs often take up $80\%$ to $90\%$ of the chip area and power**. 
- The crossbar math itself is nearly "free" (femtojoules).
- The ADC conversion is expensive (picojoules).
Furthermore, as we increase precision (e.g., from 4-bit to 8-bit), the ADC size and power consumption grow **exponentially**. This is why most CIM chips today are limited to low-precision 1-bit or 4-bit operations.

### The SRAM Alternative
While ReRAM is the future, many researchers use **SRAM-based CIM** today. By activating multiple "Wordlines" simultaneously in a standard SRAM bank, you can force the bitlines to discharge at a rate proportional to the stored bits. This allows us to use mature silicon manufacturing processes while still gaining the benefits of collocated compute.

---

## 5. Worked Example: Crossbar Math with Noise

Let's calculate the precision impact of using an analog crossbar.

**Inputs**: $V = [0.1, 0.8, 0.5]$ Volts
**Weights (Conductances)**: $G = [100, 200, 50] \ \mu\text{S}$

**Step 1: Calculate Ideal Current**
- $I_1 = 0.1 \text{ V} \times 100 \ \mu\text{S} = 10 \ \mu\text{A}$
- $I_2 = 0.8 \text{ V} \times 200 \ \mu\text{S} = 160 \ \mu\text{A}$
- $I_3 = 0.5 \text{ V} \times 50 \ \mu\text{S} = 25 \ \mu\text{A}$
- **Total Ideal $I$** = $10 + 160 + 25 = \mathbf{195 \ \mu\text{A}}$.

**Step 2: Introduce Device Variation**
Suppose the middle memristor ($G_2$) has a "programming error" of $+10\%$.
- $G_{2,noisy} = 220 \ \mu\text{S}$
- $I_{2,noisy} = 0.8 \times 220 = 176 \ \mu\text{A}$
- **Total Noisy $I$** = $10 + 176 + 25 = \mathbf{211 \ \mu\text{A}}$.

**Conclusion**: In digital logic, $2 \times 2$ is always $4.000$. In analog CIM, $2 \times 2$ might be $4.1$ or $3.9$ depending on the physical temperature and the precision of the memristor chemistry. This is why **Noise-Aware Training** is required for CIM models.

---

## Key Takeaways

- **Compute-in-Memory (CIM)** breaks the Von Neumann bottleneck by physically turning the memory storage array into the math processing unit.
- **AIMC** uses analog physics for density, while **DIMC** uses collocated digital logic for noise robustness.
- **ReRAM/Memristors** are non-volatile hardware cells whose physical resistance can be programmed to represent neural *Weights*.
- **Ohm's Law ($I = V \times G$)** and **Kirchhoff's Law** perform the dot product physically in a single step.
- The massive hardware overhead of **ADCs and DACs** is the "hidden cost" of analog CIM, often consuming >80% of total system power.

---

## Practice Problems

### Problem 1: Ohm's Law and Crossbar Logic

> **Context**: You are analyzing a $2 \times 2$ section of an analog RRAM crossbar array. 
> - Input 1 is driven at $V_1 = 0.5 \text{ V}$
> - Input 2 is driven at $V_2 = 1.0 \text{ V}$
> 
> The crossbar cells attached to the first vertical output column are programmed to conductances:
> - $G_{11} = 20 \ \mu S \ (10^{-6} \text{ Siemens})$ 
> - $G_{21} = 50 \ \mu S$
>
> **Tasks**:
> - (a) What is the physical current contribution from input 1 traversing through cell $(1,1)$? [1]
> - (b) What is the physical current contribution from input 2 traversing through cell $(2,1)$? [1]
> - (c) At the bottom of the vertical wire (column 1), what is the total current $I_{out}$ read by the ADC? [1]

<details>
<summary><b>Solution</b></summary>

**(a)** Ohm's Law for Cell 1,1:
- $I_1 = V_1 \times G_{11}$
- $I_1 = 0.5 \text{ V} \times 20 \ \mu S = \mathbf{10 \ \mu A}$

**(b)** Ohm's Law for Cell 2,1:
- $I_2 = V_2 \times G_{21}$
- $I_2 = 1.0 \text{ V} \times 50 \ \mu S = \mathbf{50 \ \mu A}$

**(c)** Kirchhoff's Current Law (Total current):
- $I_{total} = I_1 + I_2$
- $I_{total} = 10 \ \mu A + 50 \ \mu A = \mathbf{60 \ \mu A}$
- The ADC array must be calibrated to recognize `60 Micro-Amps` as the specific Integer equivalent for that neural output.

### Problem 2: The ADC Bottleneck

> **Context**: You are designing a CIM chip for an INT8 model. You have a choice between two ADCs for your crossbar columns:
> 1. **ADC-A (4-bit)**: Consumes **1 pJ** per conversion.
> 2. **ADC-B (8-bit)**: Consumes **16 pJ** per conversion.
>
> **Tasks**:
> - (a) If your model is trained for 8-bit precision, but you use ADC-A to save power, what happens to your model's accuracy? [1]
> - (b) If your crossbar has 128 columns and operates at 100 MHz, what is the power consumption of the ADC array using ADC-B? [2]

<details>
<summary><b>Solution</b></summary>

**(a)** Accuracy Loss:
- Using a 4-bit ADC means your 8-bit math is "crushed" back down to 16 levels ($2^4$) at every layer. 
- You lose information at every step, likely crashing the model's performance to $0\%$ or near-random unless you use **Quantization-Aware Training** (Module 5) to specifically learn to live with 4 bits.

**(b) Power Calculation (ADC-B):**
- Operations/sec = 128 columns $\times 100 \times 10^6 \text{ Hz} = 12,800,000,000 \text{ conversions/sec}$.
- Power = $12.8 \times 10^9 \text{ conv/s} \times 16 \times 10^{-12} \text{ J/conv}$
- **Power = 204.8 mW**.
- **Analysis**: Even though the "math" in the crossbar only uses a few milliwatts, the ADCs alone consume over $200$ mW. This is the **Conversion Tax** of analog computing.

</details>

---

[← Previous Chapter: Spiking Neural Networks](02_spiking_neural_networks.md) | [Next Module: Transformers & LLMs →](../MODULE_7_TRANSFORMERS_LLM/README.md)
