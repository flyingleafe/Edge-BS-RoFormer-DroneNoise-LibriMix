Based on the papers provided, the **"Diffusion Buffer"** model is the most suitable implementation for a drone use case.

### Selection: Diffusion Buffer (2025)

**Paper:** *Diffusion Buffer: Online Diffusion-based Speech Enhancement with Sub-Second Latency* 1\.  
**Why this fits the Drone Use Case:**

1. **Online/Streaming Capability:** Unlike standard diffusion models that require the whole file (offline), this model uses a sliding buffer to process audio as it arrives. This is critical for drone operations requiring telemetry or live audio monitoring 1, 2\.  
2. **Latency Management:** It offers a tunable trade-off between quality and latency. You can achieve sub-second latency (320–960 ms), which is acceptable for many UAV transmission scenarios, whereas standard diffusion models are too slow 2\.  
3. **Edge Optimization:** The authors explicitly reduced the model size (18M parameters vs. the standard 65M) to run on consumer-grade GPUs, achieving real-time factors (RTF) \< 1\. This aligns with the constraints of edge devices like the Jetson AGX Xavier mentioned in the *Edge-BS-RoFormer* paper 3, 4\.

### 1\. PyTorch Model Implementation

The following code implements the **Reduced NCSN++** score model and the **Diffusion Buffer** logic described in the paper.  
import torch  
import torch.nn as nn  
import numpy as np

\# Configuration based on Section 4.1 & 4.2 \[3, 5\]  
CONFIG \= {  
    'N\_freq': 256,       \# Frequency bins (F) for 16kHz audio, 510 window  
    'K\_frames': 128,     \# Chunk size (approx 2s)  
    'hidden\_dim': 96,    \# Reduced from 128  
    'buffer\_size': 20,   \# B: Trade-off parameter (e.g., 20 steps)  
    'sigma\_min': 0.03,   \# t\_epsilon  
    'sigma\_max': 1.0     \# T\_max  
}

class ReducedNCSNpp(nn.Module):  
    """  
    Reduced Noise Conditional Score Network++ as described in \[3\].  
    Reductions: Channels 128-\>96, ResBlocks 2-\>1, Up/Down Blocks 6-\>4.  
    """  
    def \_\_init\_\_(self, config):  
        super().\_\_init\_\_()  
        self.hidden\_dim \= config\['hidden\_dim'\]  
          
        \# 1\. Input Projection (Complex STFT magnitude compressed \-\> Hidden)  
        \# Input is (Batch, 2, F, K) \-\> Real/Imag parts treated as channels  
        self.input\_conv \= nn.Conv2d(2, self.hidden\_dim, kernel\_size=3, padding=1)  
          
        \# 2\. Time Embedding Adapter \[3\]  
        \# Adapts sequence of diffusion steps to channel/frame dimensions  
        self.time\_embed\_layer \= nn.Conv2d(1, self.hidden\_dim, kernel\_size=1)   
          
        \# 3\. Backbone (Simplified U-Net structure)  
        \# Note: A full NCSN++ is complex; this is a structural placeholder   
        \# representing the depth reduction described in the paper.  
        self.enc1 \= self.\_make\_block(self.hidden\_dim, self.hidden\_dim)  
        self.enc2 \= self.\_make\_block(self.hidden\_dim, self.hidden\_dim \* 2\)  
          
        self.dec2 \= self.\_make\_block(self.hidden\_dim \* 2, self.hidden\_dim)  
        self.dec1 \= self.\_make\_block(self.hidden\_dim \* 2, self.hidden\_dim) \# \+ skip  
          
        \# 4\. Output Projection  
        self.output\_conv \= nn.Conv2d(self.hidden\_dim, 2, kernel\_size=3, padding=1)

    def \_make\_block(self, in\_ch, out\_ch):  
        \# Reduced residual blocks from 2 to 1 \[3\]  
        return nn.Sequential(  
            nn.Conv2d(in\_ch, out\_ch, 3, padding=1),  
            nn.GroupNorm(8, out\_ch),  
            nn.SiLU()  
        )

    def forward(self, x, t\_embed):  
        \# x: \[Batch, 2, F, K\]  
        \# t\_embed: \[Batch, 1, 1, K\] \- diffusion time map per frame  
          
        h \= self.input\_conv(x)  
          
        \# Add time embeddings (broadcasting along F)  
        t\_map \= self.time\_embed\_layer(t\_embed)  
        h \= h \+ t\_map   
          
        \# Encoder  
        e1 \= self.enc1(h)  
        e2 \= self.enc2(e1) \# Downsample logic omitted for brevity  
          
        \# Decoder (Simplified)  
        d2 \= self.dec2(e2)  
        d1 \= self.dec1(torch.cat(\[d2, e1\], dim=1))  
          
        out \= self.output\_conv(d1)  
        return out

class DiffusionBuffer(nn.Module):  
    """  
    Implements the Diffusion Buffer logic \[6, 7\].  
    """  
    def \_\_init\_\_(self, score\_model, config):  
        super().\_\_init\_\_()  
        self.score\_model \= score\_model  
        self.B \= config\['buffer\_size'\]  
        self.K \= config\['K\_frames'\]  
        self.F \= config\['N\_freq'\]  
          
        \# Initialize Buffer V with zeros \[8\]  
        self.register\_buffer('V', torch.zeros(1, 2, self.F, self.K))  
          
        \# Pre-calculate noise schedule (Linear or Geometric)  
        \# Corresponds to t\_vec in Eq (5) \[9\]  
        self.t\_schedule \= torch.linspace(config\['sigma\_min'\], config\['sigma\_max'\], self.B)

    def process\_frame(self, new\_frame\_R, noise\_std):  
        """  
        Algorithm 1 from \[8\].  
        new\_frame\_R: \[1, 2, F, 1\] \- The incoming streaming frame  
        """  
        \# 1\. Shift Buffer (Pop oldest)  
        \# V corresponds to V\_vec in Eq (5)  
        self.V \= torch.roll(self.V, shifts=-1, dims=-1)  
          
        \# 2\. Inject noise to the new frame at t\_B \[7\]  
        \# R' \= R \+ sigma\_tB \* Z  
        noise \= torch.randn\_like(new\_frame\_R)  
        perturbed\_R \= new\_frame\_R \+ noise\_std \* noise  
          
        \# 3\. Append to end of buffer  
        self.V\[..., \-1\] \= perturbed\_R.squeeze(-1)  
          
        \# 4\. Construct Time Map for the Score Model  
        \# The buffer contains frames at steps t\_1 ... t\_B.   
        \# We need a tensor map of these times.  
        \# Pad t\_schedule to match K frames (0 for already clean frames outside B)  
        t\_map \= torch.zeros(1, 1, 1, self.K).to(self.V.device)  
        t\_map\[..., \-self.B:\] \= self.t\_schedule.view(1,1,1,-1)  
          
        \# 5\. One-Shot Score Estimation \[7\]  
        \# "Only one s\_theta call"  
        estimated\_score \= self.score\_model(self.V, t\_map)  
          
        \# 6\. Apply Reverse Step (e.g., Euler-Maruyama)  
        \# Update V in place using Eq (4) logic.   
        \# Simplified: V \= V \- step\_size \* score  
        dt \= (self.t\_schedule\[10\] \- self.t\_schedule) \# Uniform step  
          
        \# Update only the active buffer part (last B frames)  
        \# We move from t\_n to t\_{n-1}  
        update\_mask \= torch.zeros\_like(self.V)  
        update\_mask\[..., \-self.B:\] \= 1.0  
          
        self.V \= self.V \+ update\_mask \* (dt \* estimated\_score)  
          
        \# 7\. Output the frame that just finished the buffer (B-th last)  
        \# "S\_hat appends B-th last frame of V" \[7\]  
        clean\_frame \= self.V\[..., \-self.B\].clone()  
          
        return clean\_frame

### 2\. Programmer Documentation

**Subject:** Implementation of Diffusion Buffer Training on DroneNoise-LibriMix  
**Overview**We are implementing the "Diffusion Buffer" model 1 using the "DroneNoise-LibriMix" (DN-LM) dataset described in the Edge-BS-RoFormer paper 11\. This setup targets ultra-low SNR (-30dB to 0dB) drone ego-noise reduction.

#### 1\. Data Preparation (Crucial Adjustment)

The *Edge-BS-RoFormer* paper specifies the DN-LM dataset uses **1-second clips** (16,000 samples) 11\. However, the *Diffusion Buffer* paper requires a context window (chunk size $K$) of **128 frames** 5\.

* *Calculation:* With Window=510 and Hop=256 5, 128 frames $\\approx$ 2.04 seconds.  
* *Action:* You must modify the DN-LM generation script (using LibriSpeech \+ DroneAudioDataset) to generate **3-second clips** instead of 1-second clips.  
* *Preprocessing:*  
* Resample to 16 kHz.  
* STFT Specs: Window 510, Hop 256, Periodic Hann Window 5\.  
* Magnitude Compression: $c\_{compressed} \= 0.15 |c|^{0.5} e^{i\\angle c}$ 5\.

#### 2\. Training Setup

**Objective:** Denoising Score Matching (DSM) on the Diffusion Buffer 12\.

* **Constructing the Buffer for Training:**Unlike standard diffusion where you sample one $t$ for the whole file, here you must simulate the buffer structure:  
* Sample a clean/noisy pair $(X\_0, Y)$ of length $K=128$.  
* Pad with $K-1$ leading zeros (mimicking stream start) 13\.  
* Sample a sequence of times $\\vec{t} \= (t\_1, ..., t\_B)$ where $t\_1 \= \\epsilon \> 0$ 12\.  
* Generate the **Perturbed Input** $V\_{\\vec{t}}$ according to Eq (5) 9:  
* Frames $k \< K-B$: Use clean speech $S$.  
* Frames $k \\ge K-B$: Use $X\_{t\_{g(k)}}$ (noisy mixture diffused to step $t$).  
* **Loss Function:**Minimize the $L\_2$ distance between the model output and the noise target (Eq 7\) 12:$$ \\mathcal{L} \= || s\_\\theta(V\_{\\vec{t}}, Y, \\vec{t}) \+ \\frac{Z}{\\sigma} ||^2\_2 $$  
* *Note:* The model output must be cropped to the last $B$ frames before loss calculation 3\.

#### 3\. Hyperparameters

* **Optimizer:** AdamW, LR $10^{-4}$, Batch Size 32 14\.  
* **EMA:** Use Exponential Moving Average (decay 0.999) on weights 14\.  
* **SDE:** Use the "BBED" (Brownian Bridge with Exponential Diffusion) parameterization as it outperforms others for speech 15\.  
* $f(X\_t, Y) \= \\frac{Y \- X\_t}{1-t}$  
* $g(t) \= c k^t$ (Parameters: $c=0.08, k=2.6$) 15\.

#### 3.1\. Repo Config Mapping

The repo config that mirrors the Diffusion Buffer (BBED) paper setup is:

* `configs/9_Diffusion_Buffer_BBED.yaml`

It captures the explicit paper settings (16 kHz, 510/256 STFT, K=128, Adam, LR 1e-4, batch size 32, EMA 0.999, BBED c/k/0.8).

#### 4\. Edge Deployment Constraints

To match the valid benchmarks in our context:

* The model must fit on a device like the NVIDIA Jetson AGX Xavier.  
* **Target Metrics:** Latency 320–960 ms (dictated by Buffer Size $B$). Power consumption should be monitored to stay \< 30W 1, 4\.

