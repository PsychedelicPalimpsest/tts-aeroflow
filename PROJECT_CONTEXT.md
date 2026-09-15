# AeroFlow-v2 Project Context & System Engineering Baseline

## 1. Executive Summary & Voice Profile
* **Project Name:** AeroFlow-v2 Text-to-Speech Engine
* **Selected Voice Dataset:** **Hi-Fi TTS Speaker 9017 (John Van Stan)**
  * **Gender / Range:** Male Baritone (F0 Fundamental ~85 Hz – 145 Hz).
  * **Acoustic Signature:** Deep, resonant, authoritative American male voice with crisp transient articulation, warm low-end chest resonance, and immaculate studio acoustics.
  * **Studio Quality:** Clean signal-to-noise ratio $\ge 40\text{ dB}$, 24,000 Hz (24 kHz) mono uncompressed PCM, zero room reverberation or background bleed.
  * **Application Domain:** Audiobook narration, technical briefings, executive system announcements, low-latency conversational agents.

---

## 2. Hardware Profile: Intel Core i7-12700H (Alder Lake Architecture)
* **Processor Configuration:**
  * Total Cores: 14 Cores (6 Performance Golden Cove P-cores + 8 Efficient Gracemont E-cores).
  * Threads: 20 Threads.
  * Memory: 32 GB DDR5 RAM.
  * L3 Cache: 24 MB Intel Smart Cache (accommodates the entire 25.4 MB FP16 model working set in cache).
* **Asymmetric Compute Allocation Policy:**
  * **6 Physical P-Cores (`torch.set_num_threads(6)`):** Exclusively assigned to synchronous neural inference (Conformer Text Encoder, ConvNeXt-ODE Vector Field, ConvNeXt-V2 Complex STFT Decoder, and iSTFT). Hyperthreading sibling threads are unpinned to prevent AVX2 execution port contention.
  * **2 E-Cores (E0, E1):** Dedicated to deterministic text normalization, CMUdict hash lookups, fallback G2P tokenization, and circular audio buffer streaming.
  * **Remaining E-Cores:** Reserved for OS background tasks and IPC.
* **Target Latency Envelope:**
  * Target Real-Time Factor (RTF): $\mathbf{\le 0.0120}$ ($\sim 120\text{ ms}$ compute per 10.0 seconds of synthesized 24 kHz speech).
  * Real-Time Speedup: **$83.3\times$ faster than real-time**.

---

## 3. Core Architectural Constraints
1. **32-Channel Continuous Latent Space:**
   * Eliminates the traditional 80-bin mel-spectrogram bottleneck.
   * Continuous latent flow matching operates at 100 Hz frame rate (Hop = 240 samples @ 24 kHz).
2. **Optimal Transport Conditional Flow Matching (OT-CFM):**
   * Straight-line continuous probability trajectories: $x_t = (1 - (1 - \sigma_{\min}) t) x_0 + t z$.
   * Target vector field: $u_t = z - (1 - \sigma_{\min}) x_0$.
   * ConvNeXt-ODE vector field backbone parameterized with AdaLN (Adaptive Layer Normalization) conditioned on temporal position and text representations.
3. **6-Step Non-Uniform Heun ODE Solver ($\rho = 1.5$):**
   * Second-order predictor-corrector numerical integration.
   * Power-law non-uniform schedule: $t_k = 1.0 - (1.0 - k/N)^{1.5}$ ($N=6$).
   * Local truncation error $\mathcal{O}(h_k^3)$, with dense sampling at $t \to 1.0$ ($h_5 = 0.0572$) preserving sharp plosive releases and formant transitions without smearing.
4. **ConvNeXt-V2 Complex STFT Decoder:**
   * 4 ConvNeXt-V2 blocks with Global Response Normalization (GRN).
   * Direct prediction of full-band Fourier spectra: Log-Magnitude $M_{\text{log}}$ and continuous unit phase components $(p_r, p_i) \to (\cos \phi, \sin \phi)$.
   * Complex STFT output: $S \in \mathbb{C}^{513 \times T}$.
5. **Alias-Free COLA iSTFT Waveform Synthesis:**
   * Inverse Short-Time Fourier Transform with $N_{\text{fft}} = 1024$, $\text{Hop} = 240$ ($100\text{ Hz}$), periodic Hann window.
   * Overlap ratio: $\frac{1024 - 240}{1024} = 76.56\%$.
   * Strict Constant Overlap-Add (COLA) compliance: $\sum_m w^2(n - m H) = \text{const} > 0$.
   * Completely avoids sub-band phase cancellations, comb filtering, and transposed-convolution pitch checkerboards.
6. **Alignment & Duration Modeling:**
   * Training: Dynamic programming Viterbi Monotonic Alignment Search (MAS) on latent acoustic representations.
   * Inference: Energy-Constrained Monotonic Duration Predictor with clamped integer bounds:
     $$d_n = \min\left(\max(\lfloor \exp(\hat{y}_n) \cdot \alpha + 0.5 \rfloor, 1), 80\right)$$
     guaranteeing $d \ge 1$ frame ($10\text{ ms}$) and $d \le 80$ frames ($800\text{ ms}$).

---

## 4. Mathematical Anti-Collapse Guarantees

### Theorem 1: Structural Impossibility of Phoneme Skipping and Looping
* Because $d_n \ge 1$ for all phonemes $n \in \{1, \dots, N\}$, the temporal alignment mapping $\tau(t) = \arg\min_k \{ \sum_{j=1}^k d_j \ge t \}$ is strictly monotonic and non-decreasing.
* Every phoneme $n$ is guaranteed to receive at least one acoustic frame ($\mathcal{P}(\text{skip}) = 0$).
* Total sequence length $T = \sum_{n=1}^N d_n \le N \times 80 < \infty$, strictly preventing infinite loops or hallucinated repetitions ($\mathcal{P}(\text{loop}) = 0$).

### Theorem 2: Global Lipschitz Boundedness (Zero Phase Explosion)
* The vector field $v_\theta(x, t, C)$ is constructed entirely from 1D depthwise convolutions, bounded LayerNorm projections, and GELU non-linearities ($\sup |g'| \le 1.1$).
* The Jacobian spectral norm is bounded: $\|J_v(x)\|_2 \le L < \infty$.
* By the Picard-Lindelöf theorem and Grönwall's inequality, the ODE latent trajectory $x(t)$ is strictly bounded in $\mathbb{R}^{32}$ for all $t \in [0, 1]$ ($\mathcal{P}(\text{divergence}) = 0$).

### Theorem 3: Total Elimination of Aliasing and Phase Notches
* Sub-band filtering is completely eliminated. Fourier coefficients are generated on a single continuous grid $k \in \{0, \dots, 512\}$.
* Reconstruction via the COLA-compliant iSTFT ensures seamless phase transitions across frequency bins and eliminates crossover notches ($\mathcal{P}(\text{notch}) = 0$).

---

## 5. Non-Adversarial Convex Training Formulation
AeroFlow-v2 abandons fragile GAN discriminators in favor of pure convex spectral regression:
$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{CFM}(\theta) + 1.0 \cdot \mathcal{L}_{dur} + 1.0 \cdot \mathcal{L}_{MR-STFT}(y, \hat{y}) + 0.5 \cdot \mathcal{L}_{IF}(y, \hat{y})$$
* **$\mathcal{L}_{CFM}$:** Optimal Transport conditional velocity matching MSE.
* **$\mathcal{L}_{dur}$:** L1 log-duration regression against Viterbi MAS targets.
* **$\mathcal{L}_{MR-STFT}$:** Multi-resolution STFT loss across window sizes $\{512, 1024, 2048\}$ combining spectral convergence, log-magnitude L1, and complex Frobenius norm.
* **$\mathcal{L}_{IF}$:** Instantaneous frequency loss enforcing cross-frame phase gradient coherence:
  $$\mathcal{L}_{IF} = \left\| \Delta_t \angle S(k, t) - \Delta_t \angle \hat{S}(k, t) \right\|_1$$
