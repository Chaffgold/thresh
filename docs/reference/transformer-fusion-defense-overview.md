# Transformer fusion for multi-object tracking: what works, what doesn't, and why defense is different

**A single transformer-based ONNX model can theoretically track heterogeneous objects like fighter jets and ballistic missiles, but no production system does this today — and for good reason.** The autonomous driving community has proven that unified multi-class tracking works across diverse object types (pedestrians through trucks), with architectures like TransFusion and BEVFusion dominating benchmarks. However, the leap from automotive to aerospace tracking faces three hard barriers: no public training data exists for defense-domain targets, the dynamics span orders of magnitude more than any existing dataset covers, and defense systems demand certifiable uncertainty quantification that transformers cannot yet provide. The most promising path forward is hybrid architectures — transformer-based detection and data association feeding into classical Bayesian filters for state estimation — deployed as modular pipeline components rather than monolithic models.

## One model or many: the unified vs. specialist debate

The question of whether a single ONNX model can track objects as different as an F-16 (Mach 2, RCS ~1 m²), a 747 (250 m/s, RCS ~100 m²), and a ballistic missile (Mach 20+, changing RCS profile) cuts to the core of multi-task learning theory. The evidence strongly favors unified models with architectural accommodations, not pure parameter sharing or fully separate models.

Wang et al. (CVPR 2019) demonstrated this with the Universal Object Detection Benchmark spanning 11 wildly different domains — from face detection to aerial imagery to medical lesions. Their universal detector with **domain attention adapters (adding only ~7% parameters per domain) outperformed a bank of individually trained specialist models** by 1.6 mAP while using 10x fewer total parameters. CenterPoint (Yin et al., CVPR 2021) already tracks 10 object classes simultaneously on nuScenes — from traffic cones to articulated buses — using class-specific Gaussian heatmaps in a single forward pass. BEVFusion (Liu et al., ICRA 2023) goes further, performing both 3D detection and BEV map segmentation in a single architecture described by the authors as "fundamentally task-agnostic."

The arguments against pure unification are real but addressable. **Gradient conflict** — where optimizing for one object class degrades another — is well-documented. Yu et al. (NeurIPS 2020) formalized this with PCGrad (gradient surgery), and CAGrad (IJCAI 2024) demonstrated 2.2% mAP improvement in object detection by harmonizing conflicting gradients. The emerging solution is **Mixture of Experts (MoE)**: UETrack (arXiv, 2026) uses Token-Parallel MoE modules that route different modalities through specialized expert networks while sharing a common backbone. DaSSP-Net (Pattern Recognition, 2025) confirms that "co-training a single model across all domains focuses solely on domain-shared information, ignoring domain-specific information" — but lightweight domain adapters resolve this without requiring separate models.

For defense tracking of heterogeneous targets, the practical recommendation is a **shared backbone with class-specific heads**: one network extracts features from sensor data, with separate prediction heads for different target dynamics (e.g., ballistic vs. aerodynamic vs. orbital). This avoids the deployment complexity of multiple models while accommodating fundamentally different motion physics.

## Architectures that define the state of the art

Three architectural paradigms dominate transformer-based multi-sensor fusion tracking, each solving different pieces of the problem.

**Query-based cross-attention fusion** is the leading approach for combining sensor modalities. TransFusion (Bai et al., CVPR 2022) introduced a two-layer transformer decoder where the first layer generates object proposals from LiDAR BEV features, then the second layer uses those proposals as queries attending to camera image features via cross-attention. This **"soft-association"** mechanism is the critical innovation: rather than hard-projecting LiDAR points onto image pixels using calibration matrices (which fails under sensor misalignment or poor lighting), the transformer adaptively learns where to look in camera images. TransFusion achieved **1st place on the nuScenes tracking leaderboard** with 68.9% mAP.

**BEV-space unification** takes a different approach. BEVFusion (Liu et al., MIT, ICRA 2023) projects all sensor modalities into a shared Bird's-Eye View representation before fusion. Camera images are transformed to BEV via Lift-Splat-Shoot; LiDAR is naturally BEV-compatible. The key engineering contribution was **optimized BEV pooling** that reduced camera-to-BEV latency from ~500ms to ~12ms (a 40x speedup). BEVFusion exceeded TransFusion by +1.3% mAP with **1.9x lower computation**, and demonstrated +10.7 mAP improvement over LiDAR-only methods in rainy conditions.

**Track queries for temporal association** represent the end-to-end tracking paradigm. TrackFormer (Meinhardt et al., CVPR 2022), MOTR (Zeng et al., ECCV 2022), and TransTrack (Sun et al., 2020) all extend DETR by introducing persistent decoder queries that carry object identity across frames. MOTR's track queries model entire object trajectories, with a Temporal Aggregation Network for temporal reasoning and a Query Interaction Module managing track births and deaths. On DanceTrack — a benchmark with uniform appearance and diverse motion — MOTR outperformed ByteTrack by **6.5% HOTA**.

For defense applications specifically, the literature is growing but less mature. DeepAF (MDPI Aerospace, 2025) constructs a transformer-based data association and filtering network for radar multi-target tracking using **negative Mahalanobis distance** as the attention similarity metric. The TrMTT model (Information Fusion, 2023) applies encoder-decoder transformers to maneuvering target tracking, learning state transition laws without requiring predefined motion model banks. A certifiable UAM sensor fusion system (Preprints.org, 2025) uses a Perceiver IO-inspired transformer fusing **six sensor types** (LiDAR, EO/IR, GNSS, ADS-B, IMU, radar) with cross-attention, designed for DO-178C certification. Most notably, a theoretical proof by arXiv:2312.06937 (2023) established that **a causally-masked transformer can exactly represent a Kalman Filter** — providing the mathematical foundation for replacing classical state estimators with learned alternatives.

## Training: the data, losses, and tricks that make it work

The training methodology rests on three established datasets and a well-characterized loss function toolkit — but neither extends easily to aerospace domains.

**nuScenes** (Caesar et al., CVPR 2020) provides the gold standard: 1,000 scenes with 6 cameras, 1 LiDAR, 5 radars, GPS, and IMU across 23 object classes with **1.4 million 3D bounding boxes**. The Waymo Open Dataset (Sun et al., CVPR 2020) offers larger scale — 12 million 3D labels at 10 Hz — but lacks radar. **No public aerospace defense tracking dataset exists at comparable scale.** Defense research relies on simulated radar returns, small proprietary datasets, and ADS-B-labeled passive radar data from academic studies. This data gap is the single largest barrier to applying transformer tracking to defense.

The standard loss function combines: **focal loss** for classification, **L1 or smooth-L1 regression** for position/size/orientation/velocity, **Generalized IoU loss** for scale-invariant bounding box evaluation, and **Hungarian matching** for set prediction.

Handling objects with dramatically different dynamics uses: **Class-Balanced Grouping and Sampling (CBGS)** and **Interacting Multiple Model (IMM)** approaches. For the extreme velocity ranges in aerospace (hovering UAV to Mach 20 missile), no existing deep learning system has been demonstrated — this would require synthetic data generation at scale.

## ONNX export: practical but not as a single file

**No published system deploys a single ONNX file for multi-class multi-sensor tracking.**

Core challenges: **SDPA tracing fails** when attention masks are absent; **dynamic shapes** require explicit `dynamic_axes` specification; TensorRT conversion fails without optimization profiles. RT-DETR demonstrates real-time transformer inference: **217 FPS on a T4 GPU**. Multi-sensor fusion systems are deployed as modular pipelines with **multiple ONNX model chunks** converted to separate TensorRT engines.

For defense deployment, the modular approach has additional advantages: each component can be independently validated, updated, and certified. Edge deployment on SWaP-constrained platforms achieves meaningful performance through aggressive optimization: structured pruning + mixed-precision quantization can reduce inference latency by **80% and halve memory usage**.

## Where transformers win and where Kalman filters still reign

On nuScenes, current state of the art reaches **77.9% AMOTA**. Traditional methods maintain decisive advantages in speed, certifiability, and data efficiency. **SORT runs at 260 Hz** versus 8-11 FPS for transformer trackers — a 25-30x gap. Kalman filters need no training data, work from first principles, and provide mathematically characterized covariance estimates.

**No published peer-reviewed paper documents operational deployment** of transformer-based trackers in actual air defense or missile tracking systems.

## Conclusion

Three key insights: First, **unified models with class-specific adapters are architecturally superior**. Second, **hybrid architectures represent the pragmatic frontier**: transformer detection and attention-based data association feeding classical Bayesian filters for state estimation. Third, **the data problem is the binding constraint**, not the architecture. Any team pursuing this should plan to deploy modular ONNX/TensorRT pipeline components rather than a single monolithic model, build a synthetic data pipeline first, and retain Bayesian filtering for state estimation until transformer uncertainty quantification matures.
