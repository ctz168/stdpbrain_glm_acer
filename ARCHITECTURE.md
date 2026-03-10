# Neuromorphic AI Architecture: STDP-Brain Qwen3.5-0.8B

This document outlines the production-grade architectural design for the "STDP-Brain" neuromorphic AI system, strictly adhering to biological principles and O(1) computational constraints.

## 1. Core Principles
- **底座唯一 (Single Base)**: Strictly Qwen3.5-0.8B.
- **权重双轨 (Dual-Track Weights)**: 90% Static (Frozen) / 10% STDP Dynamic.
- **原生 O(1) 注意力 (O(1) Attention)**: Forced local windowing with KV-Cache truncation.
- **100Hz 刷新期 (100Hz Refresh)**: 10ms execution cycles for sensing, reasoning, and learning.

## 2. Modules
1. **TrulyIntegratedEngine**: The central controller managing the load, token generation, and coordination of neuromorphic subsystems.
2. **RefreshEngine**: Handles the 10ms execution loop, ensuring all steps (Attention, STDP update, Memory encoding) occur within the cycle.
3. **STDP System**: Implements Spike-Timing-Dependent Plasticity for online weights updating (LTP/LTD) without backpropagation.
4. **Hippocampus**: A biological memory system using EC (encoding), DG (pattern separation), and CA3 (storage) to guide attention with O(1) anchors.
5. **Self-Optimization**: Implements in-model self-generation, self-play (proposal-verification), and self-judgment.

## 3. Key Achievements
- **Mathematically Grounded Reasoning**: Fixed "zero rent" hallucinations via complex CoT grounding.
- **Telegram Stability**: Fixed Flood Control issues with time-based rate limiting (1.5s interval).
- **O(1) Hardened**: Successfully sliced `Qwen3_5DynamicCache` to a fixed 128-token window, ensuring compute time does not grow with sequence length.

---
*Date: 2026-03-10*
*Version: 2.0 (Production Stable)*
