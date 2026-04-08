---
title: SynBio-RL OpenEnv
emoji: 🧬
colorFrom: red
colorTo: teal
sdk: docker
pinned: true
tags:
  - openenv
---

# SynBio-RL — Synthetic Biology Reinforcement Learning Environment
### Meta OpenEnv Hackathon 2026 — by Bhavya

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Groq](https://img.shields.io/badge/API-Groq-orange)
![Llama 3.3](https://img.shields.io/badge/Model-Llama%203.3%2070B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Hackathon](https://img.shields.io/badge/Hackathon-2026-red)

---

## 🧬 What Is This?
**SynBio-RL** is a custom Reinforcement Learning environment where an LLM-guided agent assembles synthetic genetic circuits to solve 15 progressively complex molecular biology challenges. 

The agent places DNA parts one by one — **promoters, operators, reporter genes, repressors, CAP binding sites, enhancers, and terminators** — in the correct biological order to achieve a target fluorescence output. Every placement is scored with immediate biological feedback (**dense rewards**), and the final circuit is evaluated by a **Professor LLM judge** who grades the biological reasoning from 0.0 to 1.0. 

> **Final Score = Math Reward R × Science Grade G**

---

## 🚀 Quick Start
### Installation
```bash
git clone https://github.com/bhavyab212/biocircuit_env-RL.git
cd biocircuit_env-RL
pip install python-dotenv openai
```

### Configuration
Create a `.env` file in the root directory:
```env
API_KEY=your_groq_key_here
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.3-70b-versatile
JUDGE_MODEL=llama-3.3-70b-versatile
```

### Run Evaluation
```bash
python3 inference.py
```

---

## 🏗️ Project Architecture

| File | Role | Description |
| :--- | :--- | :--- |
| `inference.py` | **Agent Brain** | LLM calls Groq API step by step, places DNA parts, manages 15-task evaluation loop |
| `environment.py` | **RL Environment** | BioCircuitEnv with `reset()` and `step()` — Gym-compatible interface |
| `reporter_logic.py` | **Physics Engine** | Calculates fluorescence output using transcription rate formula + Math Reward R |
| `dense_rewards.py` | **Reward Shaper** | Biological feedback after every single part placement — solves sparse reward problem |
| `llm_judge.py` | **Science Judge** | Professor Llama 3.3 evaluates completed circuit, assigns Science Grade G (0.0–1.0) |
| `tasks.json` | **Task Library** | 15 challenge definitions with available parts, targets, hints |
| `dashboard.html` | **Web UI** | Real-time neumorphic dashboard — open in browser, enter API key, click Run |

---

## 📊 How The Scoring Works

### Math Reward (R) Formula
$$R = ((0.50 \times Accuracy) + (0.30 \times Complexity) - (0.20 \times Burden)) \times 10.0$$

**Where:**
- **Accuracy** = $1.0 - |fluorescence\_output - target\_fluorescence|$
- **Complexity** = $1.0 - (parts\_used - optimal\_parts) / optimal\_parts$
- **Burden** = $total\_metabolic\_cost / max\_possible\_cost$

### Science Grade (G)
Assigned by **Professor LLM (Llama 3.3 70B)** based on:
- **Structural integrity**: Promoter must be 5' of gene — violation = $G=0.0$ lethal error.
- **Mechanistic logic**: Correct explanation of CAP/cAMP, LacI repression, DNA looping.
- **Task-specific key checks**: 
    - *Level 12*: cAMP mention
    - *Level 13*: DNA looping
    - *Level 15*: Phosphorylation cascade
- **Metabolic burden**: Strong promoter where weak suffices = $G$ capped at 0.8.

> **Final Score = R × G**

---

## 🎯 The 15 Tasks

| Task | Name | Target | Key Biology | Difficulty |
| :--- | :--- | :--- | :--- | :--- |
| 01 | Basic Initiation | 1.0 | Strong promoter drives reporter gene expression | ⭐ |
| 02 | The Biological Brake | 0.0 | LacI repressor binds operator — steric hindrance blocks RNA Pol | ⭐⭐ |
| 03 | The Inducible Switch | 1.0 | Inducer binds repressor → conformational change → gene ON | ⭐⭐ |
| 04 | Signal Boosting | 0.7 | CAP protein + cAMP recruits RNA Pol — 30% transcription boost | ⭐⭐ |
| 05 | Comparing Strengths | 1.0 | Strong vs weak promoter Vmax comparison | ⭐ |
| 06 | Distant Control | 0.75 | Enhancer DNA looping brings activators to promoter | ⭐⭐⭐ |
| 07 | The Signal Relay | 1.0 | Structural gene drives reporter in polycistronic arrangement | ⭐⭐ |
| 08 | The Silencer | 0.0 | Repressor blocks RNA Pol progression at operator | ⭐⭐ |
| 09 | Drug Sensitivity | 0.5 | Operator titration — minimum drug dose for partial repression | ⭐⭐ |
| 10 | Feedback Loop | 0.5 | Repressor drives its own negative feedback OFF switch | ⭐⭐⭐ |
| 11 | Combinatorial AND Gate | 1.0 | CAP + mediator complex — both signals required for expression | ⭐⭐⭐ |
| 12 | Metabolic Switching | 1.0 | CAP-Lac dual regulation (glucose starvation logic) | ⭐⭐⭐⭐ |
| 13 | Enhancer Loop | 0.75 | Distal enhancer DNA looping mechanism | ⭐⭐⭐ |
| 14 | Intracellular Sensor | 1.0 | Medium promoter drives structural + reporter in sensor circuit | ⭐⭐ |
| 15 | Relay Amplification | 1.0 | Multi-step relay with enhancer + repressor + CAP cascade | ⭐⭐⭐⭐ |

---

## 🧬 DNA Parts Library

| Part | Biological Role | Metabolic Cost | Reward Impact |
| :--- | :--- | :--- | :--- |
| `strong_promoter` | Maximum RNA Pol affinity (Vmax 1.0) | 0.90 | +1.2 step reward |
| `medium_promoter` | Medium RNA Pol affinity (Vmax 0.5) | 0.50 | +1.0 step reward |
| `weak_promoter` | Low RNA Pol affinity (Vmax 0.2) | 0.20 | +0.8 step reward |
| `operator` | Cis-acting repressor binding site | 0.00 | +0.5 if between P and G |
| `reporter_gene` | GFP gene producing fluorescence | 0.40 | +0.5 if after promoter |
| `structural_gene` | Protein-coding relay gene | 0.40 | +0.5 if after promoter |
| `repressor_gene` | Produces LacI (trans-acting) | 0.70 | +0.5 if operator present |
| `cap_binding_site` | CAP+cAMP recruits RNA Pol (+30%) | 0.10 | +0.5 if cAMP present |
| `enhancer` | DNA looping boost (+25%) | 0.15 | +0.4 if with promoter |
| `terminator` | Ejects RNA Pol | 0.00 | +1.5 if after gene |

---

## 🧠 Dense Reward System (Biological Feedback)
The dense reward system provides immediate feedback after every single DNA part placement — **solving the Sparse Reward Problem**. 

### Key Rules Enforced:
- **RNA Pol initiation**: First part MUST be a promoter — violation = `-5.0` lethal, episode terminates.
- **Directionality**: Operator must be between promoter and gene — placed after gene = `-2.0`.
- **Structural Integrity**: Terminator placed before gene = `-4.0` lethal, episode terminates.
- **Regulatory Cohesion**: 
    - CAP Binding Site without cAMP = `-0.4` warning.
    - Repressor Gene without Operator = `-0.5` warning.
- **Completion**: Correct terminator placement after gene = `+1.5`, episode complete.

---

## 🛠️ Tech Stack

| Layer | Technology | Notes |
| :--- | :--- | :--- |
| **LLM Agent** | Llama 3.3 70B Versatile | Via Groq API free tier |
| **LLM Judge** | Llama 3.3 70B Versatile | Professor persona prompt |
| **API Provider** | Groq | 100,000 TPD / 30 RPM free tier |
| **Language** | Python 3.10+ | No heavy ML framework needed |
| **Dependencies** | `python-dotenv`, `openai` | `pip install python-dotenv openai` |
| **Dashboard** | HTML + Chart.js + Vanilla JS | Single file, no build tools |
| **UI Design** | Neumorphism | Soft shadows, #E8EBF0 background |
| **Rate Limiting** | Retry with backoff | Handles 429 errors automatically |

---

## ❓ Why LLM-as-Agent (Not Classic RL)?
This project deliberately uses an LLM inference agent rather than a trainable neural network. The environment and reward system are fully designed to support real PPO training (**BioCircuitEnv is Gym-compatible**). 

The LLM-as-Agent approach is a valid modern paradigm used in research at DeepMind, OpenAI, and Google — it tests the reasoning capability of large language models against a biologically grounded reward environment. 

> [!NOTE]
> **Round 2** will introduce a trainable PPO policy network using PyTorch + Stable-Baselines3, allowing the agent to genuinely improve over thousands of training episodes.

---

## 🗺️ Future Roadmap
- [ ] **Round 2: PPO Policy Network** — PyTorch neural net trained over 10,000+ episodes.
- [ ] **Full Gym Wrapper** — `observation_space` + `action_space` for Stable-Baselines3.
- [ ] **Multi-agent competition** — Two agents designing circuits with opposing targets.
- [ ] **Wet lab export** — Output circuits in SBOL format for DNA synthesis.
- [ ] **Drug discovery extension** — Protein binding prediction as reward signal.

---

## 📂 Repository Structure
```text
biocircuit_env-RL/
├── environment.py        # BioCircuitEnv (Gym-compatible)
├── reporter_logic.py     # Fluorescence physics + reward math
├── dense_rewards.py      # Per-step biological feedback
├── inference.py          # Main agent + evaluation loop
├── llm_judge.py          # LLM Science Judge (Professor persona)
├── tasks.json            # 15 task definitions
├── dashboard.html        # Web visualization dashboard
├── .env                  # API keys (not committed)
├── .gitignore            # Excludes .env and __pycache__
└── README.md             # This file
```

---

## 📝 License
MIT License — free to use, modify, and distribute.

---
**SynBio-RL** · Meta OpenEnv Hackathon 2026 · [github.com/bhavyab212/biocircuit_env-RL](https://github.com/bhavyab212/biocircuit_env-RL)