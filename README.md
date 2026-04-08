# 🧬 SynBio-RL — Synthetic Biology Reinforcement Learning Environment
**Meta OpenEnv Hackathon 2026**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Groq](https://img.shields.io/badge/LLM-Llama%203.3%2070B-orange)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## What Is This?
An LLM-guided agent that assembles synthetic genetic circuits to solve 15 progressive molecular biology challenges. The agent is scored by a dual system: **Mathematical Reward (R)** × **LLM Science Grade (G)**.

## Quick Start
```bash
git clone https://github.com/bhavyab212/biocircuit_env-RL.git
cd biocircuit_env-RL
pip install python-dotenv openai
echo "API_KEY=your_groq_key" > .env
echo "API_BASE_URL=https://api.groq.com/openai/v1" >> .env
echo "MODEL_NAME=llama-3.3-70b-versatile" >> .env
python3 inference.py
```

## Architecture
| File | Role |
|---|---|
| `inference.py` | LLM Agent — calls Groq API, places DNA parts |
| `environment.py` | BioCircuitEnv — Gym-compatible RL environment |
| `reporter_logic.py` | Fluorescence physics + Math Reward R |
| `dense_rewards.py` | Per-step biological dense reward shaping |
| `llm_judge.py` | Professor LLM — Science Grade G (0.0–1.0) |
| `tasks.json` | 15 challenge definitions |
| `dashboard.html` | Real-time neumorphic web dashboard |

## Final Score Formula
**Score = R × G** where R = 0.50×Accuracy + 0.30×Complexity − 0.20×Burden (×10), G = LLM Science Grade

## The 15 Tasks
From basic promoter-reporter-terminator assembly (Task 1) to multi-step phosphorylation cascades with CAP activation, enhancer DNA looping, and LacI repressor logic (Task 15).

## Built With
- **Groq API** — Llama 3.3 70B Versatile (agent + judge)
- **Python 3.10** — no heavy ML framework needed for inference
- **Chart.js + Neumorphic CSS** — dashboard visualisation