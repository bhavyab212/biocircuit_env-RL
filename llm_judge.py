"""
llm_judge.py (Groq Optimized)
============
SynBio-RL Project — Meta OpenEnv Hackathon
Task 3: LLM Science Judge — Bhavya

Connects to Groq's Llama 3 API and evaluates the RL agent's final
genetic circuit design as a Senior Professor of Synthetic Biology.
"""

from __future__ import annotations

import os
import re
import time
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import http.client

# ─────────────────────────────────────────────────────────────────────────────
# Configure logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] llm_judge: %(message)s"
)
logger = logging.getLogger("llm_judge")


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONSTANTS - Update for current Groq support
# ─────────────────────────────────────────────────────────────────────────────
_GROQ_BASE = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
LLAMA_API_BASE = _GROQ_BASE.replace("https://", "").split("/")[0]
LLAMA_API_PATH = "/" + "/".join(_GROQ_BASE.replace("https://", "").split("/")[1:])
if not LLAMA_API_PATH.endswith("/chat/completions"):
    LLAMA_API_PATH = LLAMA_API_PATH.rstrip("/") + "/chat/completions"

# Change this from llama-3.3-70b-specdec to the latest 70B model:
LLAMA_MODEL = os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile")
# Retry configuration
MAX_RETRIES      : int   = 3
RETRY_BASE_DELAY : float = 2.0   
API_TIMEOUT      : int   = 30    

# Grade boundaries (from LLM Rubric) [cite: 1064, 1071, 1078, 1080]
LETHAL_ERROR_GRADE     : float = 0.0
INCOMPLETE_GRADE_CAP   : float = 0.5
FLAWED_LOGIC_GRADE_CAP : float = 0.4
BURDEN_GRADE_CAP       : float = 0.8
EXPERT_GRADE_THRESHOLD : float = 0.9


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LLMJudgeConfig — API and Behaviour Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMJudgeConfig:
    # Point this to "API_KEY" so it uses your working Groq key
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", ""))
    temperature: float = 0.1
    max_tokens: int = 800
    task_id: int = 1
    mock_mode: bool = False


@dataclass
class JudgeVerdict:
    """
    The complete evaluation returned by the LLM Science Judge. [cite: 1084]
    """
    science_grade      : float      = 0.0
    mechanism_check    : str        = "Unknown"
    critique           : str        = ""
    lethal_errors      : list[str]  = field(default_factory=list)
    warnings           : list[str]  = field(default_factory=list)
    raw_response       : str        = ""
    final_score        : float      = 0.0
    api_used           : str        = "groq"
    parse_success      : bool       = False


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SYSTEM PROMPT — The Biology Professor Persona [cite: 1062, 1063]
# ─────────────────────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    return """You are a Senior Professor of Synthetic Biology evaluating an AI agent's genetic circuit design.
Your expertise includes:
- Gene regulation (Lac operon, CAP-cAMP activation, LacI repression)
- Cis-acting regulatory elements (Promoters, Operators, Enhancers, Terminators)
- Signal transduction cascades

Your job is to review the DNA sequence and biological explanation and assign a Science Grade (G) from 0.0 to 1.0.

STRICT RULES:
1. Lethal Structural Error (e.g., Promoter downstream of Gene) = G = 0.0 ALWAYS. [cite: 1067]
2. Missing Repressor where required = G capped at 0.4. [cite: 1078]
3. Using Strong Promoter where Weak suffices = G capped at 0.8. [cite: 1080]
4. You MUST respond in the EXACT output format requested."""


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CIRCUIT PROMPT — [cite: 1084]
# ─────────────────────────────────────────────────────────────────────────────

def _build_circuit_prompt(
    circuit_parts    : list[str],
    mechanism_trace  : list[str],
    fluorescence_out : float,
    math_reward      : float,
    task_id          : int,
    task_name        : str,
) -> str:
    parts_formatted = "  →  ".join([f"[{i+1}] {p.upper()}" for i, p in enumerate(circuit_parts)])
    trace_formatted = "\n".join([f"  {line}" for line in mechanism_trace])

    return f"""
=== AGENT CIRCUIT SUBMISSION ===
Task: Level {task_id} — {task_name}
DNA Sequence: {parts_formatted}
Fluorescence: {fluorescence_out:.4f}
Math Reward (R): {math_reward:.4f}

Mechanism Explanation:
{trace_formatted}
"""


# ─────────────────────────────────────────────────────────────────────────────
# 6.  RUBRIC PROMPT — [cite: 1065, 1072, 1081]
# ─────────────────────────────────────────────────────────────────────────────

def _build_rubric_prompt(task_id: int) -> str:
    # Task-specific key checks [cite: 1081-1083]
    task_key_checks = {
        12: "LEVEL 12: Must mention cAMP recruitment of RNA Pol by CAP. [cite: 1081]",
        13: "LEVEL 13: Must mention DNA looping. [cite: 1082]",
        15: "LEVEL 15: Must mention phosphorylation cascade or 2nd messengers. [cite: 1083]",
    }
    active_key_check = task_key_checks.get(task_id, "Standard regulatory evaluation.")

    return f"""
=== EVALUATION RUBRIC ===
- Promoter Placement: Must be upstream (5') of gene. Violation = G=0. [cite: 1067]
- Operator Logic: Must be between Promoter and Gene for steric hindrance. [cite: 1068]
- Transcription Unit: Requires Promoter + Gene + Terminator. [cite: 1070]
- Mechanistic Logic: Check for correct Inducer/Repressor use. [cite: 1076]
- Key Check: {active_key_check}

=== OUTPUT FORMAT ===
Science Grade: [0.0 - 1.0]
Mechanism Check: [Confirmed] OR [Failed]
Critique: [2-3 sentences]
Lethal Errors: [None] OR [List]
"""


def _call_llama_api(messages: list[dict], config: LLMJudgeConfig) -> str:
    host = LLAMA_API_BASE
    path = LLAMA_API_PATH
    key  = config.api_key

    payload = json.dumps({
        "model"       : LLAMA_MODEL,
        "messages"    : messages,
        "temperature" : config.temperature,
        "max_tokens"  : config.max_tokens,
    })
    headers = {
        "Content-Type"  : "application/json",
        "Authorization" : f"Bearer {key}",
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            conn = http.client.HTTPSConnection(host, timeout=API_TIMEOUT)
            conn.request("POST", path, body=payload, headers=headers)
            response     = conn.getresponse()
            response_body = response.read().decode("utf-8")
            conn.close()

            if response.status == 200:
                data = json.loads(response_body)
                return data["choices"][0]["message"]["content"]

            elif response.status in (429, 500, 502, 503):
                wait = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    f"Groq API returned {response.status} on attempt "
                    f"{attempt+1}/{MAX_RETRIES}. Retrying in {wait:.1f}s..."
                )
                time.sleep(wait)
                last_error = RuntimeError(
                    f"Groq API Error {response.status}: {response_body}"
                )

            else:
                raise RuntimeError(
                    f"Groq API Error {response.status}: {response_body}"
                )

        except RuntimeError:
            raise
        except Exception as e:
            last_error = e
            wait = RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning(
                f"Connection error on attempt {attempt+1}/{MAX_RETRIES}: "
                f"{e}. Retrying in {wait:.1f}s..."
            )
            time.sleep(wait)

    raise last_error or RuntimeError("Max retries exceeded with no response.")


def _parse_grade(response_text: str) -> tuple[float, str, str, list[str], bool]:
    grade = 0.0
    parse_success = False
    
    # Improved regex for Groq responses
    grade_match = re.search(r"Science\s+Grade:\s*([0-9]\.[0-9]+|[0-9])", response_text)
    if grade_match:
        grade = float(grade_match.group(1))
        parse_success = True

    mech_match = re.search(r"Mechanism\s+Check:\s*(\w+)", response_text)
    mechanism_check = mech_match.group(1) if mech_match else "Unknown"

    crit_match = re.search(r"Critique:\s*(.*?)(?=Lethal Errors|$)", response_text, re.DOTALL)
    critique = crit_match.group(1).strip() if crit_match else ""

    err_match = re.search(r"Lethal\s+Errors:\s*(.*)", response_text, re.DOTALL)
    lethal_errors = [e.strip("- ") for e in err_match.group(1).split("\n") if e.strip() and "None" not in e] if err_match else []

    return grade, mechanism_check, critique, lethal_errors, parse_success

def _validate_grade(
    grade        : float,
    lethal_errors: list[str],
    critique     : str,
    warnings     : list[str],
) -> float:
    if lethal_errors:
        warnings.append(
            f"Lethal structural error detected → grade forced to {LETHAL_ERROR_GRADE}"
        )
        return LETHAL_ERROR_GRADE

    crit_lower = critique.lower()

    if "silencer" in crit_lower and "repressor" not in crit_lower:
        warnings.append(
            "Silencer used instead of Repressor → grade capped at "
            f"{FLAWED_LOGIC_GRADE_CAP}"
        )
        return min(grade, FLAWED_LOGIC_GRADE_CAP)

    if "strong promoter" in crit_lower and (
        "unnecessary" in crit_lower or "weak" in crit_lower
    ):
        warnings.append(
            "Strong Promoter used where Weak suffices → grade capped at "
            f"{BURDEN_GRADE_CAP}"
        )
        return min(grade, BURDEN_GRADE_CAP)

    return max(0.0, min(grade, 1.0))

def llm_judge(circuit_parts, mechanism_trace, fluorescence_out, math_reward, task_id, task_name, config=None):
    if config is None: config = LLMJudgeConfig()
    if config.mock_mode: return JudgeVerdict(science_grade=0.8, parse_success=True)

    verdict = JudgeVerdict()
    messages = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": _build_circuit_prompt(circuit_parts, mechanism_trace, fluorescence_out, math_reward, task_id, task_name) + _build_rubric_prompt(task_id)}
    ]

    try:
        raw_response = _call_llama_api(messages, config)
        grade, mech, crit, errs, ok = _parse_grade(raw_response)
        grade = _validate_grade(grade, errs, crit, verdict.warnings)
        
        verdict.science_grade = grade
        verdict.mechanism_check = mech
        verdict.critique = crit
        verdict.lethal_errors = errs
        verdict.raw_response = raw_response
        verdict.parse_success = ok
    except Exception as e:
        logger.error(f"Judge failed: {e}")
        verdict.critique = str(e)

    return verdict

def calculate_final_score(math_reward, verdict):
    """Final Score = R × G [cite: 1064, 1121]"""
    final = round(math_reward * verdict.science_grade, 4)
    verdict.final_score = final
    return final