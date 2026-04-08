"""
reporter_logic.py
=================
SynBio-RL Project — Meta OpenEnv Hackathon
Task: Reporter System Logic — Bhavya (Corrected for Inducer Logic)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import math

# 1. ENUMERATIONS
class PartType(str, Enum):
    STRONG_PROMOTER   = "strong_promoter"
    MEDIUM_PROMOTER   = "medium_promoter"
    WEAK_PROMOTER     = "weak_promoter"
    OPERATOR           = "operator"
    REPORTER_GENE     = "reporter_gene"
    STRUCTURAL_GENE   = "structural_gene"
    REPRESSOR_GENE    = "repressor_gene"
    CAP_BINDING_SITE  = "cap_binding_site"
    TERMINATOR        = "terminator"
    ENHANCER          = "enhancer"
    INDUCIBLE_PROMOTER = "inducible_promoter"

# 2. BIOLOGICAL CONSTANTS
PROMOTER_TYPES = {PartType.STRONG_PROMOTER, PartType.MEDIUM_PROMOTER, PartType.WEAK_PROMOTER, PartType.INDUCIBLE_PROMOTER}
GENE_TYPES = {PartType.REPORTER_GENE, PartType.STRUCTURAL_GENE}

PROMOTER_STRENGTH = {
    PartType.STRONG_PROMOTER: 1.0,
    PartType.MEDIUM_PROMOTER: 0.5,
    PartType.WEAK_PROMOTER:   0.2,
    PartType.INDUCIBLE_PROMOTER: 0.8,
}

METABOLIC_COST = {
    PartType.STRONG_PROMOTER: 0.90, PartType.MEDIUM_PROMOTER: 0.50, PartType.WEAK_PROMOTER: 0.20,
    PartType.REPORTER_GENE: 0.40,   PartType.REPRESSOR_GENE: 0.70,  PartType.TERMINATOR: 0.00,
    PartType.OPERATOR: 0.00,        PartType.CAP_BINDING_SITE: 0.10, PartType.ENHANCER: 0.15,
    PartType.INDUCIBLE_PROMOTER: 0.60,
}

LEAKY_EXPRESSION_FLOOR = 0.05
W_A, W_C, W_B = 0.50, 0.30, 0.20 # Reward weights

@dataclass
class DNAPart:
    part_type       : PartType
    slot_index      : int
    inducer_present : bool = False # Corrected: Key for Inducible Switch
    cAMP_present    : bool = False

@dataclass
class ReporterResult:
    fluorescence_output : float = 0.0
    math_reward         : float = 0.0
    metabolic_burden    : float = 0.0
    complexity_score    : float = 0.0
    lethal_error        : bool  = False
    lethal_error_reason : Optional[str] = None
    is_expressing       : bool = False
    warnings            : list[str] = field(default_factory=list)
    mechanism_trace     : list[str] = field(default_factory=list)

# 3. CORE CALCULATION LOGIC
def _check_lethal_errors(sequence, result):
    # Rule: Promoter must be 5' (upstream) of Reporter
    promoter = next((p for p in sequence if p.part_type in PROMOTER_TYPES), None)
    reporter = next((p for p in sequence if p.part_type == PartType.REPORTER_GENE), None)
    terminator = next((p for p in sequence if p.part_type == PartType.TERMINATOR), None)

    if not promoter or not reporter or not terminator:
        result.lethal_error_reason = "Incomplete transcription unit (Missing P, G, or T)."
        return True
    if promoter.slot_index > reporter.slot_index:
        result.lethal_error_reason = "Promoter downstream (3') of gene."
        return True
    if terminator.slot_index < reporter.slot_index:
        result.lethal_error_reason = "Terminator upstream of gene."
        return True
    return False

def calculate_reporter_logic(dna_sequence, target_fluorescence=1.0, target_part_count=4):
    # Sort for 5' to 3' directionality
    sequence = sorted(dna_sequence, key=lambda p: p.slot_index)
    result = ReporterResult()
    
    # 1. Check Lethal
    if _check_lethal_errors(sequence, result):
        result.lethal_error = True
        return result

    # 2. Metabolic Burden
    raw_cost = sum(METABOLIC_COST.get(p.part_type, 0.0) for p in sequence)
    max_cost = sum(METABOLIC_COST.values())
    result.metabolic_burden = min(raw_cost / max_cost, 1.0)

    # 3. Transcription Rate (Inducible Logic)
    promoter = next(p for p in sequence if p.part_type in PROMOTER_TYPES)
    reporter = next((p for p in sequence if p.part_type == PartType.REPORTER_GENE), None)
    basal_rate = PROMOTER_STRENGTH[promoter.part_type]
    
    transcription_rate = basal_rate
    
    operator_part = next((p for p in sequence if p.part_type == PartType.OPERATOR), None)
    has_repressor = any(p.part_type == PartType.REPRESSOR_GENE for p in sequence)

    if operator_part is not None and has_repressor:
        promoter_slot  = promoter.slot_index
        reporter_slot  = reporter.slot_index
        operator_slot  = operator_part.slot_index
        if promoter_slot < operator_slot < reporter_slot:
            if not operator_part.inducer_present:
                # Repressor IS bound — steric hindrance
                transcription_rate = LEAKY_EXPRESSION_FLOOR
                result.mechanism_trace.append("Repressor bound to Operator: Steric Hindrance active. Gene OFF.")
            else:
                # Inducer present → conformational change → repressor detaches
                transcription_rate = basal_rate
                result.mechanism_trace.append("Inducer present: conformational change released repressor. Gene ON.")

    # CAP/cAMP Activation
    cap_part = next((p for p in sequence if p.part_type == PartType.CAP_BINDING_SITE), None)
    if cap_part is not None:
        if cap_part.cAMP_present:
            transcription_rate = min(1.0, transcription_rate + 0.30)
            result.mechanism_trace.append(f"CAP/cAMP active: RNA Pol recruited. Transcription boosted +0.30 → {transcription_rate:.2f}")
        else:
            result.warnings.append("CAP Binding Site present but cAMP absent. No activation boost.")

    # Enhancer boost (DNA looping)
    if any(p.part_type == PartType.ENHANCER for p in sequence):
        transcription_rate = min(1.0, transcription_rate + 0.25)
        result.mechanism_trace.append(f"Enhancer: DNA looping brings enhancer to Promoter proximity. Boost +0.25 → {transcription_rate:.2f}")

    # 4. Fluorescence & Reward
    fluorescence = round(transcription_rate ** (1 / 1.2), 4)
    result.fluorescence_output = fluorescence
    result.is_expressing = fluorescence > 0.05
    
    accuracy = 1.0 - abs(fluorescence - target_fluorescence)
    complexity = max(0.0, 1.0 - (len(sequence) - target_part_count)/target_part_count)
    
    raw_R = (W_A * accuracy) + (W_C * complexity) - (W_B * result.metabolic_burden)
    result.math_reward = round(max(0.0, min(raw_R, 1.0)) * 10.0, 4)
    
    return result