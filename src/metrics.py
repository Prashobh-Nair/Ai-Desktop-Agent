"""
Speech Recognition Evaluation Metrics: WER (Word Error Rate) and CER (Character Error Rate).
"""
from typing import List
import jiwer


def compute_wer(references: List[str], hypotheses: List[str]) -> float:
    """Computes Word Error Rate between references and hypotheses."""
    clean_refs = [r.strip() if r.strip() else "<empty>" for r in references]
    clean_hyps = [h.strip() if h.strip() else "<empty>" for h in hypotheses]
    try:
        return float(jiwer.wer(clean_refs, clean_hyps))
    except Exception:
        return 1.0


def compute_cer(references: List[str], hypotheses: List[str]) -> float:
    """Computes Character Error Rate between references and hypotheses."""
    clean_refs = [r.strip() if r.strip() else " " for r in references]
    clean_hyps = [h.strip() if h.strip() else " " for h in hypotheses]
    try:
        return float(jiwer.cer(clean_refs, clean_hyps))
    except Exception:
        return 1.0

