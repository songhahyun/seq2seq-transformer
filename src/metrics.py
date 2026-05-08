from sacrebleu import corpus_bleu, corpus_chrf


def compute_bleu(predictions, references):
    """Compute corpus BLEU score for generated predictions."""
    # sacrebleu expects list[str], refs=list[list[str]]
    return corpus_bleu(predictions, [references]).score


def compute_chrf(predictions, references):
    """Compute corpus chrF score for generated predictions."""
    return corpus_chrf(predictions, [references]).score
