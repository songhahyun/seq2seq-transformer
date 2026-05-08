from sacrebleu import corpus_bleu, corpus_chrf


def compute_bleu(predictions, references):
    # sacrebleu expects list[str], refs=list[list[str]]
    return corpus_bleu(predictions, [references]).score


def compute_chrf(predictions, references):
    return corpus_chrf(predictions, [references]).score
