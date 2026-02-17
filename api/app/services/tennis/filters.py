def has_edge(model_prob: float, book_prob: float, min_edge: float = 0.05) -> bool:
    return model_prob - book_prob >= min_edge
