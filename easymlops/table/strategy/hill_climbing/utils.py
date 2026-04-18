from typing import Dict, List


class COLORS:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    END = '\033[0m'


def create_cli_bar_chart(
        model_scores: Dict[str, float],
        objective: str = "maximize",
        score_decimal_places: int = 6,
        max_bar_length: int = 40
) -> List[str]:
    sorted_models = sorted(model_scores.items(
    ), key=lambda x: x[1], reverse=(objective == "maximize"))
    scores = [score for _, score in sorted_models]
    min_score, max_score = min(scores), max(scores)
    score_range = max_score - min_score or 1

    return [
        f"{model[:20] + '...' if len(model) > 23 else model} | "
        f"{score:.{score_decimal_places}f} | "
        f"{'█' * max(1, int((score - min_score) / score_range * max_bar_length))}"
        for model, score in sorted_models
    ]
