import random
from typing import Dict, List


def generate_scenario(difficulty: int = 1) -> Dict:
    """
    difficulty 1 = easy  (20% noise, 3 characters active)
    difficulty 2 = medium (40% noise, 4 characters active)  
    difficulty 3 = hard  (60% noise, all 5 characters, contradictory signals)
    """
    import random

    noise_level = difficulty * 0.2
    active_count = min(2 + difficulty, 5)

    scenarios = [
        {
            "event": "layoffs",
            "truth": {
                "happening": True,
                "teams": ["Engineering"],
                "size": random.randint(5, 30),
                "date": "Friday",
            }
        },
        {
            "event": "revenue_miss",
            "truth": {
                "missed": True,
                "percentage": random.randint(5, 20),
                "responsible_team": random.choice(["Sales", "Marketing"]),
                "consequences": "budget_freeze",
            }
        },
        {
            "event": "promotion_politics",
            "truth": {
                "candidate_a": "competent",
                "candidate_b": "politically_connected",
                "winner": "candidate_b",
                "reason": "executive_favor",
            }
        },
    ]

    scenario = random.choice(scenarios)
    scenario["noise_level"] = noise_level
    scenario["active_characters"] = active_count
    scenario["difficulty"] = difficulty

    return scenario