import random
from typing import Dict, List


def generate_scenario() -> Dict:
    """Create the hidden ground truth for a new episode."""
    scenarios: List[Dict] = [
        {
            "event": "layoffs",
            "truth": {
                "happening": True,
                "teams": ["Engineering"],
                "size": 15,
                "date": "Friday",
            },
        },
        {
            "event": "revenue_miss",
            "truth": {
                "missed": True,
                "percentage": 12,
                "responsible_team": "Sales",
                "consequences": "budget_freeze",
            },
        },
        {
            "event": "promotion_politics",
            "truth": {
                "candidate_a": "competent",
                "candidate_b": "politically_connected",
                "winner": "candidate_b",
                "reason": "executive_favor",
            },
        },
    ]
    return random.choice(scenarios)
