import random
from typing import Dict, List


def generate_scenario(difficulty: int = 1) -> Dict:
    """
    difficulty 1 = easy  (20% noise, 3 characters active)
    difficulty 2 = medium (40% noise, 4 characters active)  
    difficulty 3 = hard  (60% noise, all 5 characters, contradictory signals)
    
    Now generates a timeline of events across 10-15 days with planted contradictions
    that force the agent to reconcile conflicting signals across turns.
    """
    import random

    noise_level = difficulty * 0.2
    active_count = min(2 + difficulty, 5) 

    scenario_type = random.choice(["layoffs", "revenue_miss", "promotion_politics"])
    
    if scenario_type == "layoffs":
        timeline = {
            "day_1": {
                "event": "layoffs rumored",
                "truth": True,
                "source": "leaker",
                "details": "Engineering team, 10-15 people"
            },
            "day_4": {
                "event": "HR denies layoff rumors",
                "truth": False,  # planted contradiction
                "source": "hr_official",
                "details": "No staffing changes planned"
            },
            "day_8": {
                "event": "layoff memo leaked to press",
                "truth": True,  # contradiction resolved
                "source": "leaker",
                "details": "Memo dated day 2, kept secret for 6 days"
            },
            "day_12": {
                "event": "layoffs officially announced",
                "truth": True,
                "source": "leadership",
                "details": "12 people affected, transitions support offered"
            },
            "event_type": "layoffs",
            "core_truth": {
                "happening": True,
                "teams": ["Engineering"],
                "size": random.randint(8, 15),
                "date": "Day 12",
            }
        }
    elif scenario_type == "revenue_miss":
        timeline = {
            "day_2": {
                "event": "Q4 numbers looking strong",
                "truth": False,  # planted contradiction
                "source": "cfo",
                "details": "On track for record quarter"
            },
            "day_5": {
                "event": "Q4 revenue miss revealed",
                "truth": True,
                "source": "investor_call",
                "details": "15% below target, budget freeze imminent"
            },
            "day_9": {
                "event": "Root cause: pipeline collapsed",
                "truth": True,
                "source": "sales_vp",
                "details": "Three large deals fell through in final week"
            },
            "day_13": {
                "event": "Recovery plan announced",
                "truth": True,
                "source": "ceo",
                "details": "New sales strategy, aggressive Q1 targets"
            },
            "event_type": "revenue_miss",
            "core_truth": {
                "missed": True,
                "percentage": random.randint(12, 20),
                "responsible_team": "Sales",
                "consequences": "budget_freeze",
            }
        }
    else:  # promotion_politics
        timeline = {
            "day_1": {
                "event": "Candidate A leading for promotion",
                "truth": True,
                "source": "politician",
                "details": "More technical, better track record"
            },
            "day_5": {
                "event": "Candidate B getting the role instead",
                "truth": False,  # planted contradiction
                "source": "gossip",
                "details": "Word is leadership prefers B"
            },
            "day_9": {
                "event": "Candidate A still in final round",
                "truth": True,  # contradiction resolved
                "source": "leaker",
                "details": "B faded after executive concerns raised"
            },
            "day_14": {
                "event": "Candidate A officially promoted",
                "truth": True,
                "source": "leadership",
                "details": "VP of Engineering, effective next month"
            },
            "event_type": "promotion_politics",
            "core_truth": {
                "candidate_a": "competent",
                "candidate_b": "politically_connected",
                "winner": "candidate_a",  # corrected from B
                "reason": "executive_eventually_aligned",
            }
        }

    timeline["noise_level"] = noise_level
    timeline["active_characters"] = active_count
    timeline["difficulty"] = difficulty
    
    return timeline