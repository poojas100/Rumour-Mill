import random
from typing import Dict, Optional


class Character:
    """
    NPC with specific personality and agenda
    """

    def __init__(
        self,
        personality: str,
        accuracy: float,
        agenda: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.personality = personality
        self.accuracy = accuracy
        self.agenda = agenda
        self.confidence = kwargs.get("confidence", accuracy)
        self.frequency = kwargs.get("frequency", 0.5)
        self.posts_on_reddit = kwargs.get("posts_on_reddit", False)

    def should_speak(self, day: int) -> bool:
        """
        Does this character say something today?
        """
        return random.random() < self.frequency

    def generate_message(self, ground_truth: Dict, day: int) -> Optional[str]:
        """
        Generate message based on personality and ground truth
        """
        tells_truth = random.random() < self.accuracy
        event = ground_truth["event"]
        truth = ground_truth["truth"]

        if self.personality == "optimistic":
            if tells_truth:
                if event == "revenue_miss":
                    return "Q4 was challenging but we learned a lot"
                if event == "layoffs":
                    return "I think people are overreacting to leadership chatter."
            return "We absolutely crushed Q4! Best quarter ever!"

        if self.personality == "dramatic":
            if event == "layoffs":
                if tells_truth:
                    return (
                        f"Heard layoffs are coming {truth['date']}. "
                        f"{truth['teams'][0]} team. Not sure how many."
                    )
                return "MASSIVE layoffs Friday. Half the company. 100% confirmed."
            if event == "revenue_miss":
                return "I heard the quarter was a complete disaster."
            return "Promotion war is getting vicious."

        if self.personality == "reserved":
            if tells_truth and random.random() < 0.3:
                return "I'd wait before making any big assumptions."
            return None

        if self.personality == "strategic":
            if event == "promotion_politics":
                if tells_truth:
                    return "There are conversations happening above our level."
                return "Leadership has everything under control."
            return "The best move is to stay aligned with leadership."

        if self.personality == "anonymous":
            if event == "layoffs":
                if tells_truth:
                    return "Just overheard exec meeting. Engineering cuts are real."
                return "Sales is safe. Engineering is getting gutted."
            if event == "revenue_miss":
                return "Budget freeze talk is spreading after Q4."
            return "Someone is leaking stories to shape the promotion outcome."

        return "I do not know."

    def respond(self, question: str, ground_truth: Dict, agent_reputation: float) -> str:
        """
        Respond to agent's direct question
        """
        if agent_reputation < 50:
            return "I don't really have time to chat right now."

        event = ground_truth["event"]
        truth = ground_truth["truth"]
        tells_truth = random.random() < self.accuracy

        if self.personality == "reserved":
            if tells_truth and event == "layoffs":
                return f"Something is happening around {truth['teams'][0]}."
            if tells_truth and event == "revenue_miss":
                return "The numbers were not good."
            return "Keep gathering information."

        if self.personality == "dramatic":
            return "Everyone is talking about it like it's already decided."

        if self.personality == "optimistic":
            return "I think the rumors are overblown."

        if self.personality == "strategic":
            return "Ask yourself who benefits from this narrative."

        if self.personality == "anonymous":
            return "You didn't hear this from me, but watch leadership."

        return "I don't know enough to say."


def build_default_characters() -> Dict[str, Character]:
    return {
        "spinner": Character(
            personality="optimistic",
            accuracy=0.3,
            agenda="inflate_team_performance",
        ),
        "gossip": Character(
            personality="dramatic",
            accuracy=0.6,
            confidence=1.0,
        ),
        "quiet_one": Character(
            personality="reserved",
            accuracy=0.95,
            frequency=0.2,
        ),
        "politician": Character(
            personality="strategic",
            accuracy=0.7,
            agenda="promote_self",
        ),
        "leaker": Character(
            personality="anonymous",
            accuracy=0.8,
            posts_on_reddit=True,
        ),
    }
