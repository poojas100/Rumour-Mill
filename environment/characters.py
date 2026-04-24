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
        For timeline-based scenarios, references the event active at the current day
        """
        tells_truth = random.random() < self.accuracy #probability of telling the truth is based on accuracy parameter
        
        # Handle both old single-event format and new timeline format
        if "event_type" in ground_truth:  # Timeline format
            event = ground_truth["event_type"]
            truth = ground_truth.get("core_truth", {})
            timeline = ground_truth
        else:  # Legacy single-event format
            event = ground_truth.get("event", "")
            truth = ground_truth.get("truth", {})

        if self.personality == "optimistic":
            if tells_truth:
                if event == "revenue_miss":
                    if day <= 2:  # Before miss is revealed
                        return "Q4 is looking good so far!"
                    elif day >= 5:  # After miss revealed
                        return "We'll bounce back. This is temporary setback."
                    return "I think people are overreacting to the numbers."
                if event == "layoffs":
                    if day <= 3:  # Before denial
                        return "I think people are overreacting to leadership chatter."
                    elif day >= 8:  # After memo leaked
                        return "It's unfortunate but we'll get through this."
                    return "Don't panic about rumors."
            # Exaggerate if lying
            if event == "revenue_miss" and day <= 2:
                return "We absolutely crushed Q4! Best quarter ever!"
            return "Everything is fine, trust me!"

        if self.personality == "dramatic":
            if event == "layoffs":
                if tells_truth:
                    if day <= 3:
                        date_info = truth.get('date', 'soon')
                        size_info = truth.get('size', '10-15')
                        return f"Heard layoffs are coming {date_info}. Engineering team. About {size_info} people."
                    elif day >= 8:
                        size_info = truth.get('size', '?')
                        return f"The memo was leaked! Engineering cuts confirmed. {size_info} people."
                    elif day == 5:  # Day of contradiction (HR denial)
                        return "HR said no layoffs, but I don't believe them. Too much smoke."
                # Exaggerate if lying (especially day 5 contradiction)
                if day == 5:
                    return "MASSIVE layoffs Friday. Half the company. HR is lying!"
                return "MASSIVE layoffs coming. Everyone is doomed."
            if event == "revenue_miss":
                if tells_truth:
                    if day >= 5:
                        return "Q4 was a disaster. Revenue missed badly. Budget freeze incoming."
                    return "Whispers about Q4 being rough."
                return "I heard the quarter was a complete disaster."
            return "Promotion war is getting INSANE."

        if self.personality == "reserved":
            if tells_truth and random.random() < 0.3:
                if day <= 5:
                    return "I'd wait before making any big assumptions."
                elif day >= 8:
                    return "Things are becoming clearer. Still gathering info."
                return "I don't have enough to say."
            return None

        if self.personality == "strategic":
            if event == "promotion_politics":
                if tells_truth:
                    if day <= 3:
                        return "Candidate A seems to be winning."
                    elif day <= 8:
                        return "There are conversations happening above our level."
                    else:
                        return "Leadership made their choice. No surprises."
                if day <= 6:
                    return "Candidate B is gaining support with leadership."
                return "Leadership has everything under control."
            return "The best move is to stay aligned with leadership."

        if self.personality == "anonymous":
            if event == "layoffs":
                if tells_truth:
                    if day == 1:
                        return "Just overheard exec meeting. Engineering cuts are real."
                    elif day == 5:  # Day of contradiction
                        return "HR is in full denial mode. The cuts are definitely happening."
                    elif day >= 8:
                        return "The memo proving this was leaked. It was dated day 2!"
                # Lie convincingly at contradiction point
                if day == 5:
                    return "Actually, sales is getting hit harder. Engineering is safe."
                return "Sales is safe. Engineering is getting gutted."
            if event == "revenue_miss":
                if tells_truth:
                    if day >= 5:
                        return "Budget freeze talk is spreading after Q4."
                    return "Numbers aren't adding up for this quarter."
                return "Word is we're making massive cuts across the board."
            if tells_truth:
                return "Someone is leaking stories to shape the promotion outcome."
            return "Leaks everywhere about internal politics."

        return "I do not know."

    def respond(self, question: str, ground_truth: Dict, agent_reputation: float, day: int = 0) -> str:
        """
        Respond to agent's direct question - timeline-aware responses
        """
        if agent_reputation < 50:
            return "I don't really have time to chat right now."

        # Handle both old single-event format and new timeline format
        if "event_type" in ground_truth:  # Timeline format
            event = ground_truth["event_type"]
            truth = ground_truth.get("core_truth", {})
        else:  # Legacy single-event format
            event = ground_truth.get("event", "")
            truth = ground_truth.get("truth", {})
        
        tells_truth = random.random() < self.accuracy

        if self.personality == "reserved":
            if tells_truth:
                if event == "layoffs":
                    teams = truth.get('teams', ['Engineering'])
                    if day == 5:  # Contradiction day
                        return f"Something weird is happening with {teams[0]}. HR denied it though."
                    elif day >= 8:
                        return f"Well, it was real. {teams[0]} is being affected."
                    return f"Something is happening around {teams[0]}."
                if event == "revenue_miss":
                    if day >= 5:
                        return "Yeah, the numbers were really not good."
                    return "I've heard things. Not great."
            return "Keep gathering information."

        if self.personality == "dramatic":
            if day == 5 and event == "layoffs":
                return "They're lying! The layoffs are definitely happening!"
            if day >= 8:
                return "I told you so! It was all true!"
            return "Everyone is talking about it like it's already decided."

        if self.personality == "optimistic":
            if day >= 8:
                return "Well, it happened, but we'll recover from it."
            return "I think the rumors are overblown."

        if self.personality == "strategic":
            if day == 5 and event == "layoffs":
                return "This tells you something about what's really valued here."
            elif day >= 8:
                return "Now the smart play is clear. Adapt."
            return "Ask yourself who benefits from this narrative."

        if self.personality == "anonymous":
            if day >= 8:
                return "See? That's why I said to watch leadership."
            if day == 5 and event == "layoffs":
                return "The denial is the tell. It's definitely happening."
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
