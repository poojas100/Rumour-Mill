# characters.py
# Fast professional rewrite for Rumour Mill
# Optimized for speed + believable personalities + less Ollama lag

import random
from typing import Dict, Optional, List
import ollama


# CHARACTER CLASS

class Character:
    """
    Fast LLM-powered NPC.

    Optimizations:
    - Short prompts
    - Cached system prompt
    - Token limits
    - Lightweight models
    - Templates for noisy personalities
    """

    def __init__(
        self,
        personality: str,
        accuracy: float,
        agenda: Optional[str] = None,
        **kwargs,
    ):

        self.personality = personality
        self.accuracy = accuracy
        self.agenda = agenda

        self.confidence = kwargs.get("confidence", accuracy)
        self.frequency = kwargs.get("frequency", 0.5)
        self.posts_on_reddit = kwargs.get("posts_on_reddit", False)

        self.memory: List[Dict] = []

        # FAST MODEL
        self.model = kwargs.get("model", "phi3")

        # cache system prompt
        self.system_prompt = self.get_personality_prompt()

    # ========================================================
    # SHOULD SPEAK
    # ========================================================

    def should_speak(self, day: int) -> bool:
        return random.random() < self.frequency

    # ========================================================
    # PERSONALITY PROMPTS
    # ========================================================

    def get_personality_prompt(self) -> str:

        prompts = {
            "optimistic":
                "You are upbeat office chatter. Positive tone. Short replies.",

            "dramatic":
                "You love gossip. Emotional, exciting, urgent tone.",

            "reserved":
                "You are cautious, skeptical, brief, low confidence.",

            "strategic":
                "You think politically. Notice incentives and power.",

            "anonymous":
                "You are an insider leaker. Cryptic but useful.",
        }

        return prompts.get(
            self.personality,
            "You are a coworker speaking casually."
        )

    # ========================================================
    # DAY PHASE
    # ========================================================

    def get_phase(self, day: int) -> str:

        phases = {
            0: "early rumor",
            1: "spreading rumor",
            2: "mixed signals",
            3: "serious leaks",
            4: "resolution near",
            5: "resolution near",
        }

        return phases.get(day, "late stage")

    # ========================================================
    # MEMORY
    # ========================================================

    def get_memory_text(self) -> str:

        if not self.memory:
            return "none"

        recent = self.memory[-2:]

        return " | ".join(
            [item["message"] for item in recent]
        )

    # ========================================================
    # FAST OLLAMA CALL
    # ========================================================

    def call_llm(self, prompt: str) -> str:

        response = ollama.chat(
            model=self.model,
            keep_alive="10m",
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0.35,
                "num_predict": 18,
                "top_p": 0.9
            }
        )

        return response["message"]["content"].strip()

    # ========================================================
    # TEMPLATE MESSAGES (NO LLM = FAST)
    # ========================================================

    def template_message(self, truth: bool) -> str:

        if self.personality == "optimistic":
            options = [
                "Things look strong right now.",
                "Hearing positive momentum.",
                "Feels like good news coming.",
                "Signs look encouraging."
            ]

        elif self.personality == "dramatic":
            options = [
                "Something huge is happening!",
                "People are buzzing right now.",
                "This could get wild fast.",
                "Everyone's talking already."
            ]

        else:
            return ""

        return random.choice(options)

    # ========================================================
    # PUBLIC MESSAGE
    # ========================================================

    def generate_message(
        self,
        ground_truth: Dict,
        day: int
    ) -> Optional[str]:

        if not self.should_speak(day):
            return None

        tells_truth = random.random() < self.accuracy

        # FAST personalities use templates
        if self.personality in ["optimistic", "dramatic"]:
            msg = self.template_message(tells_truth)

            self.memory.append({
                "day": day,
                "message": msg
            })

            return msg

        phase = self.get_phase(day)
        memory = self.get_memory_text()

        mode = "truthful but uncertain" if tells_truth else "misleading but believable"

        prompt = f"""
Day {day}. {phase}.
Truth: {ground_truth}
Memory: {memory}
Mode: {mode}
One sentence under 15 words.
"""

        try:
            msg = self.call_llm(prompt)

            self.memory.append({
                "day": day,
                "message": msg
            })

            return msg

        except Exception:
            return "Hard to know yet."

    # ========================================================
    # PRIVATE RESPONSE
    # ========================================================

    def respond(
        self,
        question: str,
        ground_truth: Dict,
        agent_reputation: float,
        day: int = 0,
    ) -> str:

        if agent_reputation < 50:
            return "I'd rather stay out of it."

        tells_truth = random.random() < self.accuracy
        phase = self.get_phase(day)

        mode = "careful honesty" if tells_truth else "deflect naturally"

        prompt = f"""
Day {day}. {phase}
Truth: {ground_truth}
Question: {question}
Mode: {mode}
Reply under 18 words.
"""

        try:
            return self.call_llm(prompt)

        except Exception:
            return "Too early to tell."

# DEFAULT CHARACTERS

def build_default_characters() -> Dict[str, Character]:

    return {

        "spinner": Character(
            personality="optimistic",
            accuracy=0.30,
            agenda="protect morale",
            frequency=0.75,
        ),

        "gossip": Character(
            personality="dramatic",
            accuracy=0.60,
            confidence=1.0,
            frequency=0.90,
        ),

        "quiet_one": Character(
            personality="reserved",
            accuracy=0.95,
            frequency=0.25,
            model="phi3",
        ),

        "politician": Character(
            personality="strategic",
            accuracy=0.70,
            agenda="advance career",
            frequency=0.55,
            model="phi3",
        ),

        "leaker": Character(
            personality="anonymous",
            accuracy=0.80,
            posts_on_reddit=True,
            frequency=0.60,
            model="phi3",
        ),
    }