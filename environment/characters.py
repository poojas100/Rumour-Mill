import random
from typing import Dict, List, Optional

# ── Ollama import (graceful fallback if not available) ────────
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class Character:
    """
    Hybrid NPC: fast templates for simple personalities,
    Ollama LLM calls for complex ones.
    """

    # Characters that use Ollama
    LLM_PERSONALITIES = {"reserved", "strategic", "anonymous"}

    # Fast template messages
    TEMPLATES = {
        "optimistic": {
            "layoffs": {
                True:  ["People are overreacting to leadership chatter.",
                        "I think this rumor is overblown.",
                        "Leadership has a plan — don't panic.",
                        "It's not as bad as people are saying."],
                False: ["We absolutely crushed it this quarter!",
                        "Everything is fine, trust the process.",
                        "Best quarter ever — nothing to worry about.",
                        "Signs look encouraging, very positive momentum."],
            },
            "revenue_miss": {
                True:  ["Q4 was challenging but we will bounce back.",
                        "The numbers are rough but recoverable.",
                        "Setbacks happen — we will adapt quickly.",
                        "I think we learned a lot this quarter."],
                False: ["We absolutely crushed Q4! Record numbers.",
                        "Best quarter in company history by far.",
                        "Revenue is looking great this cycle.",
                        "Pipeline is stronger than ever right now."],
            },
            "promotion_politics": {
                True:  ["Competition is healthy — best person will win.",
                        "Leadership will make the right call here.",
                        "Both candidates are strong — good problem to have.",
                        "The process is fair, I trust it completely."],
                False: ["Everything is settled — no drama here at all.",
                        "Leadership has this completely under control.",
                        "Nothing to see here, move along.",
                        "The decision is basically already made."],
            },
        },

        "dramatic": {
            "layoffs": {
                True:  ["Heard layoffs are coming — Engineering team, not sure how many.",
                        "Something is definitely happening Friday. People are scared.",
                        "I have it on good authority cuts are real. Not huge but real.",
                        "Multiple people told me Engineering is getting hit."],
                False: ["MASSIVE layoffs Friday. Half the company. 100% confirmed.",
                        "Everyone is getting fired. They are keeping maybe 20 people.",
                        "It's a bloodbath — the whole floor is going.",
                        "HR is in full panic mode. This is catastrophic."],
            },
            "revenue_miss": {
                True:  ["I heard the quarter was rough — missed targets badly.",
                        "Finance is freaking out behind closed doors right now.",
                        "Q4 was a disaster apparently. Numbers are ugly.",
                        "Budget freeze is definitely coming after this quarter."],
                False: ["Complete disaster — worst quarter in company history.",
                        "We missed by 40%. Heads are rolling already.",
                        "The board is furious. CEO might be out.",
                        "It's over. The company is basically on life support."],
            },
            "promotion_politics": {
                True:  ["Promotion war is getting intense — two strong candidates.",
                        "The politics around this role are getting vicious.",
                        "Heard there are serious disagreements above our level.",
                        "This is getting messy — alliances are forming everywhere."],
                False: ["Someone is getting fired over this promotion fight.",
                        "It's an absolute circus up there right now.",
                        "Half the exec team is involved in this drama.",
                        "There are threats flying around — this is insane."],
            },
        },
    }

    # Ollama system prompts per personality
    SYSTEM_PROMPTS = {
        "reserved": (
            "You are a cautious, skeptical coworker. "
            "You speak rarely and briefly. "
            "You never sound certain. "
            "You use phrases like: maybe, not sure, too early, unclear. "
            "Max 15 words per reply."
        ),
        "strategic": (
            "You think politically about everything in the office. "
            "You notice incentives, power dynamics, and who benefits. "
            "You speak with measured confidence. "
            "Max 18 words per reply."
        ),
        "anonymous": (
            "You are an insider leaker. "
            "You hint at confidential information without revealing your source. "
            "You sound cryptic but useful. "
            "Max 18 words per reply."
        ),
    }

    def __init__(self, personality: str, accuracy: float,
                 agenda: Optional[str] = None, **kwargs):
        self.personality     = personality
        self.accuracy        = accuracy
        self.agenda          = agenda
        self.confidence      = kwargs.get("confidence", accuracy)
        self.frequency       = kwargs.get("frequency", 0.5)
        self.posts_on_reddit = kwargs.get("posts_on_reddit", False)
        self.ollama_model    = kwargs.get("model", "llama3")
        self.memory: List[Dict] = []

    def should_speak(self, day: int) -> bool:
        return random.random() < self.frequency

    def _get_event(self, ground_truth: Dict) -> str:
        return (ground_truth.get("event_type")
                or ground_truth.get("event", "layoffs"))

    def _uses_llm(self) -> bool:
        return (self.personality in self.LLM_PERSONALITIES
                and OLLAMA_AVAILABLE)

    # ── Ollama call ───────────────────────────────────────────

    def _call_ollama(self, user_prompt: str) -> Optional[str]:
        try:
            system = self.SYSTEM_PROMPTS.get(
                self.personality, "You are a coworker. Max 18 words."
            )
            response = ollama.chat(
                model=self.ollama_model,
                keep_alive="10m",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_prompt},
                ],
                options={
                    "temperature": 0.4,
                    "num_predict": 24,
                    "top_p": 0.9,
                },
            )
            return response["message"]["content"].strip()
        except Exception:
            return None

    # ── Template message ──────────────────────────────────────

    def _template_message(self, ground_truth: Dict, tells_truth: bool) -> Optional[str]:
        event    = self._get_event(ground_truth)
        p_temps  = self.TEMPLATES.get(self.personality, {})
        e_temps  = p_temps.get(event, p_temps.get("layoffs", {}))
        options  = e_temps.get(tells_truth, e_temps.get(True, []))
        options  = [o for o in options if o]
        return random.choice(options) if options else None

    # ── Public message ────────────────────────────────────────

    def generate_message(self, ground_truth: Dict, day: int) -> Optional[str]:
        if not self.should_speak(day):
            return None

        tells_truth = random.random() < self.accuracy
        event       = self._get_event(ground_truth)

        # Fast personalities — no LLM
        if not self._uses_llm():
            return self._template_message(ground_truth, tells_truth)

        # LLM personalities — Ollama call
        recent_memory = " | ".join(
            m["message"] for m in self.memory[-2:]
        ) or "none"

        truth_mode = "truthful but uncertain" if tells_truth else "misleading but believable"

        phase_map  = {0:"early rumors", 1:"spreading rumors",
                      2:"mixed signals", 3:"serious leaks",
                      4:"resolution near", 5:"resolution near"}
        phase      = phase_map.get(day, "late stage")

        prompt = (
            f"Day {day}. Phase: {phase}.\n"
            f"Situation: {event} is the hidden event.\n"
            f"Your recent statements: {recent_memory}\n"
            f"Mode: {truth_mode}.\n"
            f"Write one sentence under 15 words. "
            f"Do not mention the event type directly."
        )

        result = self._call_ollama(prompt)

        # Fallback to template if Ollama fails
        if not result:
            result = self._template_message(ground_truth, tells_truth)

        if result:
            self.memory.append({"day": day, "message": result})

        return result

    # ── DM response ───────────────────────────────────────────

    # Fast DM templates for non-LLM characters
    FAST_DM = {
        "optimistic": {
            "layoffs":           "I think the rumors are overblown -- leadership has a plan.",
            "revenue_miss":      "We will bounce back -- one rough quarter is not the end.",
            "promotion_politics":"I think the best person will get it -- the process is fair.",
        },
        "dramatic": {
            "layoffs":           "They are definitely happening. I have heard it from three people.",
            "revenue_miss":      "Q4 was an absolute disaster. Budget freeze is certain.",
            "promotion_politics":"Everyone is talking about it like it is already decided.",
        },
        "reserved": {
            "layoffs":           "Something is happening around Engineering. I would not ignore it.",
            "revenue_miss":      "The numbers were not good. Q4 did not go as planned.",
            "promotion_politics":"There are conversations above our level. Outcome is unclear.",
        },
        "strategic": {
            "layoffs":           "Ask yourself who benefits from this narrative. Cuts look real.",
            "revenue_miss":      "The accountability conversations will be interesting. Miss is real.",
            "promotion_politics":"Think about relationships, not just track record.",
        },
        "anonymous": {
            "layoffs":           "You did not hear this from me -- cuts are real. Engineering.",
            "revenue_miss":      "The numbers are worse than they will admit. Budget freeze coming.",
            "promotion_politics":"The decision is already made. This is theatre.",
        },
    }

    def respond(self, question: str, ground_truth: Dict,
                agent_reputation: float, day: int = 0) -> str:

        if agent_reputation < 50:
            return "I would rather not get involved right now."

        tells_truth = random.random() < self.accuracy
        event       = self._get_event(ground_truth)

        if not self._uses_llm():
            event_dm = self.FAST_DM.get(self.personality, {})
            return event_dm.get(event, "Something is going on but I cannot say more.")

        # Ollama DM for LLM characters
        phase_map = {0:"early rumors", 1:"spreading rumors",
                     2:"mixed signals", 3:"serious leaks",
                     4:"resolution near"}
        phase     = phase_map.get(day, "late stage")

        # Late-stage leaker reveals more
        if self.personality == "anonymous" and day >= 3:
            truth_mode = "reveal a strong hint carefully"
        else:
            truth_mode = "honest but cautious" if tells_truth else "deflect naturally"

        recent_memory = " | ".join(
            m["message"] for m in self.memory[-2:]
        ) or "none"

        prompt = (
            f"Day {day}. Phase: {phase}.\n"
            f"Situation: {event} is the hidden corporate event.\n"
            f"Your recent statements: {recent_memory}\n"
            f"Someone privately asks: '{question}'\n"
            f"Mode: {truth_mode}.\n"
            f"Reply in one sentence under 20 words. Stay in character."
        )

        result = self._call_ollama(prompt)
        return result or self._fallback_dm(event, tells_truth)

    def _fallback_dm(self, event: str, tells_truth: bool) -> str:
        """Used when Ollama call fails."""
        fallbacks = {
            "reserved": {
                "layoffs":           ["Something is happening around Engineering.",
                                      "I would not ignore the rumors entirely.",
                                      "Things are becoming clearer — still gathering."],
                "revenue_miss":      ["The numbers were not good.",
                                      "Q4 did not go as planned.",
                                      "Something significant happened with revenue."],
                "promotion_politics":["There are conversations above our level.",
                                      "The outcome is less certain than it looks.",
                                      "I am watching this carefully."],
            },
            "strategic": {
                "layoffs":           "Ask yourself who benefits from this narrative.",
                "revenue_miss":      "The accountability conversations will be interesting.",
                "promotion_politics":"Think about relationships, not just track record.",
            },
            "anonymous": {
                "layoffs":           "You did not hear this from me — cuts are real.",
                "revenue_miss":      "The numbers are worse than they will admit.",
                "promotion_politics":"The decision is already made. This is theatre.",
            },
        }
        fb = fallbacks.get(self.personality, {})
        result = fb.get(event, fb.get("layoffs", "Hard to say right now."))
        if isinstance(result, list):
            result = random.choice(result)
        return result or "Hard to say right now."


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
            model="llama3",
        ),
        "politician": Character(
            personality="strategic",
            accuracy=0.70,
            agenda="advance career",
            frequency=0.55,
            model="llama3",
        ),
        "leaker": Character(
            personality="anonymous",
            accuracy=0.80,
            posts_on_reddit=True,
            frequency=0.60,
            model="llama3",
        ),
    }