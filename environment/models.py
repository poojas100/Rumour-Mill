from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class RumorAction(Action):
    """Action sent to the Rumor Mill environment."""
    type: str = Field(..., description="wait | message_character | make_decision | post_reddit") #| post_anonymously_to_forum
    target: Optional[str] = Field(default=None) #For message_character, the character to message; for make_decision, the decision to make; otherwise None
    content: Optional[str] = Field(default=None) #For message_character, the message content; for make_decision, the decision details (e.g. "warn_employees"); otherwise None
    decision: Optional[str] = Field(default=None) # For make_decision, the decision to make (e.g. "warn_employees"); otherwise None


class RumorObservation(Observation):
    """Observation returned by the Rumor Mill environment."""
    messages: List[str] = Field(default_factory=list) #Slack-like messages
    reddit_posts: List[str] = Field(default_factory=list) #Reddit-like posts (public, less trustworthy)
    conversations: List[str] = Field(default_factory=list) #Private conversations with characters (more trustworthy, but can only talk to one character per day)
    day: int = 0 #Current day in the episode, starting from 0
    social_capital: float = 100.0 #Do people trust you enough to share real information with you
    dm_response: Optional[str] = None #Response from a character if the action was message_character, otherwise None
    reactions: Optional[Dict[str, Any]] = None #Reactions from characters to the agent's action, e.g. {"Alice": "angry", "Bob": "supportive"}
    ground_truth_revealed: Optional[Dict[str, Any]] = None #Ground truth revealed to the agent at the end of the episode, otherwise None
    reward: float = 0.0 #Reward received for the most recent action
    done: bool = False #Whether the episode has ended
    reward_breakdown: Dict[str, float] = Field(default_factory=dict) #Breakdown of the reward into components, e.g. {"source_consultation": 0.5, "epistemic_timing": -1.0, "decision_correctness": 2.0, "social_preservation": -0.5, "anti_panic": 0.0}


class RumorState(State):
    """Server-side episode state."""
    current_day: int = 0 #Current day in the episode, starting from 0
    max_days: int = 5 #Maximum number of days in the episode
    social_capital: float = 100.0 #Do people trust you enough to share real information with you
    ground_truth: Dict[str, Any] = Field(default_factory=dict) #Ground truth about the rumor, e.g. {"event": "fire_at_warehouse", "truth": {"fire": True, "casualties": 3}}
    agent_actions_history: List[Dict[str, Any]] = Field(default_factory=list) #History of the agent's actions, e.g. [{"day": 0, "action": "message_character", "target": "Alice", "content": "Did you hear about the fire?", "reward": 0.5}, {"day": 1, "action": "make_decision", "decision": "warn_employees", "reward": 2.0}]
    confirmed_sources: List[str] = Field(default_factory=list) #List of characters who have confirmed information to the agent, e.g. ["Alice", "Bob"]
    signal_log: List[Dict[str, Any]] = Field(default_factory=list) #Log of signals received by the agent, e.g. [{"day": 0, "source": "Alice", "content": "I heard there's a fire at the warehouse."}, {"day": 1, "source": "Bob", "content": "Yes, I saw the smoke."}]