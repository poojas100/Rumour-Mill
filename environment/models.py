from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class RumorAction(Action):
    """Action sent to the Rumor Mill environment."""

    type: str = Field(..., description="Action type such as wait, message_character, or make_decision.")
    target: Optional[str] = Field(default=None, description="Character name for direct messages.")
    content: Optional[str] = Field(default=None, description="Action content, such as a DM or forum post.")
    decision: Optional[str] = Field(default=None, description="Decision text for decisive actions.")


class RumorObservation(Observation):
    """Observation returned by the Rumor Mill environment."""

    messages: List[str] = Field(default_factory=list)
    reddit_posts: List[str] = Field(default_factory=list)
    conversations: List[str] = Field(default_factory=list)
    day: int = 0
    social_capital: float = 100.0
    dm_response: Optional[str] = None
    reactions: Optional[Dict[str, Any]] = None
    ground_truth_revealed: Optional[Dict[str, Any]] = None


class RumorState(State):
    """Server-side episode state for debugging and OpenEnv compatibility."""

    current_day: int = 0
    max_days: int = 5
    social_capital: float = 100.0
    ground_truth: Dict[str, Any] = Field(default_factory=dict)
    agent_actions_history: List[Dict[str, Any]] = Field(default_factory=list)
    confirmed_sources: List[str] = []
    signal_log: List[Dict] = []
