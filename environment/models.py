from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class RumorAction(Action):
    type: str = Field(..., description="wait | message_character | post_reddit | make_decision")
    target: Optional[str] = Field(default=None)
    content: Optional[str] = Field(default=None)
    decision: Optional[str] = Field(default=None)


class RumorObservation(Observation):
    messages: List[str] = Field(default_factory=list)
    reddit_posts: List[str] = Field(default_factory=list)
    conversations: List[str] = Field(default_factory=list)
    day: int = Field(default=0)
    social_capital: float = Field(default=100.0)
    dm_response: Optional[str] = Field(default=None)
    reactions: Optional[Dict[str, Any]] = Field(default=None)
    ground_truth_revealed: Optional[Dict[str, Any]] = Field(default=None)
    reward: float = Field(default=0.0)
    done: bool = Field(default=False)
    reward_breakdown: Dict[str, Any] = Field(default_factory=dict)  # ✅ Any not float
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class RumorState(State):
    current_day: int = Field(default=0)
    max_days: int = Field(default=5)
    social_capital: float = Field(default=100.0)
    ground_truth: Dict[str, Any] = Field(default_factory=dict)
    agent_actions_history: List[Dict[str, Any]] = Field(default_factory=list)
    confirmed_sources: List[str] = Field(default_factory=list)
    signal_log: List[Dict[str, Any]] = Field(default_factory=list)