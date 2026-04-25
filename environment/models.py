"""
Rumor Mill Data Models - IMPROVED VERSION
==========================================

OpenEnv-compatible type definitions for:
- Actions (what agent can do)
- Observations (what agent sees)
- State (server-side episode tracking)

NEW: Added metadata field for difficulty, format version, oversight analysis
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class RumorAction(Action):
    """
    Action sent to the Rumor Mill environment.
    
    The agent can:
    1. Wait and gather more information
    2. Message a character directly (DM)
    3. Post on the anonymous forum (Reddit)
    4. Make a decisive action (warn people, escalate, etc.)
    
    Attributes:
        type: One of "wait", "message_character", "post_reddit", "make_decision"
        target: Character name for DMs (e.g., "quiet_one", "gossip")
        content: Message text for DMs or forum posts
        decision: Decision text for decisive actions
    """

    type: str = Field(
        ..., 
        description="Action type: 'wait', 'message_character', 'post_reddit', or 'make_decision'"
    )
    
    target: Optional[str] = Field(
        default=None, 
        description="Character name for direct messages (e.g., 'quiet_one', 'gossip')"
    )
    
    content: Optional[str] = Field(
        default=None, 
        description="Message content for DMs or forum posts"
    )
    
    decision: Optional[str] = Field(
        default=None, 
        description="Decision text for decisive actions (e.g., 'warn engineering team')"
    )


class RumorObservation(Observation):
    """
    Observation returned by the Rumor Mill environment.
    
    This is what the agent sees after each action.
    Contains:
    - Messages from characters (Slack-style)
    - Anonymous Reddit posts
    - Current day and social capital
    - Responses to agent actions
    - Metadata (difficulty, format, oversight analysis)
    
    Attributes:
        messages: List of Slack-style messages from characters
        reddit_posts: List of anonymous forum posts
        conversations: List of 1-on-1 DM conversations
        day: Current day (0-4, week is 5 days)
        social_capital: Agent's reputation (0-100)
        dm_response: Response if agent DM'd someone
        reactions: Reddit reactions if agent posted
        ground_truth_revealed: Hidden truth (only shown at episode end)
        metadata: Extra info (difficulty level, format version, oversight)
    """

    # === CORE OBSERVATIONS (what agent sees every turn) ===
    messages: List[str] = Field(
        default_factory=list,
        description="Slack-style messages from characters"
    )
    
    reddit_posts: List[str] = Field(
        default_factory=list,
        description="Anonymous forum posts (could be from any character)"
    )
    
    conversations: List[str] = Field(
        default_factory=list,
        description="1-on-1 DM conversation history"
    )
    
    day: int = Field(
        default=0,
        description="Current day of the week (0-4)"
    )
    
    social_capital: float = Field(
        default=100.0,
        description="Agent's reputation/credibility (0-100)"
    )
    
    # === ACTION RESPONSES (only present if agent took specific actions) ===
    dm_response: Optional[str] = Field(
        default=None,
        description="Response from character if agent sent a DM"
    )
    
    reactions: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Reddit upvotes/downvotes/comments if agent posted"
    )
    
    ground_truth_revealed: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The hidden truth, revealed only at episode end"
    )

    reward: float = Field(
        default=0.0,
        description="Reward received after taking the action"
    )

    done: bool = Field(
        default=False,
        description="Whether the episode has ended"
    )

    reward_breakdown: Dict[str, float] = Field(
    default_factory=dict,
    description="Breakdown of reward into components"
    )

class RumorState(State):
    """
    Server-side episode state for debugging and OpenEnv compatibility.
    
    This is NOT shown to the agent - it's for environment tracking.
    Used for:
    - Episode ID tracking
    - Step counting
    - Internal debugging
    - OpenEnv integration
    
    Attributes:
        current_day: Which day of the week (0-4)
        max_days: Total days per episode (always 5)
        social_capital: Agent's current reputation
        ground_truth: The hidden truth (not shown to agent until end)
        agent_actions_history: All actions taken this episode
        confirmed_sources: Characters that have provided information
        signal_log: Extracted signals (positive/negative) from interactions
    """

    current_day: int = Field(
        default=0,
        description="Current day of the work week (0-4)"
    )
    
    max_days: int = Field(
        default=5,
        description="Total number of days per episode (one work week)"
    )
    
    social_capital: float = Field(
        default=100.0,
        description="Agent's current reputation score"
    )
    
    ground_truth: Dict[str, Any] = Field(
        default_factory=dict,
        description="Hidden truth about the event (layoffs, revenue, etc.)"
    )
    
    agent_actions_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Complete history of all actions taken this episode"
    )

    confirmed_sources: List[str] = Field(
        default_factory=list,
        description="List of characters that have been consulted by the agent"
    )

    signal_log: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Signals extracted from conversations (positive/negative)"
    )
