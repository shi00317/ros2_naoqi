from pydantic import BaseModel, Field
from typing import Optional, Callable, Dict, List, TypedDict, Any

# Pydantic model for animation output
class AnimationActions(BaseModel):
    """Structured output for NAO robot animation actions"""
    actions: List[str] = Field(..., description="List of animation action names for the response")


STTPrompt = "The following conversation is a user with Nao robot, that Nao robot will help user to answering question related to user's dayily tasks."

AnimationPrompt = """
Analyze the following response and select appropriate animation actions for a NAO robot.

Available actions: affirmative_context, anterior, comparison, confirmation, disappointment, diversity, exclamation, joy, people, self, user

Response to analyze: "{response_text}"

Based on the content and emotion of this response, select appropriate animation actions for every sentence. Consider:
- The emotional tone (joy, disappointment, excitement, etc.)
- The context (affirmation, comparison, self-reference, etc.) 
- The interaction type (confirmation, exclamation, diversity)

Return a list of action names that best match the content and emotion of each sentence in the response.
"""


