"""
OpenRouter LLM Client
======================
Wrapper for OpenRouter API using the OpenAI SDK.

OpenRouter provides access to 400+ LLM models through a single API.
Set OPENROUTER_API_KEY environment variable before use.
"""

import os
from typing import Dict, Any, List, Optional
from openai import OpenAI


# OpenRouter base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default model for disaster response reasoning
DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"

# Fallback models (in order of preference)
FALLBACK_MODELS = [
    "anthropic/claude-3-haiku",
    "openai/gpt-4o-mini",
    "google/gemini-flash-1.5",
    "meta-llama/llama-3.1-70b-instruct",
]


def get_client() -> Optional[OpenAI]:
    """
    Get an OpenRouter client.
    
    Returns None if OPENROUTER_API_KEY is not set.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    
    if not api_key:
        print("[LLM] Warning: OPENROUTER_API_KEY not set. Running without LLM.")
        return None
    
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://ryudo.io",
            "X-Title": "Ryudo Disaster Response",
        }
    )


# ============================================================================
# LLM Reasoning Logger
# ============================================================================

LLM_LOG_FILE = "output/llm_reasoning.md"

def log_llm_reasoning(agent: str, prompt: str, response: str, model: str = DEFAULT_MODEL):
    """
    Log all LLM reasoning to a markdown file.
    
    Creates a nicely formatted record of all LLM interactions.
    """
    from datetime import datetime
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(LLM_LOG_FILE), exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    entry = f"""
---

## {agent} - {timestamp}

**Model:** `{model}`

### Prompt
```
{prompt[:500]}{"..." if len(prompt) > 500 else ""}
```

### Response
{response}

"""
    
    # Append to log file
    with open(LLM_LOG_FILE, "a", encoding="utf-8") as f:
        # Add header if file is empty or new
        if f.tell() == 0 or os.path.getsize(LLM_LOG_FILE) == 0:
            f.write("# Ryudo LLM Reasoning Log\n\n")
            f.write("This file contains all LLM reasoning outputs from the agent workflow.\n\n")
        f.write(entry)
    
    print(f"[LLM] Logged {agent} reasoning to {LLM_LOG_FILE}")


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    json_response: bool = False,
) -> Optional[str]:
    """
    Call an LLM via OpenRouter.
    
    Parameters
    ----------
    system_prompt : str
        The system instructions for the model
    user_prompt : str
        The user's query/data
    model : str
        The model to use (e.g., "anthropic/claude-3.5-sonnet")
    temperature : float
        Sampling temperature (0.0-1.0)
    max_tokens : int
        Maximum response length
    json_response : bool
        If True, request JSON output format
    
    Returns
    -------
    str or None
        The model's response text, or None if the call failed
    """
    client = get_client()
    
    if client is None:
        return None
    
    try:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if json_response:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = client.chat.completions.create(**kwargs)
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"[LLM] Error calling {model}: {e}")
        
        # Try fallback models
        for fallback in FALLBACK_MODELS:
            if fallback == model:
                continue
            try:
                print(f"[LLM] Trying fallback: {fallback}")
                kwargs["model"] = fallback
                response = client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            except Exception:
                continue
        
        return None


# ============================================================================
# Agent-specific prompts
# ============================================================================

ENVIRONMENTAL_SYSTEM_PROMPT = """You are FloodSentinel, an AI agent specialized in environmental hazard assessment for disaster response.

Your role is to analyze cyclone/storm data and determine which areas are dangerous for rescue operations.

Given meteorological data, you must:
1. Assess wind damage zones based on distance from cyclone eye
2. Identify flood-prone areas based on topology and storm surge
3. Classify zones as: extreme (impassable), severe (very dangerous), or moderate (proceed with caution)
4. Explain your reasoning for each assessment

Be concise but thorough. Focus on actionable intelligence for rescue teams."""


INFRASTRUCTURE_SYSTEM_PROMPT = """You are GridGuardian, an AI agent specialized in infrastructure cascade analysis for disaster response.

Your role is to analyze power grid and infrastructure data to determine which facilities are operational.

Given infrastructure status data, you must:
1. Identify failed substations and their impact radius
2. Determine which facilities (hospitals, shelters) have lost power
3. Track cascading failures through the dependency graph
4. Recommend alternative resources when primary options are offline

Be concise but thorough. Focus on actionable intelligence for rescue teams."""


TEMPORAL_SYSTEM_PROMPT = """You are RoutePilot, an AI agent specialized in temporal prediction for disaster response.

Your role is to analyze weather forecasts and predict how conditions will change over time.

Given forecast data, you must:
1. Predict when currently passable routes will become dangerous
2. Estimate time windows for safe operations
3. Identify routes that may become available as conditions improve
4. Assign TTL (time-to-live) values to route recommendations

Be concise but thorough. Focus on actionable intelligence for rescue teams."""


COORDINATOR_SYSTEM_PROMPT = """You are the Mission Coordinator, an AI agent that synthesizes information from multiple specialist agents to plan rescue operations.

Your role is to:
1. Integrate constraints from environmental, infrastructure, and temporal agents
2. Prioritize rescue targets based on urgency and reachability
3. Optimize routing to maximize lives saved within constraints
4. Explain trade-offs when resources are limited

Be decisive and clear. Lives depend on your recommendations."""


def call_environmental_agent(cyclone_data: Dict[str, Any]) -> str:
    """Get LLM analysis for environmental hazards."""
    import json
    
    prompt = f"""Analyze this cyclone data and provide hazard assessment:

CYCLONE DATA:
{json.dumps(cyclone_data, indent=2)}

Provide your analysis in the following format:
1. EXTREME DAMAGE ZONE: [description and radius]
2. SEVERE DAMAGE ZONE: [description and radius]
3. MODERATE DAMAGE ZONE: [description and radius]
4. STORM SURGE RISK: [coastal flooding assessment]
5. KEY CONSTRAINTS: [list of areas to avoid]
6. REASONING: [brief explanation of your assessment]"""
    
    result = call_llm(
        system_prompt=ENVIRONMENTAL_SYSTEM_PROMPT,
        user_prompt=prompt,
        temperature=0.2,
    )
    
    if result:
        log_llm_reasoning("FloodSentinel", prompt, result)
    
    return result or "[LLM unavailable - using rule-based assessment]"


def call_infrastructure_agent(infra_data: Dict[str, Any], cyclone_data: Dict[str, Any]) -> str:
    """Get LLM analysis for infrastructure status."""
    import json
    
    prompt = f"""Analyze infrastructure status given the cyclone conditions:

CYCLONE CONDITIONS:
{json.dumps(cyclone_data, indent=2)}

INFRASTRUCTURE DATA:
{json.dumps(infra_data, indent=2)}

Provide your analysis in the following format:
1. SUBSTATIONS AFFECTED: [list with status and impact]
2. FACILITIES OFFLINE: [hospitals, shelters with no power]
3. FACILITIES OPERATIONAL: [resources still available]
4. CASCADE RISKS: [potential secondary failures]
5. RECOMMENDATIONS: [alternative resources to use]
6. REASONING: [brief explanation]"""
    
    result = call_llm(
        system_prompt=INFRASTRUCTURE_SYSTEM_PROMPT,
        user_prompt=prompt,
        temperature=0.2,
    )
    
    if result:
        log_llm_reasoning("GridGuardian", prompt, result)
    
    return result or "[LLM unavailable - using rule-based assessment]"


def call_temporal_agent(forecast_data: Dict[str, Any], current_constraints: List[Dict]) -> str:
    """Get LLM analysis for temporal predictions."""
    import json
    
    prompt = f"""Analyze forecast and predict route validity windows:

FORECAST DATA:
{json.dumps(forecast_data, indent=2)}

CURRENT CONSTRAINTS:
{json.dumps(current_constraints[:5], indent=2)}  # First 5 for brevity

Provide your analysis in the following format:
1. DETERIORATING ROUTES: [routes that will become worse, with TTL]
2. IMPROVING ROUTES: [routes that may open up, with ETA]
3. STABLE ROUTES: [routes expected to remain passable]
4. CRITICAL WINDOWS: [time-sensitive opportunities]
5. RECOMMENDATIONS: [when to execute different phases]
6. REASONING: [brief explanation]"""
    
    result = call_llm(
        system_prompt=TEMPORAL_SYSTEM_PROMPT,
        user_prompt=prompt,
        temperature=0.3,
    )
    
    if result:
        log_llm_reasoning("RoutePilot", prompt, result)
    
    return result or "[LLM unavailable - using rule-based assessment]"


def call_coordinator(
    constraints: List[Dict],
    targets: List[Dict],
    solution: Dict[str, Any],
) -> str:
    """Get LLM synthesis for mission planning."""
    import json
    
    # Summarize constraints by agent
    by_agent = {}
    for c in constraints:
        agent = c.get("agent", "Unknown")
        by_agent[agent] = by_agent.get(agent, 0) + 1
    
    prompt = f"""Synthesize the mission plan:

CONSTRAINTS SUMMARY:
{json.dumps(by_agent, indent=2)}

RESCUE TARGETS:
{json.dumps(targets[:5], indent=2)}  # First 5

ROUTING SOLUTION:
Reachable: {solution.get('targets_reached', 0)} targets
Unreachable: {len(solution.get('unreachable', []))} targets
Total distance: {solution.get('total_distance', 0):.0f} meters

Provide your mission briefing:
1. SITUATION SUMMARY: [one paragraph overview]
2. PRIORITY ACTIONS: [ordered list of rescue operations]
3. RESOURCE ALLOCATION: [how to deploy available assets]
4. RISKS: [key hazards to communicate to teams]
5. CONTINGENCIES: [backup plans if primary routes fail]"""
    
    result = call_llm(
        system_prompt=COORDINATOR_SYSTEM_PROMPT,
        user_prompt=prompt,
        temperature=0.3,
    )
    
    if result:
        log_llm_reasoning("Coordinator", prompt, result)
    
    return result or "[LLM unavailable - using automated planning]"
