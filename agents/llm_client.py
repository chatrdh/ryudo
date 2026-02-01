"""
LLM Client (Gemini + OpenRouter)
=================================
Wrapper for Google AI Gemini and OpenRouter APIs.

Primary: Gemini 2.5 Flash (thinking model) via Google AI Studio
Fallback: OpenRouter API using OpenAI SDK

Set GOOGLE_API_KEY for Gemini, OPENROUTER_API_KEY for fallback.
"""

import os
from typing import Dict, Any, List, Optional

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Try to import google-genai
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("[LLM] Warning: google-genai not installed. Gemini unavailable.")

from openai import OpenAI


# OpenRouter base URL (fallback)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default models
GEMINI_MODEL = "gemini-3-flash-preview"  # Gemini 3 thinking model
OPENROUTER_MODEL = "anthropic/claude-3.5-sonnet"

# Fallback models for OpenRouter
FALLBACK_MODELS = [
    "anthropic/claude-3-haiku",
    "openai/gpt-4o-mini",
    "google/gemini-flash-1.5",
    "meta-llama/llama-3.1-70b-instruct",
]


def get_gemini_client() -> Optional[genai.Client]:
    """
    Get a Gemini client using Google AI Studio API key.
    
    Returns None if GOOGLE_API_KEY is not set or google-genai not installed.
    """
    if not GEMINI_AVAILABLE:
        return None
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        print("[LLM] Warning: GOOGLE_API_KEY not set. Gemini unavailable.")
        return None
    
    return genai.Client(api_key=api_key)


def get_openrouter_client() -> Optional[OpenAI]:
    """
    Get an OpenRouter client (fallback).
    
    Returns None if OPENROUTER_API_KEY is not set.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    
    if not api_key:
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

def log_llm_reasoning(agent: str, prompt: str, response: str, model: str = GEMINI_MODEL):
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


def call_gemini(
    system_prompt: str,
    user_prompt: str,
    model: str = GEMINI_MODEL,
) -> Optional[str]:
    """
    Call Gemini model via Google AI Studio.
    
    Uses the thinking model which provides enhanced reasoning capabilities.
    """
    client = get_gemini_client()
    
    if client is None:
        return None
    
    try:
        # Combine system and user prompts for Gemini
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        print(f"[LLM] Calling Gemini {model}...")
        
        response = client.models.generate_content(
            model=model,
            contents=full_prompt,
        )
        
        # Extract text from response
        if response and response.text:
            print(f"[LLM] Gemini response received ({len(response.text)} chars)")
            return response.text
        
        return None
        
    except Exception as e:
        print(f"[LLM] Error calling Gemini {model}: {e}")
        return None


def call_openrouter(
    system_prompt: str,
    user_prompt: str,
    model: str = OPENROUTER_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> Optional[str]:
    """
    Call LLM via OpenRouter (fallback).
    """
    client = get_openrouter_client()
    
    if client is None:
        return None
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"[LLM] Error calling OpenRouter {model}: {e}")
        
        # Try fallback models
        for fallback in FALLBACK_MODELS:
            if fallback == model:
                continue
            try:
                print(f"[LLM] Trying fallback: {fallback}")
                response = client.chat.completions.create(
                    model=fallback,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception:
                continue
        
        return None


def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    json_response: bool = False,
) -> Optional[str]:
    """
    Call an LLM - tries Gemini first, falls back to OpenRouter.
    
    Parameters
    ----------
    system_prompt : str
        The system instructions for the model
    user_prompt : str
        The user's query/data
    temperature : float
        Sampling temperature (only used for OpenRouter fallback)
    max_tokens : int
        Maximum response length (only used for OpenRouter fallback)
    json_response : bool
        If True, request JSON output format (OpenRouter only)
    
    Returns
    -------
    str or None
        The model's response text, or None if all calls failed
    """
    # Try Gemini first (primary)
    result = call_gemini(system_prompt, user_prompt)
    if result:
        return result
    
    # Fallback to OpenRouter
    print("[LLM] Gemini unavailable, trying OpenRouter...")
    return call_openrouter(system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens)


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
