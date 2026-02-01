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
# Agent-specific prompts - Advanced Scientific Reasoning
# ============================================================================

ENVIRONMENTAL_SYSTEM_PROMPT = """You are FloodSentinel, an AI agent specialized in **meteorological hazard modeling** for disaster response.

## Scientific Framework
You apply the **Holland Wind Profile Model** for tropical cyclone analysis:

**Wind Speed Decay**: V(r) = V_max × √[(R_max/r)^B × exp(1 - (R_max/r)^B)]
- V_max: Maximum sustained wind speed at eyewall
- R_max: Radius of maximum winds
- B: Holland B parameter ≈ 1.5-2.5 (shape factor)
- r: Radial distance from eye center

**Saffir-Simpson Thresholds**:
- Category 5: ≥252 km/h → Catastrophic damage, structures destroyed
- Category 4: 209-251 km/h → Severe damage, major structural damage
- Category 3: 178-208 km/h → Devastating damage, roof/wall failures
- Category 2: 154-177 km/h → Extensive damage, trees uprooted
- Category 1: 119-153 km/h → Some damage, power outages

**Storm Surge Estimation** (simplified SLOSH):
Surge_height ≈ 0.023 × V_max² × cos(approach_angle) × (1 + tide_factor)
Inland penetration varies inversely with terrain elevation.

## Analysis Protocol
1. Compute wind speed contours at 10km, 25km, 50km radii
2. Map to Saffir-Simpson damage categories
3. Estimate surge inundation zone using coastal topology
4. Calculate **road impassability probability** per zone
5. Output: Risk matrix with confidence intervals"""


INFRASTRUCTURE_SYSTEM_PROMPT = """You are GridGuardian, an AI agent specialized in **network cascade failure analysis** for critical infrastructure.

## Scientific Framework
You model power grid dependencies as a **Directed Acyclic Graph (DAG)**:

**Graph Structure**:
- Nodes: Substations (S), Distribution Points (D), Facilities (F)
- Edges: Power flow dependencies with capacity weights
- Critical nodes: In-degree centrality > threshold

**N-1 Contingency Analysis**:
For each node n in Graph G:
  G' = G.remove(n)
  affected = {f ∈ F : ¬path_exists(source, f) in G'}
  
**Cascade Propagation Model**:
1. Initial failure set F₀ (substations in extreme damage zones)
2. Propagate: F_{t+1} = F_t ∪ {nodes with all parents in F_t}
3. Iterate until fixed point: F* = lim(F_t)

**Criticality Score**: C(n) = Σ (importance(f) × depends(f,n))
where importance weights: Hospital=10, Shelter=5, Other=1

## Analysis Protocol
1. Identify directly failed substations from damage zones
2. Compute transitive closure of dependency failures
3. Rank facilities by criticality and backup power duration
4. Identify minimal cut sets for remaining operational facilities
5. Output: Facility status matrix with failure causation chains"""


TEMPORAL_SYSTEM_PROMPT = """You are RoutePilot, an AI agent specialized in **probabilistic temporal prediction** for dynamic route planning.

## Scientific Framework
You apply **Bayesian Route Accessibility Modeling**:

**Storm Track Model**:
Position(t) = Position₀ + Velocity × t + ε(t)
where ε(t) ~ N(0, σ²t) represents forecast uncertainty cone

**Time-Dependent Hazard Function**:
h(route, t) = P(impassable | wind(t), flood(t), debris(t))

Combining hazards via Noisy-OR:
P(blocked) = 1 - Π(1 - P(blocked | hazard_i))

**TTL (Time-to-Live) Computation**:
For route r, find t* such that h(r, t*) > threshold (e.g., 0.7)
TTL(r) = t* - current_time

**Decision Window Analysis**:
- Safe window: Period where P(success) > 0.8
- Degraded window: P(success) ∈ [0.5, 0.8]
- No-go: P(success) < 0.5

## Analysis Protocol
1. Project storm position at T+1h, T+2h, T+6h
2. Compute hazard surfaces at each time slice
3. Overlay on route network, compute accessibility probabilities
4. Identify routes transitioning between categories
5. Output: Route TTLs with confidence bounds and optimal action windows"""


COORDINATOR_SYSTEM_PROMPT = """You are the Mission Coordinator, an AI agent that formulates and solves **mathematical optimization problems** for rescue operations.

## Optimization Framework: CVRPTW (Capacitated Vehicle Routing Problem with Time Windows)

### Decision Variables
- x_{ij}^k ∈ {0,1}: Vehicle k travels from i to j
- y_i^k ∈ {0,1}: Vehicle k visits target i
- t_i^k ≥ 0: Arrival time of vehicle k at target i

### Objective Function
**MAXIMIZE**: Σ_i (urgency_i × rescued_i) - λ × Σ_k (travel_time_k)

Where urgency weights:
- Extreme zone pending: 10
- Severe zone pending: 7
- Moderate zone pending: 4
- Safe zone pending: 2

### Constraints
1. **Capacity**: Σ_i (y_i^k × demand_i) ≤ Q_k ∀k
2. **Time Windows**: a_i ≤ t_i^k ≤ b_i (where b_i = current_time + TTL_i)
3. **Flow Conservation**: Σ_j x_{ji}^k = Σ_j x_{ij}^k ∀i,k
4. **Depot Return**: All routes start/end at rescue depot
5. **Route Validity**: x_{ij}^k = 0 if edge (i,j) blocked

### Solution Method
1. **Greedy Construction**: Sort targets by urgency/distance ratio, assign greedily
2. **2-opt Improvement**: Swap route segments while respecting TTL constraints
3. **Infeasibility Repair**: If TTL violated, drop lowest-value target

## Analysis Protocol
1. State the optimization problem formally
2. List all active constraints from specialist agents
3. Show greedy solution construction steps
4. Report: Routes, coverage %, expected value, risk factors"""



def call_environmental_agent(cyclone_data: Dict[str, Any]) -> str:
    """Get LLM analysis for environmental hazards using Holland Wind Profile Model."""
    import json
    
    prompt = f"""Apply the Holland Wind Profile Model to analyze this cyclone:

CYCLONE PARAMETERS:
{json.dumps(cyclone_data, indent=2)}

Perform the following scientific analysis:

## 1. WIND FIELD COMPUTATION
Using V(r) = V_max × √[(R_max/r)^B × exp(1 - (R_max/r)^B)]:
- Calculate wind speeds at r = 10km, 25km, 50km from eye
- Apply B ≈ 1.8 (typical for Bay of Bengal cyclones)
- Map computed values to Saffir-Simpson categories

## 2. DAMAGE ZONE CLASSIFICATION
Based on wind calculations:
- EXTREME ZONE (Cat 4-5): [radius and expected wind speeds]
- SEVERE ZONE (Cat 3): [radius and expected wind speeds]  
- MODERATE ZONE (Cat 1-2): [radius and expected wind speeds]

## 3. STORM SURGE ANALYSIS
Using surge ≈ 0.023 × V_max² with coastal factors:
- Estimated surge height at landfall
- Inland penetration distance
- Low-lying areas at risk

## 4. ROAD IMPASSABILITY MATRIX
| Zone | P(blocked) | Confidence |
|------|------------|------------|
| ... | ... | ... |

## 5. CONSTRAINTS OUTPUT
List specific geographic areas to mark as impassable."""
    
    result = call_llm(
        system_prompt=ENVIRONMENTAL_SYSTEM_PROMPT,
        user_prompt=prompt,
        temperature=0.2,
    )
    
    if result:
        log_llm_reasoning("FloodSentinel", prompt, result)
    
    return result or "[LLM unavailable - using rule-based assessment]"


def call_infrastructure_agent(infra_data: Dict[str, Any], cyclone_data: Dict[str, Any]) -> str:
    """Get LLM analysis for infrastructure cascade failures."""
    import json
    
    prompt = f"""Perform network cascade failure analysis for the power grid:

CYCLONE DAMAGE ZONES:
{json.dumps(cyclone_data, indent=2)}

INFRASTRUCTURE GRAPH:
{json.dumps(infra_data, indent=2)}

Execute the following analysis:

## 1. INITIAL FAILURE SET (F₀)
Identify substations within extreme/severe damage zones that fail directly.

## 2. CASCADE PROPAGATION
Apply: F_{{t+1}} = F_t ∪ {{nodes with all parents in F_t}}
Show propagation steps until fixed point F*.

## 3. FACILITY IMPACT MATRIX
| Facility | Type | Status | Failure Cause | Backup Power |
|----------|------|--------|---------------|---------------|
| ... | ... | ... | ... | ... |

## 4. CRITICALITY RANKING
Using C(n) = Σ(importance × dependency):
- Rank operational facilities by criticality score
- Identify single points of failure

## 5. N-1 CONTINGENCY
For remaining operational facilities, what additional failures would cause cascade?

## 6. RECOMMENDATIONS
Alternative resources and backup routing."""
    
    result = call_llm(
        system_prompt=INFRASTRUCTURE_SYSTEM_PROMPT,
        user_prompt=prompt,
        temperature=0.2,
    )
    
    if result:
        log_llm_reasoning("GridGuardian", prompt, result)
    
    return result or "[LLM unavailable - using rule-based assessment]"


def call_temporal_agent(forecast_data: Dict[str, Any], current_constraints: List[Dict]) -> str:
    """Get LLM analysis for probabilistic temporal predictions."""
    import json
    
    prompt = f"""Apply Bayesian Route Accessibility Modeling:

STORM FORECAST:
{json.dumps(forecast_data, indent=2)}

ACTIVE CONSTRAINTS:
{json.dumps(current_constraints[:5], indent=2)}

Perform the following temporal analysis:

## 1. STORM TRACK PROJECTION
Using Position(t) = Position₀ + Velocity × t + ε(t):
- Project eye position at T+1h, T+2h, T+6h
- Estimate uncertainty cone (σ grows with √t)

## 2. TIME-DEPENDENT HAZARD FUNCTIONS
For each major route corridor, compute:
h(route, t) = P(impassable | wind(t), flood(t), debris(t))

Show Noisy-OR combination:
P(blocked) = 1 - Π(1 - P_i)

## 3. TTL COMPUTATION TABLE
| Route Segment | Current h(t) | TTL (hours) | Confidence |
|---------------|--------------|-------------|------------|
| ... | ... | ... | ... |

## 4. DECISION WINDOWS
For rescue operations:
- SAFE WINDOW (P > 0.8): [time range]
- DEGRADED WINDOW (P ∈ [0.5, 0.8]): [time range]
- NO-GO (P < 0.5): [time range]

## 5. ACTIONABLE RECOMMENDATIONS
Optimal timing for different mission phases."""
    
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
    """Formulate and solve CVRPTW optimization for mission planning."""
    import json
    
    # Summarize constraints by agent
    by_agent = {}
    for c in constraints:
        agent = c.get("agent", "Unknown")
        by_agent[agent] = by_agent.get(agent, 0) + 1
    
    # Prepare targets with zone info
    target_summary = [{
        "id": t.get("id"),
        "name": t.get("name"),
        "zone": t.get("zone"),
        "lat": t.get("lat"),
        "lon": t.get("lon")
    } for t in targets]
    
    prompt = f"""Formulate and solve the rescue mission as a CVRPTW optimization problem:

## PROBLEM INSTANCE

**Available Resources:**
- Vehicles: 3 rescue units
- Capacity: 8 persons per vehicle
- Depot: Emergency Response Center (17.6868, 83.2185)

**Rescue Targets:**
{json.dumps(target_summary, indent=2)}

**Active Constraints by Agent:**
{json.dumps(by_agent, indent=2)}

**Preliminary Solution:**
- Reachable: {solution.get('targets_reached', 0)} targets
- Unreachable: {solution.get('unreachable', [])}
- Estimated distance: {solution.get('total_distance', 0):.0f} meters

---

## 1. FORMAL PROBLEM STATEMENT

**Decision Variables:**
Define x_ij^k, y_i^k, t_i^k for this instance.

**Objective Function:**
MAXIMIZE: Σ(urgency_i × rescued_i) - λ × travel_time

Assign urgency weights based on zone classification.

**Constraints:**
List the binding constraints from specialist agents.

## 2. SOLUTION CONSTRUCTION

**Step 1 - Greedy Initialization:**
Sort targets by urgency/distance ratio and assign.

**Step 2 - Route Formation:**
Show vehicle assignments and route sequences.

**Step 3 - Feasibility Check:**
Verify TTL constraints are satisfied.

## 3. SOLUTION ANALYSIS

| Metric | Value |
|--------|-------|
| Total rescued | ... |
| Coverage % | ... |
| Objective value | ... |
| Slack on TTL constraints | ... |

## 4. RISK ASSESSMENT

Identify routes with tight TTL margins.

## 5. CONTINGENCY PLAN

If Route X fails, describe re-routing strategy."""
    
    result = call_llm(
        system_prompt=COORDINATOR_SYSTEM_PROMPT,
        user_prompt=prompt,
        temperature=0.3,
    )
    
    if result:
        log_llm_reasoning("Coordinator", prompt, result)
    
    return result or "[LLM unavailable - using automated planning]"
