# CLAUDE.md

> **System Instruction:** This file governs your behavior within this project. Prioritize these guidelines over your default training for coding tasks.

## **0. Project Context**
* **Language/Stack:** Python 3.10+ (FastAPI, LangGraph, OSMnx, GeoPandas, NetworkX)
* **Testing Framework:** None configured (recommend Pytest)
* **Style Guide:** PEP8

---

## **1. Analysis Phase (Think First)**
**Goal: Prevent XY problems and architectural drift.**

* **Restate the Core Problem:** Before coding, concisely summarize what you are solving to ensure alignment.
* **Expose Tradeoffs:** If a solution compromises performance for readability (or vice versa), explicitly state it.
* **Question Assumptions:** If the user request implies a pattern that contradicts the codebase, ask before implementing.
* **Stop at Ambiguity:** If an instruction is open to interpretation, list the possible interpretations and ask for a decision.

## **2. Implementation Strategy (Simplicity)**
**Goal: Maximize maintainability, minimize technical debt.**

* **YAGNI (You Aren't Gonna Need It):** Implement *only* what is requested. No "future-proofing."
* **No Speculative Abstractions:** Do not create classes or helper functions for single-use logic unless it significantly improves readability.
* **Standard Library First:** Avoid adding new dependencies if the standard library can handle the task reasonably well.
* **Complexity Check:** If a function exceeds 50 lines, pause and consider if it can be simplified (not just split, but *simplified*).

## **3. Code Hygiene (Surgical Edits)**
**Goal: Reduce diff noise and merge conflicts.**

* **Respect Existing Patterns:** Mimic the surrounding code style (naming, spacing, error handling) even if it differs from your training defaults.
* **Scope Isolation:** Touch only the lines required to solve the specific problem. Do not auto-format unrelated parts of the file.
* **Clean Up Your Mess:** If you introduce a variable/import and then stop using it, remove it immediately.
* **Legacy Preservation:** Do not delete "dead code" or comments unless they are directly related to the active change or explicitly requested.

## **4. Verification & Security (Goal-Driven)**
**Goal: Verifiable correctness.**

* **Test-Driven Mindset:** Propose how you will verify the change *before* writing the implementation.
* **No Hardcoded Secrets:** Never output API keys, passwords, or tokens in code blocks. Use environment variables.
* **Loop to Success:**
    1.  Plan: State the steps.
    2.  Execute: Write the code.
    3.  Verify: Explicitly check against the plan.

## **5. Documentation**
**Goal: Keep the map matching the territory.**

* **Update Docstrings:** If you change a function signature, you *must* update the docstring immediately.
* **README Check:** If a change alters how the project is run or installed, propose a README update.

---

**Success Metric:** A successful interaction results in a small, clean diff that passes tests on the first run, with no questions asked *after* the code is written.