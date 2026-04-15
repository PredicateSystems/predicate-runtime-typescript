# PlannerExecutorAgent: Deferred Features

**Date:** 2026-04-13
**Status:** Documentation for post-MVP implementation

## Overview

This document outlines features from the Python `PlannerExecutorAgent` that were deferred from the TypeScript MVP port. These features add reliability and flexibility but are not required for basic browser automation tasks.

## MVP Implementation Summary

The TypeScript MVP includes:

1. **Core Agent (~600 lines)**
   - Stepwise (ReAct-style) planning loop
   - Action parsing (CLICK, TYPE, SCROLL, PRESS, DONE)
   - Compact context formatting for small models
   - Token usage tracking by role and model

2. **Reliability Features (~200 lines)**
   - Snapshot escalation (progressive limit increase)
   - Pre-action authorization hook (for sidecar policy)
   - Basic error handling and retry

3. **Configuration (~250 lines)**
   - `PlannerExecutorConfig` with presets
   - `SnapshotEscalationConfig`, `RetryConfig`, `StepwisePlanningConfig`
   - Factory helpers for provider creation

## Deferred Features

### 1. Modal/Overlay Dismissal

**Python Reference:** `ModalDismissalConfig`, `_attempt_modal_dismissal()`

**Description:** Automatically dismiss blocking overlays after DOM changes:

- Product protection/warranty upsells
- Cookie consent banners
- Newsletter signup popups
- Promotional overlays
- Cart upsell drawers

**Implementation Effort:** ~150 lines

**Config Interface:**

```typescript
interface ModalDismissalConfig {
  enabled: boolean;
  dismissPatterns: string[]; // e.g., ['close', 'no thanks', 'skip']
  dismissIcons: string[]; // e.g., ['×', '✕', 'x']
  roleFilter: string[]; // e.g., ['button', 'link']
  maxAttempts: number;
  minNewElements: number; // Minimum DOM changes to trigger
}
```

**Key Logic:**

- Detect DOM changes after CLICK actions
- Find buttons matching dismissal patterns (word-boundary matching)
- Click dismissal button and verify modal closed
- Skip if checkout-related buttons are present

---

### 2. Captcha Handling

**Python Reference:** `CaptchaConfig`, `_detect_captcha()`, `_handle_captcha()`

**Description:** Detect and handle CAPTCHAs during automation:

- Policy options: `abort`, `callback`, `pause`
- Support for external solving services
- Detection via element text and patterns

**Implementation Effort:** ~100 lines

**Config Interface:**

```typescript
interface CaptchaConfig {
  enabled: boolean;
  policy: 'abort' | 'callback' | 'pause';
  detectionPatterns: string[];
  solverCallback?: (imageBase64: string) => Promise<string>;
  maxWaitMs: number;
}
```

**Key Logic:**

- Check snapshot elements for CAPTCHA indicators
- Based on policy: abort task, call external solver, or pause for human
- Resume automation after CAPTCHA solved

---

### 3. Vision Fallback

**Python Reference:** `VisionFallbackConfig`, vision_executor, vision_verifier

**Description:** Use vision-capable models when DOM-based automation fails:

- Canvas pages with no accessible elements
- Low element confidence scores
- Complex visual layouts

**Implementation Effort:** ~200 lines

**Config Interface:**

```typescript
interface VisionFallbackConfig {
  enabled: boolean;
  maxVisionCalls: number;
  triggerRequiresVision: boolean;
  triggerCanvasOrLowActionables: boolean;
  canvasDetectionThreshold: number;
  lowActionablesThreshold: number;
}
```

**Key Logic:**

- Detect snapshot failures (low elements, canvas pages)
- Switch to vision executor with screenshot input
- Use vision verifier for state verification
- Fall back gracefully to DOM mode when possible

---

### 4. Intent Heuristics

**Python Reference:** `IntentHeuristics` protocol, `_try_intent_heuristics()`

**Description:** Pluggable domain-specific element selection without LLM:

- E-commerce: "Add to Cart", "Checkout" buttons
- Authentication: login forms, password fields
- Search: search boxes, result links

**Implementation Effort:** ~100 lines

**Interface:**

```typescript
interface IntentHeuristics {
  findElementForIntent(
    intent: string,
    elements: SnapshotElement[],
    url: string,
    goal: string
  ): number | null;

  priorityOrder(): string[];
}

// Example implementation
class EcommerceHeuristics implements IntentHeuristics {
  findElementForIntent(intent, elements, url, goal) {
    if (intent.toLowerCase().includes('add to cart')) {
      const btn = elements.find(el => el.text?.toLowerCase().includes('add to cart'));
      return btn?.id ?? null;
    }
    return null; // Fall back to LLM
  }

  priorityOrder() {
    return ['add_to_cart', 'checkout', 'search'];
  }
}
```

**Key Logic:**

- Check heuristics before calling executor LLM
- Reduces token usage for common patterns
- Improves reliability for known sites

---

### 5. Recovery Navigation

**Python Reference:** `RecoveryNavigationConfig`, `_last_known_good_url`

**Description:** Track and recover from off-track navigation:

- Remember last URL where verification passed
- Navigate back when subsequent steps fail
- Detect when agent is lost

**Implementation Effort:** ~80 lines

**Config Interface:**

```typescript
interface RecoveryNavigationConfig {
  enabled: boolean;
  maxRecoveryAttempts: number;
  trackSuccessfulUrls: boolean;
}
```

**Key Logic:**

- Store URL after successful verification
- On repeated failures, navigate back to last good URL
- Replan from recovered state

---

### 6. Checkout/Auth Boundary Detection

**Python Reference:** `CheckoutDetectionConfig`, `AuthBoundaryConfig`

**Description:** Detect when agent reaches boundaries that require human intervention:

- Checkout pages requiring payment info
- Login/signup pages requiring credentials
- Age verification gates

**Implementation Effort:** ~60 lines

**Config Interface:**

```typescript
interface CheckoutDetectionConfig {
  enabled: boolean;
  urlPatterns: string[]; // e.g., ['/checkout', '/payment']
  elementPatterns: string[]; // e.g., ['credit card', 'payment']
  stopOnDetection: boolean;
}

interface AuthBoundaryConfig {
  enabled: boolean;
  urlPatterns: string[]; // e.g., ['/login', '/signin']
  elementPatterns: string[]; // e.g., ['sign in', 'log in']
  stopOnDetection: boolean;
}
```

---

### 7. Executor Override

**Python Reference:** `ExecutorOverride` protocol

**Description:** Validate or override executor's element choices before action:

- Safety checks (block delete buttons)
- Domain-specific corrections
- Audit logging

**Implementation Effort:** ~50 lines

**Interface:**

```typescript
interface ExecutorOverride {
  validateChoice(
    elementId: number,
    action: string,
    elements: SnapshotElement[],
    goal: string
  ): {
    valid: boolean;
    overrideElementId?: number;
    rejectionReason?: string;
  };
}
```

---

### 8. Upfront Planning Mode

**Python Reference:** `plan()`, `replan()` methods

**Description:** Generate full execution plan upfront (alternative to stepwise):

- Better for known workflows
- Supports plan patching on failure
- More efficient for simple tasks

**Implementation Effort:** ~200 lines

**Key Functions:**

- `plan(task, startUrl)` - Generate full plan
- `replan(task, failedStep, reason)` - Patch plan after failure
- `run(runtime, task)` - Execute with upfront planning

---

### 9. Task Category Pruning

**Python Reference:** `PruningTaskCategory`, `prune_with_recovery()`

**Description:** Category-specific element filtering to reduce context size:

- Shopping: prioritize product/cart elements
- Search: prioritize search box/results
- Auth: prioritize form fields

**Implementation Effort:** ~150 lines

**Categories:**

- `shopping`, `checkout`, `search`, `auth`, `form_filling`, `extraction`, `navigation`

---

## Implementation Priority

Recommended order based on impact and complexity:

1. **Intent Heuristics** - High impact, low complexity, reduces token usage
2. **Modal Dismissal** - Common pain point, medium complexity
3. **Vision Fallback** - Required for canvas/complex pages
4. **Captcha Handling** - Needed for production use
5. **Recovery Navigation** - Improves reliability
6. **Upfront Planning** - Alternative mode for simple tasks
7. **Boundary Detection** - Nice to have for graceful stops
8. **Executor Override** - Nice to have for safety
9. **Task Category Pruning** - Optimization for large pages

## References

- Python implementation: `sdk-python/predicate/agents/planner_executor_agent.py`
- Design doc: `docs/sdk-ts-doc/2026-03-28_planner_executor_agent_port.md`
- Chrome extension feasibility: `docs/sdk-python-doc/2026-04-13_predicate_chrome_extension_agent_feasibility.md`
