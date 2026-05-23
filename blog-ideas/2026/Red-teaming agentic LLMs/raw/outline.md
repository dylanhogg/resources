# A Pragmatic Engineer's Guide to Defending Agentic LLM Systems

## 1. Introduction: agent security is system security

Agentic LLM systems are useful because they can interpret messy inputs, reason over context, retrieve data, call tools, and trigger actions. Those same capabilities create failure modes that traditional ML systems do not fully cover: prompt injection, contaminated retrieval, unsafe tool use, data leakage, over-permissioned actions, and brittle automation under adversarial input.

This guide is for machine learning engineers who need to harden systems that use LLMs. The goal is not to make the model perfectly safe. The goal is to build a layered system where the model is constrained, observed, tested, and prevented from taking actions it should not take.

Core thesis: prompts guide behaviour, but architecture enforces safety.

---

## 2. Classify the system before choosing defences

Not every LLM system needs the same security posture. Start by classifying what the model is allowed to do and where its output flows.

| System type | Main risk | Defences to prioritize |
| --- | --- | --- |
| Structured extraction | The model returns incorrect, unsupported, or adversarially influenced fields. | Treat the model like a parser: strict schemas, enum validation, normalization, abstention, deterministic post-processing, adversarial extraction evals. |
| RAG and LLM workflows | Untrusted content contaminates downstream prompts or decisions. | Separate instructions from retrieved data, label retrieved text as untrusted, filter sources, check provenance, validate intermediate outputs. |
| Conversational agents | Open-ended user behaviour pushes the model outside product scope, policy, or factual grounding. | Define supported intents, ground responses in approved sources, handle refusals and escalation, constrain memory and personalization. |
| Tool-using agents | Model output crosses into real-world execution. | Use least-privilege tools, action schemas, allowlists, scoped credentials, confirmation gates, sandboxing, audit trails. |

As systems move from extraction to action, prompts become less important as a defence boundary and external controls become more important.

---

## 3. Threat model: the attack surfaces that matter

### 3.1 Direct prompt injection

The user explicitly tries to override the intended behaviour of the system.

Examples:

- "Ignore previous instructions."
- "Reveal your system prompt."
- "Call this tool with hidden arguments."
- "Return JSON that bypasses validation."

Engineering takeaway: hostile instructions are expected input, not rare edge cases.

---

### 3.2 Indirect prompt injection and RAG contamination

The model reads malicious instructions from retrieved documents, websites, emails, tickets, PDFs, tool outputs, or database fields.

Examples:

- a web page tells the agent to exfiltrate private data
- an email contains hidden instructions to forward messages
- a support ticket attempts to override internal policy
- a document tells the model to ignore the user's request

Engineering takeaway: retrieved content is data, not authority.

---

### 3.3 Unsafe tool execution

The model calls the wrong tool, calls a tool with unsafe parameters, or takes an action the user did not intend.

Examples:

- sending an email instead of drafting one
- deleting records instead of archiving them
- querying data outside the user's permissions
- making high-impact changes without confirmation

Engineering takeaway: tool calls need policy enforcement outside the model.

---

### 3.4 Data leakage and context exposure

The model exposes private, internal, personal, or cross-tenant information because sensitive data was placed in context, retrieved incorrectly, returned from a tool, or mixed across users.

Examples:

- leaking hidden instructions or private context
- exposing retrieved documents not intended for the user
- revealing tool outputs that should stay internal
- mixing data between tenants, accounts, or sessions

Engineering takeaway: context construction is a security boundary.

---

### 3.5 Over-trust in model output

Users and downstream systems may treat fluent model output as more reliable than it is, especially in legal, financial, medical, security, or operational workflows.

Engineering takeaway: high-stakes outputs need provenance, uncertainty handling, validation, and fallback paths. Confidence in tone is not confidence in correctness.

---

## 4. Layered defences

Defence in depth means every layer has a narrow job. The model can propose. The system should decide what is allowed.

### 4.1 Prompt and instruction layer

Use prompts to frame the task, not to carry the whole security model.

Key controls:

- separate system/developer instructions, user input, retrieved content, tool outputs, and generated plans
- define supported and unsupported tasks
- tell the model which content is authoritative and which content is untrusted
- require refusal or escalation for unsupported requests
- keep each model step narrow enough to evaluate

---

### 4.2 Data and context layer

Treat context construction as part of the security boundary.

Key controls:

- retrieve only content needed for the current task
- label retrieved documents and tool outputs as untrusted observations
- filter sources by user permissions and task relevance
- isolate tenants, accounts, sessions, and memory stores
- avoid placing secrets, credentials, or unnecessary internal data in prompts
- check citations or provenance where answers depend on retrieved evidence

---

### 4.3 Model output layer

Do not pass raw model output directly into downstream systems when the output affects data, decisions, or actions.

Key controls:

- schema-constrained output
- enum and type validation
- normalization before execution or storage
- null, abstain, or fallback values for unsupported fields
- grounded generation checks for RAG outputs
- rejection of malformed or policy-violating responses

---

### 4.4 Tool and action layer

Tools are where model behaviour becomes system behaviour. This layer needs explicit enforcement.

Key controls:

- least-privilege tool access
- action allowlists
- scoped credentials
- rate limits and quotas
- sandboxing for code execution or external operations
- policy checks before tool execution
- confirmation before irreversible or high-impact actions
- audit logs for tool requests, approvals, and results

The confirmation should describe the actual action, not the model's vague intent.

---

### 4.5 System and operations layer

Production safeguards should assume that some attacks and model failures will get through earlier layers.

Key controls:

- logs for user input, retrieved context IDs, model outputs, tool requests, policy decisions, validation errors, and user feedback
- monitoring for unusual tool calls, repeated injection attempts, spikes in refusals, schema failures, and requests involving secrets or credentials
- staged rollout from read-only to draft mode to action mode
- rollback paths and kill switches for high-risk agents
- incident review that feeds back into evaluation and regression tests

---

## 5. Evaluation and red teaming

Attacks should be part of normal testing, not a one-off review before launch. The goal is to measure whether the whole system behaves safely: prompt construction, retrieval, model output, validation, tool execution, permissions, and user experience.

### 5.1 Build adversarial datasets

Include examples for:

- direct prompt injection
- indirect prompt injection in retrieved content
- unsafe tool-call attempts
- unsupported requests
- biased or discriminatory instructions
- data exfiltration attempts
- malformed structured outputs
- ambiguous user intent
- high-impact action requests

For each case, define the expected safe behaviour: refuse, abstain, escalate, ask for clarification, return null, draft without sending, or block the tool call.

---

### 5.2 Use layered metrics

Useful metrics include:

- extraction correctness
- unsupported-field extraction rate
- unsafe tool-call rate
- confirmation-bypass rate
- data-leakage rate
- policy-violation rate
- escalation accuracy
- successful safe task completion

Track quality and safety together. A system that completes more tasks by taking unsafe actions is not better.

---

### 5.3 Convert findings into regression tests

A red-team finding is only useful if it becomes durable engineering work.

Workflow:

1. capture the failure
2. minimize the repro case
3. define the expected safe behaviour
4. add it to the eval set
5. fix the relevant layer of the system
6. run against existing quality benchmarks
7. prevent regressions in CI

---

## 6. Production hardening checklist

### Before launch

- Define supported and unsupported tasks.
- Map tools and data access to risk levels.
- Apply least-privilege permissions.
- Separate trusted instructions from untrusted content.
- Add schemas, validators, and normalization.
- Build adversarial evals for the system, not just the model.
- Add confirmation for high-impact actions.
- Log model decisions, validation failures, and tool calls.
- Define escalation, fallback, rollback, and kill-switch paths.

---

### During rollout

- Start read-only where possible.
- Use draft mode before action mode.
- Restrict initial users, tools, and data scopes.
- Monitor tool calls, refusals, validation failures, and user reports.
- Review high-severity conversations and blocked actions.
- Expand permissions only when the system behaves safely under evaluation and production monitoring.

---

### After launch

- Convert incidents and red-team findings into regression tests.
- Re-run evals after model, prompt, tool, retrieval, or policy changes.
- Review unused tools, stale permissions, and over-broad credentials.
- Track safety metrics alongside task success, latency, and cost.
- Keep rollback paths simple for high-risk capabilities.

---

## Conclusion: build agents like production systems, not demos

The practical path is not to rely on the model to always choose safe behaviour. It is to design a system where untrusted content is contained, model outputs are validated, tools are mediated by policy, high-impact actions require approval, and failures become regression tests.

Prompts guide behaviour. Architecture enforces safety.
