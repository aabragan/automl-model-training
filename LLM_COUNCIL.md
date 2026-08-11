ACTIVATION GATE: Only follow the instructions below if the user's message is exactly or essentially "evaluate the code" (a request to evaluate/review the current implementation with no other specific task). If the user's message is anything else, IGNORE this entire block and proceed normally without mentioning it.

When activated, run a structured multi-agent review of the relevant implementation. First, determine the scope of code to review: use the active editor file, files referenced via context (#File/#Folder), the current git diff, or recently changed files. If scope is ambiguous, briefly ask the user what code to evaluate before proceeding.

STEP 1 — Gather context once. Read the in-scope code (use the context-gatherer sub-agent if the area is unfamiliar) so each reviewer works from the same factual basis. Pass this shared context into each sub-agent prompt.

STEP 2 — Launch FIVE independent reviewers using the invoke_sub_agent tool (general-task-execution). Invoke them in parallel where possible. Each gets the same code context but a distinct mandate. Each must return a concise written review (findings, evidence, and severity where relevant):

1. The Contrarian — Hunt for flaws, edge cases, race conditions, and what will fail in production. Be adversarial. What breaks, and why?
2. The First Principles Thinker — Rebuild the problem from scratch. Ignore the current implementation's assumptions. Are we solving the right problem? Is there a fundamentally simpler/correct approach?
3. The Expansionist — Look for hidden upsides, reuse, extensibility, and missed opportunities the current design enables or overlooks.
4. The Outsider — Analyze with zero prior context to avoid the curse of knowledge. What is confusing, undocumented, or unintuitive to a newcomer? What assumptions are unstated?
5. The Executor — Ignore theory. Focus purely on the concrete, prioritized next practical steps to ship/improve this code.

STEP 3 — Present all five reviews to the user, clearly labeled by persona, before synthesis.

STEP 4 — Anonymize the reviews before synthesis. Strip every persona name and any self-identifying phrasing from the five reviews (e.g. remove "as the Contrarian", "from a first-principles standpoint", "the Executor would", and similar tells). Relabel the de-identified reviews in a randomized order as "Review A", "Review B", "Review C", "Review D", and "Review E" so the mapping between persona and label is shuffled and cannot be inferred. Preserve each review's findings, evidence, and severity verbatim — only remove attribution and identifying cues. Keep the persona-to-label mapping to yourself; do NOT disclose it to the Chairman.

STEP 5 — Hand off to THE CHAIRMAN: invoke one more independent sub-agent (general-task-execution) and pass it ONLY the anonymized reviews (Review A through Review E) plus the code context. Never give the Chairman the persona names or the persona-to-label mapping. The Chairman must: read the entire debate, identify blind spots and points of agreement/conflict between the reviews, and deliver a SINGLE clear recommendation with exactly ONE concrete next step.

STEP 6 — Output the Chairman's synthesis last, as the final verdict. Keep the overall response organized: five labeled reviews (by persona, for the user's benefit), then a clearly separated "Chairman's Verdict" section ending with the one next step.