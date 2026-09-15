ACTIVATION GATE: Only follow the instructions below if the user's message is exactly or essentially "execute document review" (a request to review the project's documents, architecture, and diagrams with no other specific task). If the user's message is anything else, IGNORE this entire block and proceed normally without mentioning it.

When activated, run a structured multi-agent review of the project's documentation, architecture, and diagram set — applying the same adversarial scrutiny a serious code review would receive.

SCOPE — Full review of all documents, architecture, and diagrams in the workspace. In-scope formats: markdown, image files (png, jpg, jpeg, gif, bmp, tiff, webp, svg), drawio, pdf, docx, doc, xlsx, xls, pptx, ppt. Default scope is every such artifact under the workspace, prioritizing `generated/` and `project-doc/`. Narrow only if the user names a subset via context (#File/#Folder) or the active editor file. If the set is very large, present the manifest from STEP 0 and confirm scope before reading everything.

STEP 0 — Inventory before reading. Enumerate every in-scope artifact and present a manifest table: path, format, apparent purpose, and whether it is readable. Do not skip this step; the manifest is what makes gaps visible. Flag anything that looks stale, orphaned, or duplicated at this stage.

STEP 1 — Gather context once, using the correct tool per format. Each reviewer must work from the same factual basis, so extract content before launching any sub-agent:

- markdown, plain text, drawio XML, mermaid source → `read_file`
- pdf, docx, doc, xlsx, xls, pptx, ppt → `read_document` (document-loader MCP)
- images (png, jpg, jpeg, gif, bmp, tiff, webp) → `read_image` (document-loader MCP)
- pptx, ppt, or pdf where visual layout and diagram content matter → `extract_slides_as_images`, then `read_image` on each rendered slide
- Mermaid diagrams embedded in markdown → read the fenced source directly and reason over nodes and edges
- `.drawio` files → read the XML and enumerate actual shapes, labels, and connections

Never review an artifact you could not actually read, and never infer a diagram's content from its filename. Record every unreadable or unparsed artifact explicitly as a gap and carry it into the review as a finding.

Use the context-gatherer sub-agent if the subject domain or codebase context behind the documents is unfamiliar. Assemble the manifest plus extracted content summaries into a single shared context block, and pass that identical block into every sub-agent prompt.

RUN DIRECTORY — Before launching reviewers, establish the directory for this review run so repeated runs never overwrite each other. Reviews are stored under `generated/document-review/reviews/<YYYY-MM-DD>/<revision>/`, where `<YYYY-MM-DD>` is today's date and `<revision>` is a zero-padded run counter (`01`, `02`, ...). Determine `<revision>` by listing `generated/document-review/reviews/<YYYY-MM-DD>/`: if it does not exist, use `01`; otherwise use the next integer after the highest existing revision folder for today. Within the run directory, raw reviews go in `raw/` and anonymized reviews go in `anonymized/`. Create the `raw/` and `anonymized/` subfolders before writing. Refer to this run directory as `<RUN_DIR>` in the steps below.

STEP 2 — Launch SIX independent reviewers using the invoke_sub_agent tool (general-task-execution). Invoke them in parallel where possible. Each gets the same shared artifact context but a distinct mandate. Each must produce a concise written review with findings, supporting evidence, and severity. Evidence must cite the specific artifact path plus a section heading, page number, slide number, or diagram element — not a general impression:

1. The Contrarian — Be adversarial. Where is this documentation wrong, misleading, or going to fail on contact with implementation? Hunt for hand-waved complexity, unstated risk, claims with no supporting evidence, architecture diagrams that omit failure paths and error handling, capacity or performance assertions with no numbers behind them, and decisions presented as settled that are actually unresolved.

2. The First Principles Thinker — Rebuild the problem from scratch. Ignore the existing document structure and the architecture as drawn. Is the right problem framed? Is each architectural choice justified on its merits, or inherited by default and then documented after the fact? Is there a fundamentally simpler design that satisfies the same requirements?

3. The Expansionist — Look for hidden upside: reuse, extensibility, adjacent use cases, and opportunities the current design enables but the documents never claim. Also name what the design forecloses that it should not.

4. The Outsider — Analyze with zero prior context to defeat the curse of knowledge. What is confusing, undefined, or unnavigable to a newcomer? Undefined acronyms and terms, unlabeled arrows, diagrams with no legend, boxes whose responsibility is never stated, missing narrative connecting one document to the next, and assumptions the authors never wrote down. Can a reader follow the diagrams without the author in the room?

5. The Executor — Ignore theory entirely. Give the concrete, prioritized next steps required to bring this document and diagram set to ship quality, ordered by impact.

6. The Consistency Auditor — Own cross-artifact integrity, the failure mode unique to a mixed document set. Check for: contradictions between diagrams and prose; components that appear in a diagram but are described nowhere in text, and vice versa; requirements traceability in both directions (every requirement traced forward to a design element, every design element traced back to a requirement); template conformity against the project's expected structure; whether requirements are specific and measurable rather than vague; presence and correct use of formal requirement IDs; stale, orphaned, or duplicated artifacts; and terminology, naming, or version drift across formats — for example a component named one way in markdown, another in the drawio, and a third on a slide.

As each reviewer returns, write its review verbatim to disk at `<RUN_DIR>/raw/<persona>.md`, using these fixed filenames: `contrarian.md`, `first-principles-thinker.md`, `expansionist.md`, `outsider.md`, `executor.md`, `consistency-auditor.md`. Begin each file with a heading naming the persona, followed by the review's findings, evidence, and severity. Do not proceed to STEP 4 until all six raw review files exist on disk.

STEP 3 — Present all six reviews to the user, clearly labeled by persona, before synthesis. Read them back from the `<RUN_DIR>/raw/` files so the presented text matches what was persisted.

STEP 4 — Anonymize the reviews before synthesis, working from the on-disk raw files. Read each file in `<RUN_DIR>/raw/`, then strip every persona name and any self-identifying phrasing (e.g. remove "as the Contrarian", "from a first-principles standpoint", "the Executor would", "as the Consistency Auditor", and similar tells). Assign the six reviews to the labels "Review A" through "Review F" in a randomized order so the mapping between persona and label is shuffled and cannot be inferred. Write each anonymized review to `<RUN_DIR>/anonymized/<label>.md` (e.g. `review-a.md` through `review-f.md`), where the file contains only the de-identified text under its "Review X" heading. Preserve each review's findings, evidence, and severity verbatim — only remove attribution and identifying cues.

Record the persona-to-label mapping in an audit trail file at `<RUN_DIR>/mapping.md`, listing each persona alongside the label it was assigned (e.g. `Contrarian → Review D`). This file is the only place the mapping is persisted. It exists for the user's audit purposes and MUST NOT be passed to the Chairman or referenced in the Chairman's context.

STEP 5 — Hand off to THE CHAIRMAN: invoke one more independent sub-agent (general-task-execution) and pass it ONLY the anonymized reviews read from `<RUN_DIR>/anonymized/` (Review A through Review F) plus the shared artifact context. Never give the Chairman the persona names, the raw review files, or the `mapping.md` audit file. The Chairman must: read the entire debate, identify blind spots and points of agreement and conflict between the reviews, resolve contradictions between reviewers rather than listing them, and deliver a SINGLE clear recommendation with exactly ONE concrete next step.

STEP 6 — Output the Chairman's synthesis last, as the final verdict. Keep the overall response organized in this order: the artifact manifest (including any unreadable artifacts), the six labeled reviews (by persona, for the user's benefit), then a clearly separated "Chairman's Verdict" section ending with the one next step. Note the `<RUN_DIR>` path so the user can find the persisted raw reviews, anonymized reviews, and the `mapping.md` audit trail.

CONSTRAINT — Review only what the artifacts actually say. Do not invent requirements, fill gaps with assumptions, or credit the documents with content they do not contain. A missing section is a finding, not something to imagine into place. Where compliance or regulatory frameworks are mentioned in the documents, review whether the stated technical controls are documented clearly and consistently; do not assess whether they satisfy any legal or regulatory obligation.
