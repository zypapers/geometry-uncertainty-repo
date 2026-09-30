# Prompt classification task — instructions

Thank you for helping with this annotation task. You will classify **200 short English prompts** into one of **three categories**, using only the definitions below. Expect roughly **2-4 hours** total (about a minute per prompt).

## Ground rules

1. **Work independently.** Do not discuss the prompts or your decisions with anyone until you have returned the completed sheet.
2. **Reference lookups are allowed and encouraged.** You may consult Wikipedia, encyclopedias, dictionaries, or other reference sources to check whether a stated fact is true.
3. **Do not use AI assistants** (ChatGPT, Claude, Gemini, Copilot, or any similar tool) for any part of this task, including fact checking. This study concerns how AI models handle these prompts, so AI involvement would contaminate the annotation. Use conventional reference sources only.
4. **Classify every prompt** (forced choice). If you are genuinely torn, still pick the best fit and set `unsure = 1`; you may add a brief note in `comment`.
5. Base every decision on the **prompt text and the definitions below** — nothing else.

## The three categories

### 1. DF — Determinate-factual

The prompt asks for a **verifiable world fact with a determinate answer**. It contains no false presupposition and admits one clear reading.

Notes:
- Obscurity does not matter. Some prompts ask very common facts, others extremely specialized or obscure ones (rare identifiers, technical nomenclature, recent events). **If a determinate answer exists in principle, it is DF** — whether you (or anyone) happen to know the answer is irrelevant.
- A question that *asks whether* something is true (for example, *"Did Shakespeare write the Iliad?"*) is DF: the possibly-false statement is the **object of inquiry**, not a hidden assumption. Compare category FP below.

### 2. AMB — Ambiguous

The prompt **admits multiple valid readings**. From the published definition:

> A prompt that admits multiple valid readings, either referential ("What did Jordan say?", where Jordan could be Michael Jordan, the country, or a person named Jordan), scope ("Three boys lifted the piano", either together as one event or each individually), temporal ("Did you eat?", taken as recently or today or ever), or genuine multi-meaning ("The bank is closing").

The test: could two reasonable readers, with the same knowledge, legitimately take the prompt to be asking **different questions**?

### 3. FP — False-premise

The prompt **embeds a false statement as a presupposition** — something assumed in passing rather than asked about. From the published definition:

> A prompt that embeds a false premise declaratively. Example: "How did Shakespeare's writing of the Iliad influence later epic poetry?" (not "Did Shakespeare write the Iliad?").
>
> Three rules keep this category sharp. First, the false statement must be **presupposed**, not asserted as the object of inquiry: "Did Shakespeare write the Iliad?" is a factual query (DF), not FP. Second, the answer space must remain coherent **only if the false premise is accepted**: "How did Shakespeare's writing of the Iliad influence later epic poetry?" has no defensible answer except correcting the premise. Third, the falsehood must concern **verifiable world fact**, not contested interpretation or hypothetical reasoning. Counterfactual prompts ("If Shakespeare had written the Iliad...") are excluded by construction.

Use reference sources to verify whether an embedded premise is actually false. If a prompt presupposes something and that something is **true**, the prompt is not FP.

## Suggested decision procedure per prompt

1. Does the prompt **assume** (not ask) something you can check — and is that assumption **false**? → **FP**
2. Does the prompt admit **multiple valid readings** (referential / scope / temporal / multi-meaning)? → **AMB**
3. Otherwise → **DF**

## How to fill the sheet

Open `annotation_sheet.csv`. For each row, fill:

| column | value |
|---|---|
| `label` | `DF`, `AMB`, or `FP` |
| `unsure` | `0` (confident) or `1` (torn between categories) |
| `comment` | optional; brief note on hard cases |

Please return the completed CSV when done. Every row must have a `label` and an `unsure` value.
