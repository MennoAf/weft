# Single-session preference failure classification

Verdict is `judge.label`; contexts are captured `weft_recall` turn content. Source quotes are lexical matches in answer-designated dataset sessions. See JSON for full answers, sentence-match detail, and run-level costs.

| Question ID | Per-run verdict (main/run1/run2) | Statement found? | Context presence (main/run1/run2) | Stage | Evidence quote |
|---|---|---|---|---|---|
| `09d032c9` | incorrect/incorrect/incorrect | yes | yes/no/yes | **READER** | “I'm looking for some advice on the best way to organize my tech accessories, like my new portable power bank and wireless charging pad, when I'm traveling.” |
| `0a34ad58` | incorrect/incorrect/incorrect | yes | no/no/yes | **RETRIEVAL** | “I have downloaded the TripIt app to stay organized, but I'm still nervous about the trip.” |
| `0edc2aef` | incorrect/incorrect/incorrect | yes | yes/no/yes | **READER** | “Besides great views, I also like hotels with unique features, such as a rooftop pool or a hot tub on the balcony.” |
| `1c0ddc50` | incorrect/incorrect/incorrect | yes | no/yes/no | **RETRIEVAL** | “I'm particularly interested in the history podcasts.” |
| `32260d93` | incorrect/incorrect/incorrect | yes | no/no/no | **RETRIEVAL** | “Can you recommend some stand-up comedy specials on Netflix with strong storytelling abilities like John Mulaney's 'Kid Gorgeous'?” |
| `35a27287` | incorrect/incorrect/incorrect | yes | yes/yes/no | **READER** | “I'm looking for some new language learning resources, specifically podcasts in French that can help me improve my listening skills.” |
| `75832dbd` | incorrect/incorrect/incorrect | yes | no/no/no | **RETRIEVAL** | “I'd like to explore some more research papers and articles on the topic of explainable AI in medical image analysis, specifically on techniques for visualizing and interpreting deep learning models.” |
| `afdc33df` | incorrect/incorrect/incorrect | yes | yes/no/yes | **READER** | “I recently bought a new utensil holder to keep countertops clutter-free.” |
| `d6233ab6` | incorrect/incorrect/incorrect | yes | no/no/no | **RETRIEVAL** | “I still remember the happy high school experiences such as being part of the debate team and taking advanced placement courses in economics.” |

**Dominant stage:** RETRIEVAL ({'STORAGE': 0, 'RETRIEVAL': 5, 'READER': 4, 'JUDGE': 0}).

## Evidence limits

Checkpoint receipts show the source session was marked ingested but do not include a raw post-write database readback. Sentence matching is a lexical heuristic against answer-designated dataset sessions; manually audit weak matches before treating the stage label as ground truth.

## Focused validation (not executed)

Treatment: 9 stable failures. Control: 9 flicker items. Answerer: `gpt-6-luna`; judge: `gpt-4o`; total: 18 items. In this FULL_S run profile, source sessions are ingested locally rather than sent to a paid writer model.
Main-run actual cost mean/item: $0.001996; median/item: $0.001790; 18-item observed-cost estimate: $0.035922.
Reserved-estimate cost mean/item: $0.010297; 18-item reserved estimate: $0.185340.
Use the manifest's model, turn-retrieval tier, tool-round policy, embedding settings, owner identity, and operational budget; include exactly the 18 IDs recorded in the JSON. Compare treatment judge-label lift against its three-run baseline, and check that controls do not materially regress. No provider call or benchmark execution is performed here.

## Reader-side proposal

Hypothesis: the faithful answerer has no preference-specific instruction to actively search for direct preference evidence; its generic ‘answer only from recall context’ rule can terminate with an ungrounded or generic answer when the relevant turn is not surfaced on the first recall. A falsifiable treatment should improve judged accuracy on stable failures without materially changing flicker controls.

At `benchmarks/longmemeval/faithful_agent.py:652-656`, append this exact sentence to `instructions`:

> When a question asks what the user prefers, search the public recall tools for the user's specific preference and its earlier session before answering. Prefer direct user statements and user-confirmed choices over assistant-generated suggestions or topic associations; apply the retrieved preference to the current question, and say “I don't know” if no directly relevant preference is retrieved.
