---
title: "Same Question, Four Phrasings: Prompt Sensitivity, Consistency and Hallucination of Four 7-9B Open-Weight Language Models on Short Factual Questions"
author: "Istiak Hossain Saikot"
affiliation: "Independent researcher, Bangladesh"
date: "25 November 2026"
abstract: |
  Small open-weight language models now run on ordinary laptops, and people ask them everyday factual questions. We wrote 120 short English factual questions, each tied to a pinned Wikipedia revision, and asked each in four styles (plain, detailed, polite, and step-by-step) to Llama-3.1-8B, Mistral-7B, Gemma-2-9B and Qwen2.5-7B, run locally at temperature 0 (1,920 answers and a 480-answer repeat). The study speaks mainly to evaluation and answer form, not to large swings in correctness. A strict first grader rejected long but correct sentences and scored Mistral-7B at 88.8% rather than 99.0%, which created a spurious style effect. A first-sentence rule and three aliases, added after we saw the outputs, removed that artifact. Style changed form more than correctness: Mistral-7B gave full-sentence final answers in 46-99% of replies, and only 10% of its questions had the same answer string in all four styles, against 97% with the same correctness label. False answers were rare (0.2-1.0%) but patterned, mostly on counting and rule questions. Step-by-step prompting sometimes talked a model out of a fact it already had. Temperature-0 decoding was not fully repeatable (84-100% of repeated responses were identical), although no correctness label changed. Accuracy was near ceiling (98.3-99.6%), so the study is under-powered for small correctness differences and should not be read as showing insensitivity to wording in general. These results apply to everyday short English factual questions on these four 4-bit local models via Ollama, not to harder, long-form, or higher-temperature settings. We release the questions, prompts, raw outputs and code.

---

# Introduction

Large language models (LLMs) are often asked short factual questions. A growing share of these questions go to small open-weight models that run on a personal computer, such as the 7-9 billion parameter versions of Llama, Mistral, Gemma and Qwen [@grattafiori2024llama3; @jiang2023mistral; @gemma2024gemma2; @qwen2024qwen25]. Users do not phrase their questions in a fixed way. Some type the bare question, some add instructions, some are polite, and some ask the model to "think step by step". If the answer to a factual question depends on such incidental choices, a single-prompt accuracy number says little about how reliable the model will be in use.

Two lines of work bear on this. Studies of prompt sensitivity show that small changes in wording or formatting can move benchmark accuracy by large amounts and even reorder models [@sclar2024formatting; @mizrahi2024multiprompt; @cao2024worst; @zhuo2024prosa; @errica2025whatdid]. Studies of hallucination measure how often models state false facts, usually under one fixed prompt [@lin2022truthfulqa; @li2023halueval; @wei2024simpleqa]. Ganesh et al. [@ganesh2026rethinking] argue that the two should be studied together, because a model that answers correctly under one prompt and falsely under another is unreliable in a way that single-prompt accuracy hides. Hua et al. [@hua2025flaw] add a warning in the other direction: much reported prompt sensitivity comes from rigid answer matching rather than from the model.

This paper is a small, fully reproducible study at the meeting point of these lines. We ask four questions:

- **RQ1 (sensitivity).** How much does accuracy change across four everyday prompt styles for each model?
- **RQ2 (consistency).** For what share of questions does a model give the same answer under all four styles?
- **RQ3 (hallucination).** How often does each model commit to a false answer, and does the style, especially step-by-step prompting, change this?
- **RQ4 (errors).** What kinds of errors occur, and where?

Before the run we stated three hypotheses: H1, accuracy differs across styles within each model; H2, questions answered inconsistently across styles have a higher hallucination rate than consistent ones; H3, chain-of-thought (CoT) prompting changes the hallucination rate compared with the plain prompt.

Our contributions are modest but concrete. (1) A new set of 120 short factual questions with gold answers, accepted aliases and revision-pinned sources. (2) Results for four widely used 7-9B open-weight models under four prompt styles, with bootstrap confidence intervals, paired tests and a repeat run. (3) A row-by-row audit of the automatic grader, which shows that a strict but plausible grading rule would have produced a large, spurious style effect. (4) All prompts, raw responses, grading code and analysis scripts. The study is small and, because accuracy is near ceiling, under-powered for small differences in correctness; the results apply to everyday short English factual questions on these 4-bit local models.

# Related work

**Hallucination and factuality.** TruthfulQA [@lin2022truthfulqa] measures whether models repeat common misconceptions, and HaluEval [@li2023halueval] tests whether models recognize hallucinated content. SimpleQA [@wei2024simpleqa] grades short factual answers as correct, incorrect or not attempted, which is close to our setting; its questions are deliberately hard. FActScore [@min2023factscore] scores long-form text fact by fact, and SelfCheckGPT [@manakul2023selfcheckgpt] uses disagreement between sampled answers as a hallucination signal. Huang et al. [@huang2025survey] survey the area and its taxonomy. Kalai et al. [@kalai2025hallucinate] argue that evaluations which never reward abstention push models toward confident guessing. Mallen et al. [@mallen2023trust] show that facts about popular entities are recalled far more reliably than facts about rare ones, which bears on how hard a factual question set is.

**Consistency.** Elazar et al. [@elazar2021consistency] showed that pretrained language models often give different answers to paraphrases of the same factual query. Ganesh et al. [@ganesh2026rethinking] extend this to hallucination evaluation and report high inconsistency across prompts on existing benchmarks.

**Prompt sensitivity.** Sclar et al. [@sclar2024formatting] found that formatting choices alone can change accuracy by tens of points in open models. Mizrahi et al. [@mizrahi2024multiprompt] showed that model rankings change with instruction wording and argued for multi-prompt evaluation; Polo et al. [@polo2024efficient] make such evaluation cheaper. Cao et al. [@cao2024worst] recommend reporting worst-prompt performance. ProSA [@zhuo2024prosa] and POSIX [@chatterjee2024posix] define instance-level sensitivity scores, and Errica et al. [@errica2025whatdid] propose separate sensitivity and consistency metrics for classification. Razavi et al. [@razavi2025benchmarking] generate variations of TriviaQA and HotpotQA questions and show that predicting which variation a model will answer correctly is hard. Yin et al. [@yin2024respect] study politeness, and CoT prompting [@wei2022cot; @kojima2022zeroshot] is the basis of our step-by-step style. Hua et al. [@hua2025flaw] find that much measured sensitivity disappears when answers are judged more flexibly. Atil et al. [@atil2024nondeterminism] show that nominally deterministic LLM settings still give varying outputs.

**Gap.** Most hallucination studies use one prompt, and most sensitivity studies use multiple-choice or classification tasks with larger or closed models. We measure sensitivity, consistency and hallucination together, on the same short-answer factual questions, for four 7-9B models that people actually run locally, with natural prompt styles and a hand-audited grader.

# Method

## Questions

We wrote 120 English questions, 30 in each of four domains (science, geography, history, general knowledge), each tagged easy (44), medium (48) or hard (28) when the questions were written, before any model was run. Each question has one gold answer and a list of accepted aliases (for example "Bern" and "Berne", "11" and "eleven", or "ohm" and "Ω"). Each item is linked to a Wikipedia article pinned to a specific revision (`oldid`), and a script checked that a sentence in that revision contains the answer. All 120 items passed this check. The AI assistant used in this project then reviewed every item and its aliases; this review led to three alias fixes before the run. Gold answers were verified by this script and by the AI assistant only; there was no human verification of the answer key (see Limitations). The questions are deliberately everyday facts rather than obscure ones, so that the effect of wording would not be swamped by missing knowledge.

## Prompt styles

Every question was asked in four styles. All four require the reply to end with a line "Final answer: <answer>".

- **Simple:** the question, then the final-answer instruction.
- **Detailed:** "Please answer the following factual question accurately. Keep the answer short, and end your reply with a line of the form "Final answer: <answer>"." followed by the question.
- **Polite:** "Could you kindly tell me the answer to this question? ... If you don't mind, please end your reply with ... Thank you!"
- **CoT:** "Think step by step and answer the following question. Briefly show your reasoning, then end your reply with ..." [@kojima2022zeroshot].

The exact templates are in `config.json`.

## Models and decoding

We used four instruction-tuned open-weight models through Ollama 0.40.18 with its default 4-bit quantization (`Q4_K_M`): `llama3.1:8b`, `mistral:7b`, `gemma2:9b` and `qwen2.5:7b`. The short model IDs reported by `ollama list` on the same laptop on 25 November 2026 were `46e0c10c039e` (Llama-3.1-8B), `6577803aa9a0` (Mistral-7B), `ff02c3702f32` (Gemma-2-9B) and `845dbda0ea48` (Qwen2.5-7B). Decoding was greedy and fixed: temperature 0, top-k 1, top-p 1, repeat penalty 1, seed 42 and a 2,048-token context. The output limit was 128 tokens for the simple, detailed and polite styles and 512 for CoT. Each prompt was sent as a single user message with no system prompt. All runs used one laptop (Intel Core i7-14650HX, NVIDIA GeForce RTX 5060 Laptop GPU with 8 GB, Windows 11) on 24-25 November 2026. The main run (pass 1) covered 120 questions x 4 styles x 4 models = 1,920 calls. A repeat run (pass 2) re-asked a random 25% subset (30 questions, subset seed 2026) in all four styles, giving 480 more calls. Median latency was 2.3-4.6 s per call depending on model and style.

## Answer extraction and grading

The answer is the text after the last "Final answer:" on the same line. If no such line exists, we take the last line (`fallback_short` when it has at most 8 words, otherwise `fallback_long`). Answers and aliases are normalized in the SQuAD/TriviaQA manner [@joshi2017triviaqa]: lower case, punctuation removed, articles removed and thousands separators removed. The final grader (v2) takes the first sentence of the extracted answer and marks it correct if it equals an alias or contains an alias as a whole phrase, with a 40-word limit. Each row receives one error type: *none* (correct), *refusal*, *format* (no usable answer, for example output cut off before any answer), *incomplete* (a strict, non-matching part of an alias) or *wrong fact* (a committed answer that matches no alias). A row is a **hallucination** if it is *wrong fact* or *incomplete*; refusals and format failures are counted separately.

The first-sentence rule and three alias additions (H29, S22 and K09) were introduced after the run, as a result of the audit described in the Results. The final grader was not pre-registered. We report the original strict rule (v1: whole extracted answer, at most 15 words) alongside the final one in Table 5. The pre-registered aliases already included both "Astana" and "Nur-Sultan" for G24 (Kazakhstan's capital).

## Metrics

For model $m$ and style $s$, accuracy $\mathrm{Acc}_{m,s}$ and hallucination rate $H_{m,s}$ are the shares of the 120 questions answered correctly or falsely. Accuracy, hallucination, refusal and format-failure rates sum to one. The **accuracy spread** is $\mathrm{PS}_m = \max_s \mathrm{Acc}_{m,s} - \min_s \mathrm{Acc}_{m,s}$ [@cao2024worst]. The **question sensitivity** $\mathrm{QS}_m$ is the share of questions whose correctness differs across styles. The **consistency rate** $\mathrm{CR}_m$ is the share of questions whose four answers are equivalent: all correct, all refusals, all unusable, or the same normalized wrong answer. Because CR treats every correct answer as equivalent, we also report **surface consistency**, the share of questions whose four normalized first-sentence answers are identical strings. This separates changes in *what* the model claims from changes in *how* it phrases the claim.

## Statistics

Confidence intervals are 95% percentile bootstrap intervals [@efron1979] with B = 10,000 resamples of questions (seed 2026). PS is recomputed inside each resample. Within each model, we test for a style effect on accuracy with Cochran's Q [@cochran1950], compare all six style pairs with exact McNemar tests [@mcnemar1947] with Holm correction [@holm1979], and test H3 with an exact McNemar test on hallucination (simple vs CoT). For H2 we compare hallucination rates of consistent and inconsistent questions with Fisher's exact test. Pooled over models, we ran Cochran's Q on the 480 model-question units; because the four models answer the same questions, we also report a cluster permutation test that applies one random permutation of the style labels per question to all four models (10,000 permutations). A pre-run simulation of the exact McNemar test, and a post-hoc power calculation at the observed ceiling, are reported with the results.

# Results

## Accuracy and hallucination by style (RQ1, RQ3)

All four models were close to ceiling in every style (Table 1, Figure 1). Accuracy over all four styles was 99.6% for Gemma-2-9B and Llama-3.1-8B, 99.0% for Mistral-7B and 98.3% for Qwen2.5-7B (Table 2). No model's accuracy differed significantly across styles: Cochran's Q gave p = 0.572 (Gemma-2-9B), 0.112 (Llama-3.1-8B), 0.101 (Mistral-7B) and 0.494 (Qwen2.5-7B), and no pairwise McNemar test reached p < 0.05 even before Holm correction (smallest p = 0.25). The largest accuracy spread was 2.5 points (Mistral-7B, 95% CI 0.8-5.8), and at most 3.3% of a model's questions changed correctness across styles. These non-significant tests should not be read as evidence that wording has no effect on correctness.

A pre-run simulation of an exact McNemar test (120 questions, $\alpha = 0.05$, two-sided, $B = 10{,}000$, seed 2026) had power 0.80, 0.64 and 0.53 for a 10-point difference at 15, 20 and 25% discordance, and power 0.43 for an 8-point difference at 20% discordance. At the observed ceiling, the most favourable pattern for a test is one-sided discordance: the better style never fails uniquely. In that case the test reaches significance only when the worse style has at least six unique failures, a 5.0 percentage-point difference. The observed maximum spread of 2.5 points (3 of 120) then has power of only 0.08. Eighty percent power would require a difference of 8 of 120, or 6.7 percentage points (power 0.82).

Table: **Table 1.** Accuracy with 95% CI, hallucination rate and format-failure rate (%), per model and style (n = 120 questions each). No refusals occurred. Each cell reports a single quantity.

{{T_STYLE}}

![Accuracy by prompt style with 95% bootstrap CIs. The y-axis starts at 90%.](figures/fig1_accuracy_by_style.png){width=95%}

Hallucination was rare: 1 of 480 answers for Gemma-2-9B and Llama-3.1-8B, 5 for Mistral-7B and 4 for Qwen2.5-7B (Table 2, Figure 2). Per model, CoT did not change the hallucination rate significantly (exact McNemar, simple vs CoT: p = 1.0 for Gemma-2-9B, Llama-3.1-8B and Mistral-7B; p = 0.25 for Qwen2.5-7B), so H3 is not supported at the model level. Pooled over models, however, false answers were most frequent under CoT (7 of 480, 1.5%) and absent under the polite style (0 of 480) (Table 3). The pooled Cochran's Q was significant for hallucination (Q = 11.90, cluster-permutation p = 0.039) but not for accuracy once clustering by question was respected (naive p = 0.011, cluster-permutation p = 0.113). With so few events we treat the CoT result as exploratory.

Table: **Table 2.** Model-level results over all four styles. Accuracy and hallucination over 480 answers; CR = label consistency; surface consistency = identical normalized answer string in all four styles; PS = accuracy spread in percentage points; QS = % of questions whose correctness changes; Cochran's Q across styles (df = 3). Brackets: 95% bootstrap CIs.

{{T_MODEL}}

Table: **Table 3.** Pooled over the four models (480 answers per style) with question-clustered 95% bootstrap CIs, and pooled Cochran's Q with naive and question-clustered permutation p-values.

{{T_POOLED}}

![Hallucination rate (committed false answers) by prompt style with 95% bootstrap CIs.](figures/fig2_hallucination_by_style.png){width=95%}

## Consistency (RQ2)

Label consistency was high: 98.3% of questions for Gemma-2-9B and Llama-3.1-8B and 96.7% for Mistral-7B and Qwen2.5-7B received the same correctness under all four styles (Table 2).^[The pre-specified test of H2 (that inconsistent questions have a higher hallucination rate than consistent ones) was statistically significant (Fisher's exact test, *p* from 0.017 for Gemma-2-9B and Llama-3.1-8B to below $10^{-6}$ for Mistral-7B and Qwen2.5-7B). Every hallucination fell on an inconsistent question, and no consistent question received a false answer. This is close to definitional at this ceiling: a question can only be inconsistent if at least one style gets it wrong, and no question was answered with the same wrong answer in all four styles. It should not be read as evidence that inconsistency predicts hallucination in general.]

Surface consistency told a different story (Figure 3). The four normalized answer strings were identical for 88.3% of Gemma-2-9B's questions, 80.8% of Llama-3.1-8B's and 70.8% of Qwen2.5-7B's, but for only 10.0% of Mistral-7B's. The style mainly changed the form of Mistral-7B's answers, not their content.

![Label consistency versus surface consistency across the four styles, with 95% bootstrap CIs.](figures/fig7_label_vs_surface_consistency.png){width=85%}

## What the styles did change: answer form

Table 4 shows how often the extracted final answer was longer than five words, and the median number of output tokens. Mistral-7B wrote a full sentence after "Final answer:" in 45.8% (detailed) to 99.2% (polite) of replies. The extra words were often true but unasked-for details. Under the polite style, for example, Mistral-7B answered the deepest-lake question with "Final answer: The deepest lake in the world is the Lake Baikal in Siberia, Russia. It reaches a maximum depth of approximately 5,387 feet (1,641 meters)." The effect of style on verbosity was large and significant for Mistral-7B (Cochran's Q = 96.9, p < $10^{-20}$) and also significant for Llama-3.1-8B (p < 0.001) and Qwen2.5-7B (p = 0.043). Llama-3.1-8B ignored the final-answer instruction in 8 of 120 simple-style replies (6.7%), answering in one plain sentence instead. CoT multiplied the median output length by roughly 4-9 times and raised median latency from about 2.4 s to 3.8-4.6 s without raising accuracy.

Table: **Table 4.** Share of final answers longer than five words (%) / median output tokens, by style, with Cochran's Q for the verbosity indicator. Medians of an even number of integer counts can end in .5.

{{T_FORMAT}}

![Share of final answers longer than five words, by model and style.](figures/fig6_verbosity_by_style.png){width=85%}

## Grader audit

We audited the automatic grader in two steps.

*Step 1: all non-correct rows.* After the first grading (v1), we read every one of the 74 answers marked wrong. 58 of them (78%) were grader errors. In 49 cases, all from Mistral-7B, the model gave the right entity inside a full sentence longer than the 15-word limit ("The first President of the United States was George Washington. He served two terms from 1789 to 1797."). Four were Llama-3.1-8B's "Trygve Halvdan Lie" for "Trygve Lie". Five came from two ambiguous questions. "Who published *On the Origin of Species* in 1859?" (S22) was meant to ask for the author, but Llama correctly named the publisher, John Murray. "How many squares are there on a standard chessboard?" (K09) has both 64 and the well-known puzzle answer 204 as defensible readings. We changed the grader to score the first sentence of the extracted answer, added the full name as an alias, accepted both readings of the two ambiguous items, and flagged those items for rewording. The AI assistant then re-read all 58 rows whose label changed and agreed with every new label. These changes affect the answer key only; the question text and prompts sent to the models are unchanged (the as-run question file is released with SHA-256 `88b78829...b1647`).

*Step 2: stratified sample.* We drew a fresh audit sample from the v2 grades: 60 rows per model (240 in total), consisting of every non-correct row (17) plus correct rows stratified by style. The AI assistant acting as first coder read each row's gold answer, extracted answer and full response. It agreed with the automatic correctness label on 240 of 240 rows (Cohen's kappa = 1.00 [@cohen1960]) and with the automatic error type on 239 of 240 (kappa = 0.97). The one disagreement was Qwen2.5-7B's polite reply to the snooker question, which ran out of tokens before a final answer (auto code: *format*) but had already committed to false claims ("theoretically unlimited", "75 points"); the first coder coded it as *wrong fact*. There was no human second coder, so no inter-human agreement is reported.

The audit changed the conclusions. Table 5 shows accuracy under the strict v1 rule and under v2, and under three further robustness choices. Under v1, Mistral-7B's overall accuracy would have been 88.8% rather than 99.0%, and its style spread would have been driven by how verbose its answers were. This is the artifact Hua et al. [@hua2025flaw] warn about. The v2 results are stable under the other choices: excluding the two ambiguous items, treating replies without a "Final answer:" line as failures, or rejecting Kazakhstan's former capital name "Nur-Sultan" (a pre-registered alias) each move accuracy by at most 1.7 points.

Table: **Table 5.** Overall accuracy (%) under alternative grading choices.

{{T_ROBUST}}

## Repeat-run stability

Table 6 compares pass 1 and pass 2 on the same 120 prompts per model. Although decoding was greedy with a fixed seed, only Gemma-2-9B reproduced every response byte for byte. For Llama-3.1-8B 91.7%, Mistral-7B 93.3% and Qwen2.5-7B 84.2% of raw responses were identical. The differences were mostly in the wording of the reasoning or explanation. They reached the extracted answer in 4 of 120 cases for Mistral-7B and Qwen2.5-7B, and in one case they changed a fact: Qwen2.5-7B's CoT answer to "What is the capital city of Kazakhstan?" was "Nur-Sultan" in pass 1 and "Astana" in pass 2, while its polite answer moved the other way. No correctness label changed between passes for any model. This matches reports that "deterministic" LLM settings are not fully deterministic [@atil2024nondeterminism].

Table: **Table 6.** Pass 1 vs pass 2 on 30 questions x 4 styles (% identical).

{{T_PASS2}}

![Repeat-run stability at temperature 0.](figures/fig8_pass2_stability.png){width=85%}

# Error analysis (RQ4)

There were 17 non-correct answers out of 1,920: 11 *wrong fact* and 6 *format*. There were no refusals and no *incomplete* answers. Errors were concentrated on 6 questions; the other 114 questions were answered correctly by every model in every style. By domain, Qwen2.5-7B's errors fell mostly on general knowledge (94.2% accuracy there) and Mistral-7B's on science (96.7%). Errors did not track our difficulty tiers: no model fell below 97.3% on any tier. Table 7 lists every error.

Table: **Table 7.** All non-correct answers in pass 1 (v2 grader). Extracted answers are abbreviated for display; the full text is in the released raw outputs.

{{T_ERRORS}}

Five patterns stand out.

**Faulty reasoning toward a confident wrong number.** The snooker question (K21, gold 147) produced 6 of the 17 errors. Gemma-2-9B's CoT reply summed "15 reds x 1 point each = 15 points" and "6 + 7 + 8 + 9 + 10 + 15 = 55 points" to reach "Final answer: 70"; Qwen2.5-7B answered 302 (detailed) and 42 (CoT). In the cricket question (K14, gold 11), Llama-3.1-8B's CoT first stated the correct fact ("each team has 11 players on the field at a time"), then reasoned that "the bowling team ... has 9 players on the field" and gave 9. Here step-by-step reasoning talked the model out of a correct answer it had already written down.

**Redefining the question.** Mistral-7B's CoT reply to "What is the largest organ of the human body?" reasoned that the skin "is not considered an organ in the traditional sense" and answered "the liver". Asked which metal has the highest melting point, Mistral-7B's CoT listed tungsten at 3422 °C and osmium at 3032 °C, then concluded that osmium was highest. Its simple-style reply named carbon, a non-metal, and then said in the next sentence that tungsten has the highest melting point among common metals. Because we grade the first sentence, this counts as a false answer.

**Formatting collapse.** All six *format* failures were replies that hit the token limit before a final answer. Llama-3.1-8B's CoT reply to K21 degenerated into the word "Red" repeated until the limit, and Gemma-2-9B and Qwen2.5-7B began enumerating the squares of every size on a chessboard and ran out of tokens.

**Stale facts.** Several models gave "Nur-Sultan", Kazakhstan's capital name from 2019 to 2022, instead of "Astana". We accepted this alias, so these are not counted as errors, but they show that "correct" depends on when the model's training data ends.

**Invented detail in otherwise correct answers.** Many correct replies, especially Mistral-7B's full sentences, added facts that were not asked for. We coded a stratified random sample of 100 automatically correct pass-1 answers (25 per model, seed 2026) for *invented detail*: a code of Y only when the response asserted a false factual claim in the reasoning or in extra text, despite a correct final answer. Extra true facts were coded N. The coder was the AI assistant used in this project; there was no human verification of these codes (see Limitations). Five of the 100 answers (5.0%, Wilson 95% CI 2.2-11.2%) received a Y: 0 of 25 for Gemma-2-9B, 1 of 25 for Llama-3.1-8B, 1 of 25 for Mistral-7B and 3 of 25 for Qwen2.5-7B. The five cases were a self-contradictory claim that the gas giants other than Jupiter lack notable rings; a misattribution of the 1903 Nobel Prize in Physics to the discovery of polonium and radium; two answers that gave incorrect dates for Trygve Lie's term as UN Secretary-General; and a claim that a rugby union team has seven reserves (a current match-day squad is 15 plus eight replacements). Because the sample is modest and was coded only by the AI assistant, we treat the 5% rate as a descriptive indication that extra false claims do occur in otherwise correct replies, not as a precise model-by-model estimate. Our automatic metrics do not count such errors.

# Discussion

The four 7-9B models we tested were highly accurate on these everyday short factual questions (98.3-99.6%). Because accuracy was near ceiling, the study is under-powered for small differences in correctness, and the non-significant per-model style tests should not be read as showing that the models are insensitive to wording. This is a different setting from the large prompt effects reported for harder or multiple-choice tasks [@sclar2024formatting; @mizrahi2024multiprompt], and it is consistent with ProSA's finding that sensitivity falls as model confidence rises [@zhuo2024prosa]. Most of our questions are about popular entities, which models recall well [@mallen2023trust]. With at most three non-correct answers per model per style, only differences of about 5-7 percentage points could be detected with reasonable power (see the power note in Results).

What the study does support is more specific. First, grading was a large source of apparent sensitivity: a strict but reasonable exact-match rule turned Mistral-7B's verbosity into a 10.2-point accuracy deficit and inflated its style spread. The first-sentence rule and the three added aliases were decided after seeing the outputs; the final grader was not pre-registered. We therefore report the v1 and v2 results side by side (Table 5). If we had not read every answer the first grader marked wrong, we would have reported a strong prompt-sensitivity effect that is really an effect of answer format. This supports Hua et al.'s argument [@hua2025flaw] that prompt-sensitivity claims need flexible, audited grading, and it means that reported differences between styles should be checked against how the answers were matched. Second, the style changed the form of the answer, especially for Mistral-7B, far more than it changed correctness. Third, committed errors were rare but patterned: they clustered on a few counting and rule questions, and CoT sometimes talked a model out of a fact it already had. CoT also rescued a few items that a shorter style missed (for example, Mistral-7B's half-life and chessboard answers), so its net effect on accuracy was close to zero. On recall questions like these, asking the model to reason costs 4-9 times the tokens and opens new ways to go wrong. Fourth, temperature 0 with a fixed seed did not guarantee identical outputs on consumer hardware, although no correctness label changed. Studies that rely on a single deterministic run should check repeatability, as we did. Finally, for a careful user, a stated final answer may still come wrapped in extra unverified claims.

# Limitations

- **Quantization.** All models ran in 4-bit `Q4_K_M` form through Ollama. Full-precision models may behave differently.
- **Single hardware and software setup.** One laptop GPU, one operating system and one Ollama version (0.40.18). The repeat-run differences suggest that results can vary slightly across hardware and software. Model tags and the 12-character `ollama list` IDs recorded on 25 November 2026 are given in the Reproducibility statement. Full 64-character weight digests were saved in the pre-run environment snapshot only for Llama-3.1-8B (`llama3.1:8b`) and Qwen2.5-7B (`qwen2.5:7b`); those prefixes match the IDs above. Mistral-7B (`mistral:7b`) and Gemma-2-9B (`gemma2:9b`) were pulled after that snapshot, so only their short IDs are recorded.
- **Partial repeat.** Only 25% of the questions (30 x 4 styles per model) were repeated, so the stability estimates rest on 120 pairs per model.
- **Gold answers verified by script and an AI assistant only.** Every gold answer was checked against a pinned source by script and reviewed by the AI assistant. There was no human verification of the answer key. Two items (S22, K09) turned out to be ambiguous and were handled by accepting both readings. The robustness analysis excludes them.
- **Post-hoc grader change.** The first-sentence rule and three alias additions were made after seeing the outputs, as a result of the audit. We report both versions, and every changed label was re-read, but the final grader was not pre-registered.
- **Single coder, no human verification of the audit.** The audit and the invented-detail sample were coded by the AI assistant only. There was no human second coder and no human verification of those codes.
- **Ceiling effect and scope.** Accuracy was near 100%, so the study is under-powered for small style effects on correctness (see the power note in Results). The questions are English, short-answer and mostly about well-known facts, and they may appear in training data. The results apply to everyday short English factual questions on these four 4-bit local models via Ollama; they do not carry over to long-form, multilingual, harder, or higher-temperature settings.
- **Latency.** Sleep prevention failed to start on the laptop, so the machine slept during the run. This caused wall-clock latency outliers (maximum 1,955 s for one Mistral-7B call) but did not change any answer, since each response is produced by a single request. We report median latency.

# Conclusion

We asked 120 everyday short English factual questions in four phrasings to four 7-9B open-weight models running locally in 4-bit form through Ollama. The results that the study supports are mainly about evaluation and answer form. A strict first grader turned Mistral-7B's verbose but often correct sentences into a 10.2-point accuracy deficit (88.8% rather than 99.0%) and a spurious style effect. Prompt style changed how answers were written far more than whether they were right. Committed false answers were rare (0.2-1.0%) but patterned, clustering on a few counting and rule questions, where step-by-step prompting sometimes led the model astray. Temperature-0 decoding was not fully repeatable, although no correctness label changed. Accuracy was near ceiling (98.3-99.6%), so the study is under-powered for small differences in correctness and should not be read as showing that these models are insensitive to wording in general. These results apply to everyday short English factual questions on these four 4-bit local models via Ollama, not to harder, long-form, or higher-temperature settings. Future work should use harder and multilingual questions, more models and quantization levels, and repeated runs on more than one machine.

# Reproducibility statement

All code is in `src/` (Python 3.12 on the laptop for querying, Python 3.13 for analysis; querying uses only the standard library). Configuration, including exact prompt templates, decoding parameters (temperature 0, top-k 1, top-p 1, repeat penalty 1, seed 42, num_ctx 2048, num_predict 128/512), model tags and the analysis seed (2026), is in `config.json`. Ollama 0.40.18 model IDs from `ollama list` on 25 November 2026 (all default Q4_K_M tags): `llama3.1:8b` `46e0c10c039e`, `mistral:7b` `6577803aa9a0`, `gemma2:9b` `ff02c3702f32`, `qwen2.5:7b` `845dbda0ea48`. Full 64-character digests recorded at the start of the run: `llama3.1:8b` `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e`, `qwen2.5:7b` `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`. The question file used for the run has SHA-256 `88b7882986da0a09af9e3e227f0b45896449167ad0647fb6bb1dfa0ea38b1647`. The post-audit answer key (aliases only changed) has SHA-256 `bf70fd3d8870c85a97364e17e7eae771815174ebf29e6a6355d9bbaa0b12f21d`. The raw responses of both passes (with per-call latency, token counts, Ollama version and timestamps), graded results, audit sheets, tables and figures are released with the code. Running `run_analysis.sh` (grading, audit tables, metrics, statistics, figures and the tables of this paper) on the released raw files reproduces every computed number reported here; the invented-detail Y/N codes are judgements stored in the released coding sheet and are summarised, not re-assigned, by the script. Code is released under the MIT licence and data under CC BY-SA 4.0. Code, questions, raw outputs and graded results are available at <https://github.com/istiak7ok/llm-prompt-sensitivity>. A Zenodo archive DOI will be minted for the v1.0 release of that repository; the DOI has not been issued yet and will be inserted at the following marked spot: [Zenodo DOI for v1.0: to be added after minting]. The same files are also in `deliverables.zip`. The arXiv LaTeX sources are in `arxiv_submission.tar.gz`.

# AI-use disclosure

An AI assistant (Grok Bot) was used extensively in this project, under the direction and oversight of the author. It drafted and checked the literature review (every reference was verified against arXiv, Crossref, OpenReview or the ACL Anthology), drafted the question set and checked each item against its source, wrote the experiment and analysis code, ran the grading and statistical analysis on the real model outputs, served as the only coder in the grading audit and in a 100-row invented-detail sample of automatically correct answers, and drafted this manuscript, including its tables and figures. The model runs themselves were carried out on the author's laptop. No result, citation or data point was invented: every number in this paper is computed from the released raw outputs. Gold answers and the audit were verified by script and by the AI assistant only; there was no human verification of the answer key, the audit codes, or the invented-detail codes. That is a limitation of this study. The author is responsible for the content.

# References

::: {#refs}
:::
