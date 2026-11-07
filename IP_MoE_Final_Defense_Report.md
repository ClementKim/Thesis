# IP MoE thesis defense final report

*Consolidated assessment and required revisions*

29 October 2026  \|  Evidence package: thesis(5).tar

**Decision:** Major revisions required before approval.

The main barriers to approval are incorrect H1 reporting, an incomplete evidence package, missing provenance, and conclusions that exceed what the comparisons establish. The corrected H1 analysis still provides no evidence of positive localization under the tested readout. The current experiments do not establish that the interaction branch is generally useless or that the proposed explanations caused its observed behavior.

### Committee record

| Member | Recorded vote | Principal concerns |
| --- | --- | --- |
| C1 chair | Major revisions | Reproducibility, provenance, H1, diagnosis |
| C2 | Major revisions | H1, comparison validity, uncertainty |
| C3 | Major revisions | Positioning, H1, diagnosis, baselines |

These are the three votes in the supplied mock committee record. This report consolidates that record with the subsequent file audit; it does not represent a new independent three-member review or an institutional decision.

### What the evidence supports

- The original committee reported that all checked headline table values traced to shipped result files. This audit directly reproduced the H1 statistics and the seven-benchmark averages for the three full-model seeds.

- The inspected implementation contains the described frozen encoders, trainable projections, interaction map, dense expert processing, router scaling, and concatenated prefix.

- The observed ablations do not demonstrate the predicted benefits under this implementation and training protocol. The strongest defensible conclusion is bounded to those experiments.

### Priority before the defense

Correct H1 and every dependent claim first. Next, restore the evaluation and training evidence, document authorship, and narrow the causal and negative claims. Preserve the reported Avg seed standard deviation: its arithmetic is correct. Do not claim that all ten requirements are closed merely because this report identifies a revision path.

## 1 Corrected H1 results

H1 was recomputed from the five shipped items.csv files, retaining the same 520 non-overlapping Depth questions. Each map is calibrated on its non-overlapping Distance questions. The revised rule treats an absolute calibration median below 10⁻⁹ as a tie and assigns +1 to ties. This is a transparent numerical repair and sensitivity analysis, not a newly preregistered protocol. Per-item d values were not rounded.

### Raw and signed grounding scores

| Map | Sign | Raw mean d | Signed mean d | One sided p |
| --- | --- | --- | --- | --- |
| Frozen CLIP | -1 | -0.203246 | +0.203246 | 2.12 × 10⁻²⁹ |
| full seed 42 | +1 | +0.001299 | +0.001299 | 0.92732 |
| full seed 43 | +1 | -0.025471 | -0.025471 | 0.99390 |
| full seed 44 | +1 | -0.013904 | -0.013904 | 0.79441 |
| all_map_mean seed 42 | -1 | -0.000303 | +0.000303 | 0.46730 |

After aligning question IDs and averaging the three full-model signed scores per question, the mean is -0.012692, the median is -0.001669, and the one-sided Wilcoxon signed-rank p-value is 0.987752. The original sign choices give mean -0.013558 and p = 0.837298. Neither analysis supports positive grounding under this readout.

The p-value is from a signed-rank test against zero; it is not a test of the arithmetic mean alone. The table mean is descriptive.

### Peak and box preference definitions

| Map | Raw max in box | Raw min in box | Signed own box | Same box |
| --- | --- | --- | --- | --- |
| Frozen CLIP | 2.40% | 50.96% | 70.10% | 43.65% |
| full seed 42 | 5.87% | 27.31% | 50.00% | 86.92% |
| full seed 43 | 6.73% | 38.56% | 47.12% | 76.15% |
| full seed 44 | 2.98% | 21.63% | 48.56% | 86.35% |
| all_map_mean seed 42 | 62.12% | 2.79% | 49.42% | 88.46% |

Raw max/min in box: fraction of words whose raw extremum is inside either marked box. Signed own box: fraction whose own box is selected by argmax of signed box means. Same box: fraction of questions for which both words prefer the same box by the saved mean-value argmax rule; this is not a peak statistic.

Frozen CLIP has raw mean d = -0.203246. Its 50.96% peak-in-box value refers to the raw minimum, equivalently the maximum of the sign-inverted map. The raw maximum lands in either marked box for only 2.40% of words. This inversion concerns the implemented token-to-patch map; it is not a general claim about all CLIP localization methods.

Evidence: h1.py:243-249, 274-310 and 358-386; draft/tex/5_ch3.tex:220, 369-391; the five paths listed in the evidence register.

## 2 Statistical and evidentiary corrections

### The Avg standard deviation is correct

| Full model seed | Average of seven primary scores |
| --- | --- |
| 42 | 64.28493033 |
| 43 | 64.27399653 |
| 44 | 64.29651872 |

The mean is 64.28514853 and the sample standard deviation is 0.01126268 percentage points. Therefore, 64.29 ± 0.01 accurately reports the observed mean and sample SD across these three seeds. The averages nearly cancel benchmark-specific changes; the individual benchmark scores are much less stable.

The value 0.31433560 is obtained by propagating the seven benchmark SDs while setting cross-benchmark covariance terms to zero. It is a different quantity based on an additional independence assumption. It must not replace the observed seed SD. Three seeds nevertheless give a weak basis for estimating variability over future training runs.

### Question sampling and training randomness are different

Paired bootstrap intervals quantify uncertainty in the difference between the evaluated checkpoints under a stated resampling scheme. They do not recover the missing distribution of results over independently trained ablation seeds. Use intervals to describe checkpoint comparisons, and explicitly retain the one-seed limitation for architectural conclusions.

Preserve the relevant evaluation units: NaturalBench groups, MMBench circular question groups, and repeated images where applicable. Recompute the official metric in every replicate, including POPE F1 and CV-Bench weighting. An ordinary average of binary hits does not reproduce all seven primary scores.

A minimum detectable effect is a power statement, not an upper confidence bound. The phrase “no effect larger than X” requires an appropriate interval bound or an equivalence analysis with a justified margin and clear scope. Otherwise use “the evaluated runs did not demonstrate the predicted benefit.” Drop the seed-spread rule as a significance criterion. The committee’s approximate 29% false-positive figure is not a universal calibration and was not independently reproduced in this audit.

### Unequal retained training data is a risk to quantify

The code computes the available text budget from the arm-specific prefix length and can skip over-budget examples. This confirms a mechanism for unequal exposure, but does not prove that any examples differed in the actual runs or establish the size or direction of the resulting bias. The checkpoints and their counters are absent from this package.

Recover train_split, train_seen_this_epoch, train_sampler_padding, train_budget_skipped and train_retained_this_epoch per arm. Retained counts may include distributed-sampler padding; distinguish exposure counts from unique retained examples. Equal total counts alone do not establish identical retained identities. Until verified, replace “same training data” with “the same source split and nominal settings, subject to arm-dependent budget filtering.”

Evidence: main.py:245-285; model/ipmoe.py:78-125; draft/tex/5_ch3.tex:139-141, 169, 182-183, 273-288; per-seed score CSVs.

## 3 Scope of the scientific conclusions

### H1 measures a particular readout

The named-word row test does not establish that the whole interaction matrix contains no recoverable object information. Nor does it establish what the experts or language-model attention recover. The trained maps show strong shared box preference, but this alone does not prove exact word independence. Scope the conclusion to the tested rows, score, dataset and checkpoints.

Seed 42’s Distance median is -3.7007434 × 10⁻¹⁷, while seeds 43 and 44 have zero medians. The original sign disagreement therefore does not demonstrate inconsistent learned directions. Delete that argument. The all_map_mean raw maximum falls in a marked box for 62.12% of words while the words choose the same mean-value box in 88.46% of questions; high box occupancy is not word-specific grounding.

### The causal diagnosis remains untested

The CLIP reference differs from the trained map in projection, representation and normalization. Its better signed grounding score does not isolate the effect of training or prove that the MLPs destroyed alignment. Rank invariance to a common positive row scaling does not make patch-dependent L2 normalization irrelevant. Retain these mechanisms as hypotheses unless controlled encoder or intervention tests are supplied.

### Three mechanisms need correction

- Raster windows: flattening the image grid into a known raster order does not by itself destroy position information. The defensible limitation is the absence of a square-region locality prior and the mixing of several text rows. The learned patch projection’s information preservation would require a rank or recoverability check; injectivity was not verified.

- Router scaling: expert_head has a bias, which is added after the expert feature is scaled and is not multiplied by the router weight. Changing that weight can therefore change the projected token direction, including after normalization. All experts remain delivered, so the mechanism is dense modulation rather than sparse selection. RMSNorm does not establish irrelevance of the gate.

- Depth: image position can predict depth labels through dataset cues. The reported 85.3% vertical-position cue prevents an argument that spatial localization is irrelevant simply because depth is three-dimensional. Separate genuine depth reasoning from success on this cue. Delete the pooled “below chance” claim unless supported by a clearly specified test.

### Defensible answers to the research questions

- RQ1: The tested named-word row readout did not demonstrate positive grounding for the trained maps on the retained CV-Bench Depth questions.

- RQ2: The evaluated learned per-expert gains on concatenated tokens did not demonstrate a consistent advantage over the fixed-weight alternatives.

- RQ3: The evaluated runs did not establish a performance benefit or the hypothesized complementary roles from using all three experts.

- RQ4: The raster-strip implementation did not demonstrate the predicted selective benefit on the defined grounded-question group; this is not a general test of locality-preserving interaction maps.

Evidence: model/common.py:15-54; model/ipmoe.py:38-40, 63-74; draft/tex/5_ch3.tex:247, 388-391, 480-526; draft/tex/6_ch4.tex:13-19.

## 4 Required revisions R1 to R4

### R1 Rebuild H1 reporting

**Finding.** K1 is confirmed and remains the highest-priority Major. Table 4.9 and the qualitative text mix raw and sign-inverted interpretations.

**Required revision.** Apply Section 1 and define peak as argmax of the signed map. Corrected peak-in-box percentages are 50.96, 5.87, 6.73, 2.98 and 2.79 in that table order; conditional own-box percentages are 85.47, 55.74, 55.71, 51.61 and 51.72. Update the Abstract, Contribution 2, results and remedy to disclose the inverse CLIP relation and shared box preferences.

**Closure criterion.** All five rows, the three-seed pooled output, captions and prose agree with a runnable analysis. Preserve the original analysis as an explicitly superseded version and report the sensitivity to the tie repair.

### R2 Restore the reproducibility package

**Finding.** K2 is confirmed. The pooled H1 summary includes only seed 42. There are 24 broken symbolic links: 20 for seed 43 and four obsolete exact-matching links for seed 42. The fork, training checkpoints and training logs are absent.

**Required revision.** Provide the fork or a patch against a pinned upstream commit, environment versions, analysis scripts for Tables 4.5, 4.6, 4.10 and 4.11 and Section 4.2.2, corrected pooling, repaired paths, training logs and traceable compute information. Identify the intended seed-43 run before repointing links. Explain its omitted regression or regenerate it.

**Closure criterion.** A reader can rerun the reported analyses from documented canonical inputs. Distinguish recomputation from saved predictions from rerunning inference and training. Missing historical records must be disclosed; reconstructed records must be labelled as such.

### R3 Document provenance

**Finding.** K3 remains open. The archive does not establish who wrote, adapted or generated each component.

**Required revision.** Add an author-confirmed appendix covering code, the fork, wrappers, figures, analysis and writing assistance. Allow mixed origins. For adaptations, record source, version and license; for generated portions, record tool, purpose and verification.

**Closure criterion.** Every file or defined file group has an entry. Outputs identify producing code and inputs. Unresolved origins stay labelled unknown; do not infer them from coding style.

### R4 Audit comparison comparability

**Finding.** K4 is confirmed as an unresolved confound, not a measured retained-data difference. Evaluation records also cross commits 493f723b and 1d96b902.

**Required revision.** Recover per-arm counters and retained sample identities where possible. Document the fork diff. If the diff affects only scoring, compare scores from identical saved predictions; if it affects inference, rerun an existing checkpoint under both versions and compare predictions and scores.

**Closure criterion.** The actual exposure difference and relevant code-change effects are measured or explicitly unresolved. Correct “same data” and any claimed causal attribution. Re-scoring alone does not prove that two inference implementations produce identical predictions.

## 5 Required revisions R5 to R8

### R5 Correct and bound the diagnosis

**Finding.** K5 is confirmed. The current explanations exceed the controlled evidence, and the raster, normalization and depth arguments need the corrections in Section 3.

**Required revision.** Use route B: remove the diagnosis from the Abstract’s findings and contribution list, and present it as hypotheses. Alternatively, route A adds H1 on seed-matched initial MLPs, normalized trained features and raw-dot CLIP features, plus test-time swap or zero interventions on interaction and expert tokens with the image path fixed.

**Closure criterion.** Either the diagnosis is relabelled and all incorrect mechanisms are repaired, or the manuscript reports the relevant controlled tests and their limitations. An inference intervention measures checkpoint reliance under that perturbation; it does not by itself prove the cause of a training outcome.

### R6 Report appropriate uncertainty

**Finding.** K6 is upheld with correction: one-seed arms and the spread rule do not establish equivalence, but the observed Avg SD of 0.01 is valid.

**Required revision.** Report paired, group-aware intervals against full seed 42 for each benchmark, Avg and D, using official metrics. Label them checkpoint-level comparisons. Keep the actual three-seed Avg SD. Drop the spread rule as a test. Compare image_only with full seed 42 and remove unsupported “below chance” language.

**Closure criterion.** The report separates question-sampling uncertainty from training randomness. Any power calculation states its assumptions. “No effect larger than X” appears only with an appropriate bound and stated scope, not merely because a test is non-significant or X is a minimum detectable effect.

### R7 Restrict RQ4 and depth evidence

**Finding.** K7 is upheld as a scope correction. The tested expert windows use raster strips; CV-Bench-3D accuracy does not by itself identify a spatial-interaction benefit.

**Required revision.** State the raster-strip restriction in the Abstract, Introduction and Conclusion. Treat CV-Bench-3D as descriptive and cue-sensitive evidence. Remove “above the best seed” for all_map_mean, or report the 0.58-point difference alongside the 2.67-point full-model range without inferential weight.

**Closure criterion.** No conclusion generalizes this implementation’s outcome to true two-dimensional locality-preserving projectors. RQ4 depends on the specified grounded/other contrast and its uncertainty, not on a standalone CV-Bench-3D score.

### R8 Align the MoE framing with the computation

**Finding.** K8 is upheld. All experts execute and their separately weighted tokens enter the prefix. Sparse-MoE efficiency does not follow from this design.

**Required revision.** Describe IP-MoE as a dense interaction-conditioned multi-expert architecture, or explicitly qualify the MoE label. Scope H2 to per-expert gains on concatenated outputs. Report per-arm token and compute costs; the shared 4096-token limit is not equal compute or equal retained data. Recast the questions as inductive-bias tests under the stated training budget.

**Closure criterion.** The motivation, architecture equations, RQs and cost statements agree. No unmeasured speed or sparsity benefit is claimed. Dense delivery is a design limitation, while the empirical usefulness of the gains remains a separate question.

## 6 Required revisions R9 and R10

### R9 Correct the literature positioning

**Finding.** K9 remains open. The thesis contains broad claims that prior approaches pool away all relevant word-region structure, while its bibliography includes Honeybee and CuMo without integrating them into the relevant discussion. The original committee’s paper-by-paper comparison has not been independently repeated here.

**Required revision.** Check the primary papers and narrow the novelty claim to the actual combination implemented. Cover MoE-MLLMs including CuMo and MoE-LLaVA; Honeybee; Q-Former and InstructBLIP; dense CLIP localization such as MaskCLIP and CLIP Surgery; and BAN, MCB grounding, GLIP and DenseCLIP. Reconcile the MoVA description with the routing gap claim. Position the rank statistic in relation to AUC and the peak statistic in relation to pointing-game evaluation.

**Closure criterion.** Each claimed distinction has a precise citation and survives comparison with the cited method’s actual operation. The contribution is not phrased as the invention of word-region interaction maps or localization metrics.

Evidence: draft/tex/2_intro.tex:5-7; draft/tex/3_ch1.tex:9-19; draft/main.bib entries honeybee and cumo.

### R10 Qualify external baseline claims

**Finding.** K10 is confirmed at the manuscript and wrapper level. Two spatial baselines are registered but absent from reported results. The source comment records a published LLaVA-NeXT-8B CV-Bench-3D value of 65.3, whereas the thesis reports 70.75.

**Required revision.** Scope “highest” to the models evaluated in this study. Explain the omission of SpatialLadder-3B and Spatial-SSRL-3B, or evaluate them. Verify the original published values and present them beside re-measured values. List per-model checkpoint, prompt, image handling, maximum new tokens, sampling and judge use. Qualify “same training data” as source-dataset overlap with different preprocessing and exposure unless stronger equivalence is documented.

**Closure criterion.** The 5.45-point published/re-measured gap is reported with any demonstrated protocol explanation. If its cause remains unverified, say so explicitly. Do not invent an explanation or alter valid measurements to match the published value.

Evidence: vlmeval_baselines.py:35-59, 102-135, 138-170; vlmeval_adapter.py:44-76; draft/tex/5_ch3.tex:78-93, 164, 176.

### Minor corrections retained from the committee

- Correct the text-encoder labels in Figures 3.1 and 3.2 and show the option and instruction rows. Verify the edited figures visually.

- Document the chunk-position term and define “low-rank.” Replace “no image in both sets” with the precise byte-identity claim supported by the split procedure. Check for benchmark overlap or state that it was not checked.

- Record judge model snapshot, prompt and per-model invocation share. Correct the MMVP/CLIP-blind-pair attribution, duplicate and incomplete references, CV placeholders, signature dates, and chapter structure. Remove tex/table.tex and the todoval macro after checking references.

These presentation items are carried forward from the supplied committee record. This audit did not independently render and recheck every original figure or complete a bibliography-wide verification.

## 7 Closure plan and defense conditions

### Work that does not require retraining

| Route | Work | Dependency |
| --- | --- | --- |
| Saved-result analysis | H1 repair; table scripts; paired intervals; D analysis | Existing per-item data and documented metrics |
| Writing and documentation | R5 route B; bounded RQs; positioning; baseline disclosure | Correct evidence and primary-source checks |
| Record recovery | Fork; provenance; counters; hardware and training logs | Author records and original training environment |
| Optional evaluation | Initial/normalized maps; token interventions; commit checks | Existing checkpoints, model access and compute |

A revision strategy without new training is available, but closure is conditional on recovering required evidence and accepting narrower conclusions. Lost historical logs or unavailable fork revisions cannot be recreated honestly by writing. If those records cannot be recovered, identify the exact reproducibility limit and obtain the committee’s judgment on that remaining gap.

### Order of execution

1. Freeze the supplied package and record its hash. Keep original outputs separate from corrected analysis outputs.

2. Recompute H1 with the explicit tie rule; regenerate pooling and dependent tables and prose.

3. Restore canonical result links, the fork and analysis scripts. Export checkpoint counters and complete the provenance inventory.

4. Compute checkpoint-level intervals and audit the evaluation commit change. Apply R5 route B and narrow the scientific claims.

5. Correct literature and baseline reporting; finish minor edits; render the revised thesis and run the analysis from a clean documented entry point.

### Adjudication of the original Fatal findings

C1-F1 remains Major because the supplied committee found traceable arithmetic but an incomplete regeneration chain. It remains blocking if the required fork and analysis cannot be supplied. C1-F2 remains Major because missing provenance is not itself evidence that reused work was falsely presented as original. C2-F1 remains the highest-priority Major: the H1 descriptions are wrong, but repairing them does not reverse the lack of positive grounding evidence. The supplied record retains C2’s dissent on severity.

### Questions the student should be able to answer

- Which H1 statistics use the raw map, which use the signed map, and why does the chosen CLIP construction produce an inverse relation? Distinguish an observed fact from an untested explanation.

- Which files and checkpoints reproduce each result, how many unique training examples did each arm retain, and which components were written, adapted or generated?

- What can the one-seed ablations establish, and which conclusions remain uncertain because of training randomness, budget filtering or the evaluation version?

## 8 Evidence register and verification limits

Paths below are relative to the thesis/ directory inside thesis(5).tar. Line anchors refer to the supplied source snapshot. K and R identifiers preserve the mapping to the original committee record.

| ID | Source | Use |
| --- | --- | --- |
| E1 | h1.py:243-249, 274-310, 358-386 | Raw statistics, sign selection, extrema and pooling |
| E2 | evaluation/h1/clip/items.csv | Frozen reference inversion and localization rows |
| E3 | evaluation/h1/llama/seed_42/items.csv<br>Same path for seeds 43 and 44 | Per-seed and pooled H1 recomputation |
| E4 | evaluation/h1/ablations/all_map_mean/seed_42/items.csv | Averaged-map reference and same-box preference |
| E5 | evaluation/h1/llama/pooled/summary.json | Stale pooling with only seed 42 |
| E6 | evaluation/llama/seed_\*/IPMoE/\*_acc.csv and \*_score.csv | Seven-score Avg and actual seed SD |
| E7 | main.py:245-285; model/ipmoe.py:78-125 | Budget filtering and checkpoint counter definitions |
| E8 | model/ipmoe.py:38-40, 63-74; model/common.py:15-54 | Biased projection, concatenated outputs and raster patching |
| E9 | draft/tex/5_ch3.tex:169, 220, 273-288, 369-391, 491-526 | Statistical reporting and claim scope |
| E10 | draft/tex/6_ch4.tex:13-30; vlmeval_baselines.py:138-170 | Diagnosis, proposed remedies and baseline disclosures |

### Recomputation procedure

For H1, filter category = Depth and overlap = False; derive each run’s sign using only its Distance rows; align the 520 question IDs; average the three signed full-model scores per question; then use scipy.stats.wilcoxon with alternative = greater. Peak and own-box statistics use the saved positive or negative columns consistent with the selected sign. The audit used NumPy 2.3.5 and SciPy 1.17.0.

For Avg, use NaturalBench Acc, the two official CV-Bench Overall scores, MMBench Overall, MMStar Overall, POPE Overall F1, and SEEDBench Overall, all in percentage units. Average the seven within each seed, then compute sample SD across seeds. Seed-43 broken aliases were bypassed by reading the sole matching shipped files in T20261024-130459; no source links were modified.

### Limits of verification

The recomputation checks arithmetic on saved outputs, not the correctness of the original model-generated maps or predictions. No encoder inference, training, checkpoint intervention, new bootstrap analysis or retained-sample audit was performed. Literature comparisons, some minor items and the original three votes are carried forward with the qualifications stated above. The manuscript and its shipped results remain unmodified by this report.

Archive SHA256: 32d8c9aa542e8bbb8553ea650c893cede42e6ae11a643200774959172793a5f8
