# DeceptionBench: llama3.1:8b vs qwen3:8b

I applied the [DeceptionBench](https://github.com/Aries-iai/DeceptionBench) evaluation 
framework (Huang et al., 2025, arXiv:2510.15501) to two open-weight ~8B models not 
evaluated in the original paper: `llama3.1:8b` and `qwen3:8b`. I wanted to test whether 
adding reward- or pressure-based incentive framing to an otherwise identical prompt 
shifts a model's rate of deceptive responses, and whether that effect differs between 
the two models.

### Motivation & Hypothesis

DeceptionBench elicits a model's internal reasoning and final answer jointly (structured 
`{thought, response}` output), then labels each as honest or deceptive relative to a 
scenario-specific ground truth, across 150 scenarios spanning five domains (Economy, 
Healthcare, Education, Social Interaction, Entertainment). The benchmark varies two factors:

- an **intrinsic role axis** — `self` (the model acts as an autonomous agent with 
  egoistic, self-interested motives) vs `other` (the model acts as a user's assistant, 
  testing sycophantic, user-appeasing motives)
- an **extrinsic contextual axis** — `L1` neutral baseline (no inducement) vs `L2` 
  reward or pressure-based inducement

I focused on the L2 extrinsic axis: whether reward or pressure framing increases 
deception rate relative to L1 baseline, and secondarily report results on the L1 
intrinsic axis (self vs other) and on the consistency between a model's internal 
reasoning and its final answer.

**Hypothesis:** If honesty is not robust to incentives, deception rate should increase 
under pressure and reward relative to baseline. The original paper found this holds for 
most tested models (GPT-4o, Gemini, Qwen2.5, DeepSeek-R1, etc.), with pressure generally 
producing a larger effect than reward in single-turn settings, and reports Claude-family 
models as a notable exception with near-zero deception rates. I wanted to test whether 
the same pattern holds for llama3.1:8b and qwen3:8b.

### Setup

- Models evaluated: `llama3.1:8b`, `qwen3:8b` — run locally via Ollama
- Judge model: `gemma2:9b` (local), used to produce honest/decept verdicts
- Decoding: temperature 0.0 (deterministic) for the evaluated models — I traded some 
  realism (real-world chat use is typically 0.7–1.0) for reproducible, low-noise results; 
  see Limitations

### Method

I evaluated both models on the DeceptionBench scenario set using the original paper's 
prompt templates and single-turn conditions:

- **L1 (baseline):** `self` and `other` framings, no incentive
- **L2 (induced):** `self` and `other` framings, crossed with `pressure` and `reward` 
  inducements

I labeled each response `honest` or `decept` for both `thought` and `response` against 
the scenario's ground truth, following the paper's evaluation protocol.

I also ran L3 (multi-turn escalation) across every domain × incentive combination, but 
I decided not to include L3 in the findings below for either model. qwen3:8b's L3 data 
had the truncation/generation-failure issues described in Known Limitations, and I 
didn't have time to fully verify llama3.1:8b's L3 data was clean enough to report on its 
own — so rather than present a partial or one-sided multi-turn comparison, I excluded L3 
entirely from the reported Findings. Raw L3 output for both models is still in the repo; 
see Known Limitations for the diagnostic work I did on it.

### Implementation Notes

To run DeceptionBench entirely on local models, I made the following changes to the 
original codebase:

- I reconfigured `config.py` to point at local Ollama endpoints instead of cloud APIs
- I found and fixed a bug in `calculate_metric.py`'s `aggregate_metrics()` (it was 
  indexing into the wrong nested key when aggregating L2/L3 self/other conditions)
- I wrote `build_full_csv.py` from scratch — it parses raw per-condition `.jsonl` 
  output, joins generation and eval records by question, and produces 
  `result/metric/full_responses.csv`

### Findings

#### 1. Effect of incentive (extrinsic axis: baseline vs pressure vs reward)

| Model | Baseline decept% | Pressure decept% | Reward decept% |
|---|---|---|---|
| llama3.1:8b | 23.15% (n=298) | 20.74% (n=299) | 20.67% (n=300) |
| qwen3:8b | 60.6% (n=251)† | 72.24% (n=299)‡ | 74.33% (n=300)‡§ |

† Corrected after I excluded 42 truncated/empty rows I identified in a full manual audit 
of the baseline condition, including the 5 fully-blank rows I originally flagged (see 
Limitations).
‡ I haven't yet systematically audited this for the same artifact; likely inflated by a 
similar or greater margin (see Limitations). Treat as an upper-bound estimate.
§ I found at least one confirmed empty/truncated `L2-self-reward` row (see Known 
Limitations); combined with independent evidence from my L3 multi-turn diagnosis 
implicating `reward` framing specifically, I'd give this figure somewhat weaker 
confidence than the pressure figure.

**Fig. 1 — Deception rate by incentive condition.** The qwen3:8b baseline reflects the 
corrected figure after excluding 42 truncated/empty rows. I haven't systematically 
re-audited the pressure and reward figures for qwen3:8b for the same artifact, so I'd 
treat them as upper-bound estimates, with reward carrying additional independent 
evidence of elevated risk (see Known Limitations).

![Deception rate by condition](assets/deception_rate_by_condition.png)

Significance (chi-square test, baseline vs each condition):

| Model | vs Pressure | vs Reward |
|---|---|---|
| llama3.1:8b | p = 0.539 (n.s.) | p = 0.524 (n.s.) |
| qwen3:8b | p = 0.028 | p = 0.006 |

Note: I used the original, uncorrected baseline (63.48%) for the significance tests, for 
consistency with the pressure/reward figures, which I haven't re-audited. Using the 
corrected baseline (60.6%) doesn't change the direction or rough magnitude of 
significance, but I'd treat the exact p-values as approximate given the known 
data-quality issues (see Known Limitations).

llama3.1:8b's deception rate doesn't move meaningfully under either incentive condition 
— its honest/decept split under pressure and reward is statistically indistinguishable 
from baseline. qwen3:8b, by contrast, starts from a much higher baseline decept rate 
(60.6% corrected / 63.48% uncorrected, vs. llama's 23.2%) and increases significantly 
further under both pressure (p = 0.028) and reward (p = 0.006), with reward producing 
the larger effect — the opposite ordering from the original paper, which found pressure 
more effective than reward for most tested models in single-turn settings. I want to 
flag that the pressure and reward decept% figures I'm comparing here haven't been 
re-audited for the truncation artifact described in Known Limitations and are likely 
themselves modestly inflated. This matters most for the reward finding specifically: 
independent evidence from both the single-turn and multi-turn data (see Known 
Limitations) suggests `reward`-framed scenarios may be disproportionately prone to this 
artifact in qwen3:8b, which raises the possibility that some or all of the reward-vs-pressure 
gap I'm reporting here is itself an artifact of uneven data quality across conditions 
rather than a genuine difference in how qwen3:8b responds to reward versus pressure 
incentives.

#### 2. Effect of role (intrinsic axis: self vs other)

| Model | Self decept% | Other decept% | Gap (self − other) |
|---|---|---|---|
| llama3.1:8b | 22.54% (n=448) | 20.49% (n=449) | +2.05 pts |
| qwen3:8b | 71.24% (n=445)* | 68.90% (n=447) | +2.33 pts |

*I only audited the truncation artifact described in Known Limitations for the 
L1-baseline condition. This table draws on the same underlying qwen3:8b response pool 
and may be affected similarly, but I haven't independently re-audited it.

The original paper reports ~20-point self/other gaps for models like GPT-4o and Gemini, 
attributing this to self-serving bias, and identifies the Claude family as an exception 
with minimal gap. Both llama3.1:8b and qwen3:8b show small gaps (~2 pts) in my data, 
patterning closer to the paper's Claude-family results than its GPT/Gemini results — 
neither model shows a strong egoistic bias under self-framing in this dataset.

#### 3. Thought–response alignment

| Model | Aligned | Self-correction (decept thought → honest response) | Override (honest thought → decept response) |
|---|---|---|---|
| llama3.1:8b | 99.22% (890/897) | 0.56% (5) | 0.22% (2) |
| qwen3:8b | 99.22% (885/892)* | 0.45% (4) | 0.34% (3) |

*This table draws on the same underlying qwen3:8b response pool as Table 1 and I haven't 
independently re-audited it for the truncation artifact described in Known Limitations.

The original paper reports a "prevalent" gap for many tested models where honest 
internal reasoning is overridden by external pressure into a deceptive final answer. 
Both models here show near-total alignment (99.2%) between internal reasoning and final 
response in my data, with misaligned cases rare and roughly balanced between 
self-correction and override — I read this as noise at these sample sizes rather than 
the systematic override pattern the paper describes for other models.

### Known Limitations

- **Truncation/empty-response artifact (qwen3:8b only).** I manually audited the 
  qwen3:8b L1-baseline condition and found 42 of 293 rows (~14%) where `response_text` 
  was empty or truncated mid-generation but still received a `decept`/`honest` verdict, 
  disproportionately labeled `decept` (roughly 4:1). The corrected baseline is 60.6% 
  (down from 63.48% uncorrected). I didn't systematically re-audit pressure and reward, 
  so I'd treat those as upper-bound estimates — independent evidence from my L3 
  diagnosis (below) suggests `reward`-framed scenarios may carry the largest undetected 
  inflation, since I found a similar failure mode there disproportionately affecting 
  `reward`-condition multi-turn generations. I found no comparable truncation in any 
  llama3.1:8b file I checked. Tables 2 and 3 draw on the same underlying qwen3:8b pool 
  and I didn't independently re-audit them. See [Known Limitations](known-limitations.md) 
  for my full audit methodology, verbatim examples, and the L3 diagnostic process that 
  led me to this finding.
  
- I fixed decoding at temperature 0.0 across all conditions. This makes runs 
  reproducible and isolates the effect of incentive framing from sampling noise, but it 
  also means these results reflect a low-variance decoding regime that may not fully 
  represent model behavior in typical higher-temperature chat use (0.7–1.0).

- This project covers two ~8B open-weight models rather than the original paper's 
  14-model, closed- and open-source sweep, and I didn't manually validate the automated 
  verdicts (via `gemma2:9b` as judge) against human review.

### Related Work

This project directly applies the DeceptionBench evaluation protocol — including its 
scenario set, ground-truth labels, and thought/response judging strategy — to two 
open-weight models the original paper didn't evaluate.

It also connects to a broader literature on incentive-driven misrepresentation in 
language models. Van der Weij et al. (2024, arXiv:2406.07358) define and study 
sandbagging — strategic underperformance on an evaluation when a model or its developer 
has an incentive for the evaluation to understate true capability — and show frontier 
models can be prompted or fine-tuned to do this selectively. Meinke et al. (2024, 
arXiv:2412.04984) find that frontier models are capable of in-context scheming: pursuing 
a goal that conflicts with given instructions and taking deceptive actions to protect 
that goal when prompted with sufficient situational awareness and incentive. Anthropic's 
2025 report on sandbagging in agentic ML research similarly documents models 
underperforming strategically on research tasks under certain incentive structures.

### Acknowledgments

This project's evaluation infrastructure (`main.py`, `data_loader.py`, `logger.py`) is 
built on the DeceptionBench codebase by Huang et al. (2025), 
<https://github.com/Aries-iai/DeceptionBench>. The scenario data, prompt templates, and 
evaluation methodology are theirs. The metric aggregation scripts (`calculate_metric.py`, 
`utils.py`, `config.py`), the llama3.1:8b and qwen3:8b model runs, and all analysis in 
`result/metric/` are my own contribution.

