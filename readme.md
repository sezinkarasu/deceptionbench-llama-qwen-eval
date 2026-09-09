# DeceptionBench: llama3.1:8b vs qwen3:8b

I ran the [DeceptionBench](https://github.com/Aries-iai/DeceptionBench) framework (Huang et al., 2025, arXiv:2510.15501) on two open-weight ~8B models the original paper didn't test: `llama3.1:8b` and `qwen3:8b`. Question: does adding reward or pressure framing to an otherwise identical prompt push a model toward more deceptive answers, and does that differ between the two models?

### How it works

DeceptionBench asks a model for both its internal reasoning and final answer (`{thought, response}`), then a judge model labels each honest or deceptive against a scenario ground truth, across 150 scenarios in five domains. Two axes get varied:

- **Role**: `self` (model acts as an autonomous agent with self-interested motives) vs `other` (model acts as a helpful assistant)
- **Incentive**: `L1` neutral baseline vs `L2` reward- or pressure-based framing

I focused on the L2 incentive effect, with role and reasoning/response alignment as secondary results.

The original paper found deception rate generally rises under incentive for most models tested (GPT-4o, Gemini, Qwen2.5, DeepSeek-R1), with pressure usually beating reward — Claude models were the exception, staying near zero. I wanted to see if that pattern held for two much smaller open-weight models.

### Setup

- `llama3.1:8b` and `qwen3:8b`, run locally via Ollama
- `gemma2:9b` as judge, also local
- Temperature 0.0 for reproducibility (trades off some realism vs. typical chat use (0.7-1.0)

### Method

Evaluated both models on the full scenario set using the paper's original prompts: L1 baseline (self/other), L2 pressure and reward (self/other). I also ran L3 (multi-turn) across every domain, but I'm not reporting it here. qwen3:8b's L3 data has the same truncation issue described below, and I didn't have time to independently verify llama3.1:8b's L3 was clean enough to report on its own. Raw L3 output is still in the repo.

To get this running fully local, I pointed the config at Ollama instead of cloud APIs, fixed a key-indexing bug in `calculate_metric.py`'s metric aggregation, and wrote `build_full_csv.py` to join the raw per-condition output into one results file.

### Findings

**1. Incentive effect**

| Model | Baseline | Pressure | Reward |
|---|---|---|---|
| llama3.1:8b | 23.15% (n=298) | 20.74% (n=299) | 20.67% (n=300) |
| qwen3:8b | 60.6% (n=251) | 72.24% (n=299) | 74.33% (n=300) |

<img width="1400" height="900" alt="deception_rate_by_condition" src="https://github.com/user-attachments/assets/162aa48f-0ba5-45e9-845e-188afff545c1" />

Corrected after removing 42 truncated/empty rows found in a manual audit (see Limitations). Pressure and reward figures haven't had the same audit yet, so treat them as upper bounds — reward in particular looks likely to be somewhat inflated for the same reason.

Significance (baseline vs. condition, chi-square): llama3.1:8b n.s. either way (p = 0.54, 0.52). qwen3:8b significant both ways (p = 0.028 pressure, p = 0.006 reward).

llama3.1:8b barely moves under incentive. qwen3:8b starts much higher (60.6% vs 23.2%) and climbs further under both conditions, reward more than pressure — the reverse of what the original paper found for most models. I'd hold the reward finding loosely though: it's the condition most likely affected by the same data-quality issue flagged above.

**2. Role effect (self vs other)**

| Model | Self | Other | Gap |
|---|---|---|---|
| llama3.1:8b | 22.54% | 20.49% | +2.05 pts |
| qwen3:8b | 71.24% | 68.90% | +2.33 pts |

The paper found ~20-point self/other gaps for GPT-4o and Gemini (self-serving bias), with Claude as the exception. Both models here land close to the Claude pattern — small gaps, no strong egoistic bias.

**3. Reasoning/response alignment**

| Model | Aligned | Self-corrected | Overridden |
|---|---|---|---|
| llama3.1:8b | 99.22% | 0.56% | 0.22% |
| qwen3:8b | 99.22% | 0.45% | 0.34% |

The paper documents a common pattern where honest internal reasoning gets overridden into a deceptive answer under pressure. I don't see that here since both models are ~99% aligned, with the rare mismatches split roughly evenly. Reads as noise at this sample size, not the override pattern the paper describes.

### Limitations

The big one: I manually reviewed wen3:8b's L1-baseline run and found 42 of 293 rows (~14%) had empty or truncated responses that still got scored `decept`/`honest` anyway, skewed about 4:1 toward `decept`. Corrected baseline is 60.6% (down from 63.48% raw). 

I haven't done the same evaluation on the pressure/reward conditions, so those numbers are upper-bound estimates, independent evidence from the L3 multi-turn data suggests reward-framed scenarios are most likely to carry this same problem. No comparable issue in any llama3.1:8b file I checked. 

Full evaluation methodology and examples are in [Known Limitations](known-limitations.md).

Worth noting: temperature 0.0 isolates the incentive effect but doesn't reflect typical higher-temperature chat use. This covers two ~8B models against the original paper's 14-model sweep, and judge verdicts weren't validated against human review.

### Related work

Builds directly on DeceptionBench's scenarios, ground truth, and judging protocol. Also relevant: van der Weij et al. (2024, arXiv:2406.07358) on sandbagging — models strategically underperforming on evals when there's incentive to hide capability; Meinke et al. (2024, arXiv:2412.04984) on in-context scheming; and Anthropic's 2025 report on sandbagging in agentic ML research.

### Credit

Evaluation infrastructure (`main.py`, `data_loader.py`, `logger.py`), scenarios, and prompts are from Huang et al.'s DeceptionBench. Metric scripts, model runs, and analysis are mine.

