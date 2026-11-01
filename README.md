# Experiment Decision Intelligence

**From experiment results to decisions you can defend.**

Expert-designed evidence establishes what an experiment supports. Decision modeling translates uncertainty into business consequences. AI makes those verified results easier to interpret, challenge and explore through controlled tools; an accountable human owns the decision.

The promise is **better reasoning under uncertainty**, not automatic product decisions. The demo illustrates analytical judgment, not production impact. Everything in the interface and exports is in English.

## Quick start

Python 3.12 is recommended. The statistical analysis, simulations and decision documents work without an API key; AI assistance is optional.

```bash
git clone https://github.com/Alexmmarin94/experiment-decision-lab.git
cd experiment-decision-lab
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

On Windows, activate with `.venv\Scripts\activate`. For local AI features, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and enter your OpenRouter key in that private file. It is ignored by Git.

## Deploy on Streamlit Community Cloud

1. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub and choose **Create app**.
2. Select repository `Alexmmarin94/experiment-decision-lab`, branch `main`, and main file `streamlit_app.py`.
3. In **Advanced settings**, choose Python **3.12** and enter the following TOML in **Secrets**, replacing only the key placeholder:

```toml
OPENROUTER_API_KEY = "YOUR_OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "google/gemini-2.5-flash"
```

4. Choose **Deploy**. For an existing app, update these values under **App settings → Secrets**, then save/reboot as prompted.
5. Start the example walkthrough to validate the statistical workflow. Request an AI strategic review to validate your key, credits and model access; this sends a paid provider request.

Use an **OpenRouter** API key, rather than a direct OpenAI or Google key. The model can be changed to another OpenRouter model supporting tool calling. GitHub Actions secrets are separate and are not automatically supplied to Streamlit. No key belongs in a commit or in the public example file.

**Hosting limits:** decision versions use local SQLite and can disappear when a cloud instance is replaced. Export important decisions. This app does not provide per-user authentication or user-isolated durable storage; saved history on a shared instance is not a private account vault. Use the synthetic example for public demos.

A live demo URL will be available after the app is created in Streamlit Cloud.

## Repository structure

```text
streamlit_app.py                 Streamlit interface and six-stage workflow
edi/                            Statistical inference, decision models, AI and reporting
data/experiment_users.csv       Synthetic example with 40,000 users
tests/                          Unit and Streamlit workflow tests
scripts/generate_examples.py    Reproducible synthetic dataset generator
.streamlit/config.toml          Theme and upload configuration
.streamlit/secrets.toml.example  Placeholder-only local/cloud secret template
.github/workflows/tests.yml     Python 3.12 validation on pushes and pull requests
requirements.txt                Runtime dependencies
```

## Six steps, one experiment

| Step | What the team gets |
| --- | --- |
| 01 · Welcome & Upload | A compact product introduction, immediate example/upload entry, actionable file checks, guided column mapping and an explicit owner design confirmation. |
| 02 · Can we trust the data? | Visual pass/warning/block/missing-evidence cards, exact inspectable facts, optional AI explanations and a case-specific trust commentary. |
| 03 · What did we learn? | Overall effect, Bayesian benefit/harm probabilities, guardrails, multiplicity, robustness versus heterogeneity, partial pooling, timing and optional target mix. |
| 04 · What should we do? | Ready-to-run starting assumptions with sources, alternative budgets, Monte Carlo, downside, thresholds, sensitivity and a shared-policy EVSI. |
| 05 · Explore with AI | Explicitly requested initial reading, question shortcuts, company context, read-only comparisons and confirmed business changes. |
| 06 · Decision brief | A usable decision document, local saved versions, assumption/decision comparisons and auditable exports. |

The distinction from a one-off CSV conversation is a reusable evidence and decision contract: explicit methods, preserved statistical limits, validated business tools, confirmation, deterministic recalculation and version history. A general model can analyze a CSV; it does not automatically provide this workflow or organizational context.

## CSV: minimal input and optional capabilities

One row per independent assigned user, including inactive users. Minimum:

- `unit_id`: unique ID; `unir_id` is accepted as an alias.
- `variant`: two assigned arms, mapped to control/treatment.
- One numeric outcome column selected as the primary metric in the app.

The primary metric is a column in the **same CSV**, not a second table. Missing measurements are never replaced with zeros; rows are never silently removed or deduplicated.

Optional columns:

| Input | Capability / limit |
| --- | --- |
| Any `*_guardrail` | Safety assessment with declared harmful direction and deterioration limit. Without guardrails, safety is unassessed. |
| `assigned_at` or first-exposure information | Time/context diagnostics where supported. The example uses **only assignment time**, not both assignment and first exposure. |
| `window_end_at`, `observed_through` | Window ordering/maturity checks; without them, owner confirmation is necessary. |
| Baseline categorical splits such as `platform` | Subgroups, robustness, heterogeneity, partial pooling and target-mix scenarios. |
| Pre-treatment numeric attributes | Optional balance diagnostics. |
| Equal-duration early/late outcomes | Within-user time-effect contrast; dates alone do not identify novelty. |
| Separate numerator and denominator | Ratio-of-sums inference preserving within-user covariance. |

The only example, `data/experiment_users.csv`, contains 40,000 synthetic users, conversion, a crash guardrail, assignment/maturity dates, baseline splits, pre-period sessions, experimental sessions, heavy-tailed revenue and first-purchase outcomes for days 1–3 and 4–6. Its positive primary estimate has p≈0.098: it is not a conventionally significant win. The case supports useful uncertainty and planning analysis without retrospectively “rescuing” the primary result.

Regenerate it with `python -m scripts.generate_examples`. All data are fictional.

### Guided uploads and state

The first screen presents product purpose and the two entry paths before the detailed dictionary. The minimal format is illustrated with four fictional records (a structural template, not a statistically adequate sample). The full CSV specification and optional-column guide are expandable. The example’s p-value is not revealed in onboarding.

After load, the workspace confirms rows/columns, previews records and groups configuration into identity, assignment, outcome and design. Custom ID/assignment names can be mapped; `unit_id`, `unir_id` and `variant` remain compatible. IDs are initially read as strings, preserving leading zeros. The normalized analysis copy is separate from the supplied records; no missing values are imputed or users dropped.

File/encoding/separator errors, unmapped columns, unusable outcomes and design limitations receive specific messages. A new uploaded file requires its own design confirmation. A confirmation is owner-reported evidence, not independent verification by the app. The planned allocation remains an original-design input, not a value inferred to pass SRM.

Click **Start the example walkthrough** or **Start analysis of my CSV** to begin. Before that click only preview/mapping is available: quality checks, inference, decision simulations and AI actions do not run. Start permission is tied to the source file and assignment mapping, so a different uploaded file must be started separately. Continue buttons guide the six stages. Business context survives metric changes within the same source. Business changes show changed inputs, before/after modeled value and loss risk, and any change in preferred alternative; protected statistical evidence remains unchanged. AI confirmations and local version comparisons are retained.

### Why no CUPED?

CUPED is intentionally **not implemented**. The project focuses on uncertainty, decision thresholds, alternatives and the value of evidence. Requiring additional pre-treatment covariates would complicate that purpose and the minimum input contract. Covariates are optional diagnostics, not prerequisites for the primary analysis.

## Human-friendly goals and defaults

Enter a **relative percentage improvement**, such as 5%: an 8% conversion rate becomes 8.4%, or a 200 ms delay becomes 190 ms. Internally the goal is translated to a metric-scale threshold. Observed-effect charts show original-scale effect intervals: percentage points for binary outcomes and defined metric units for numeric/ratio outcomes. Control/treatment values and relative improvement are shown separately; an absolute-effect interval is never relabeled as an exact relative interval. A zero baseline needs an explicit planning reference. A negative baseline uses its magnitude and is not interpreted as a conventional growth rate.

Binary posterior useful-effect probabilities retain uncertainty in the paired control posterior draws. Numeric/ratio posteriors condition on the estimated control reference and sampling variance; their approximation is disclosed.

The business model starts ready to run. Traffic is an explicitly labeled extrapolation of assignment pace, or observed denominator volume for ratios, when timestamps permit it. Margins, engineering budgets, maintenance, persistence and reversibility cannot be identified by this CSV: defaults are low-confidence expert/hypothetical starting assumptions, not a universal best case.

Every modeled economic parameter has a source, confidence label and rationale. Available sources include observed history, observed experiment, finance estimate, expert judgment, hypothetical scenario and unverified AI suggestion. Confidence is provenance metadata; it does not silently change distribution widths. Manual edits with unchanged original rationales are relabeled for validation. Advanced controls are secondary to the common business inputs.

## Reading the charts and filling inputs

Every editable widget has an information tooltip explaining the value and a criterion for filling it. Editable tables have column-level guidance. The statistical settings remain optional/secondary to the business questions.

- The overall result first explains the experiment comparison, then introduces **prior → data → posterior** before displaying the Bayesian histogram. Colored regions distinguish harm, sub-goal improvement and meeting the business goal. Bars count modeled possibilities, not users, on a shared probability scale.
- Probability/method details use readable question-based charts. Segment drilldown separates multiple-testing protection, evidence of differences between groups, and partial pooling. Hollow-to-filled points show how sharing information changes estimates; ranges show remaining model uncertainty. A separate before/after curve explains the assumptions about group variation.
- **Total eligible people** is the applicable population before deployment coverage. **Expected affected share** is the fraction expected to receive the change. Their product is recipients/month, not purchasers. Do not apply coverage again for non-conversion or behavior already averaged into an ITT effect.
- The comparison period is explicitly the number of months of benefits and costs, not test duration. The durability input is in the projection-assumptions expander, explicitly distinguished from customer retention and tested in sensitivity.
- The sensitivity view labels current value and break-even, highlights reach, and exposes the input at each endpoint. A coverage threshold shows when central expected economics can cross break-even; it is not a safety or downside-risk guarantee.
- Iteration/retest budget is always a main input. Gradual launch is opt-in, defined as a fraction of intended recipients, and never enabled by an AI tool.
- Financial payback and additional-sample views explain what is projected, what the uncertainty band means, and why information has value only if better choices repay collection/delay costs.

## Trust and statistical evidence

Checks distinguish blockers, warnings and unavailable evidence. They cover duplicate/cross-arm IDs, exact repeated records, SRM (block at p<0.001), finite/missing values, domains, ratio consistency/stability, outliers, baseline balance, exposure where supplied, and mature/comparable windows. The workspace supports independent-user, complete-assignment, fixed-horizon analysis; an owner confirms this scope. Clustered, filtered or sequential analysis requires different methods and is not implemented.

Baseline standardized differences are retained in technical diagnostics rather than shown as an unexplained chart. Thresholds such as |SMD|>0.1, exposure gaps >2 pp and unusual tail shares are investigation heuristics, not certificates or automatic exclusions. Missing first-exposure data in the simpler example is identified as an unavailable check, not a discovered exposure defect.

The optional AI quality layer receives aggregate computed facts, translates non-pass details into findings/implications/next checks, and comments on trust. It cannot change computed statuses or certify causality. Original facts remain visible even if the explanation is unavailable or wrong.

Inference methods:

- Binary: pooled two-proportion z test for adequate expected cells; Fisher exact test for sparse cells; Newcombe-Wilson difference interval.
- Numeric/count mean: Welch unequal-variance t test and interval.
- Ratio: arm ratio of sums with unit-level delta linearization and Welch-Satterthwaite uncertainty; not a mean of per-user ratios. Denominator limitations are flagged.
- Numeric outlier sensitivity: original mean plus pooled upper-p99 capped mean. The capped mean is a different exploratory estimand; its interval conditions on the estimated cutoff. Original outcomes still feed decisions.
- Guardrails: protected harmful directions and limits; Bonferroni-adjusted confidence intervals over all supplied guardrails. No significant deterioration is not a safety demonstration.

Binary Bayesian inference uses independent beta-binomial posteriors and a weak Beta(0.5,0.5) prior. Historical rate/strength requires source and independence confirmation. Numeric/ratio effects use a zero-centered normal prior and approximate normal effect likelihood with plug-in variance. Prior sensitivity is shown together; the AI cannot change priors or pick the favorable result.

The evidence contract separates `raw_probabilities` (treatment minus control) from `directional_probabilities` (benefit, harm and useful improvement using the metric’s beneficial direction). The copilot is instructed to use the latter for recommendations. A latency increase is harm, even when the raw-effect probability is positive.

## Robustness, heterogeneity and partial pooling

These are distinct interpretations:

- **Robustness:** do estimates tell a broadly consistent story, allowing for uncertainty and small groups?
- **Heterogeneity:** is there evidence that effects differ across groups?

Every displayed subgroup remains in a declared exploratory family across selected attributes, even when drilling down. Holm and BH p-values are available; BH requires appropriate dependence assumptions. Omnibus inverse-variance Q tests have a separate Holm family across attributes. Group confidence intervals remain marginal. None of this erases post hoc selection or independently validates targeting.

The hierarchical model uses `estimated_effect_s ~ Normal(delta_s, SE_s²)`, `delta_s ~ Normal(mu, tau²)`, a zero-centered normal prior on mu and half-normal prior on tau. Hyperprior scales are explicitly tied to the declared useful-effect scale (mu SD=5×threshold; tau SD=2×threshold). Mu is analytically marginalized and tau is integrated with nonuniform quadrature; posterior simulation retains uncertainty in both hyperparameters. A grid-tail diagnostic is exported.

Outputs include no-pooling effect, pooled posterior mean, credible interval, probability of useful benefit and posterior heterogeneity magnitude. The likelihood is an approximation on subgroup effects, not an exact binary hierarchy. Few groups make tau prior-sensitive. Shrinkage is **not** a substitute for frequentist multiplicity correction.

The primary weekly view compares the overall effect among all users assigned in each UTC Monday–Sunday week, across all segments. Rows show actual calendar ranges in chronological order, earliest at the top; observed partial-week spans remain in tooltips/audit. It is not a cumulative-user or daily-event effect. The example’s numeric weekly results are unchanged; cross-year dates are now kept in their true shared calendar week.

A separate optional time analysis contrasts each user’s late-minus-early outcome, retaining within-user covariance and zeros. Equal complete windows require confirmation. Assignment cohorts are a separate calendar view; cohort p-values are Holm-corrected in the export. Timing, purchase acceleration, instrumentation and novelty can resemble one another; a seven-day export does not establish long-run persistence.

Optional target standardization requires nonnegative sum-to-one user shares, overlap and within-group transportability. Ratios require target denominator weights and are not transported by user shares. EVSI is disabled for a transported posterior until a corresponding future stratified sampling plan is specified.

## One shared decision model

`edi/decision_policy.py` owns the reusable action and readiness policy. The default numerical action set is keep control and global launch. Gradual launch is optional and disabled by default; enabling it manually adds it to both the comparison and EVSI. Iteration has a technical budget and requires a new experiment, but is unranked because its future effect is unknown. The table explicitly distinguishes “not estimated” from zero.

All quantified actions use the same simulated worlds. Gradual rollout gets a literal fraction of the same opportunity, a separate launch budget and unchanged maintenance. Rollback is an assumption, not recovery of sunk technical costs. New causal evidence requires a planned randomized holdout; monitoring by itself is not enough.

Monte Carlo combines posterior effect draws with declared lognormal population/margin/cost uncertainty, bounded reach and monthly carryover of the product improvement (benefit durability, not user retention), and a mean-one transport multiplier. The latter scales the effect and cannot model arbitrary shifts or sign reversals. Optional margin/reach dependence exists in the engine; starting inputs are independent. Maintenance/opportunity costs are explicit; no discounting. Sources and distributions are assumptions unless independently validated.

The shared objective is:

`E[net value − loss_weight × max(−net value, 0)]`

PM-facing preferences map to expected money, balanced downside weighting or more cautious weighting. Reversibility and current validity/safety determine feasible actions. Expected net value, loss probability and the preference-adjusted **decision score** remain separate; a score is not a profit forecast. Quality warnings keep the result conditional rather than being silently ignored.

Sensitivity uses common random numbers and explicit one-at-a-time ranges. Break-even is a central-input threshold, not a success probability. The perfect-information value uses the same feasible actions and utility.

### EVSI now shares that model

The active workspace uses `shared_evsi()` in `edi/scenarios.py`, not the earlier launch/no-launch central-economics helper retained as a reference in `edi/uncertainty.py`.

Nested posterior-predictive simulation updates the primary-effect posterior, integrates the **same economic distributions**, compares the **same quantified feasible actions**, and applies the **same loss-weighted utility**. Study/delay costs enter future payoffs **before** utility, so a downside-sensitive decision is not approximated by simply subtracting cash from utility afterward.

Binary samples use beta-binomial updating; numeric/ratio effects use an approximate normal-normal update. Evidence readiness and reversibility stay fixed. The model learns only the primary effect, not guardrail validity, engineering costs, transportability or the return to an unbuilt iteration. Iteration is unranked in both optimizers. The horizon is held fixed and delay costs are explicit.

Finite inner simulation can bias the maximization upward. The UI exports outer SE and typical inner-score simulation error; these do not bound model error or nested bias. Small/negative estimates can reflect numerical noise. A study designed to learn guardrails or economics is a different sampling model, not a hidden capability of this EVSI.

## AI: explanations, Strategic Review and controlled scenarios

Configure `.streamlit/secrets.toml` (ignored by Git):

```toml
OPENROUTER_API_KEY = "your-api-key"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "google/gemini-2.5-flash"
```

Quotes are correct. `.example` is a shareable placeholder, not the private configuration. Restart Streamlit after editing secrets. Environment variables are also supported.

AI assistance is selected by default in Welcome and can be turned off. Starting the dataset unlocks the analysis; it does not itself send an AI request. Enabling it, opening a tab or changing assumptions makes **no automatic provider request**. Request check explanations, request a Strategic Review, click a question shortcut or send a chat message when you want AI. Those actions can incur provider charges; tool loops can require several requests. Deterministic summaries and the complete evidence/decision/report workflow remain usable without AI. Requests that fail do not block calculations.

Click **Get an AI strategic review** to obtain a first AI reading without filling in any extra fields. Company context is in a collapsed optional section. Click **Save context** to use the notes in later questions/reports, or **Get an AI strategic review** to submit fresh text and generate the review together. Both buttons submit the same form: there is no need to press Enter or save separately before generation. Enter adds a new line. Save alone makes no provider request. The submitted notes are processed before generation and personalize the review. The review identifies the real decision tension, prioritizes binding constraints over economic sensitivities, challenges material assumptions, uses computed counterfactuals, and suggests a next action. It is distinct from the final Decision Brief, which records evidence, conditions and owner judgment. The first review is read-only and cannot prepare a pending assumption update. Its instructions require a Months of impact driver, and the UI also displays the deterministic month comparison with the review, even if the provider omits it. No reliable change or an eligibility block is stated explicitly; a longer financial period is not treated as observed follow-up or proof of durability. Concrete follow-up requests can prepare proposals. No company finance or engineering facts are inferred from user counts.

`edi/decision_drivers.py` reuses the authoritative `scenarios.run()` and decision policy. It distinguishes rollout eligibility, expected-profit break-even and downside-adjusted preferences. A bounded grid tests up to six existing sensitivity ranges with at most 4,000 selected posterior worlds per setting, seed 23 and shared random numbers. It reports tested preference-switch brackets, paired score-gap simulation errors and cases with no switch; it does not invent exact thresholds, assume monotonicity or extrapolate. The comparison baseline may differ slightly from the full simulation; this is explicitly recorded. Hard constraints remain binding even when hypothetical economics improve. The grid does not estimate the research value of an assumption. A separate impact-period comparison tests every supported whole month from 1 to 36, using the same posterior worlds, seed, policy and fixed monthly durability. Both fading benefits and maintenance accumulate, so the preferred choice can switch more than once. It reports tested whole-month choices and simulation-ambiguous intervals without inventing a continuous-time cutoff. The cache includes the driver-schema version so this calculation is refreshed after an analytical capability change.

Review caching includes evidence, confirmed scenario, policy, current user context and prompt version. Material changes mark the previous reading stale; no automatic provider request is made. The deterministic decision-driver summary remains available without an API key. OpenRouter insufficient-credit errors (402), authentication, rate limits and connection failures receive separate actionable messages without exposing provider response bodies or keys.

Supported tools:

| Tool | Effect |
| --- | --- |
| `get_decision_summary` | Read current authoritative outputs. |
| `get_decision_drivers` | Read binding constraints, payoff trade-offs, bounded counterfactuals and limitations. |
| `inspect_quality_check` | Read one computed check. |
| `compare_scenario` | Calculate an allowlisted hypothetical; active assumptions stay unchanged. |
| `propose_scenario_update` | Validate business changes, calculate a preview and prepare confirmation. |

The loop is bounded. Unknown/protected inputs, nonfinite numbers, invalid ranges and stale proposals are rejected. No arbitrary code, file access or external service actions are available. The AI cannot change data, priors, significance rules, guardrail limits, metric direction or decision preferences. Annual maintenance is converted by the server; an unpriced two-week project delay remains a constraint until the user supplies its financial meaning.

The UI explains tool permissions in plain language without displaying raw JSON; machine-readable evidence remains a separate audit download. The preview explains the business assumption first, using annual maintenance, currency, affected percentages and improvement durability in consistent units. It shows current versus hypothetical net value, loss probability, downside-adjusted score and preferred feasible action, with an explanation of whether the choice changes or a blocker remains. **Use this scenario** applies the preview; **Keep my current scenario** discards it. Detailed tables remain optional. Confirmation saves the previous decision version, applies only validated business inputs, recalculates and saves the next version. Before/after scenarios, results, reasons, timestamps and fingerprints are retained. A proposal is not an applied change. Model instructions guide behavior but do not make generated prose infallible.

No individual rows or IDs are sent. Selected group labels, aggregate computed facts, optional business context and recent conversation are included. Avoid sensitive information or secrets in those fields. The chat itself stays in the session; tool proposals and confirmed changes have the scoped audit behavior above.

## A real decision brief and local history

The concise brief also summarizes corrected subgroup improvements/deterioration separately from omnibus evidence of different effects. Uncertain groups are not labeled failed/nonresponding. Strategically interesting groups are candidates for business-context investigation and independent confirmation, not an automatic targeted launch. Segment-specific value and implementation-policy payoffs are not newly modeled.

The Decision Brief tab begins with a clearly labeled disclaimer box, before its heading and report controls: secondary outcomes/interactions, learning from iterative releases, strategic bets and goals such as acquisition or margin can broaden a decision beyond immediate modeled profit. It points to Explore with AI for evidence-grounded discussion, without implying that untested interactions are established or safety constraints disappear. The disclaimer belongs to the interface and is excluded from both exported report formats.

The default brief always includes a **Months of impact** condition, alongside the priority list, with the current period and computed choices that change it—or an explicit no-switch, numerical-uncertainty or eligibility limitation.

The default brief uses plain English and a numbered **What would change the choice?** list generated by `edi/brief_drivers.py`. Hard data/safety/rollback constraints come first. Without a hard gate, assumptions with credible tested preference changes rank first; within that class, priority follows the change in global-launch downside-adjusted score over each declared range. This is a quantified scenario ranking, not arbitrary importance weights, a probability that an assumption is wrong, or research ROI. Each item names the current input, its source/confidence, a computed alternative setting and its consequences. A switch between grid settings is a range, never a claim that every value below one endpoint must change the decision. Close simulation results and disagreement with the full decision run are identified. Numeric scores and exact parameters remain in the appendix/audit; default bullets show financial consequences and the preferred action in business units. Ratio traffic remains denominator volume, not people.

The deterministic brief remains complete without AI. An optional **Generate an AI explanation for the brief** action adds a short interpretation of the recorded decision and ranked conditions. It cannot call tools or update the recommendation. No automatic call occurs on start, navigation, context save or report export. Generated explanations are associated with evidence, confirmed scenario, saved context, owner notes and prompt version. Material changes remove an outdated explanation from the current report until regenerated. When included, the explanation and its provenance appear in the Markdown, appendix, JSON and saved version; previous versions retain their own explanation. A confirmed business change preserves the valid pre-change explanation in the before version and does not carry it into the changed scenario.

A separate detailed appendix retains all assumptions, source quality, methods, scores, alternatives, limitations and audit references. Company/owner statements and AI interpretation stay distinct from observed evidence.

Save versions under the same experiment ID and compare changed values, source quality, statistical-contract fields and decisions. Versions are append-only; repeating an identical save reuses the latest version. JSON retains the full evidence contract and seeds. AI-confirmed changes save both the before and after versions automatically.

History is local SQLite at `.local/decisions.sqlite3`, excluded from Git. It holds aggregate evidence and entered report context, not raw records or the conversation. Export/back it up if needed. A hosted app needs durable private storage; ephemeral Streamlit hosting would not guarantee local history persistence.

## Verify

```bash
python -m unittest discover -s tests -v
```

Tests cover statistical identities, diagnostics, ratio covariance, direction-aware probabilities, hierarchy/shrinkage and quadrature, shared policy/EVSI utility, protected inputs, stale/read-only proposals, bounded provider tool calls, version persistence, report sections and complete Streamlit flows—including before/after confirmation without altering the statistical posterior. Provider calls are mocked; the suite performs no paid requests. Added journeys cover valid/custom-name uploads, invalid/missing fields, unchanged source data, dataset-specific design confirmation, context preservation, scenario updates and explicit AI requests. Golden checks preserve the example primary estimate, p-value and confidence interval.

Validation limitation: browser automation and browser tools were not available in this environment. Streamlit AppTest, schema/inference/report tests and Plotly structure/format checks were performed; visual inspection at desktop/narrow viewport was **not** performed or claimed.

Presentation helpers in `presentation.py` and contextual input guidance in `input_help.py` keep charts and editable fields readable without changing inference. Onboarding/schema logic is isolated in `onboarding.py` and `schema.py`; `interpretation.py` summarizes existing segment evidence without adding an estimator. A state-controlled workflow selector and keyed panels make bottom Continue buttons select the actual visible step. All panels continue registering their widgets so navigation preserves assumptions and evidence. Native tabs remain inside Evidence.

`proposal_ui.py` owns business-unit proposal presentation; it does not mutate or confirm scenarios. Active calculation modules: `quality.py`, `inference.py`, `hierarchy.py`, `decision_policy.py`, `scenarios.py`, `tools.py`, `copilot.py`, `reporting.py`. Streamlit coordinates presentation/state while calculation and policy remain callable without UI or LLM.

Not implemented: CUPED, cluster/sequential estimators, exact generative heavy-tail/ratio hierarchies, a general daily-panel model, automatic data repair, arbitrary transport correction, iteration-effect forecasts or production impact validation.

Deployment uses the source, dependencies, synthetic CSV and placeholder-only secret template. Private secrets, virtual environments and local decision history are excluded from Git.

Method references: [Stan on hierarchical partial pooling](https://mc-stan.org/docs/stan-users-guide/regression.html), [OpenRouter tool calling](https://openrouter.ai/blog/tutorials/tool-calling/), [value-of-information methods](https://www.valueinhealthjournal.com/article/S1098-3015%2820%2930036-X/fulltext), [SciPy Welch test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html).

OpenRouter error semantics: [official error reference](https://github.com/OpenRouterTeam/docs/blob/main/api_reference/errors-and-debugging.mdx). Statistical algorithms and the synthetic example data are unchanged by this UX revision.
