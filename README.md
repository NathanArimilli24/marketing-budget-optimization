# Optimizing Marketing Budgets with Integer Programming

Allocating a **$10M marketing budget** across **10 advertising platforms** to
maximize return, when each platform's ROI is a **tiered (piecewise-constant)**
function of how much you spend. Built with **Gurobi** in Python.

This was a graduate Optimization course project (Optimization-I, Project 2).
Team: Abhay Puri, Nathan Arimilli, Shruti Chembu Kuppuswamy, Simoni Dalal.

---

## The problem

A CMO must spread $10M across Print, TV, SEO, AdWords, Facebook, LinkedIn,
Instagram, Snapchat, Twitter, and Email. An outside firm estimates each
platform's ROI as a step function of investment: the first dollars earn one
rate, later dollars earn another. Management adds three rules:

1. Print + TV spend ≤ Facebook + Email spend
2. Social (Facebook, LinkedIn, Instagram, Snapchat, Twitter) ≥ 2 × Search (SEO, AdWords)
3. No more than **$3M** in any single platform

The catch: when ROI **declines** with spend (concave total return) a plain
**linear program** works. When a second firm's estimates **rise** in later
tiers (non-concave), the LP would cheat by skipping cheap early tiers — so the
model becomes a **mixed-integer program with SOS2** piecewise constraints.

## Approach

| Part | Question | Method |
|------|----------|--------|
| 3 | Concave ROI (firm 1) | Linear program over per-tier variables |
| 4 | Non-concave ROI (firm 2) | MIP, SOS2 convex-combination of breakpoints |
| 5 | What if the ROI estimate is wrong? | Cross-evaluate each allocation on the other firm's data |
| 6 | Minimum spend per channel | Add a binary on/off variable per platform |
| 7 | Reinvest ½ of each month's return | 12 sequential MIPs with a growing budget |
| 8 | Is the monthly plan stable? | Check month-to-month swings; describe a joint model |

## Results

All figures below are produced by the code (no hand-entered numbers).

- **Part 3 — concave ROI (LP).** Optimal spend: **TV $3M, Instagram $3M,
  Email $3M, AdWords $1M**. Return **$543,640** (5.44% on $10M).
- **Part 4 — non-concave ROI (MIP).** Optimal spend shifts to **Print $3M,
  Facebook $3M, AdWords $2.33M, LinkedIn $1.67M**. Return **$452,827** (4.53%).
- **Part 5 — robustness.** The two allocations are very different. Using the
  wrong one costs **38.7%–49.4%** of the achievable return — strong evidence
  that the $3M cap (forced diversification) is a useful hedge against bad ROI
  estimates.
- **Part 6 — minimum investment.** Adding "invest at least $X or nothing"
  thresholds leaves the firm-2 allocation **unchanged** — the optimizer already
  funded only 4 channels at meaningful levels.
- **Part 7 — reinvestment.** Rolling half of each month's return forward grows
  the budget from **$10M to $13.23M (+32.3%)** over the year, for a total
  annual return of **$7.08M** (70.8% of the initial $10M; 5.12% average monthly
  ROI). Print, Facebook, and LinkedIn are funded almost every month.
- **Part 8 — stability.** The month-by-month plan is **not stable**: 9 of 10
  platforms swing by more than $1M in some month. A stable plan would require a
  single 12-month model with `|x[p,t] − x[p,t−1]| ≤ $1M` constraints, trading a
  little return for an executable schedule.

Figures: [`output/cross_evaluation_analysis.png`](output/cross_evaluation_analysis.png),
[`output/minimum_investment_analysis.png`](output/minimum_investment_analysis.png),
[`output/monthly_allocation.png`](output/monthly_allocation.png).
Monthly numbers: [`output/monthly_allocation.csv`](output/monthly_allocation.csv).

## Repository structure

```
data/     roi_company1.csv, roi_company2.csv, min_amount.csv, roi_monthly.csv
code/     solve.py                 — generalized end-to-end script (one command)
          Group_8_Optimization_Project2.ipynb — annotated notebook with figures
output/   generated figures + monthly_allocation.csv
report/   Project_2_Writeup.pdf    — the full written report
          assignment_prompt.pdf    — the original assignment
README.md
requirements.txt
```

## How to reproduce

```bash
pip install -r requirements.txt        # needs a Gurobi license (free for academics)
cd code
python3 solve.py                       # prints every result, writes figures to ../output/
```

To run on **new data**, change only the four file paths at the top of
`code/solve.py` (or cell 2 of the notebook). The code is fully generalized — no
results are hard-coded.

## Honest caveats

- ROI values are course-provided estimates, not real market data; the point is
  the optimization method, not the specific channel rankings.
- Parts 6 and 7 cap total spend with `≤ $10M` (rather than `= $10M`) so the
  on/off and minimum-investment rules can never make the model infeasible; in
  practice the optimizer still spends the full budget every period.
- Part 8 quantifies instability and specifies the stabilizing model but does not
  re-solve it (a joint 12-month model with reinvestment is nonlinear because each
  month's budget depends on the previous month's return).
