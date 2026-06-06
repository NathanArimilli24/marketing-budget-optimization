"""
Marketing Budget Allocation via Integer Programming
===================================================
End-to-end, reproducible solution for Optimization Project 2.

Allocates a $10M marketing budget across 10 platforms whose ROI is a tiered
(piecewise-constant) function of the amount invested, subject to the boss's
business rules. Covers:

  Part 3  Concave ROI (firm 1)      -> Linear Program
  Part 4  Non-concave ROI (firm 2)  -> MIP with SOS2 piecewise formulation
  Part 5  Cross-evaluation          -> cost of using the wrong allocation
  Part 6  Minimum investment        -> binary "invest or don't" decisions
  Part 7  Monthly reinvestment       -> 12-period dynamic budget
  Part 8  Stability check            -> month-to-month change analysis

The code is fully generalized: to run on new data, change only the four paths
in the DATA INPUT block below. No results are hard-coded.

Run:  python3 solve.py            (from the code/ directory)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: save figures without a display
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

# =====================================================================
# DATA INPUT — paths to the ROI CSV files.
# To run on new data, change ONLY these four paths. Everything else is
# generalized and references variable names, with no hard-coded numbers.
# =====================================================================
ROI1_PATH        = "../data/roi_company1.csv"
ROI2_PATH        = "../data/roi_company2.csv"
MIN_AMOUNT_PATH  = "../data/min_amount.csv"
ROI_MONTHLY_PATH = "../data/roi_monthly.csv"

OUTPUT_DIR = "../output"

# Model constants
BUDGET = 10.0   # $M total budget
CAP    = 3.0    # $M maximum per platform

# Boss's strategic groupings
PT     = ["Print", "TV"]
FE     = ["Facebook", "Email"]
SOCIAL = ["Facebook", "LinkedIn", "Instagram", "Snapchat", "Twitter"]
SEARCH = ["SEO", "AdWords"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def load_roi(path, ub_col="UpperBound"):
    """Load a tiered-ROI table and turn 'inf' upper bounds into np.inf."""
    df = pd.read_csv(path)
    df[ub_col] = pd.to_numeric(df[ub_col], errors="coerce").fillna(np.inf)
    return df


def build_breakpoints(df, lb_col="LowerBound", ub_col="UpperBound", cap=CAP):
    """
    Convert each platform's piecewise-constant ROI table into (x, F(x))
    breakpoints for an SOS2 formulation, capped at `cap` per platform.

    Returns dict[platform] -> (list_of_spend_breakpoints, list_of_return_values)
    """
    bp, val = {}, {}
    for p in df["Platform"].unique():
        tiers = df[df["Platform"] == p].sort_values(lb_col)
        b_list, v_list = [0.0], [0.0]
        cur_b, cur_v = 0.0, 0.0
        for _, r in tiers.iterrows():
            lb = float(r[lb_col])
            ub = float(min(r[ub_col], cap))
            if ub <= cur_b:
                continue
            seg_lb, seg_ub = max(cur_b, lb), ub
            if seg_ub > seg_lb:
                cur_v += (seg_ub - seg_lb) * float(r["ROI"])
                cur_b = seg_ub
                b_list.append(cur_b)
                v_list.append(cur_v)
            if cur_b >= cap:
                break
        bp[p], val[p] = b_list, v_list
    return bp, val


def actual_return(roi_df, allocation, lb_col="LowerBound", ub_col="UpperBound"):
    """
    Compute the TRUE return of a dollar allocation under a given ROI table by
    filling each platform's tiers from the bottom up. Works for concave and
    non-concave ROI alike (dollars in a tier earn that tier's ROI regardless).
    """
    total = 0.0
    for platform, invest in allocation.items():
        if invest <= 0:
            continue
        dfp = roi_df[roi_df["Platform"] == platform].sort_values(lb_col)
        remaining = invest
        for _, row in dfp.iterrows():
            lb, ub, roi = row[lb_col], row[ub_col], row["ROI"]
            tier_cap = ub - lb if np.isfinite(ub) else remaining
            amount = min(remaining, tier_cap)
            if amount > 0:
                total += amount * roi
                remaining -= amount
            if remaining <= 1e-6:
                break
    return total


# ---------------------------------------------------------------------
# Part 3 — Concave ROI (firm 1): Linear Program
# ---------------------------------------------------------------------
def solve_lp_concave(roi1):
    """LP over per-tier continuous variables. Valid because firm-1 ROI is
    non-increasing within each platform => total return is concave."""
    # Per-tier available width, capped so per-platform total cannot exceed CAP
    widths = np.array([
        max(min(r["UpperBound"], CAP) - r["LowerBound"], 0.0)
        for _, r in roi1.iterrows()
    ])

    m = gp.Model("LP_concave")
    m.Params.OutputFlag = 0
    x = m.addMVar(len(roi1), lb=0.0, ub=widths)
    m.setObjective(roi1["ROI"].values @ x, GRB.MAXIMIZE)

    m.addConstr(x.sum() == BUDGET)
    for p in roi1["Platform"].unique():
        idx = [i for i, plat in enumerate(roi1["Platform"]) if plat == p]
        m.addConstr(x[idx].sum() <= CAP)
    pt = [i for i, p in enumerate(roi1["Platform"]) if p in PT]
    fe = [i for i, p in enumerate(roi1["Platform"]) if p in FE]
    m.addConstr(x[pt].sum() <= x[fe].sum())
    soc = [i for i, p in enumerate(roi1["Platform"]) if p in SOCIAL]
    sea = [i for i, p in enumerate(roi1["Platform"]) if p in SEARCH]
    m.addConstr(x[soc].sum() >= 2.0 * x[sea].sum())

    m.optimize()
    alloc = {}
    for p in roi1["Platform"].unique():
        idx = [i for i, plat in enumerate(roi1["Platform"]) if plat == p]
        alloc[p] = float(x[idx].sum().getValue())
    return alloc, m.objVal


# ---------------------------------------------------------------------
# Part 4 / 6 / 7 — SOS2 MIP (handles non-concave ROI, optional min-invest)
# ---------------------------------------------------------------------
def solve_sos2(bp, val, budget=BUDGET, min_map=None, budget_equality=True):
    """
    SOS2 piecewise-linear MIP. If `min_map` is given, each platform gets a
    binary on/off variable enforcing spend >= minimum when active.
    """
    platforms = list(bp.keys())
    m = gp.Model("SOS2")
    m.Params.OutputFlag = 0

    lam, Spend, Return = {}, {}, {}
    z = m.addVars(platforms, vtype=GRB.BINARY, name="z") if min_map else None

    for p in platforms:
        K = len(bp[p])
        lam[p] = m.addVars(K, lb=0.0, name=f"lam[{p}]")
        rhs = z[p] if min_map else 1.0
        m.addConstr(gp.quicksum(lam[p][k] for k in range(K)) == rhs)
        m.addSOS(GRB.SOS_TYPE2, [lam[p][k] for k in range(K)], bp[p])

        Spend[p]  = m.addVar(lb=0.0, ub=CAP, name=f"Spend[{p}]")
        Return[p] = m.addVar(lb=0.0, name=f"Return[{p}]")
        m.addConstr(Spend[p]  == gp.quicksum(bp[p][k]  * lam[p][k] for k in range(K)))
        m.addConstr(Return[p] == gp.quicksum(val[p][k] * lam[p][k] for k in range(K)))
        if min_map:
            m.addConstr(Spend[p] >= min_map[p] * z[p])

    spend_sum = gp.quicksum(Spend[p] for p in platforms)
    if budget_equality:
        m.addConstr(spend_sum == budget)
    else:
        m.addConstr(spend_sum <= budget)

    m.addConstr(gp.quicksum(Spend[p] for p in PT if p in platforms)
                <= gp.quicksum(Spend[p] for p in FE if p in platforms))
    m.addConstr(gp.quicksum(Spend[p] for p in SOCIAL if p in platforms)
                >= 2.0 * gp.quicksum(Spend[p] for p in SEARCH if p in platforms))

    m.setObjective(gp.quicksum(Return[p] for p in platforms), GRB.MAXIMIZE)
    m.optimize()

    alloc = {p: Spend[p].X for p in platforms}
    return alloc, m.objVal


def fmt_alloc(alloc):
    return ", ".join(f"{p} ${v:.2f}M" for p, v in
                     sorted(alloc.items(), key=lambda kv: -kv[1]) if v > 1e-6)


# =====================================================================
# Main
# =====================================================================
def main():
    roi1 = load_roi(ROI1_PATH)
    roi2 = load_roi(ROI2_PATH)
    min_map = dict(pd.read_csv(MIN_AMOUNT_PATH).itertuples(index=False, name=None))

    print("=" * 70)
    print("MARKETING BUDGET ALLOCATION — INTEGER PROGRAMMING")
    print("=" * 70)

    # ----- Part 3: LP on concave firm-1 ROI -----
    alloc_lp, obj_lp = solve_lp_concave(roi1)
    print("\n[Part 3] Concave ROI (firm 1) — Linear Program")
    print("  Allocation:", fmt_alloc(alloc_lp))
    print(f"  Optimal return: ${obj_lp*1e6:,.0f}  ({obj_lp/BUDGET*100:.2f}% avg ROI)")

    # ----- Part 4: SOS2 MIP on non-concave firm-2 ROI -----
    bp2, val2 = build_breakpoints(roi2)
    alloc_mip, obj_mip = solve_sos2(bp2, val2)
    print("\n[Part 4] Non-concave ROI (firm 2) — SOS2 MIP")
    print("  Allocation:", fmt_alloc(alloc_mip))
    print(f"  Optimal return: ${obj_mip*1e6:,.0f}  ({obj_mip/BUDGET*100:.2f}% avg ROI)")

    # ----- Part 5: cross-evaluation -----
    ret_lp_on_roi2  = actual_return(roi2, alloc_lp)
    ret_mip_on_roi1 = actual_return(roi1, alloc_mip)
    loss_a = obj_lp  - ret_mip_on_roi1          # ROI1 true, used firm-2 alloc
    loss_b = obj_mip - ret_lp_on_roi2           # ROI2 true, used firm-1 alloc
    print("\n[Part 5] Cross-evaluation (cost of the wrong allocation)")
    print(f"  ROI1 true, firm-2 allocation: ${ret_mip_on_roi1*1e6:,.0f} "
          f"(loss ${loss_a*1e6:,.0f}, {loss_a/obj_lp*100:.2f}%)")
    print(f"  ROI2 true, firm-1 allocation: ${ret_lp_on_roi2*1e6:,.0f} "
          f"(loss ${loss_b*1e6:,.0f}, {loss_b/obj_mip*100:.2f}%)")

    _plot_cross_eval(obj_lp, obj_mip, ret_lp_on_roi2, ret_mip_on_roi1,
                     loss_a, loss_b)

    # ----- Part 6: minimum-investment MIP (firm-2 ROI) -----
    alloc_min, obj_min = solve_sos2(bp2, val2, min_map=min_map,
                                    budget_equality=False)
    print("\n[Part 6] Minimum-investment rule (firm 2) — SOS2 MIP + binaries")
    print("  Allocation:", fmt_alloc(alloc_min))
    print(f"  Optimal return: ${obj_min*1e6:,.0f}  "
          f"({sum(1 for v in alloc_min.values() if v>1e-6)}/{len(alloc_min)} platforms funded)")

    # ----- Part 7: monthly reinvestment -----
    roi_m = load_roi(ROI_MONTHLY_PATH, ub_col="UpperBoundM")
    months = list(roi_m["Month"].unique())
    budgets = {months[0]: BUDGET}
    rows = []
    for i, month in enumerate(months):
        dfm = roi_m[roi_m["Month"] == month].reset_index(drop=True)
        bpm, valm = build_breakpoints(dfm, lb_col="LowerBoundM",
                                      ub_col="UpperBoundM")
        b = budgets[month]
        alloc_m, ret_m = solve_sos2(bpm, valm, budget=b, min_map=min_map,
                                    budget_equality=False)
        row = {"Month": month, "Budget": b, "Return($M)": ret_m,
               "ROI(%)": ret_m / b * 100}
        row.update(alloc_m)
        rows.append(row)
        if i + 1 < len(months):
            budgets[months[i + 1]] = b + 0.5 * ret_m

    results = pd.DataFrame(rows).fillna(0).round(4)
    results.to_csv(os.path.join(OUTPUT_DIR, "monthly_allocation.csv"), index=False)
    total_return = results["Return($M)"].sum()
    print("\n[Part 7] Monthly reinvestment (12 periods)")
    print(f"  Budget: ${results['Budget'].iloc[0]:.2f}M (Jan) -> "
          f"${results['Budget'].iloc[-1]:.2f}M (Dec), "
          f"{(results['Budget'].iloc[-1]/results['Budget'].iloc[0]-1)*100:.1f}% growth")
    print(f"  Total annual return: ${total_return*1e6:,.0f}  "
          f"({total_return/BUDGET*100:.1f}% of initial $10M; "
          f"{results['ROI(%)'].mean():.2f}% avg monthly ROI)")

    _plot_monthly(results, months)

    # ----- Part 8: stability check -----
    plat_cols = [c for c in results.columns
                 if c not in ("Month", "Budget", "ROI(%)", "Return($M)")]
    unstable = [p for p in plat_cols if (results[p].diff().abs()[1:] > 1.0).any()]
    print("\n[Part 8] Stability check (month-to-month change <= $1M)")
    if unstable:
        print(f"  NOT stable. {len(unstable)}/{len(plat_cols)} platforms swing >$1M: "
              + ", ".join(unstable))
    else:
        print("  Stable — all monthly changes <= $1M.")

    print("\n" + "=" * 70)
    print(f"Figures and monthly_allocation.csv written to {OUTPUT_DIR}/")
    print("=" * 70)


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------
def _plot_cross_eval(obj_lp, obj_mip, ret_lp_on_roi2, ret_mip_on_roi1,
                     loss_a, loss_b):
    scenarios = ["Company 1\nROI Data", "Company 2\nROI Data"]
    c1 = [obj_lp, ret_mip_on_roi1]
    c2 = [ret_lp_on_roi2, obj_mip]
    x = np.arange(len(scenarios)); w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.bar(x - w/2, c1, w, label="Company 1 Allocation", color="#3498db", edgecolor="black")
    ax1.bar(x + w/2, c2, w, label="Company 2 Allocation", color="#e74c3c", edgecolor="black")
    for xs, ys in [(x - w/2, c1), (x + w/2, c2)]:
        for xi, yi in zip(xs, ys):
            ax1.text(xi, yi, f"${yi:.4f}M", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Expected Return ($M)", fontweight="bold")
    ax1.set_xlabel("True ROI Data", fontweight="bold")
    ax1.set_title("Cross-Evaluation: Allocation Performance")
    ax1.set_xticks(x); ax1.set_xticklabels(scenarios)
    ax1.legend(); ax1.grid(axis="y", alpha=0.3, ls="--"); ax1.set_ylim(0, 0.65)

    losses = [loss_a, loss_b]
    pcts = [loss_a/obj_lp*100, loss_b/obj_mip*100]
    labels = ["Company 2 Allocation\nUsed on ROI1 Data", "Company 1 Allocation\nUsed on ROI2 Data"]
    bars = ax2.bar(labels, losses, color=["#e74c3c", "#3498db"], edgecolor="black", alpha=0.8)
    for bar, pct in zip(bars, pcts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"${bar.get_height():.4f}M\n({pct:.1f}% loss)",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Opportunity Loss ($M)", fontweight="bold")
    ax2.set_title("Cost of Using the Wrong Allocation")
    ax2.grid(axis="y", alpha=0.3, ls="--"); ax2.set_ylim(0, 0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cross_evaluation_analysis.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_monthly(results, months):
    plat_cols = [c for c in results.columns
                 if c not in ("Month", "Budget", "ROI(%)", "Return($M)")]
    plot_df = results.set_index("Month").reindex(months)
    fig = plt.figure(figsize=(16, 6))
    for p in plat_cols:
        plt.plot(plot_df.index, plot_df[p], marker="o", label=p)
    plt.title("Monthly Budget Allocation per Platform", fontsize=15, fontweight="bold")
    plt.xlabel("Month"); plt.ylabel("Investment ($M)")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
    plt.grid(True, ls="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "monthly_allocation.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
