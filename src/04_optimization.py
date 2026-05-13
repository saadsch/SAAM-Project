"""
Part 1 Methodology

Optimization Statement

minimize w^\top \Sigma w
subject to \sum_i w_i = 1
and w_i \ge 0 for long-only (plus any max-weight cap if used).

Assumptions: return frequency is monthly; annualization convention is to multiply mean returns by 12 and multiply the covariance matrix by 12; transaction costs are ignored.
"""

# This module currently serves as the canonical reference for the Part 1
# optimization formulation and assumptions used across the project.
