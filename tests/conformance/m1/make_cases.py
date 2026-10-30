#!/usr/bin/env python3
"""Generate cases.json, the M1 conformance cases (spec/toy-problems.md §2).

Usage: python tests/conformance/m1/make_cases.py   (rewrites cases.json beside this file)
"""

import json
from pathlib import Path

V = "0.1.0"

# Method -> (order on forced_decay, family). WRF_RK3 is second order on a
# non-autonomous problem. KPP_* methods are left to M7 (WRF-Chem parity).
METHODS = {
    "ForwardEuler": (1, "explicit"),
    "Heun": (2, "explicit"),
    "SSPRK33": (3, "explicit"),
    "SSPRK43": (3, "explicit"),
    "RK4": (4, "explicit"),
    "Tsit5": (5, "explicit"),
    "BS3": (3, "explicit"),
    "SSPRK104": (4, "explicit"),
    "WRF_RK3": (2, "explicit"),
    "Rosenbrock23": (2, "rosenbrock"),
    "ROS2": (2, "rosenbrock"),
    "ROS3P": (3, "rosenbrock"),
    "Rodas3": (3, "rosenbrock"),
    "Rodas4": (4, "rosenbrock"),
    "Rodas5P": (5, "rosenbrock"),
    "BackwardEuler": (1, "sdirk"),
    "TRBDF2": (2, "sdirk"),
    "SDIRK2": (2, "sdirk"),
    "Kvaerno3": (3, "sdirk"),
    "Kvaerno4": (4, "sdirk"),
    "IMEXEuler": (1, "imex"),
    "Theta": (2, "theta"),
}
# Methods OrdinaryDiffEq does not implement in the same arithmetic form (D31),
# so they have no oracle check.
NO_ORACLE = {"WRF_RK3", "IMEXEuler", "Theta"}
# OrdinaryDiffEq's Kvaerno3/4 evaluate stage 2 at the wrong time (spec/methods.md
# §5.4), so they are oracles only on autonomous problems.
NO_ORACLE_NONAUTONOMOUS = NO_ORACLE | {"Kvaerno3", "Kvaerno4"}


# OrdinaryDiffEq's SDIRK methods reuse J and W across steps, take the first
# stage from the previous step's last one and start Newton from other guesses
# (spec/methods.md §5.1-§5.3). On a nonlinear problem their stages then differ
# from ours by up to the Newton tolerance, κ·reltol with κ = 1/100, on every
# step, and the differences accumulate; `reltol` is the case's. On a linear
# problem (reltol None) Newton converges in one iteration and the two agree
# to roundoff. The same bound applies to reverse_diff, which differentiates
# OrdinaryDiffEq's iterations.
def oracle_rtol(alg, tight, reltol=None):
    if METHODS[alg][1] != "sdirk":
        return tight
    return 1e-8 if reltol is None else times(10, reltol)


def times(factor, reltol):
    """factor * reltol, rounded so cases.json shows 1e-05, not 9.999999999999999e-06."""
    return float(f"{factor * reltol:.1g}")


GRAD = [
    {"check": "dot_product", "rtol": 1e-12},
    {"check": "fd_gradient", "rtol": 1e-6, "epsilons": [1e-3, 1e-4, 1e-5]},
    {"check": "replay"},
]
XL = {"check": "cross_language", "rtol": 1e-10}


# Adaptive runs can take different step sequences after a roundoff-level
# accept/reject flip; their states then differ by about the global error, which
# is several times reltol (spec/methods.md §5.5). With the same steps, SDIRK
# runs can still differ by a Newton iteration, as for oracle_rtol.
def xl_adaptive(alg, reltol):
    return {
        "check": "cross_language",
        "rtol": oracle_rtol(alg, 1e-10, reltol),
        "rtol_steps_differ": times(100, reltol),
    }


def cfg(model, schedule, step, adjoint=None):
    c = {"esm_integrator": V, "model": model, "schedule": schedule, "step": step}
    if adjoint:
        c["adjoint"] = adjoint
    return c


def leaf(op, alg, **opts):
    node = {"leaf": op, "alg": alg}
    family = METHODS[alg][1]
    if family in ("rosenbrock", "sdirk", "imex", "theta"):
        node["jacobian"] = "symbolic"
        node.setdefault("linsolve", "dense_lu")
    if family == "imex":
        node["imex"] = {"implicit": op}
    if family == "theta":
        node["theta"] = 0.5
    node.update(opts)
    return node


def adaptive(reltol, abstol):
    return {"control": {"reltol": reltol, "abstol": abstol}}


def grad(quantity, over, how="sum"):
    return {"functional": {how: quantity, "over": over}, "wrt": ["u0", "params:numeric"]}


cases = []

# forced_decay: order and oracle for every method at fixed dt. ROS2 is
# pre-asymptotic at the common step sizes (observed order 1.71 there, 1.92 at
# these), so it is measured at smaller ones.
DTS = {"ROS2": [0.025, 0.0125, 0.00625, 0.003125]}
for alg, (order, _) in METHODS.items():
    checks = [
        {
            "check": "order",
            "vary": "leaf",
            "dts": DTS.get(alg, [0.1, 0.05, 0.025, 0.0125]),
            "expected": order,
            "against": "exact",
        }
    ]
    if alg not in NO_ORACLE_NONAUTONOMOUS:
        checks.append({"check": "oracle", "rtol": oracle_rtol(alg, 1e-12)})
    checks.append(XL)
    cases.append(
        {
            "id": f"forced_decay.{alg}.fixed",
            "run_config": cfg("toy:forced_decay", leaf("decay", alg, dt=0.1), {"dt": 2}),
            "tspan": [0, 2],
            "checks": checks,
        }
    )

# forced_decay: gradients, one method per family.
for alg in ["Tsit5", "Rodas4", "Kvaerno3", "IMEXEuler", "Theta"]:
    checks = (
        GRAD
        + ([] if alg in NO_ORACLE_NONAUTONOMOUS else [{"check": "reverse_diff", "rtol": 1e-10}])
        + [XL]
    )
    cases.append(
        {
            "id": f"forced_decay.{alg}.gradient",
            "run_config": cfg(
                "toy:forced_decay", leaf("decay", alg, dt=0.05), {"dt": 2}, grad("u", "final")
            ),
            "tspan": [0, 2],
            "checks": checks,
        }
    )

# van_der_pol, nonstiff: explicit methods, fixed and adaptive.
for alg in ["SSPRK43", "Tsit5", "BS3", "RK4"]:
    cases.append(
        {
            "id": f"van_der_pol.nonstiff.{alg}.fixed",
            "run_config": cfg(
                "toy:van_der_pol/nonstiff", leaf("oscillator", alg, dt=0.01), {"dt": 6.3}
            ),
            "tspan": [0, 6.3],
            "checks": [{"check": "oracle", "rtol": 1e-12}, XL],
        }
    )
cases.append(
    {
        "id": "van_der_pol.nonstiff.Tsit5.adaptive",
        "run_config": cfg(
            "toy:van_der_pol/nonstiff",
            leaf("oscillator", "Tsit5", **adaptive(1e-8, 1e-10)),
            {"dt": 6.3},
        ),
        "tspan": [0, 6.3],
        "saveat": [1.0, 3.0, 6.3],
        "checks": [
            {"check": "oracle", "rtol": 1e-12},
            {"check": "reference", "rtol": 1e-6},
            xl_adaptive("Tsit5", 1e-8),
        ],
    }
)

# van_der_pol, stiff: implicit methods, adaptive, with gradients (nonzero second-order terms).
for alg in ["Rosenbrock23", "Rodas4", "Rodas5P", "TRBDF2", "Kvaerno4"]:
    checks = [
        {"check": "oracle", "rtol": oracle_rtol(alg, 1e-10, 1e-6)},
        {"check": "reference", "rtol": 1e-4},
    ] + GRAD
    checks += [
        {"check": "reverse_diff", "rtol": oracle_rtol(alg, 1e-8, 1e-6)},
        xl_adaptive(alg, 1e-6),
    ]
    cases.append(
        {
            "id": f"van_der_pol.stiff.{alg}.adaptive",
            "run_config": cfg(
                "toy:van_der_pol/stiff",
                leaf("oscillator", alg, **adaptive(1e-6, 1e-8)),
                {"dt": 1},
                grad("y", "final"),
            ),
            "tspan": [0, 1],
            "checks": checks,
        }
    )

# robertson: stiff, adaptive, invariant, gradient of the final y3.
for alg in ["Rosenbrock23", "Rodas4", "Rodas5P", "TRBDF2", "Kvaerno4"]:
    checks = [
        {"check": "invariant", "rtol": 1e-12},
        {"check": "oracle", "rtol": oracle_rtol(alg, 1e-10, 1e-6)},
        {"check": "reference", "rtol": 1e-4},
    ]
    if alg == "Rodas4":
        checks += GRAD + [{"check": "reverse_diff", "rtol": 1e-8}]
    checks.append(xl_adaptive(alg, 1e-6))
    cases.append(
        {
            "id": f"robertson.{alg}.adaptive",
            "run_config": cfg(
                "toy:robertson",
                leaf("kinetics", alg, **adaptive(1e-6, 1e-10)),
                {"dt": 40},
                grad("y3", "final") if alg == "Rodas4" else None,
            ),
            "tspan": [0, 40],
            "saveat": [0.4, 4, 40],
            "checks": checks,
        }
    )

# photostationary_cells: block_diagonal layout, batched block LU, integral functional.
for alg in ["Rosenbrock23", "Rodas4", "Kvaerno4"]:
    # photostationary_cells is non-autonomous.
    checks = [{"check": "invariant", "rtol": 1e-12}, {"check": "reference", "rtol": 1e-4}]
    if alg not in NO_ORACLE_NONAUTONOMOUS:
        checks.insert(1, {"check": "oracle", "rtol": oracle_rtol(alg, 1e-10, 1e-6)})
    if alg == "Rosenbrock23":
        checks += GRAD + [{"check": "reverse_diff", "rtol": 1e-8}]
    checks.append(xl_adaptive(alg, 1e-6))
    node = leaf(
        "photochemistry", alg, linsolve={"block_lu": {"pivot": False}}, **adaptive(1e-6, 1e-6)
    )
    cases.append(
        {
            "id": f"photostationary_cells.{alg}.adaptive",
            "run_config": cfg(
                "toy:photostationary_cells",
                node,
                {"dt": 600},
                grad("O3", "time", "mean") if alg == "Rosenbrock23" else None,
            ),
            "tspan": [0, 3600],
            "saveat": [1800, 3600],
            "checks": checks,
        }
    )

# advection_reaction: splitting order, subcycle, adjoints through sequence and strang.
transport = {"leaf": "transport", "alg": "SSPRK43", **adaptive(1e-10, 1e-12)}
reaction = leaf(
    "reaction", "Rodas4", linsolve={"block_lu": {"pivot": False}}, **adaptive(1e-10, 1e-12)
)
for name, node, order in [
    ("lie", {"sequence": [transport, reaction]}, 1),
    ("strang", {"strang": [transport, reaction]}, 2),
]:
    cases.append(
        {
            "id": f"advection_reaction.{name}.order",
            "run_config": cfg("toy:advection_reaction", node, {"dt": 0.05}),
            "tspan": [0, 0.5],
            "checks": [
                {
                    "check": "order",
                    "vary": "macro",
                    "dts": [0.05, 0.025, 0.0125, 0.00625],
                    "expected": order,
                    "against": "reference",
                },
                {"check": "invariant", "rtol": 1e-12},
                xl_adaptive("Rodas4", 1e-10),
            ],
        }
    )
    cases.append(
        {
            "id": f"advection_reaction.{name}.gradient",
            "run_config": cfg("toy:advection_reaction", node, {"dt": 0.025}, grad("B", "final")),
            "tspan": [0, 0.5],
            "checks": GRAD + [xl_adaptive("Rodas4", 1e-10)],
        }
    )
sub = {
    "sequence": [
        {"subcycle": {"leaf": "transport", "alg": "SSPRK33", "dt": "step"}, "n": 4},
        leaf("reaction", "Rodas4", dt="step", linsolve={"block_lu": {"pivot": False}}),
    ]
}
cases.append(
    {
        "id": "advection_reaction.subcycle.fixed",
        "run_config": cfg("toy:advection_reaction", sub, {"dt": 0.01}, grad("B", "final")),
        "tspan": [0, 0.5],
        "checks": [{"check": "invariant", "rtol": 1e-12}] + GRAD + [XL],
    }
)

# source_decay_map: map, bind_dt, exact discrete solution.
cases.append(
    {
        "id": "source_decay_map.sequence.fixed",
        "run_config": cfg(
            "toy:source_decay_map",
            {
                "sequence": [
                    {"leaf": "source", "alg": "ForwardEuler", "dt": "step"},
                    {"map": "decay"},
                ]
            },
            {"dt": 0.5},
            grad("u", "final"),
        ),
        "tspan": [0, 4],
        "checks": [{"check": "exact_discrete", "rtol": 1e-14}] + GRAD + [XL],
    }
)

out = {"esm_integrator": V, "generated_by": "tests/conformance/m1/make_cases.py", "cases": cases}
path = Path(__file__).with_name("cases.json")
path.write_text(json.dumps(out, indent=1) + "\n")
print(f"{len(cases)} cases -> {path}")
