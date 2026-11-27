#!/usr/bin/env python3
"""Write the data files of the M2 data-input cases and their reference results.

spec/toy-problems.md §4.4–§4.6: `data_decay` (two hourly records bracketing t,
across a daily file boundary), `data_monthly` (a calendar cadence, a custom
substitution and a mirror) and `data_points` (a CSV points source with codes,
a record filter and an extent). Each has an exact solution, evaluated here in
binary64 at the cases' output times and written as a reference result file
(spec/schemas/result.schema.json) under tests/conformance/m2/reference/.

The netCDF files are written with netCDF4 (HDF5 stamps them, so CI compares
their decoded values, not their bytes).

Usage: python tests/conformance/toys/make_data.py [--check]
  --check   regenerate into a temporary directory and compare decoded values
            and references with the checked-in files instead of writing.
"""

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

HERE = Path(__file__).resolve().parent
M2 = HERE.parent / "m2"
HOUR = 3600.0
DAY = 86400.0

# --------------------------------------------------------------------------
# data_decay (§4.4)


def K(h, i):
    """K_{h,i} = 1e-4 (1 + i/2 + (h mod 5)/4), hour h from the origin, i = 0, 1, 2."""
    return 1.0e-4 * (1.0 + i / 2.0 + (h % 5) / 4.0)


def S(i):
    return 1.0 + i / 4.0


DECAY_T0, DECAY_TF = 79200.0, 93600.0
DECAY_SAVEAT = [82800.0, 86400.0, 90000.0, 93600.0]


def decay_integral(i, t0, t):
    """The integral of k_i over [t0, t]: k_i is linear between K_h at 3600 h and K_{h+1}."""
    total = 0.0
    a = t0
    while a < t:
        h = math.floor(a / HOUR)
        b = min((h + 1) * HOUR, t)
        # k(s) = K_h + (K_{h+1} - K_h) (s - 3600 h) / 3600: a trapezoid.
        ka = K(h, i) + (K(h + 1, i) - K(h, i)) * (a - h * HOUR) / HOUR
        kb = K(h, i) + (K(h + 1, i) - K(h, i)) * (b - h * HOUR) / HOUR
        total += 0.5 * (ka + kb) * (b - a)
        a = b
    return total


def decay_exact(t, u0=None):
    u0 = u0 or [1.0, 1.0, 1.0]
    return [u0[i] * math.exp(-S(i) * decay_integral(i, DECAY_T0, t)) for i in range(3)]


def write_decay(root):
    d = root / "data" / "data_decay"
    d.mkdir(parents=True, exist_ok=True)
    for day in (1, 2):
        with Dataset(d / f"rate_201601{day:02d}.nc", "w", format="NETCDF4") as nc:
            nc.createDimension("time", 24)
            nc.createDimension("x", 3)
            tv = nc.createVariable("time", "f8", ("time",))
            tv.units = "hours since 2016-01-01 00:00:00"
            tv.calendar = "standard"
            hours = [24 * (day - 1) + k for k in range(24)]
            tv[:] = np.array(hours, dtype=float)
            kv = nc.createVariable("K", "f8", ("time", "x"))
            kv.units = "s-1"
            kv[:, :] = np.array([[K(h, i) for i in range(3)] for h in hours])
    with Dataset(d / "scale.nc", "w", format="NETCDF4") as nc:
        nc.createDimension("x", 3)
        sv = nc.createVariable("S", "f8", ("x",))
        sv.units = "1"
        sv[:] = np.array([S(i) for i in range(3)])


# --------------------------------------------------------------------------
# data_monthly (§4.5)

MONTH_STARTS = [
    0.0,
    31 * DAY,
    60 * DAY,
    91 * DAY,
]  # 2016-01-01, 02-01, 03-01, 04-01 from the origin
MONTHLY_T0, MONTHLY_TF = 19 * DAY, 69 * DAY  # 2016-01-20 to 2016-03-10
MONTHLY_SAVEAT = [k * DAY for k in range(20, 70)]


def E(m, i):
    """E_{m,i} = 1e-6 (m + i), month m = 1, 2, 3, file cell i = 1..5 (per day)."""
    return 1.0e-6 * (m + i)


def monthly_exact(t):
    out = []
    for i in range(3):
        c = 0.0
        for m in (1, 2, 3):
            a = max(MONTH_STARTS[m - 1], MONTHLY_T0)
            b = min(MONTH_STARTS[m], t)
            if b > a:
                c += (
                    E(m, i + 2) / DAY * (b - a)
                )  # model cell i+1 is file cell i+2 (select cells 2–4)
        out.append(c)
    return out


def write_monthly(root):
    d = root / "data_monthly"
    d.mkdir(parents=True, exist_ok=True)
    for m in (1, 2, 3):
        with Dataset(d / f"inv_point_2016{m:02d}.nc", "w", format="NETCDF4") as nc:
            nc.createDimension("time", 1)
            nc.createDimension("x", 5)
            tv = nc.createVariable("time", "f8", ("time",))
            tv.units = "days since 2016-01-01 00:00:00"
            tv.calendar = "standard"
            tv[:] = np.array([MONTH_STARTS[m - 1] / DAY])
            ev = nc.createVariable("E_day", "f8", ("time", "x"))
            ev.units = "day-1"
            ev[:, :] = np.array([[E(m, i) for i in range(1, 6)]])


# --------------------------------------------------------------------------
# data_points (§4.6)

# The two records the filter drops carry NaN: EarthSciIO's CSV readers never read a blank cell as NaN.
STATIONS = [
    ("1.0e-3", "a"),
    ("NaN", "b"),
    ("2.0e-3", "b"),
    ("5.0e-4", "c"),
    ("NaN", "a"),
    ("3.0e-3", "a"),
]
POINTS_T0, POINTS_TF = 0.0, 3600.0
POINTS_SAVEAT = [600.0 * k for k in range(1, 7)]


def points_exact(t):
    sums = {"a": 0.0, "b": 0.0, "c": 0.0}
    for r, c in STATIONS:
        if r != "NaN":
            sums[c] += float(r)
    return [t * sums[c] for c in ("a", "b", "c")]


def write_points(root):
    d = root / "data_points"
    d.mkdir(parents=True, exist_ok=True)
    lines = ["name,rate,cell"] + [f"s{k + 1},{r},{c}" for k, (r, c) in enumerate(STATIONS)]
    (d / "stations.csv").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# reseact_mini (spec/toy-problems.md §5; spec/partitioning.md §9)
#
# reseact's shape at toy size: NO2/NO/O3 chemistry with photolysis that varies
# by longitude (local time) and level, a thermal rate read from hourly
# two-record data, an NO source at the surface, O3 deposition, periodic upwind
# advection along lon and a column mean along lev. The document is written
# here so the expressions stay readable; the data files beside it.

RM_NLON, RM_NLAT, RM_NLEV = 6, 4, 5
RM_T0, RM_TF = 79200.0, 90000.0  # 22:00 on 1 January to 01:00 on 2 January


def TS(h, i, j):
    """Surface temperature at hour h from the origin, lon cell i, lat cell j (1-based)."""
    return 285.0 + 2.0 * i + 1.5 * j + 3.0 * math.sin(2.0 * math.pi * h / 24.0)


def _op(op, *args, **kw):
    d = {"op": op, "args": list(args)}
    d.update(kw)
    return d


def _ix(name, *idx):
    return _op("index", name, *idx)


def _cell_faq(expr):
    return _op(
        "faq",
        output_idx=["i", "j", "k"],
        ranges={"i": [1, RM_NLON], "j": [1, RM_NLAT], "k": [1, RM_NLEV]},
        expr=expr,
    )


def reseact_mini_doc():
    n, m = RM_NLON, RM_NLAT
    # The model's own variables by their local names, the form every binding
    # validates (EarthSciAST.jl 0.14 does not count `D(Mini.NO2)` inside model
    # `Mini` as an equation for `NO2`).
    X = {s: s for s in ("NO2", "NO", "O3")}
    # Photolysis: half-wave with a phase per longitude, attenuated with level.
    jrate = _op(
        "*",
        "j0",
        _op(
            "max",
            0,
            _op(
                "sin",
                _op(
                    "+",
                    _op("*", "omega", "t"),
                    _op("*", math.pi / 3, _op("-", "i", 1)),
                ),
            ),
        ),
        _op("-", 1, _op("*", 0.15, _op("-", "k", 1))),
    )
    # Thermal rate from the two-record temperature blend.
    tblend = _op(
        "+",
        _ix("TS", 1, "i", "j"),
        _op("*", "w", _op("-", _ix("TS", 2, "i", "j"), _ix("TS", 1, "i", "j"))),
    )
    krate = _op("*", "k0", _op("+", 1, _op("*", 0.01, _op("-", tblend, 290))))
    r = _op(
        "-",
        _op("*", jrate, _ix(X["NO2"], "i", "j", "k")),
        _op("*", krate, _ix(X["NO"], "i", "j", "k"), _ix(X["O3"], "i", "j", "k")),
    )
    chem = {
        "NO2": _cell_faq(_op("-", r)),
        "NO": _cell_faq(
            _op("+", r, _op("*", "e0", _op("max", 0, _op("-", 2, "k"))))
        ),
        "O3": _cell_faq(_op("-", r, _op("*", "vd", _ix(X["O3"], "i", "j", "k")))),
    }

    def advection(x):
        c = _op("/", "u", "dx")
        interior = _op(
            "faq",
            output_idx=["i", "j", "k"],
            ranges={"i": [2, n], "j": [1, m], "k": [1, RM_NLEV]},
            expr=_op(
                "*", -1, c, _op("-", _ix(x, "i", "j", "k"), _ix(x, _op("-", "i", 1), "j", "k"))
            ),
        )
        seam = _op(
            "faq",
            output_idx=["j", "k"],
            ranges={"j": [1, m], "k": [1, RM_NLEV]},
            expr=_op("*", -1, c, _op("-", _ix(x, 1, "j", "k"), _ix(x, n, "j", "k"))),
        )
        return _op(
            "makearray",
            regions=[
                [[2, n], [1, m], [1, RM_NLEV]],
                [[1, 1], [1, m], [1, RM_NLEV]],
            ],
            values=[interior, seam],
        )

    def mixing(x):
        mean = _op(
            "/",
            _op("faq", output_idx=[], ranges={"kk": [1, RM_NLEV]}, expr=_ix(x, "i", "j", "kk")),
            RM_NLEV,
        )
        return _cell_faq(_op("*", "kmix", _op("-", mean, _ix(x, "i", "j", "k"))))

    variables = {
        "NO2": {"type": "unknown", "shape": ["lon", "lat", "lev"], "default": 10.0},
        "NO": {"type": "unknown", "shape": ["lon", "lat", "lev"], "default": 5.0},
        "O3": {"type": "unknown", "shape": ["lon", "lat", "lev"], "default": 40.0},
        "j0": {"type": "parameter", "default": 0.01},
        "omega": {"type": "parameter", "default": 2 * math.pi / 10800},
        "k0": {"type": "parameter", "default": 4.4e-4},
        "e0": {"type": "parameter", "default": 0.002},
        "vd": {"type": "parameter", "default": 1.5e-5},
        "u": {"type": "parameter", "default": 10.5},
        "dx": {"type": "parameter", "default": 20000.5},
        "kmix": {"type": "parameter", "default": 5.0e-4},
        "dtrec": {"type": "parameter", "default": 3600.0},
        "TS": {
            "type": "parameter",
            "shape": ["rec", "lon", "lat"],
            "default": 290.0,
            "update": {"kind": "data", "source": "Met", "from": {"file_variable": "TS"}},
        },
        "w": {"type": "unknown"},
    }
    equations = [
        {
            "lhs": "w",
            "rhs": _op(
                "-",
                _op("/", "t", "dtrec"),
                _op("floor", _op("/", "t", "dtrec")),
            ),
        }
    ]
    for s in ("NO2", "NO", "O3"):
        x = X[s]
        equations.append(
            {
                "lhs": _op("D", x, wrt="t"),
                "rhs": _op("+", chem[s], advection(x), mixing(x)),
            }
        )
    return {
        "esm": "1.3.0",
        "metadata": {
            "name": "reseact_mini",
            "description": "spec/toy-problems.md §5: reseact's shape at toy size (6 x 4 x 5 cells): NO2/NO/O3 chemistry with photolysis varying by longitude and level and a thermal rate from hourly two-record data, periodic upwind advection along lon and a column mean along lev. The M3 fixture for shards, lane pools and per-cell dt (spec/partitioning.md §9).",
        },
        "domain": {"temporal": {"reference_time": "2016-01-01T00:00:00Z"}},
        "index_sets": {
            "lon": {"kind": "interval", "size": RM_NLON},
            "lat": {"kind": "interval", "size": RM_NLAT},
            "lev": {"kind": "interval", "size": RM_NLEV},
            "rec": {"kind": "interval", "size": 2, "records": True},
        },
        "data_sources": {
            "Met": {
                "kind": "grid",
                "source": {"url_template": "data/reseact_mini/met_{date:%Y%m%d}.nc"},
                "temporal": {
                    "start": "2016-01-01T00:00:00Z",
                    "frequency": "PT1H",
                    "file_period": "P1D",
                    "records_per_file": 24,
                    "time_variable": "time",
                },
                "metadata": {"esio_format": "netcdf"},
            }
        },
        "models": {"Mini": {"variables": variables, "equations": equations}},
        "solver": {"time_span": {"start": RM_T0, "end": RM_TF}},
    }


def write_reseact_mini(root):
    (root / "reseact_mini.esm").write_text(json.dumps(reseact_mini_doc(), indent=2) + "\n")
    d = root / "data" / "reseact_mini"
    d.mkdir(parents=True, exist_ok=True)
    for day in (1, 2):
        with Dataset(d / f"met_201601{day:02d}.nc", "w", format="NETCDF4") as nc:
            nc.createDimension("time", 24)
            nc.createDimension("lon", RM_NLON)
            nc.createDimension("lat", RM_NLAT)
            tv = nc.createVariable("time", "f8", ("time",))
            tv.units = "hours since 2016-01-01 00:00:00"
            tv.calendar = "standard"
            hours = [24 * (day - 1) + k for k in range(24)]
            tv[:] = np.array(hours, dtype=float)
            v = nc.createVariable("TS", "f8", ("time", "lon", "lat"))
            v.units = "K"
            v[:, :, :] = np.array(
                [
                    [[TS(h, i, j) for j in range(1, RM_NLAT + 1)] for i in range(1, RM_NLON + 1)]
                    for h in hours
                ]
            )


# --------------------------------------------------------------------------
# Reference results


def reference(case, t0, u0, times, exact):
    ts = [t0, *times]
    return {
        "case": case,
        "producer": "reference",
        "t": ts,
        "u": [list(u0)] + [exact(t) for t in times],
        "stats": {"accepted": 0, "rejected": 0},
    }


def references():
    decay = reference("m2.data_decay.tsit5", DECAY_T0, [1.0] * 3, DECAY_SAVEAT, decay_exact)
    out = {
        "m2.data_decay.tsit5": decay,
        "m2.data_decay.tsit5_dt600": dict(decay, case="m2.data_decay.tsit5_dt600"),
        "m2.data_monthly.tsit5": reference(
            "m2.data_monthly.tsit5", MONTHLY_T0, [0.0] * 3, MONTHLY_SAVEAT, monthly_exact
        ),
        "m2.data_points.tsit5": reference(
            "m2.data_points.tsit5", POINTS_T0, [0.0] * 3, POINTS_SAVEAT, points_exact
        ),
    }
    s0 = [S(i) for i in range(3)]
    out["data_decay_ic"] = reference(
        "data_decay_ic", DECAY_T0, s0, DECAY_SAVEAT, lambda t: decay_exact(t, s0)
    )
    return out


def write_all(root, refdir):
    write_decay(root)
    write_monthly(root)
    write_points(root)
    write_reseact_mini(root)
    refdir.mkdir(parents=True, exist_ok=True)
    for case, ref in references().items():
        (refdir / f"{case}.json").write_text(json.dumps(ref, indent=1) + "\n")


def decoded(path):
    if path.suffix == ".nc":
        with Dataset(path) as nc:
            return {k: np.array(v[:]).tolist() for k, v in nc.variables.items()}
    return path.read_text()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not args.check:
        write_all(HERE, M2 / "reference")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_all(tmp / "toys", tmp / "reference")
        bad = []
        for f in sorted((tmp / "toys").rglob("*")):
            if f.is_file():
                rel = f.relative_to(tmp / "toys")
                mine = HERE / rel
                if not mine.is_file() or decoded(f) != decoded(mine):
                    bad.append(str(rel))
        for f in sorted((tmp / "reference").glob("*.json")):
            mine = M2 / "reference" / f.name
            if not mine.is_file() or json.loads(f.read_text()) != json.loads(mine.read_text()):
                bad.append(f"m2/reference/{f.name}")
        for b in bad:
            print(f"differs from its generator: {b}", file=sys.stderr)
        return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
