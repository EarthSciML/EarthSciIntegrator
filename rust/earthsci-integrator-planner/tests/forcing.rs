//! Data-fed parameters in the Rust planner (spec/plan.md §4.1, §5.3,
//! §5.5.1, §6.3, D87), on a model of the planner's own: a discrete two-record
//! array parameter read through a time blend by a pointwise operator, a
//! const scalar reached through a `param_to_var` coupling onto a shaped
//! target read by a column operator, and an unused data-fed parameter with no
//! `default`. The shared conformance fixture `forced_column` and its refusal
//! variants are the Julia planner's (§9); these checks are the ones §9 leaves
//! to each planner's unit tests (a default with a zero element) and the
//! derived documents' declarations.

use earthsci_integrator_planner::astdiff::AstDiffBands;
use earthsci_integrator_planner::{plan, Options, PlanError, Planned};
use serde_json::{json, Value};
use std::path::PathBuf;

const MODEL: &str = r#"{
 "esm": "1.3.0",
 "metadata": {
  "name": "forced_unit",
  "description": "Rust planner unit test of data-fed parameters (spec/plan.md §4.1, §5.3, §6.3)."
 },
 "domain": {
  "temporal": {
   "reference_time": "2016-01-01T00:00:00Z"
  }
 },
 "index_sets": {
  "x": {
   "kind": "interval",
   "size": 3
  },
  "lev": {
   "kind": "interval",
   "size": 4
  },
  "rec": {
   "kind": "interval",
   "size": 2
  }
 },
 "data_sources": {
  "Rate": {
   "kind": "grid",
   "source": {
    "url_template": "data/rate_{date:%Y%m%d}.nc"
   },
   "temporal": {
    "start": "2016-01-01T00:00:00Z",
    "frequency": "PT1H",
    "file_period": "P1D",
    "records_per_file": 24,
    "time_variable": "time"
   },
   "metadata": {
    "esio_format": "netcdf"
   }
  },
  "Scale": {
   "kind": "static",
   "source": {
    "url_template": "data/scale.nc"
   },
   "metadata": {
    "esio_format": "netcdf"
   }
  }
 },
 "models": {
  "Chem": {
   "variables": {
    "c": {
     "type": "unknown",
     "shape": [
      "x",
      "lev"
     ],
     "default": 1.5
    },
    "F": {
     "type": "parameter",
     "shape": [
      "rec",
      "x",
      "lev"
     ],
     "default": 0.0001,
     "update": {
      "kind": "data",
      "source": "Rate",
      "from": {
       "file_variable": "K",
       "records_per_sample": 2
      }
     }
    },
    "w": {
     "type": "unknown"
    }
   },
   "equations": [
    {
     "lhs": {
      "op": "D",
      "args": [
       "c"
      ],
      "wrt": "t"
     },
     "rhs": {
      "op": "makearray",
      "args": [],
      "regions": [
       [
        [
         1,
         3
        ],
        [
         1,
         4
        ]
       ]
      ],
      "values": [
       {
        "op": "faq",
        "args": [],
        "output_idx": [
         "i",
         "k"
        ],
        "ranges": {
         "i": [
          1,
          3
         ],
         "k": [
          1,
          4
         ]
        },
        "expr": {
         "op": "-",
         "args": [
          {
           "op": "*",
           "args": [
            {
             "op": "+",
             "args": [
              {
               "op": "*",
               "args": [
                {
                 "op": "-",
                 "args": [
                  1.0,
                  "Chem.w"
                 ]
                },
                {
                 "op": "index",
                 "args": [
                  "Chem.F",
                  1,
                  "i",
                  "k"
                 ]
                }
               ]
              },
              {
               "op": "*",
               "args": [
                "Chem.w",
                {
                 "op": "index",
                 "args": [
                  "Chem.F",
                  2,
                  "i",
                  "k"
                 ]
                }
               ]
              }
             ]
            },
            {
             "op": "index",
             "args": [
              "Chem.c",
              "i",
              "k"
             ]
            }
           ]
          }
         ]
        }
       }
      ]
     }
    },
    {
     "lhs": "w",
     "rhs": {
      "op": "-",
      "args": [
       {
        "op": "/",
        "args": [
         "t",
         3600.0
        ]
       },
       {
        "op": "floor",
        "args": [
         {
          "op": "/",
          "args": [
           "t",
           3600.0
          ]
         }
        ]
       }
      ]
     }
    }
   ]
  },
  "Mix": {
   "variables": {
    "kz": {
     "type": "parameter",
     "shape": [
      "x"
     ],
     "default": 0.25
    }
   },
   "equations": [
    {
     "lhs": {
      "op": "D",
      "args": [
       "_var"
      ],
      "wrt": "t"
     },
     "rhs": {
      "op": "makearray",
      "args": [],
      "regions": [
       [
        [
         1,
         3
        ],
        [
         1,
         4
        ]
       ]
      ],
      "values": [
       {
        "op": "faq",
        "args": [],
        "output_idx": [
         "i",
         "k"
        ],
        "ranges": {
         "i": [
          1,
          3
         ],
         "k": [
          1,
          4
         ]
        },
        "expr": {
         "op": "*",
         "args": [
          {
           "op": "index",
           "args": [
            "Mix.kz",
            "i"
           ]
          },
          {
           "op": "-",
           "args": [
            {
             "op": "/",
             "args": [
              {
               "op": "faq",
               "args": [],
               "output_idx": [],
               "ranges": {
                "kk": [
                 1,
                 4
                ]
               },
               "expr": {
                "op": "index",
                "args": [
                 "_var",
                 "i",
                 "kk"
                ]
               }
              },
              4
             ]
            },
            {
             "op": "index",
             "args": [
              "_var",
              "i",
              "k"
             ]
            }
           ]
          }
         ]
        }
       }
      ]
     }
    }
   ]
  },
  "Met": {
   "variables": {
    "Kz": {
     "type": "parameter",
     "default": 0.002,
     "update": {
      "kind": "data",
      "source": "Scale",
      "from": {
       "file_variable": "S"
      }
     }
    },
    "Q": {
     "type": "parameter",
     "shape": [
      "x"
     ],
     "update": {
      "kind": "data",
      "source": "Scale",
      "from": {
       "file_variable": "Q"
      }
     }
    }
   },
   "equations": []
  }
 },
 "coupling": [
  {
   "type": "operator_compose",
   "systems": [
    "Chem",
    "Mix"
   ],
   "lifting": "pointwise"
  },
  {
   "type": "variable_map",
   "from": "Met.Kz",
   "to": "Mix.kz",
   "transform": "param_to_var"
  }
 ],
 "solver": {
  "stiffness": "high",
  "time_span": {
   "start": 0,
   "end": 7200
  }
 }
}"#;

// The chem leaf names TRBDF2: its two-record forcing is blended in time, so
// chem depends on t, and "auto" would resolve to Rosenbrock23, which is
// refused at M2 (unsupported_time_derivative, D94; see `time_dependent_rosenbrock`).
const RUN: &str = r#"{
 "esm_integrator": "0.1.0",
 "model": "model.esm",
 "axes": {
  "horizontal": [
   "x"
  ],
  "vertical": "lev"
 },
 "operators": {
  "chem": {
   "select": {
    "components": "Chem"
   }
  },
  "mix": {
   "select": {
    "components": "Mix"
   }
  }
 },
 "step": {
  "dt": 600,
  "hit": [
   "forcing_refresh"
  ]
 },
 "schedule": {
  "sequence": [
   {
    "leaf": "chem",
    "alg": "TRBDF2"
   },
   {
    "leaf": "mix",
    "alg": "TRBDF2"
   }
  ]
 }
}"#;

/// Writes `model` and `run` into a directory of their own and plans them.
fn plan_of(name: &str, model: &Value, run: &Value) -> Result<Planned, PlanError> {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("forcing")
        .join(name);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("model.esm"), model.to_string()).unwrap();
    std::fs::write(dir.join("run.json"), run.to_string()).unwrap();
    plan(&dir.join("run.json"), &Options::default(), &AstDiffBands)
}

fn inputs() -> (Value, Value) {
    (
        serde_json::from_str(MODEL).unwrap(),
        serde_json::from_str(RUN).unwrap(),
    )
}

fn refused(name: &str, model: &Value, run: &Value) -> (String, String) {
    match plan_of(name, model, run) {
        Err(PlanError::Refused(r)) => (r.code.to_string(), r.pointer.to_string()),
        Err(e) => panic!("{name}: {e}"),
        Ok(_) => panic!("{name}: planned"),
    }
}

#[test]
fn forcing_in_the_plan() {
    let (m, r) = inputs();
    let p = plan_of("planned", &m, &r).unwrap();
    let plan = p.plan_json();
    assert_eq!(
        plan["forcing"],
        json!({
            "Chem.F": {"shape": ["rec", "x", "lev"], "source": "Rate", "file_variable": "K",
                       "cadence": "discrete", "records_per_sample": 2},
            "Met.Kz": {"shape": ["x"], "source": "Scale", "file_variable": "S",
                       "cadence": "const", "records_per_sample": 1},
        })
    );
    assert_eq!(plan["operators"]["chem"]["forcing"], json!(["Chem.F"]));
    assert_eq!(plan["operators"]["mix"]["forcing"], json!(["Met.Kz"]));
    assert_eq!(plan["operators"]["mix"]["locality"]["class"], "column");
    // §6.3: each document declares its own forcing with the data rule and
    // the stub, a used data-fed parameter it does not read as an ordinary
    // parameter, and the unused one not at all.
    for (op, fed, plain, source, kind) in [
        ("chem", "F", ("Met", "Kz"), "Rate", "grid"),
        ("mix", "Kz", ("Chem", "F"), "Scale", "static"),
    ] {
        let docs = &p.documents[op];
        for bytes in [&docs.rhs, &docs.jacobian] {
            let d: Value = serde_json::from_slice(bytes).unwrap();
            assert_eq!(
                d["data_sources"],
                json!({source: {"kind": kind, "source": {"url_template": "{esm_integrator_forcing}"}}}),
                "{op}"
            );
            let comp = if fed == "F" { "Chem" } else { "Met" };
            assert_eq!(
                d["models"][comp]["variables"][fed]["update"]["source"], source,
                "{op}"
            );
            let v = &d["models"][plain.0]["variables"][plain.1];
            assert!(
                v.get("update").is_none() && v.get("default").is_some(),
                "{op}: {v}"
            );
            assert!(d["models"]["Met"]["variables"].get("Q").is_none(), "{op}");
        }
    }
    assert!(!String::from_utf8(p.plan_bytes().unwrap())
        .unwrap()
        .contains("Met.Q"));
}

#[test]
fn forcing_refusals() {
    let (m, r) = inputs();
    let mut cases: Vec<(&str, Value, Value, &str, &str)> = Vec::new();
    let mut no_default = m.clone();
    no_default["models"]["Chem"]["variables"]["F"]
        .as_object_mut()
        .unwrap()
        .remove("default");
    cases.push((
        "no-default",
        no_default,
        r.clone(),
        "forcing_without_scale",
        "/model",
    ));
    // A zero element of an array default (§9 leaves it to the unit tests).
    let mut zero = m.clone();
    zero["models"]["Chem"]["variables"]["F"]["default"] = json!([
        [
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0]
        ],
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0]
        ]
    ]);
    cases.push((
        "zero-element",
        zero,
        r.clone(),
        "forcing_without_scale",
        "/model",
    ));
    let mut auto = r.clone();
    auto["schedule"] = json!("auto");
    cases.push((
        "auto",
        m.clone(),
        auto,
        "auto_stiffness_reads_forcing",
        "/schedule",
    ));
    let mut no_refresh = r.clone();
    no_refresh["step"]["hit"] = json!(["output"]);
    cases.push((
        "no-refresh",
        m.clone(),
        no_refresh,
        "forcing_refresh_not_hit",
        "/step/hit",
    ));
    let mut no_hit = r.clone();
    no_hit["step"].as_object_mut().unwrap().remove("hit");
    cases.push((
        "no-hit",
        m.clone(),
        no_hit,
        "forcing_refresh_not_hit",
        "/step",
    ));
    let mut records = m.clone();
    records["data_sources"]["Rate"]["temporal"] =
        json!({"start": "2016-01-01T00:00:00Z", "time_variable": "time"});
    cases.push((
        "records",
        records,
        r.clone(),
        "forcing_records_invalid",
        "/model",
    ));
    let mut sched = m.clone();
    let rule = sched["models"]["Chem"]["variables"]["F"]["update"].clone();
    sched["models"]["Chem"]["variables"]["F"]["update"] =
        json!([rule, {"kind": "schedule", "times": [0.0], "expression": 1.0}]);
    cases.push((
        "schedule",
        sched,
        r.clone(),
        "unsupported_parameter_update",
        "/model",
    ));
    let mut wrong = Vec::new();
    for (name, model, run, code, at) in &cases {
        let got = refused(name, model, run);
        if got != (code.to_string(), at.to_string()) {
            wrong.push(format!("{name}: want {code} at {at}, got {got:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// §5.5.3, D94: chem reads a two-record forcing blended in time, so it
/// depends on t; a Rosenbrock method on its leaf, given or chosen for
/// "auto" (the pointwise stiff row), is refused, at the leaf's alg or at the
/// leaf.
#[test]
fn time_dependent_rosenbrock() {
    let (m, r) = inputs();
    let p = plan_of("chem-trbdf2", &m, &r).unwrap();
    assert_eq!(
        p.plan_json()["operators"]["chem"]["structure"]["depends_on_t"],
        true
    );
    for (name, alg, at) in [
        ("chem-auto", "auto", "/schedule/sequence/0"),
        ("chem-rodas4", "Rodas4", "/schedule/sequence/0/alg"),
    ] {
        let mut run = r.clone();
        run["schedule"]["sequence"][0]["alg"] = json!(alg);
        assert_eq!(
            refused(name, &m, &run),
            ("unsupported_time_derivative".to_string(), at.to_string()),
            "{name}"
        );
    }
}

/// `adjoint.wrt`'s prefixes (§7.6): a data-fed parameter matches only
/// `forcing:`, and only a used one.
#[test]
fn wrt_prefixes() {
    let (m, r) = inputs();
    for (wrt, ok) in [
        ("forcing:Chem.*", true),
        ("forcing:Met.Kz", true),
        ("forcing:Met.Q", false),
        ("params:Met.*", false),
        ("fields:Chem.*", false),
    ] {
        let mut run = r.clone();
        run["adjoint"] = json!({"functional": {"sum": "Chem.c", "over": "final"}, "wrt": [wrt]});
        match plan_of("wrt", &m, &run) {
            Ok(_) => assert!(ok, "{wrt} planned"),
            Err(PlanError::Refused(e)) => {
                assert!(!ok && e.code == "wrt_matches_nothing", "{wrt}: {e:?}")
            }
            Err(e) => panic!("{wrt}: {e}"),
        }
    }
}

/// A right-hand-side program built with every used data-fed parameter bound
/// to its `default` (to build `u_0`, §4) still reads each evaluation's
/// forcing, so pass 8 differences at the sampled forcing, as the Jacobian
/// program does.
#[test]
fn bound_programs_read_the_sampled_forcing() {
    use earthsci_integrator_planner::numeric::{self, Dofs, Program};
    let (m, r) = inputs();
    let p = plan_of("bound", &m, &r).unwrap();
    let dofs = Dofs::new(&p.model);
    let tspan = (p.interval.start, p.interval.end);
    for (op, docs) in &p.documents {
        let rhs: Value = serde_json::from_slice(&docs.rhs).unwrap();
        let (bound, u0) = Program::with_initial_state(&p.model, &rhs, tspan, &dofs).unwrap();
        let unbound = Program::new(&rhs, tspan, &dofs).unwrap();
        assert_eq!(
            bound.forcing_slot_names(),
            unbound.forcing_slot_names(),
            "{op}"
        );
        // A state that varies from cell to cell, so mixing is not zero.
        let u: Vec<f64> = u0
            .iter()
            .enumerate()
            .map(|(i, x)| (if *x != 0.0 { *x } else { 1.0 }) * (1.0 + 0.1 * i as f64))
            .collect();
        let samples = numeric::sampled_forcing(&p.model).unwrap();
        for f in &samples {
            let a = bound.eval(&dofs, &u, tspan.0, f).unwrap();
            let b = unbound.eval(&dofs, &u, tspan.0, f).unwrap();
            assert_eq!(a, b, "{op}");
        }
        // The sampled forcings differ, and the right-hand side follows them.
        let a = bound.eval(&dofs, &u, tspan.0, &samples[0]).unwrap();
        let b = bound.eval(&dofs, &u, tspan.0, &samples[2]).unwrap();
        assert_ne!(a, b, "{op}: {:?}", bound.forcing_slot_names());
    }
}
