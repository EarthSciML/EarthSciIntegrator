---
title: "planned"
description: "Stable type · The planner"
type: "api"
api_tabs: true
---


**Tier:** stable · **Kind:** type · **Topic:** [The planner](/generated/api/topics/planner)

| Julia | Rust |
|---|---|
| not exported | `earthsci_integrator_planner::Planned` |


<div class="api-tabs">

<div class="api-tab" data-lang="julia" data-label="Julia">
<p class="api-tab-label">Julia</p>

<p class="api-absent">Julia does not export this type.</p>

</div>

<div class="api-tab" data-lang="rust" data-label="Rust">
<p class="api-tab-label">Rust</p>

```rust
pub struct Planned {
    pub run_config: RunConfig,
    pub model: ModelInfo,
    pub axes: Axes,
    pub interval: Interval,
    pub terms: Vec<Term>,
    pub term_localities: Vec<Locality>,
    pub operators: BTreeMap<String, Operator>,
    pub structures: BTreeMap<String, Structure>,
    pub schedule: Node,
    pub step: Value,
    pub partition: Value,
    pub precision: Value,
    pub checkpoint_schedule: Option<Value>,
    pub inputs: Value,
    pub reference_steps: BTreeMap<String, f64>,
    pub stiffness: BTreeMap<String, Stiffness>,
    pub facts: StructureFacts,
    pub resolved_schedule: Value,
    pub documents: BTreeMap<String, Documents>,
    pub validation: Vec<Value>,
    pub report: Value,
}

impl Planned { pub fn time_json(&self) -> Value }

impl Planned { pub fn terms_json(&self, op: &Operator) -> Value }

impl Planned { pub fn locality_json(&self, op: &Operator) -> Value }

impl Planned { pub fn operator_json(&self, name: &str) -> Value }

impl Planned { pub fn leaf_facts(&self, n: &Node, facts: Option<&StructureFacts>) -> LeafFacts }

impl Planned { pub fn leaf_strategy(&self, n: &Node, facts: Option<&StructureFacts>) -> Result<LeafStrategy> }

impl Planned { pub fn resolve_schedule(&self, facts: Option<&StructureFacts>) -> Result<Value> }

impl Planned { pub fn dofs(&self) -> Dofs }

impl Planned { pub fn band_jacobian_at(&self, op: &str, u: &[f64], t: f64, forcing: &Forcing) -> Result<BTreeMap<(String, String), f64>> }

impl Planned { pub fn derived_path(op: &str, kind: &str) -> String }

impl Planned { pub fn derived_json(&self) -> Value }

impl Planned { pub fn plan_json(&self) -> Value }

impl Planned { pub fn plan_bytes(&self) -> Result<Vec<u8>> }

impl Planned { pub fn write(&self, dir: &Path) -> Result<()> }
```

Everything the planner computes for a run config.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/2f2815c8579399275fdf110ee093aeb2fed8c2f1/rust/earthsci-integrator-planner/src/plan.rs#L47) · [Rust docs](https://docs.rs/earthsci-integrator-planner/latest/earthsci_integrator_planner/plan/struct.Planned.html)

</div>

</div>
