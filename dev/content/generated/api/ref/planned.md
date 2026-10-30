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
    pub schedule: Node,
    pub step: Value,
    pub partition: Value,
    pub precision: Value,
    pub checkpoint_schedule: Option<Value>,
    pub inputs: Value,
    pub reference_steps: BTreeMap<String, f64>,
    pub stiffness: BTreeMap<String, Option<Stiffness>>,
}

impl Planned { pub fn time_json(&self) -> Value }

impl Planned { pub fn terms_json(&self, op: &Operator) -> Value }

impl Planned { pub fn locality_json(&self, op: &Operator) -> Value }

impl Planned { pub fn operator_json_stage1(&self, name: &str) -> Value }

impl Planned { pub fn leaf_facts(&self, n: &Node, facts: Option<&StructureFacts>) -> LeafFacts }

impl Planned { pub fn leaf_strategy(&self, n: &Node, facts: Option<&StructureFacts>) -> Result<LeafStrategy> }

impl Planned { pub fn resolve_schedule(&self, facts: Option<&StructureFacts>) -> Result<Value> }

impl Planned { pub fn plan_json(&self) -> Result<Value> }
```

Everything stage 1 computes for a run config.

[Source](https://github.com/EarthSciML/EarthSciIntegrator/blob/c4b80d912f4c4a30c93aa38491a417343a023c98/rust/earthsci-integrator-planner/src/plan.rs#L36) · [Rust docs](https://docs.rs/earthsci-integrator-planner/latest/earthsci_integrator_planner/plan/struct.Planned.html)

</div>

</div>
