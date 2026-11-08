//! Pass 6, the derived documents (spec/plan.md §6, D68, D69).
//!
//! Both documents are JSON objects written by EarthSciASTDiff's
//! `stable_json` (§2.2). The right-hand-side document (§6.1) declares the
//! flattened model's states, parameters and live observeds in models named
//! after their components and holds one operator's terms; the Jacobian
//! document (§6.2) is EarthSciASTDiff's evaluation document of it, which
//! zeroes every state equation and adds the band fields `JacobianEval.J_k`.

use crate::error::{PlanError, Result};
use crate::model::{defined_observed, differentiated, lifted_ranges, node, ModelInfo};
use crate::terms::{component_of, Term};
use earthsci_ast::{Domain, Expr, ModelVariable, VariableType};
use earthsci_astdiff::{JacEntry, SysView};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};

/// The esm version a derived document declares (§6.1).
pub const ESM_VERSION: &str = "1.3.0";

/// The model the Jacobian document adds for its band fields (§6.2), and the
/// top-level component name a model may not use (§6.1.4). EarthSciASTDiff's
/// `evaluation_document` chooses it and exports no constant for it, so
/// [`jacobian_document`] checks the document it writes against this one.
pub const JACOBIAN_MODEL: &str = "JacobianEval";

/// The document `doc` as parsing its bytes ([`to_bytes`]) gives it: every
/// object's keys in code-point order, the order `stable_json` writes them
/// in. EarthSciAST keeps a document's key order, so a document loaded from
/// this form is laid out as one loaded from its bytes. The keys are sorted
/// in place: nothing is copied.
pub fn wire_form(mut doc: Value) -> Value {
    doc.sort_all_objects();
    doc
}

/// A derived document's bytes: EarthSciASTDiff's `stable_json` (§2.2).
pub fn to_bytes(doc: &Value) -> Vec<u8> {
    earthsci_astdiff::stable_json(doc).into_bytes()
}

/// A model or subsystem of the component tree (§6.1.1).
#[derive(Default)]
struct Component {
    variables: Map<String, Value>,
    equations: Vec<Value>,
    subsystems: BTreeMap<String, Component>,
}

impl Component {
    fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert("variables".into(), Value::Object(self.variables.clone()));
        out.insert("equations".into(), Value::Array(self.equations.clone()));
        if !self.subsystems.is_empty() {
            let subs: Map<String, Value> = self
                .subsystems
                .iter()
                .map(|(k, c)| (k.clone(), c.to_json()))
                .collect();
            out.insert("subsystems".into(), Value::Object(subs));
        }
        Value::Object(out)
    }
}

/// The component tree: top-level models by name.
#[derive(Default)]
struct Tree(BTreeMap<String, Component>);

impl Tree {
    /// The component at dotted path `path`, created with its ancestors.
    fn at(&mut self, path: &str) -> &mut Component {
        let mut segs = path.split('.');
        let first = segs.next().unwrap_or_default();
        let mut c = self.0.entry(first.to_string()).or_default();
        for s in segs {
            c = c.subsystems.entry(s.to_string()).or_default();
        }
        c
    }

    /// Declares the flattened variable `name` in its component.
    fn declare(&mut self, name: &str, var: Value) {
        let local = name.rsplit_once('.').map(|(_, v)| v).unwrap_or(name);
        self.at(component_of(name))
            .variables
            .insert(local.to_string(), var);
    }

    fn push(&mut self, subject: &str, eq: Value) {
        self.at(component_of(subject)).equations.push(eq);
    }

    fn to_json(&self) -> Value {
        Value::Object(
            self.0
                .iter()
                .map(|(k, c)| (k.clone(), c.to_json()))
                .collect(),
        )
    }
}

fn expr_json(e: &Expr) -> Value {
    serde_json::to_value(e).unwrap_or(Value::Null)
}

/// A variable's declaration (§6.1.1): `{"type": T}` plus `shape` and
/// `default` when present.
fn variable_json(v: &ModelVariable, shape: &[String]) -> Value {
    let mut out = Map::new();
    let t = if v.var_type == VariableType::Parameter {
        "parameter"
    } else {
        "unknown"
    };
    out.insert("type".into(), json!(t));
    if !shape.is_empty() {
        out.insert("shape".into(), json!(shape));
    }
    if let Some(d) = &v.default {
        out.insert(
            "default".into(),
            serde_json::to_value(d).unwrap_or(Value::Null),
        );
    }
    Value::Object(out)
}

/// The zero right-hand side of a state (§6.1.2): `0.0` for a scalar, a
/// `makearray` of `0.0` over the whole array otherwise.
pub fn zero(info: &ModelInfo, state: &str) -> Value {
    let shape = &info.shapes[state];
    if shape.is_empty() {
        return json!(0.0);
    }
    let region: Vec<Value> = shape
        .iter()
        .map(|s| json!([1, info.index_sets[s]]))
        .collect();
    json!({"op": "makearray", "args": [], "regions": [region], "values": [0.0]})
}

/// The left fold `((t_1 + t_2) + t_3) + …` of `terms` as nested binary `+`
/// (§6.1.2); `None` for no term.
fn fold<'a>(terms: impl IntoIterator<Item = &'a Expr>) -> Option<Value> {
    terms.into_iter().fold(None, |acc, t| {
        Some(match acc {
            None => expr_json(t),
            Some(a) => json!({"op": "+", "args": [a, expr_json(t)]}),
        })
    })
}

/// A state equation's left-hand side, `D(x)` on the whole state.
fn d_of(x: &str) -> Value {
    json!({"op": "D", "args": [x], "wrt": "t"})
}

/// The right-hand side of state `x`'s equation `i` holding `terms` (§6.1.2).
fn state_rhs(info: &ModelInfo, i: usize, x: &str, terms: &[&Term]) -> Value {
    state_rhs_of(info, i, x, terms.iter().map(|t| &t.expr))
}

/// [`state_rhs`] of the terms' expressions `exprs`.
fn state_rhs_of<'a>(
    info: &ModelInfo,
    i: usize,
    x: &str,
    exprs: impl IntoIterator<Item = &'a Expr>,
) -> Value {
    let lifted = info.flat.equation_lifted.get(i).copied().unwrap_or(false);
    let Some(f) = fold(exprs) else {
        return zero(info, x);
    };
    if !lifted {
        return f;
    }
    let output_idx: Vec<String> = node(&info.flat.equations[i].rhs)
        .and_then(|n| n.output_idx.clone())
        .unwrap_or_default();
    json!({
        "op": "faq",
        "args": [],
        "output_idx": output_idx,
        "ranges": Value::Object(lifted_ranges(&output_idx, &info.shapes[x])),
        "expr": f,
    })
}

/// The observeds `names` reference, with every observed their definitions
/// reference, transitively: the live observeds (§6.1.1, §6.2 item 2).
pub fn live_observeds(info: &ModelInfo, names: &BTreeSet<String>) -> BTreeSet<String> {
    info.close_over_observeds(names)
        .into_iter()
        .filter(|n| info.is_observed(n))
        .collect()
}

/// The right-hand-side document's component tree: every state and parameter, the live
/// observeds, the state equations (right-hand sides from `rhs`), the live
/// observeds' definitions and every `ic` equation (§6.1.1–§6.1.2).
fn tree(
    info: &ModelInfo,
    live: &BTreeSet<String>,
    forcing: &BTreeSet<String>,
    rhs: &dyn Fn(usize, &str) -> Value,
) -> Tree {
    let flat = &info.flat;
    let mut tree = Tree::default();
    for s in &info.states {
        tree.declare(s, variable_json(&flat.state_variables[s], &info.shapes[s]));
    }
    for (p, v) in &flat.parameters {
        let shape = v.shape.clone().unwrap_or_default();
        // §6.3: an unused data-fed parameter is not declared; one this
        // document reads is declared with its `data` rule; any other is an
        // ordinary parameter.
        if let Some(d) = info.data_fed.get(p) {
            if !info.used.contains(p) && !forcing.contains(p) {
                continue;
            }
            if forcing.contains(p) {
                let mut var = variable_json(v, &shape);
                var["update"] = json!({
                    "kind": "data",
                    "source": d.source,
                    "from": {"file_variable": d.file_variable},
                });
                tree.declare(p, var);
                continue;
            }
        }
        tree.declare(p, variable_json(v, &shape));
    }
    for o in live {
        let v = &flat.observed_variables[o];
        let shape = v.shape.clone().unwrap_or_default();
        tree.declare(o, variable_json(v, &shape));
    }
    for (i, eq) in flat.equations.iter().enumerate() {
        if let Some(x) = differentiated(&eq.lhs).filter(|x| info.is_state(x)) {
            tree.push(x, json!({"lhs": d_of(x), "rhs": rhs(i, x)}));
        } else if let Some(o) = defined_observed(&eq.lhs).filter(|o| live.contains(*o)) {
            tree.push(
                o,
                json!({"lhs": expr_json(&eq.lhs), "rhs": expr_json(&eq.rhs)}),
            );
        }
    }
    for (x, e) in &flat.field_ics {
        tree.push(x, ic_equation(x, e));
    }
    tree
}

/// The `ic` equation of state `x` whose initial value is `e` (§6.1.1).
fn ic_equation(x: &str, e: &Expr) -> Value {
    json!({"lhs": {"op": "ic", "args": [x]}, "rhs": expr_json(e)})
}

/// The URL template of a derived document's data source stub (§6.3).
pub const FORCING_STUB_URL: &str = "{esm_integrator_forcing}";

/// A derived document's `data_sources` (§6.3): one stub per source named by
/// a data-fed parameter of `forcing`; `None` when `forcing` is empty.
pub fn data_sources_stub(info: &ModelInfo, forcing: &BTreeSet<String>) -> Option<Value> {
    let stubs: BTreeMap<&str, Value> = forcing
        .iter()
        .filter_map(|p| info.data_fed.get(p))
        .map(|d| {
            (
                d.source.as_str(),
                json!({"kind": d.source_kind, "source": {"url_template": FORCING_STUB_URL}}),
            )
        })
        .collect();
    (!stubs.is_empty()).then(|| json!(stubs))
}

/// The data-fed parameters a document of terms with these `names` reads:
/// those among the names and, transitively, the names of the observeds'
/// definitions (§5.3).
pub fn forcing_of(info: &ModelInfo, names: &BTreeSet<String>) -> BTreeSet<String> {
    info.close_over_observeds(names)
        .into_iter()
        .filter(|n| info.data_fed.contains_key(n))
        .collect()
}

/// The top-level keys of a derived document (§6.1).
fn document(info: &ModelInfo, name: &str, models: Value, data_sources: Option<Value>) -> Value {
    let sets: Map<String, Value> = info
        .index_sets
        .iter()
        .map(|(k, n)| (k.clone(), json!({"kind": "interval", "size": n})))
        .collect();
    let mut out = Map::new();
    out.insert("esm".into(), json!(ESM_VERSION));
    out.insert("metadata".into(), json!({"name": name}));
    out.insert("index_sets".into(), Value::Object(sets));
    if let Some(d) = &info.domain {
        out.insert("domain".into(), d.clone());
    }
    out.insert("models".into(), models);
    if let Some(d) = data_sources {
        out.insert("data_sources".into(), d);
    }
    Value::Object(out)
}

/// A derived document's `domain` (§6.1), built as EarthSciASTDiff builds the
/// evaluation document's, so both documents carry the same one: `temporal`
/// holding the `start`, `end` and `reference_time` the source gives, then
/// `element_type` and `array_type` when given, each as EarthSciAST
/// serializes it. No `independent_variable`: the derived equations are
/// written with respect to `t`. `None` when nothing is left; an internal
/// error when EarthSciAST cannot serialize the domain.
pub fn domain_json(domain: &Domain) -> Result<Option<Value>> {
    let src = serde_json::to_value(domain)
        .map_err(|e| PlanError::Internal(format!("serializing the flattened domain: {e}")))?;
    let given = |m: &Map<String, Value>, k: &str| m.get(k).filter(|v| !v.is_null()).cloned();
    let mut out = Map::new();
    if let Some(Value::Object(tmp)) = src.get("temporal") {
        let t: Map<String, Value> = ["start", "end", "reference_time"]
            .iter()
            .filter_map(|k| given(tmp, k).map(|v| (k.to_string(), v)))
            .collect();
        if !t.is_empty() {
            out.insert("temporal".into(), Value::Object(t));
        }
    }
    if let Value::Object(src) = &src {
        for k in ["element_type", "array_type"] {
            if let Some(v) = given(src, k) {
                out.insert(k.into(), v);
            }
        }
    }
    Ok((!out.is_empty()).then_some(Value::Object(out)))
}

/// The right-hand-side document of an operator named `name` whose terms are
/// `terms` (§6.1). Also the document of a single term (§5.2.2).
pub fn rhs_document(info: &ModelInfo, name: &str, terms: &[&Term]) -> Value {
    term_document(info, name, terms, None)
}

/// The component of the probe document's new states (§5.2.2): the name
/// §6.1.4 reserves for the Jacobian document, so no name of the model can
/// clash with them.
pub const PROBE_COMPONENT: &str = JACOBIAN_MODEL;

/// The probe document's new state for the term at position `k` (0-based)
/// of the slice passed to [`probe_document`]: `JacobianEval.term_<k+1>`.
pub fn probe_state(k: usize) -> String {
    format!("{PROBE_COMPONENT}.term_{}", k + 1)
}

/// §5.2.2's probe document: §6.1's document of `terms` in which every
/// state of the model has the zero right-hand side (§6.1.2), and the `k`-th
/// term is instead the whole right-hand side of a new state
/// [`probe_state`]`(k)`. That state is declared as its equation's state is
/// (the same index-set names and `default`), its equation is the one that
/// state would have holding the term alone (a lifted term inside the same
/// `faq`), and it has that state's `ic` equations, if any. Its
/// bands with row `probe_state(k)` are the `k`-th term's bands, row renamed.
/// Subexpressions the terms share are written once each, as templates
/// ([`shared_templates`]).
pub fn probe_document(info: &ModelInfo, terms: &[Term]) -> Value {
    probe_document_with(info, terms, PROBE_TEMPLATE_MIN_NODES)
}

/// [`probe_document`], naming as templates the shared subexpressions of at
/// least `min_nodes` operator nodes.
pub fn probe_document_with(info: &ModelInfo, terms: &[Term], min_nodes: u64) -> Value {
    let terms: Vec<&Term> = terms.iter().collect();
    term_document(info, "probe", &terms, Some(min_nodes))
}

/// The fewest operator nodes in one occurrence of a subexpression the probe
/// document names as a template (§5.2.2).
pub const PROBE_TEMPLATE_MIN_NODES: u64 = 8;

/// EarthSciAST's `MAX_TEMPLATE_EXPANSION_DEPTH` (esm-spec §9.7.3): the most
/// templates on a chain of template bodies referencing templates.
pub const MAX_TEMPLATE_CHAIN: u32 = 32;

/// §5.2.2's shared subexpressions of `exprs`: a subexpression of at least
/// `min_nodes` operator nodes that would still be written at least twice in
/// the result, with a template's body written once however often the
/// template is referenced, is named as a zero-parameter template, `shared_k`
/// (`k` from 1, children before parents), and each of its occurrences, in
/// `exprs` and in other templates' bodies, is replaced by an
/// `apply_expression_template` reference to it. So every template is
/// referenced at least twice. A subexpression whose template would end a
/// chain of more than [`MAX_TEMPLATE_CHAIN`] templates is left in place.
/// Expanding every reference gives `exprs` back, structurally.
/// Subexpressions are compared structurally and exactly; the walk is
/// memoized on node identity, so it costs time in proportion to the
/// distinct nodes of `exprs` rather than to their size as trees. Returns
/// the templates as `(name, body)` pairs and `exprs` with the references.
pub fn shared_templates(exprs: &[&Expr], min_nodes: u64) -> (Vec<(String, Expr)>, Vec<Expr>) {
    use earthsci_ast::ExpressionNode;
    use std::collections::HashMap;
    use std::hash::{DefaultHasher, Hash, Hasher};
    use std::sync::Arc;
    type Ptr = *const ExpressionNode;
    #[derive(Default)]
    struct Interner {
        /// The structural id of each node seen, by identity.
        id_of: HashMap<Ptr, usize>,
        /// The ids whose shells have each hash.
        by_hash: HashMap<u64, Vec<usize>>,
        /// One node of each id, its operator nodes, its operator children.
        rep: Vec<Arc<ExpressionNode>>,
        nodes: Vec<u64>,
        kids: Vec<Vec<usize>>,
    }
    // A node's shell: the node with each operator child replaced by an
    // operator node holding the child's id. Leaf children are kept, and a
    // leaf never equals an operator node, so two nodes have equal shells
    // exactly when they are structurally equal. The children must be
    // interned.
    fn shell(it: &Interner, node: &ExpressionNode) -> ExpressionNode {
        node.map_children(&mut |c| match c {
            Expr::Operator(n) => Expr::Operator(Arc::new(ExpressionNode {
                value: Some(json!(it.id_of[&Arc::as_ptr(n)])),
                ..Default::default()
            })),
            leaf => leaf.clone(),
        })
    }
    // Structural ids, children before parents. Ids are 1-based; 0 is a
    // leaf. Shells are looked up by a hash of their `stable_json` and then
    // compared exactly, so a hash collision, or a shell that does not
    // serialize, costs only a comparison.
    fn intern(it: &mut Interner, e: &Expr) -> usize {
        let Expr::Operator(node) = e else {
            return 0;
        };
        let ptr = Arc::as_ptr(node);
        if let Some(&id) = it.id_of.get(&ptr) {
            return id;
        }
        let mut cs = Vec::new();
        node.for_each_child(&mut |c| {
            let cid = intern(it, c);
            if cid != 0 {
                cs.push(cid);
            }
        });
        let sh = shell(it, node);
        let mut h = DefaultHasher::new();
        match serde_json::to_value(&sh) {
            Ok(v) => earthsci_astdiff::stable_json(&v).hash(&mut h),
            Err(_) => sh.op.hash(&mut h),
        }
        let h = h.finish();
        let found = it.by_hash.get(&h).and_then(|ids| {
            ids.iter()
                .copied()
                .find(|&id| shell(it, &it.rep[id - 1]) == sh)
        });
        let id = match found {
            Some(id) => id,
            None => {
                let n = cs
                    .iter()
                    .fold(1u64, |a, &c| a.saturating_add(it.nodes[c - 1]));
                it.rep.push(node.clone());
                it.nodes.push(n);
                it.kids.push(cs);
                let id = it.rep.len();
                it.by_hash.entry(h).or_default().push(id);
                id
            }
        };
        it.id_of.insert(ptr, id);
        id
    }
    let mut it = Interner::default();
    let roots: Vec<usize> = exprs.iter().map(|e| intern(&mut it, e)).collect();
    let n = it.rep.len();
    // Shared subexpressions, parents first (parents have larger ids than
    // children): `written[id]` is how often the subexpression is written in
    // the result, a shared parent's body counting once.
    let mut written = vec![0u64; n + 1];
    for &r in &roots {
        if r != 0 {
            written[r] = written[r].saturating_add(1);
        }
    }
    let mut shared = vec![false; n + 1];
    for id in (1..=n).rev() {
        shared[id] = written[id] >= 2 && it.nodes[id - 1] >= min_nodes;
        let each = if shared[id] { 1 } else { written[id] };
        for &c in &it.kids[id - 1] {
            written[c] = written[c].saturating_add(each);
        }
    }
    // Templates, children first, within the chain bound. Leaving a shared
    // parent in place only writes its children more often, so every
    // template is still referenced at least twice.
    let mut name: Vec<Option<String>> = vec![None; n + 1];
    let mut chain = vec![0u32; n + 1];
    let mut k = 0;
    for id in 1..=n {
        let mut d = it.kids[id - 1].iter().map(|&c| chain[c]).max().unwrap_or(0);
        if shared[id] && d < MAX_TEMPLATE_CHAIN {
            k += 1;
            name[id] = Some(format!("shared_{k}"));
            d += 1;
        }
        chain[id] = d;
    }
    // Rewrite, memoized on structural id: one rewritten node per distinct
    // subexpression.
    struct Rewriter<'a> {
        it: &'a Interner,
        name: &'a [Option<String>],
        memo: Vec<Option<Expr>>,
    }
    impl Rewriter<'_> {
        fn body(&mut self, node: &ExpressionNode) -> Expr {
            Expr::Operator(Arc::new(node.map_children(&mut |c| self.rw(c))))
        }
        fn rw(&mut self, e: &Expr) -> Expr {
            let Expr::Operator(node) = e else {
                return e.clone();
            };
            let id = self.it.id_of[&Arc::as_ptr(node)];
            if let Some(x) = &self.memo[id] {
                return x.clone();
            }
            let x = match &self.name[id] {
                Some(nm) => Expr::Operator(Arc::new(ExpressionNode {
                    op: "apply_expression_template".into(),
                    name: Some(nm.clone()),
                    bindings: Some(HashMap::new()),
                    ..Default::default()
                })),
                None => self.body(node),
            };
            self.memo[id] = Some(x.clone());
            x
        }
    }
    let mut rw = Rewriter {
        it: &it,
        name: &name,
        memo: vec![None; n + 1],
    };
    let rewritten: Vec<Expr> = exprs.iter().map(|e| rw.rw(e)).collect();
    let mut templates = Vec::new();
    for (id, nm) in name.iter().enumerate() {
        if let Some(nm) = nm {
            templates.push((nm.clone(), rw.body(&it.rep[id - 1])));
        }
    }
    (templates, rewritten)
}

/// [`rhs_document`] (`probe` `None`) or [`probe_document`] (`probe` the
/// least size of a template) of `terms`, so the two documents are built by
/// the same code.
fn term_document(info: &ModelInfo, name: &str, terms: &[&Term], probe: Option<u64>) -> Value {
    let mut by_eq: BTreeMap<usize, Vec<&Term>> = BTreeMap::new();
    let mut names = BTreeSet::new();
    for t in terms {
        by_eq.entry(t.equation).or_default().push(t);
        names.extend(t.names.iter().cloned());
    }
    let live = live_observeds(info, &names);
    let forcing = forcing_of(info, &names);
    let rhs = |i: usize, x: &str| {
        let ts = match by_eq.get(&i) {
            Some(ts) if probe.is_none() => ts.as_slice(),
            _ => &[],
        };
        state_rhs(info, i, x, ts)
    };
    let mut tree = tree(info, &live, &forcing, &rhs);
    let mut templates = Vec::new();
    if let Some(min_nodes) = probe {
        // Subexpressions the terms share, written once each (§5.2.2).
        let exprs: Vec<&Expr> = terms.iter().map(|t| &t.expr).collect();
        let (tpl, exprs) = shared_templates(&exprs, min_nodes);
        templates = tpl;
        for (k, t) in terms.iter().enumerate() {
            let x = probe_state(k);
            let v = &info.flat.state_variables[&t.variable];
            tree.declare(&x, variable_json(v, &info.shapes[&t.variable]));
            let rhs = state_rhs_of(info, t.equation, &t.variable, [&exprs[k]]);
            tree.push(&x, json!({"lhs": d_of(&x), "rhs": rhs}));
            for (_, e) in info.flat.field_ics.iter().filter(|(s, _)| *s == t.variable) {
                tree.push(&x, ic_equation(&x, e));
            }
        }
    }
    let mut models = tree.to_json();
    if !templates.is_empty() {
        let tpl: Map<String, Value> = templates
            .iter()
            .map(|(nm, body)| (nm.clone(), json!({"params": [], "body": expr_json(body)})))
            .collect();
        models[PROBE_COMPONENT]["expression_templates"] = Value::Object(tpl);
    }
    document(info, name, models, data_sources_stub(info, &forcing))
}

/// The Jacobian document of an operator (§6.2): EarthSciASTDiff's
/// evaluation document (`evaluation_document`, coefficient hoisting off) of
/// `sv`, the view of the operator's flattened right-hand-side document, for
/// its band list `entries`, without its `ic` equations, and with a
/// `default` of `0.0` on every state that has none (§6.2 item 4). Band `k`
/// (1-based) is the field `JacobianEval.J_k`; the document's `esm` version
/// and `metadata.name` are EarthSciASTDiff's.
pub fn jacobian_document(sv: &SysView, entries: &[JacEntry]) -> Result<Value> {
    let (mut doc, fields) = earthsci_astdiff::evaluation_document(sv, entries, false, 0)
        .map_err(|e| PlanError::Internal(format!("evaluation_document: {}", e.0)))?;
    // Band `k` is the field `J_k` (§6.2).
    let want: Vec<String> = (1..=entries.len()).map(|k| format!("J_{k}")).collect();
    if fields != want {
        return Err(PlanError::Internal(format!(
            "evaluation_document wrote the band fields {fields:?} for {} bands",
            entries.len()
        )));
    }
    let Some(Value::Object(models)) = doc.get_mut("models") else {
        return Err(PlanError::Internal(
            "evaluation_document wrote no models".into(),
        ));
    };
    // EarthSciASTDiff names the model; §6.1.4 reserves the name it uses.
    if !models.contains_key(JACOBIAN_MODEL) {
        return Err(PlanError::Internal(format!(
            "evaluation_document wrote no model {JACOBIAN_MODEL}: its models are {:?}",
            models.keys().collect::<Vec<_>>()
        )));
    }
    models.values_mut().for_each(drop_ic);
    default_states_to_zero(models)?;
    Ok(doc)
}

/// Gives every state of `models` without a `default` the `default` `0.0`
/// (§6.2 item 4). With its `ic` equations dropped, such a state (one only
/// an `ic` equation or `field_ics` entry starts) has no starting value,
/// which EarthSciAST refuses when it builds the document
/// (`E_TREEWALK_MISSING_INITIAL_VALUE`). No evaluation reads the value. A
/// state is a variable a `D` equation defines: by its fully scoped name
/// (§6.2 item 2), or by its name in the equation's own model or subsystem.
fn default_states_to_zero(models: &mut Map<String, Value>) -> Result<()> {
    fn states(node: &Value, path: &[String], out: &mut Vec<Vec<String>>) {
        for eq in node["equations"].as_array().into_iter().flatten() {
            if eq["lhs"]["op"] == "D" {
                if let Some(x) = eq["lhs"]["args"][0].as_str() {
                    let mut local = path.to_vec();
                    local.push(x.to_string());
                    out.push(local);
                }
            }
        }
        for (k, sub) in node["subsystems"].as_object().into_iter().flatten() {
            let mut p = path.to_vec();
            p.push(k.clone());
            states(sub, &p, out);
        }
    }
    // The declaration of the variable `name` (dot-separated) under `models`.
    fn variable<'a>(models: &'a mut Map<String, Value>, name: &[&str]) -> Option<&'a mut Value> {
        let (var, path) = name.split_last()?;
        let (top, subs) = path.split_first()?;
        let mut node = models.get_mut(*top)?;
        for s in subs {
            node = node.get_mut("subsystems")?.get_mut(*s)?;
        }
        node.get_mut("variables")?.get_mut(*var)
    }
    let mut found = Vec::new();
    for (k, m) in models.iter() {
        states(m, std::slice::from_ref(k), &mut found);
    }
    for local in found {
        let x = local.last().expect("pushed above").clone();
        let scoped: Vec<&str> = x.split('.').collect();
        let local: Vec<&str> = local[..local.len() - 1]
            .iter()
            .map(String::as_str)
            .chain(std::iter::once(x.as_str()))
            .collect();
        let decl = if variable(models, &scoped).is_some() {
            variable(models, &scoped)
        } else {
            variable(models, &local)
        };
        let Some(Value::Object(decl)) = decl else {
            return Err(PlanError::Internal(format!(
                "the Jacobian document declares no state {x}"
            )));
        };
        decl.entry("default").or_insert(json!(0.0));
    }
    Ok(())
}

/// Removes the `ic` equations of a model and of its subsystems.
fn drop_ic(model: &mut Value) {
    if let Some(Value::Array(eqs)) = model.get_mut("equations") {
        eqs.retain(|eq| eq["lhs"]["op"] != "ic");
    }
    if let Some(Value::Object(subs)) = model.get_mut("subsystems") {
        subs.values_mut().for_each(drop_ic);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astdiff::FlatView;

    fn domain(v: Value) -> Domain {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn domain_keeps_what_evaluation_reads() {
        let d = domain(json!({
            "independent_variable": "time",
            "temporal": {"start": "2020-01-01T00:00:00Z", "reference_time": "2019-12-31T00:00:00Z"},
            "element_type": "Float64",
            "array_type": "Array",
        }));
        assert_eq!(
            domain_json(&d).unwrap(),
            Some(json!({
                "temporal": {"start": "2020-01-01T00:00:00Z", "reference_time": "2019-12-31T00:00:00Z"},
                "element_type": "Float64",
                "array_type": "Array",
            }))
        );
        // Nothing left: no `domain`.
        assert_eq!(
            domain_json(&domain(json!({"independent_variable": "t"}))).unwrap(),
            None
        );
        assert_eq!(domain_json(&domain(json!({"temporal": {}}))).unwrap(), None);
    }

    /// The right-hand-side document's `domain` is the Jacobian document's,
    /// EarthSciASTDiff's evaluation document of the same source.
    #[test]
    fn domain_matches_evaluation_document() {
        let src = json!({
            "esm": "1.3.0",
            "metadata": {"name": "d"},
            "domain": {
                "independent_variable": "t",
                "temporal": {"start": "2020-01-01T00:00:00Z", "end": "2020-01-02T00:00:00Z"},
                "element_type": "Float64",
            },
            "models": {"M": {
                "variables": {"x": {"type": "unknown", "default": 1.0}, "k": {"type": "parameter", "default": 2.0}},
                "equations": [{"lhs": {"op": "D", "args": ["x"], "wrt": "t"},
                               "rhs": {"op": "*", "args": [{"op": "-", "args": ["k"]}, "x"]}}],
            }},
        });
        let view = FlatView::new(&src.to_string()).unwrap();
        let ours = domain_json(view.flat.domain.as_ref().unwrap()).unwrap();
        assert_eq!(
            ours,
            Some(json!({
                "temporal": {"start": "2020-01-01T00:00:00Z", "end": "2020-01-02T00:00:00Z"},
                "element_type": "Float64",
            }))
        );
        let sv = view.sysview().unwrap();
        let jac = jacobian_document(&sv, &crate::astdiff::bands(&sv).unwrap()).unwrap();
        assert_eq!(jac.get("domain").cloned(), ours);
    }
}
