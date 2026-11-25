//! What to emit, derived from the approved rule set so the facts match the rules that will read them.
//!
//! Floor positions are uncertain, so `in`, `near` and `under` are three-valued: *definitely*,
//! *possibly*, or no. A fact is a single bit, so for each relation we pick the threshold that errs
//! towards raising the alert:
//!
//! * used only positively (`a worker is in a height-zone`) → emit when **possibly** true;
//! * used only under negation (`... is not in a walkway`) → emit only when **definitely** true;
//! * used both ways → emit when possibly true, and report a warning (one of those rules leans
//!   towards silence under uncertainty).
//!
//! `formal/SceneFacts.lean` proves that this choice never loses an alert that the true positions
//! would raise, given the stated bound on position error.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Emit { Possible, Definite }

#[derive(Clone, Debug)]
pub struct RelNames { pub inside: String, pub near: String, pub under: String, pub on: String, pub wear: String, pub carry: String, pub hold: String, pub r#use: String, pub drive: String, pub have: String }
impl Default for RelNames {
    /// The prepositions and verbs of iogp-editor's `vg360_catalog.json`.
    fn default() -> Self {
        let s = |x: &str| x.to_string();
        RelNames { inside: s("in"), near: s("near"), under: s("under"), on: s("on"), wear: s("wear"), carry: s("carry"), hold: s("hold"), r#use: s("use"), drive: s("drive"), have: s("have") }
    }
}

#[derive(Clone, Debug)]
pub struct Policy {
    pub inside: Emit,
    pub near: Emit,
    pub under: Emit,
    /// distance for `near`, metres
    pub near_m: f64,
    /// detector classes that take part in `near` conditions (empty = every class)
    pub near_classes: Vec<String>,
    pub names: RelNames,
    pub warnings: Vec<String>,
}

impl Default for Policy {
    fn default() -> Self { Policy { inside: Emit::Possible, near: Emit::Possible, under: Emit::Possible, near_m: 3.0, near_classes: vec![], names: RelNames::default(), warnings: vec![] } }
}

impl Policy {
    /// Derive the policy from (relation, negated, metres, subject detector, object detector) per condition.
    pub fn from_conditions<'a>(conds: impl IntoIterator<Item = (&'a str, bool, Option<f64>, Option<&'a str>, Option<&'a str>)>, names: RelNames) -> Policy {
        let mut pol = Policy { names, ..Default::default() };
        let (mut pos, mut neg) = ([false; 3], [false; 3]);
        let mut metres: Vec<f64> = vec![];
        let mut near_classes: Vec<String> = vec![];
        let mut near_any_class = false;
        for (rel, negated, m, sd, od) in conds {
            let i = if rel == pol.names.inside { 0 } else if rel == pol.names.near { 1 } else if rel == pol.names.under { 2 } else { continue };
            if negated { neg[i] = true } else { pos[i] = true }
            if i == 1 {
                if let Some(m) = m { if !metres.iter().any(|x| (x - m).abs() < 1e-9) { metres.push(m); } }
                for d in [sd, od] {
                    match d { Some(d) => if !near_classes.iter().any(|c| c == d) { near_classes.push(d.to_string()) }, None => near_any_class = true }
                }
            }
        }
        let pick = |i: usize, name: &str, w: &mut Vec<String>| -> Emit {
            match (pos[i], neg[i]) {
                (false, true) => Emit::Definite,
                (true, true) => { w.push(format!("'{name}' is used both with and without negation; facts are emitted when possibly true, so the negated rules lean towards silence under position uncertainty")); Emit::Possible }
                _ => Emit::Possible,
            }
        };
        pol.inside = pick(0, &pol.names.inside.clone(), &mut pol.warnings);
        pol.near = pick(1, &pol.names.near.clone(), &mut pol.warnings);
        pol.under = pick(2, &pol.names.under.clone(), &mut pol.warnings);
        if let Some(mx) = metres.iter().cloned().reduce(f64::max) {
            pol.near_m = mx;
            if metres.len() > 1 { pol.warnings.push(format!("rules use different 'near' distances {metres:?}; one 'near' fact cannot carry them all, so the largest ({mx} m) is used")); }
        }
        pol.near_classes = if near_any_class { vec![] } else { near_classes };
        pol
    }

    #[cfg(feature = "iogp")]
    pub fn from_rules(rules: &[iogp_runtime::Rule]) -> Policy {
        use iogp_runtime::{CondKind, NounKind};
        let det = |r: &'_ iogp_runtime::Rule, v: &str| -> Option<String> {
            r.entities.iter().find(|e| e.var == v).and_then(|e| match &e.kind { NounKind::Class { detector, .. } => Some(detector.clone()), _ => None })
        };
        let mut rows: Vec<(String, bool, Option<f64>, Option<String>, Option<String>)> = vec![];
        for r in rules {
            for c in &r.conditions {
                let m = match &c.kind { CondKind::Spatial { metres, .. } => *metres, _ => None };
                rows.push((c.kind.relation().to_string(), c.negated, m, det(r, &c.subject), det(r, &c.object)));
            }
        }
        Policy::from_conditions(rows.iter().map(|(a, b, c, d, e)| (a.as_str(), *b, *c, d.as_deref(), e.as_deref())), RelNames::default())
    }

    pub fn near_applies(&self, class: &str) -> bool { self.near_classes.is_empty() || self.near_classes.iter().any(|c| c == class) }
}

/// The three-valued test shared by every uncertain relation: `margin` is how far inside the
/// threshold the estimate is (positive = inside), `slack` is k·σ.
#[inline]
pub fn decide(emit: Emit, margin: f64, slack: f64) -> bool {
    match emit { Emit::Possible => margin + slack >= 0.0, Emit::Definite => margin - slack >= 0.0 }
}
