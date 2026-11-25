//! The floor rules as Datalog, with a proof for every alarm.
//!
//! The rules are written as controlled-English sentences in yappu-fsm's style and compiled
//! to its clause schema (`{rid, head, body, stratum}`, variables as `{"v": "C"}`):
//!
//! ```text
//! SEED     If a cell C is a hazard-cell then the cell C is reached.
//! SPREAD   If the cell C is reached and the cell C is adjacent to a cell D and the cell D is
//!          not a barrier-cell then the cell D is reached.
//! BREACH   If the cell C is reached and the cell C is a border-cell then a perimeter-breach
//!          alert is raised.
//! SHIELD   If a cell C lies on the ray from a person T to a forklift K and the cell C is a
//!          barrier-cell then the person T is shielded from the forklift K.
//! LOF      If a person T is within 3.0 metres of a forklift K and the person T is not
//!          shielded from the forklift K then a line-of-fire alert is raised for the person T.
//! REDZONE  If a person T stands on a cell C and the cell C is a red-zone-cell then a red-zone
//!          alert is raised for the person T.
//! ```
//!
//! Evaluation is bottom-up, stratum by stratum, semi-naive, with an index on every argument
//! position: the same least fixpoint as yappu-fsm's own engine (checked fact for fact
//! against it in the notebook), about thirty times faster than its naive loop there. Every
//! derived fact remembers the rule and the facts that made it. For a barricade breach that
//! derivation is the path a fill took from the hazard to the edge: it passes through the gap,
//! and [`valid_path`] re-checks it (`proofs/Grammar.lean`, `valid_all_free`).
//!
//! Geometry stays outside: Datalog has no arithmetic, so distances, the radius test and the
//! cells on a ray come in as facts, the way yappu-fsm's timers hand in durations.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::grammar::{line4, Cell, Grid, RED_ZONE};

/// A clause argument: a variable or a constant.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Term {
    Var { v: String },
    Const(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Literal {
    pub pred: String,
    pub args: Vec<Term>,
    pub neg: bool,
}

/// One rule, in yappu-fsm's schema.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Clause {
    pub rid: String,
    #[serde(default)]
    pub base: String,
    pub head: Literal,
    pub body: Vec<Literal>,
    pub stratum: i64,
    #[serde(default)]
    pub text: String,
}

#[derive(Clone, Copy)]
enum Arg {
    Var(usize),
    Const(u32),
}

struct Lit {
    pred: u32,
    args: Vec<Arg>,
    neg: bool,
}

struct Rule {
    rid: String,
    head: Lit,
    body: Vec<Lit>,
    nvars: usize,
    stratum: i64,
}

#[derive(Clone, Debug)]
enum Used {
    Fact(usize),
    Not(Vec<u32>),
}

/// A computed model: every fact, and why each one holds.
pub struct Model {
    names: Vec<String>,
    sym: HashMap<String, u32>,
    facts: Vec<Vec<u32>>,
    at: HashMap<Vec<u32>, usize>,
    by: HashMap<u32, Vec<usize>>,
    idx: HashMap<(u32, usize, u32), Vec<usize>>,
    why: Vec<(String, Vec<Used>)>,
}

type Binding = Vec<Option<u32>>;

impl Model {
    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&i) = self.sym.get(s) {
            return i;
        }
        let i = self.names.len() as u32;
        self.names.push(s.to_string());
        self.sym.insert(s.to_string(), i);
        i
    }

    fn add(&mut self, f: Vec<u32>, w: (String, Vec<Used>)) -> bool {
        if self.at.contains_key(&f) {
            return false;
        }
        let k = self.facts.len();
        self.by.entry(f[0]).or_default().push(k);
        for (j, &v) in f[1..].iter().enumerate() {
            self.idx.entry((f[0], j, v)).or_default().push(k);
        }
        self.at.insert(f.clone(), k);
        self.facts.push(f);
        self.why.push(w);
        true
    }

    fn compile(&mut self, c: &Clause) -> Rule {
        let mut vars: Vec<String> = Vec::new();
        let mut lit = |m: &mut Model, l: &Literal| Lit {
            pred: m.intern(&l.pred),
            neg: l.neg,
            args: l.args.iter().map(|a| match a {
                Term::Var { v } => Arg::Var(vars.iter().position(|x| x == v).unwrap_or_else(|| { vars.push(v.clone()); vars.len() - 1 })),
                Term::Const(s) => Arg::Const(m.intern(s)),
            }).collect(),
        };
        let body: Vec<Lit> = c.body.iter().map(|l| lit(self, l)).collect();
        let head = lit(self, &c.head);
        Rule { rid: c.rid.clone(), head, body, nvars: vars.len(), stratum: c.stratum }
    }

    /// Run `clauses` over `facts` to their least fixpoint.
    pub fn run(clauses: &[Clause], facts: Vec<Vec<String>>) -> Model {
        let mut m = Model { names: vec![], sym: HashMap::new(), facts: vec![], at: HashMap::new(), by: HashMap::new(), idx: HashMap::new(), why: vec![] };
        for f in facts {
            let f: Vec<u32> = f.iter().map(|s| m.intern(s)).collect();
            m.add(f, ("fact".into(), vec![]));
        }
        let rules: Vec<Rule> = clauses.iter().map(|c| m.compile(c)).collect();
        let mut strata: Vec<i64> = rules.iter().map(|r| r.stratum).collect();
        strata.sort();
        strata.dedup();
        for s in strata {
            let rs: Vec<&Rule> = rules.iter().filter(|r| r.stratum == s).collect();
            let mut delta: Option<HashMap<u32, Vec<usize>>> = None;
            loop {
                let mut pending: HashMap<Vec<u32>, (String, Vec<Used>)> = HashMap::new();
                let mut new: Vec<Vec<u32>> = Vec::new();
                for r in &rs {
                    let pos: Vec<usize> = (0..r.body.len()).filter(|&i| !r.body[i].neg).collect();
                    let dps: Vec<Option<usize>> = if delta.is_none() { vec![None] } else { pos.into_iter().map(Some).collect() };
                    for dp in dps {
                        if let (Some(i), Some(d)) = (dp, &delta) {
                            if d.get(&r.body[i].pred).is_none_or(|v| v.is_empty()) {
                                continue;
                            }
                        }
                        let mut out = Vec::new();
                        m.matches(r, 0, &mut vec![None; r.nvars], &mut Vec::new(), dp, delta.as_ref(), &mut out);
                        for (b, used) in out {
                            let h = m.ground(&r.head, &b);
                            if !m.at.contains_key(&h) && !pending.contains_key(&h) {
                                pending.insert(h.clone(), (r.rid.clone(), used));
                                new.push(h);
                            }
                        }
                    }
                }
                if new.is_empty() {
                    break;
                }
                let mut d: HashMap<u32, Vec<usize>> = HashMap::new();
                for h in new {
                    let w = pending.remove(&h).expect("pending");
                    let p = h[0];
                    m.add(h, w);
                    d.entry(p).or_default().push(m.facts.len() - 1);
                }
                delta = Some(d);
            }
        }
        m
    }

    fn ground(&self, l: &Lit, b: &Binding) -> Vec<u32> {
        let mut g = vec![l.pred];
        g.extend(l.args.iter().map(|a| match a {
            Arg::Var(i) => b[*i].expect("a head or negated variable is bound by the body"),
            Arg::Const(c) => *c,
        }));
        g
    }

    #[allow(clippy::too_many_arguments)]
    fn matches(&self, r: &Rule, i: usize, b: &mut Binding, used: &mut Vec<Used>, dp: Option<usize>,
               delta: Option<&HashMap<u32, Vec<usize>>>, out: &mut Vec<(Binding, Vec<Used>)>) {
        if i == r.body.len() {
            out.push((b.clone(), used.clone()));
            return;
        }
        let l = &r.body[i];
        if l.neg {
            let g = self.ground(l, b);
            if !self.at.contains_key(&g) {
                used.push(Used::Not(g));
                self.matches(r, i + 1, b, used, dp, delta, out);
                used.pop();
            }
            return;
        }
        let empty = Vec::new();
        let cand: &Vec<usize> = if dp == Some(i) {
            delta.and_then(|d| d.get(&l.pred)).unwrap_or(&empty)
        } else {
            let mut c = None;
            for (j, a) in l.args.iter().enumerate() {
                let v = match a { Arg::Var(k) => b[*k], Arg::Const(s) => Some(*s) };
                if let Some(v) = v {
                    c = Some(self.idx.get(&(l.pred, j, v)).unwrap_or(&empty));
                    break;
                }
            }
            c.unwrap_or_else(|| self.by.get(&l.pred).unwrap_or(&empty))
        };
        for &k in cand {
            let f = &self.facts[k];
            if f.len() != l.args.len() + 1 {
                continue;
            }
            let saved = b.clone();
            let ok = l.args.iter().enumerate().all(|(j, a)| match a {
                Arg::Var(x) => match b[*x] {
                    Some(v) => v == f[j + 1],
                    None => { b[*x] = Some(f[j + 1]); true }
                },
                Arg::Const(c) => *c == f[j + 1],
            });
            if ok {
                used.push(Used::Fact(k));
                self.matches(r, i + 1, b, used, dp, delta, out);
                used.pop();
            }
            *b = saved;
        }
    }

    fn lookup(&self, f: &[&str]) -> Option<usize> {
        let g: Option<Vec<u32>> = f.iter().map(|s| self.sym.get(*s).copied()).collect();
        g.and_then(|g| self.at.get(&g).copied())
    }

    /// Whether a fact holds.
    pub fn has(&self, f: &[&str]) -> bool {
        self.lookup(f).is_some()
    }

    fn text(&self, f: &[u32]) -> String {
        format!("{}({})", self.names[f[0] as usize], f[1..].iter().map(|&s| self.names[s as usize].as_str()).collect::<Vec<_>>().join(", "))
    }

    /// yappu-style proof lines: the fact, the rule that made it, then its premises.
    pub fn explain(&self, f: &[&str]) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(k) = self.lookup(f) {
            self.explain_at(k, 0, &mut std::collections::HashSet::new(), &mut out);
        }
        out
    }

    fn explain_at(&self, k: usize, depth: usize, seen: &mut std::collections::HashSet<usize>, out: &mut Vec<String>) {
        let (rid, used) = &self.why[k];
        out.push(format!("{}{}  [{}]", "  ".repeat(depth), self.text(&self.facts[k]), if rid == "fact" { "given" } else { rid }));
        if !seen.insert(k) {
            return;
        }
        for u in used {
            match u {
                Used::Not(g) => out.push(format!("{}not {}  [absent]", "  ".repeat(depth + 1), self.text(g))),
                Used::Fact(j) => {
                    let p = self.names[self.facts[*j][0] as usize].as_str();
                    if !["adjacent", "border", "hazard", "red", "on_ray"].contains(&p) {
                        self.explain_at(*j, depth + 1, seen, out);
                    }
                }
            }
        }
    }

    /// The breach's derivation, read back from the hazard to the edge: the escape path.
    pub fn escape_path(&self) -> Option<Vec<Cell>> {
        let alert = *self.sym.get("alert")?;
        let (scene, breach, reached) = (*self.sym.get("scene")?, *self.sym.get("perimeter-breach")?, *self.sym.get("reached")?);
        let hit = *self.by.get(&alert)?.iter().find(|&&k| self.facts[k][1] == scene && self.facts[k][2] == breach)?;
        let next = |k: usize| self.why[k].1.iter().find_map(|u| match u { Used::Fact(j) if self.facts[*j][0] == reached => Some(*j), _ => None });
        let mut cur = next(hit)?;
        let mut path = Vec::new();
        loop {
            let name = &self.names[self.facts[cur][1] as usize];
            let (x, y) = name.split_once(':')?;
            path.push((x.parse().ok()?, y.parse().ok()?));
            if self.why[cur].0 == "SEED" {
                break;
            }
            cur = next(cur)?;
        }
        path.reverse();
        Some(path)
    }
}

fn cid(c: Cell) -> String {
    format!("{}:{}", c.0, c.1)
}

/// The facts for one frame: the fill region's adjacency and edge, the red zone, the
/// barrier cells, and for each person their cell and, for each forklift within reach,
/// the cells on the ray between them.
pub fn grid_facts(grid: &Grid, bar: &[Vec<bool>], fks: &[Cell], workers: &[(u32, Cell)]) -> Vec<Vec<String>> {
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<String>>();
    let mut f = vec![s(&["hazard", &cid(grid.seed)])];
    let [x0, y0, x1, y1] = grid.site.fill_box;
    for x in x0..=x1 {
        for y in y0..=y1 {
            if x == x0 || x == x1 || y == y0 || y == y1 {
                f.push(s(&["border", &cid((x, y))]));
            } else {
                for n in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] {
                    f.push(s(&["adjacent", &cid((x, y)), &cid(n)]));
                }
            }
        }
    }
    for y in 0..grid.h {
        for x in 0..grid.w {
            if grid.tokens[y as usize][x as usize] == RED_ZONE {
                f.push(s(&["red", &cid((x, y))]));
            }
        }
    }
    for (y, row) in bar.iter().enumerate() {
        for (x, &b) in row.iter().enumerate() {
            if b {
                f.push(s(&["barrier", &cid((x as i64, y as i64))]));
            }
        }
    }
    let radius = format!("{:?}", grid.site.r_lof);
    for &(tid, wc) in workers {
        let t = format!("T{tid}");
        f.push(s(&["stands_on", &t, &cid(wc)]));
        let w = grid.c2f(wc);
        for (k, &fc) in fks.iter().enumerate() {
            let p = grid.c2f(fc);
            if (w[0] - p[0]).hypot(w[1] - p[1]) <= grid.site.r_lof {
                let kk = format!("F{k}");
                f.push(s(&["within", &t, &kk, &radius]));
                let ray = line4(wc, fc);
                for &c in ray.iter().take(ray.len().saturating_sub(1)).skip(1) {
                    f.push(s(&["on_ray", &t, &kk, &cid(c)]));
                }
            }
        }
    }
    f
}

/// The check behind a breach proof (`proofs/Grammar.lean`: `validPath`, `valid_all_free`):
/// it starts at the hazard, ends on the edge of the fill box, moves one 4-neighbour at a
/// time and never stands on a barrier.
pub fn valid_path(grid: &Grid, path: &[Cell], bar: &[Vec<bool>]) -> bool {
    let [x0, y0, x1, y1] = grid.site.fill_box;
    let (Some(&first), Some(&last)) = (path.first(), path.last()) else { return false };
    first == grid.seed
        && (last.0 == x0 || last.0 == x1 || last.1 == y0 || last.1 == y1)
        && path.iter().all(|&(x, y)| !bar[y as usize][x as usize])
        && path.windows(2).all(|w| (w[0].0 - w[1].0).abs() + (w[0].1 - w[1].1).abs() == 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clause(json: &str) -> Clause {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn recursion_negation_and_provenance() {
        // reach(D) over free cells in a 1-D corridor with a wall: stratified negation, recursion
        let cs = vec![
            clause(r#"{"rid":"SEED","head":{"pred":"reached","args":[{"v":"C"}],"neg":false},"body":[{"pred":"hazard","args":[{"v":"C"}],"neg":false}],"stratum":0}"#),
            clause(r#"{"rid":"SPREAD","head":{"pred":"reached","args":[{"v":"D"}],"neg":false},"body":[{"pred":"reached","args":[{"v":"C"}],"neg":false},{"pred":"adjacent","args":[{"v":"C"},{"v":"D"}],"neg":false},{"pred":"barrier","args":[{"v":"D"}],"neg":true}],"stratum":0}"#),
            clause(r#"{"rid":"OUT","head":{"pred":"alert","args":["scene","perimeter-breach"],"neg":false},"body":[{"pred":"reached","args":[{"v":"C"}],"neg":false},{"pred":"border","args":[{"v":"C"}],"neg":false}],"stratum":1}"#),
        ];
        let mut facts = vec![vec!["hazard".to_string(), "0:0".into()], vec!["border".into(), "3:0".into()]];
        for x in 0..3 {
            facts.push(vec!["adjacent".into(), format!("{x}:0"), format!("{}:0", x + 1)]);
        }
        let open = Model::run(&cs, facts.clone());
        assert!(open.has(&["alert", "scene", "perimeter-breach"]));
        assert_eq!(open.escape_path().unwrap(), vec![(0, 0), (1, 0), (2, 0), (3, 0)]);
        facts.push(vec!["barrier".into(), "2:0".into()]);
        let closed = Model::run(&cs, facts);
        assert!(!closed.has(&["alert", "scene", "perimeter-breach"]));
        assert!(closed.has(&["reached", "1:0"]) && !closed.has(&["reached", "3:0"]));
    }
}
