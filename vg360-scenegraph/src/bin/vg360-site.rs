//! vg360-site: fit a site file from a calibration worksheet (see `vg360_scenegraph::fit`).
//!
//! ```text
//! vg360-site fit WORKSHEET.json [--catalog vg360_catalog.json] [-o site.json]
//! ```
//! Prints each camera's fit (marks, RMS and worst residual) to stderr. Exits 1 with every problem
//! listed, and writes nothing, when a camera does not fit or the catalog's zones are not all drawn.
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: vg360-site fit WORKSHEET.json [--catalog CATALOG.json] [-o SITE.json]";
    if args.first().map(String::as_str) != Some("fit") || args.len() < 2 { eprintln!("{usage}"); return ExitCode::from(2); }
    let (mut catalog, mut out) = (None, None);
    let mut i = 2;
    while i < args.len() {
        match (args[i].as_str(), args.get(i + 1)) {
            ("--catalog", Some(p)) => { catalog = Some(p.clone()); i += 2; }
            ("-o", Some(p)) => { out = Some(p.clone()); i += 2; }
            _ => { eprintln!("{usage}"); return ExitCode::from(2); }
        }
    }
    let read = |p: &str| std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"));
    let ws: vg360_scenegraph::fit::Worksheet = match read(&args[1]).and_then(|t| serde_json::from_str(&t).map_err(|e| format!("{}: {e}", args[1]))) {
        Ok(w) => w, Err(e) => { eprintln!("{e}"); return ExitCode::from(1); }
    };
    let cat: Option<serde_json::Value> = match catalog.as_deref().map(|p| read(p).and_then(|t| serde_json::from_str(&t).map_err(|e| format!("{p}: {e}")))) {
        None => None, Some(Ok(v)) => Some(v), Some(Err(e)) => { eprintln!("{e}"); return ExitCode::from(1); }
    };
    match vg360_scenegraph::fit::build_site(&ws, cat.as_ref()) {
        Ok((site, reports)) => {
            for r in &reports { eprintln!("{}", serde_json::to_string(r).unwrap()); }
            let text = serde_json::to_string_pretty(&site).unwrap() + "\n";
            match out {
                Some(p) => if let Err(e) = std::fs::write(&p, text) { eprintln!("{p}: {e}"); return ExitCode::from(1); },
                None => print!("{text}"),
            }
            ExitCode::SUCCESS
        }
        Err(errs) => {
            eprintln!("{} problem(s), nothing written:", errs.len());
            for e in errs { eprintln!("  - {e}"); }
            ExitCode::from(1)
        }
    }
}
