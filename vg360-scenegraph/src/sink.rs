//! Where facts go. The builder only calls `begin`, `object` and `fact`, mirroring
//! `iogp_runtime::Engine`'s zero-copy frame builder; the caller then calls `Engine::end()`.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjKind { Class, Zone, Signal }

pub trait FactSink {
    fn begin(&mut self, t: f64, camera: &str);
    fn object(&mut self, id: &str, kind: ObjKind, ty: &str, camera: Option<&str>);
    fn fact(&mut self, rel: &str, a: &str, b: &str);
}

/// Discards everything (benchmarks, allocation tests).
#[derive(Default)]
pub struct NullSink { pub objects: usize, pub facts: usize }
impl FactSink for NullSink {
    fn begin(&mut self, _: f64, _: &str) { self.objects = 0; self.facts = 0; }
    fn object(&mut self, _: &str, _: ObjKind, _: &str, _: Option<&str>) { self.objects += 1; }
    fn fact(&mut self, _: &str, _: &str, _: &str) { self.facts += 1; }
}

#[cfg(feature = "iogp")]
mod iogp {
    use super::*;
    use iogp_runtime::fast::Kind;

    fn kind(k: ObjKind) -> Kind { match k { ObjKind::Class => Kind::Class, ObjKind::Zone => Kind::Zone, ObjKind::Signal => Kind::Signal } }

    impl FactSink for iogp_runtime::Engine {
        fn begin(&mut self, t: f64, camera: &str) { iogp_runtime::Engine::begin(self, t, Some(camera)); }
        fn object(&mut self, id: &str, k: ObjKind, ty: &str, camera: Option<&str>) { iogp_runtime::Engine::object(self, id, kind(k), ty, camera); }
        fn fact(&mut self, rel: &str, a: &str, b: &str) { iogp_runtime::Engine::fact(self, rel, a, b); }
    }

    /// Collects an owned `iogp_runtime::Frame` (JSON logs, replay, the reference engine).
    impl FactSink for iogp_runtime::Frame {
        fn begin(&mut self, t: f64, camera: &str) { self.t = t; self.camera = Some(camera.to_string()); self.objects.clear(); self.facts.clear(); }
        fn object(&mut self, id: &str, k: ObjKind, ty: &str, camera: Option<&str>) {
            let mut o = iogp_runtime::Object { id: id.to_string(), camera: camera.map(str::to_string), ..Default::default() };
            match k { ObjKind::Class => o.class = Some(ty.into()), ObjKind::Zone => o.zone = Some(ty.into()), ObjKind::Signal => o.signal = Some(ty.into()) }
            self.objects.push(o);
        }
        fn fact(&mut self, rel: &str, a: &str, b: &str) { self.facts.push((rel.into(), a.into(), b.into())); }
    }
}
