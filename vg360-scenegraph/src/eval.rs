//! Matthews correlation coefficient of emitted facts against labelled facts, per relation.
//! Feed it frames where a person has labelled which (relation, a, b) truly hold, over the same
//! candidate pairs the builder considered; MCC is reported because the classes are very unbalanced
//! (most pairs are not near, most people are not under a load).

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Confusion { pub tp: u64, pub fp: u64, pub tn: u64, pub fn_: u64 }

impl Confusion {
    pub fn add(&mut self, predicted: bool, truth: bool) {
        match (predicted, truth) { (true, true) => self.tp += 1, (true, false) => self.fp += 1, (false, false) => self.tn += 1, (false, true) => self.fn_ += 1 }
    }
    /// MCC in [-1, 1]; 0 when any marginal is empty (the usual convention).
    pub fn mcc(&self) -> f64 {
        let (tp, fp, tn, fn_) = (self.tp as f64, self.fp as f64, self.tn as f64, self.fn_ as f64);
        let den = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
        if den == 0.0 { 0.0 } else { (tp * tn - fp * fn_) / den }
    }
    pub fn recall(&self) -> f64 { let d = self.tp + self.fn_; if d == 0 { 1.0 } else { self.tp as f64 / d as f64 } }
    pub fn precision(&self) -> f64 { let d = self.tp + self.fp; if d == 0 { 1.0 } else { self.tp as f64 / d as f64 } }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mcc_known_values() {
        let c = Confusion { tp: 90, fp: 10, tn: 880, fn_: 20 };
        let want = (90.0 * 880.0 - 10.0 * 20.0) / ((100.0f64) * 110.0 * 890.0 * 900.0).sqrt();
        assert!((c.mcc() - want).abs() < 1e-12);
        assert_eq!(Confusion { tp: 5, fp: 0, tn: 5, fn_: 0 }.mcc(), 1.0);
        assert_eq!(Confusion { tp: 0, fp: 5, tn: 0, fn_: 5 }.mcc(), -1.0);
        assert_eq!(Confusion { tp: 3, ..Default::default() }.mcc(), 0.0);
    }
}
