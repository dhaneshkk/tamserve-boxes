//! YOLOX on the CPU through OpenVINO: one call for a batch of frames from any cameras.

use std::path::Path;

use openvino::{Core, DeviceType, Dimension, ElementType, InferRequest, PartialShape, Rank, RwPropertyKey, Shape,
               Tensor};

use crate::image::{letterbox, PAD};
use crate::{iou, Detect, Error, Finds, GrayFrame, Result, COCO};

const STRIDES: [usize; 3] = [8, 16, 32];

/// How a [`Detector`] runs. `Default` is YOLOX-Nano's: 416 x 416, COCO's classes.
#[derive(Clone, Debug)]
pub struct DetectorOptions {
    /// The side of the square YOLOX looks at: 416 for YOLOX-Nano, as it was trained.
    pub size: usize,
    /// CPU threads OpenVINO may use. Leave a core for decoding and tracking: 3 on a
    /// 4-core N100. `None` lets OpenVINO decide.
    pub threads: Option<usize>,
    /// Keep OpenVINO's threads on their cores, so decoders do not push them about.
    pub pin: bool,
    /// The confidence below which a find is ignored.
    pub threshold: f32,
    /// Finds of one class that overlap a more confident one by more than this go.
    pub overlap: f32,
    /// The class names to keep, such as `["person"]`; all of them when `None`.
    pub classes: Option<Vec<String>>,
    /// The model's class names, in order. COCO's for YOLOX's released models.
    pub names: Vec<String>,
    /// OpenVINO's inference precision, such as `"f32"`. By default OpenVINO picks:
    /// float32 on an N100, bfloat16 on CPUs that have it.
    pub precision: Option<String>,
    /// The OpenVINO device: `"CPU"`.
    pub device: String,
    /// Inference requests in flight at once. A look's frames are split between them, and
    /// each request letterboxes its share on a thread of its own and runs in an OpenVINO
    /// stream of its own, so several cores work on one look. 1 suits a light load; more
    /// keep up with more cameras. Splitting does not change any frame's finds.
    pub requests: usize,
}

impl Default for DetectorOptions {
    fn default() -> Self {
        DetectorOptions {
            size: 416,
            threads: None,
            pin: true,
            threshold: 0.1,
            overlap: 0.45,
            classes: None,
            names: COCO.iter().map(|s| s.to_string()).collect(),
            precision: None,
            device: "CPU".into(),
            requests: 1,
        }
    }
}

/// One inference request, with what it letterboxes into.
struct Lane {
    request: InferRequest,
    input: Option<(usize, Tensor)>,
    scratch: Vec<u8>,
}

/// What turns the model's raw output into finds: plain data, shared by the requests' threads.
struct Head {
    opts: DetectorOptions,
    keep: Option<Vec<usize>>,    // class indices kept, in the order of `classes`
    cells: Vec<(f32, f32, f32)>, // each anchor's cell x, cell y and stride, as YOLOX's head lists them
}

/// YOLOX-Nano (or any YOLOX) taking 1-channel frames, through OpenVINO.
pub struct Detector {
    head: Head,
    lanes: Vec<Lane>,
    _compiled: openvino::CompiledModel,
    _core: Core,
}

fn ov<E: std::fmt::Display>(e: E) -> Error {
    Error::OpenVino(e.to_string())
}

impl Detector {
    /// Load a 1-channel YOLOX model (ONNX, or OpenVINO IR `.xml` beside its `.bin`) and
    /// compile it for the CPU. OpenVINO's C library is found at run time: put its folder
    /// on `LD_LIBRARY_PATH` or set `OPENVINO_INSTALL_DIR`.
    pub fn new(model: impl AsRef<Path>, opts: DetectorOptions) -> Result<Self> {
        let model = model.as_ref();
        let keep = match &opts.classes {
            None => None,
            Some(cs) => Some(
                cs.iter()
                    .map(|c| opts.names.iter().position(|n| n == c).ok_or_else(|| Error::Invalid(format!("no class {c:?} in this model"))))
                    .collect::<Result<Vec<_>>>()?,
            ),
        };
        let mut core = Core::new().map_err(ov)?;
        let path = model.to_str().ok_or_else(|| Error::Invalid("the model's path is not UTF-8".into()))?;
        let weights = if path.ends_with(".xml") { path.trim_end_matches(".xml").to_string() + ".bin" } else { String::new() };
        let mut m = core.read_model_from_file(path, &weights).map_err(ov)?;
        let channels = m.get_input_by_index(0).map_err(ov)?.get_partial_shape().map_err(ov)?.get_dimensions()[1];
        if channels.get_min() != 1 || channels.get_max() != 1 {
            return Err(Error::Invalid("Detector takes a 1-channel model: make one with fold::gray_model \
                                       or tamserve.boxes.gray_model".into()));
        }
        let size = opts.size as i64;
        let shape = PartialShape::new_dynamic(Rank::new(4, 4), &[Dimension::new(1, 256), Dimension::new(1, 1),
                                                                 Dimension::new(size, size), Dimension::new(size, size)])
            .map_err(ov)?;
        m.reshape_single_input(&shape).map_err(ov)?;
        let device = DeviceType::from(opts.device.as_str()).to_owned();
        if opts.requests == 0 {
            return Err(Error::Invalid("requests must be at least 1".into()));
        }
        // One request: lowest latency. Several: one OpenVINO stream each, so they run at once.
        let hint = if opts.requests == 1 { "LATENCY" } else { "THROUGHPUT" };
        let mut props = vec![(RwPropertyKey::HintPerformanceMode, hint.to_string()),
                             (RwPropertyKey::NumStreams, opts.requests.to_string())];
        if opts.device == "CPU" {
            props.push((RwPropertyKey::HintEnableCpuPinning, if opts.pin { "YES" } else { "NO" }.to_string()));
            if let Some(t) = opts.threads {
                props.push((RwPropertyKey::InferenceNumThreads, t.to_string()));
            }
        }
        if let Some(p) = &opts.precision {
            props.push((RwPropertyKey::HintInferencePrecision, p.clone()));
        }
        for (k, v) in &props {
            core.set_property(&device, k, v).map_err(ov)?;
        }
        let mut compiled = core.compile_model(&m, device).map_err(ov)?;
        let lanes = (0..opts.requests)
            .map(|_| Ok(Lane { request: compiled.create_infer_request().map_err(ov)?, input: None, scratch: Vec::new() }))
            .collect::<Result<Vec<_>>>()?;
        let mut cells = Vec::new();
        for s in STRIDES {
            let n = opts.size / s;
            for y in 0..n {
                for x in 0..n {
                    cells.push((x as f32, y as f32, s as f32));
                }
            }
        }
        Ok(Detector { head: Head { opts, keep, cells }, lanes, _compiled: compiled, _core: core })
    }

}

impl Head {
    /// One frame's raw head output (anchors x (5 + classes)) as finds.
    fn decode(&self, raw: &[f32], scale: f32, width: usize, height: usize) -> Finds {
        let row = 5 + self.opts.names.len();
        let mut cand: Vec<([f32; 4], f32, usize)> = Vec::new();
        for (a, r) in raw.chunks_exact(row).enumerate() {
            let cls = &r[5..];
            let (best, p) = match &self.keep {
                None => cls.iter().enumerate().fold((0, f32::MIN), |m, (i, &v)| if v > m.1 { (i, v) } else { m }),
                Some(keep) => keep.iter().map(|&i| (i, cls[i])).fold((0, f32::MIN), |m, x| if x.1 > m.1 { x } else { m }),
            };
            let score = r[4] * p;
            if score <= self.opts.threshold {
                continue;
            }
            let (cx, cy, s) = self.cells[a];
            let (x, y) = ((r[0] + cx) * s, (r[1] + cy) * s);
            let (w, h) = (r[2].min(10.0).exp() * s, r[3].min(10.0).exp() * s);
            let clip = |v: f32, hi: usize| v.clamp(0.0, hi as f32);
            cand.push(([clip((x - w / 2.0) / scale, width), clip((y - h / 2.0) / scale, height),
                        clip((x + w / 2.0) / scale, width), clip((y + h / 2.0) / scale, height)], score, best));
        }
        // Greedy non-maximum suppression, the most confident first, within each class.
        cand.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut out = Finds::default();
        let mut gone = vec![false; cand.len()];
        for i in 0..cand.len() {
            if gone[i] {
                continue;
            }
            for j in i + 1..cand.len() {
                if !gone[j] && cand[j].2 == cand[i].2 && iou(&cand[i].0, &cand[j].0) > self.opts.overlap {
                    gone[j] = true;
                }
            }
            out.boxes.push(cand[i].0);
            out.scores.push(cand[i].1);
            out.labels.push(cand[i].2);
        }
        out
    }
}

impl Detect for Detector {
    fn names(&self) -> &[String] {
        &self.head.opts.names
    }

    fn detect(&mut self, frames: &[GrayFrame]) -> Result<Vec<Finds>> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        // Split the frames between the requests, in order, as evenly as they go.
        let n = frames.len();
        let k = self.lanes.len().min(n);
        let mut chunks = Vec::with_capacity(k);
        let mut start = 0;
        for i in 0..k {
            let len = n / k + usize::from(i < n % k);
            chunks.push(&frames[start..start + len]);
            start += len;
        }
        let head = &self.head;
        let size = head.opts.size;
        let work = |lane: &mut Lane, chunk: &[GrayFrame]| -> Result<Vec<Finds>> {
            let (raw, per, scales) = run_lane(lane, chunk, size)?;
            Ok(chunk.iter().enumerate()
                .map(|(i, f)| head.decode(&raw[i * per..(i + 1) * per], scales[i], f.width, f.height))
                .collect())
        };
        let results: Vec<Result<Vec<Finds>>> = if k == 1 {
            vec![work(&mut self.lanes[0], chunks[0])]
        } else {
            std::thread::scope(|s| {
                let handles: Vec<_> = self.lanes.iter_mut().zip(&chunks)
                    .map(|(lane, chunk)| s.spawn(move || work(lane, chunk)))
                    .collect();
                handles.into_iter()
                    .map(|h| h.join().unwrap_or_else(|_| Err(Error::OpenVino("a request's thread panicked".into()))))
                    .collect()
            })
        };
        let mut out = Vec::with_capacity(n);
        for r in results {
            out.extend(r?);
        }
        Ok(out)
    }
}

/// Letterbox `frames` into a lane's input and run its request: (raw output, floats per frame, scales).
fn run_lane(lane: &mut Lane, frames: &[GrayFrame], s: usize) -> Result<(Vec<f32>, usize, Vec<f32>)> {
    let n = frames.len();
    if lane.input.as_ref().map(|(k, _)| *k) != Some(n) {
        // Not padded to fixed sizes: OpenVINO's CPU plugin takes any batch, and padding is wasted work.
        let t = Tensor::new(ElementType::F32, &Shape::new(&[n as i64, 1, s as i64, s as i64]).map_err(ov)?).map_err(ov)?;
        lane.input = Some((n, t));
    }
    let (_, tensor) = lane.input.as_mut().unwrap();
    let data = tensor.get_data_mut::<f32>().map_err(ov)?;
    data.fill(PAD);
    let mut scales = Vec::with_capacity(n);
    for (i, f) in frames.iter().enumerate() {
        scales.push(letterbox(f, s, &mut data[i * s * s..(i + 1) * s * s], &mut lane.scratch));
    }
    lane.request.set_input_tensor(tensor).map_err(ov)?;
    lane.request.infer().map_err(ov)?;
    let out = lane.request.get_output_tensor().map_err(ov)?;
    let raw = out.get_data::<f32>().map_err(ov)?.to_vec();
    let per = raw.len() / n;
    Ok((raw, per, scales))
}
