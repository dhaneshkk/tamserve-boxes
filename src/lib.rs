//! Find and follow boxes on many cameras with YOLOX-Nano on gray frames, on the CPU
//! through OpenVINO: the Rust port of `tamserve.boxes`.
//!
//! It serves the way tamserve does. One copy of the model serves every camera, and the
//! cameras that look on a given round go through it in one batch. YOLOX looks at each
//! camera every `look_every` frames, the looks are spread so that they do not all land
//! in the same round, and between looks a Kalman filter per object carries each box
//! forward. Objects are matched to what a look finds as ByteTrack matches them.
//!
//! The model takes one channel: frames go in as the gray picture itself (a camera's
//! gray stream, or the Y plane of its colour stream). [`fold::gray_model`] (feature
//! `fold`) turns YOLOX's own 3-channel ONNX file into one, exactly: on a picture whose
//! three channels are the same gray image `g`, the first convolution gives
//! `Σ_c W_c * g = (Σ_c W_c) * g`.
//!
//! The behaviour is the Python version's, step for step: the same matching, the same
//! scheduling of looks, the same letterbox and decoding, so the two give the same ids
//! on the same frames.
//!
//! ```no_run
//! use tamserve_boxes::{BoxTracker, Detector, DetectorOptions, GrayFrame, TrackerOptions};
//!
//! let detector = Detector::new("yolox_nano_gray.onnx", DetectorOptions {
//!     threads: Some(3), classes: Some(vec!["person".into()]), ..Default::default() })?;
//! let mut tracker = BoxTracker::new(detector, TrackerOptions::default())?;
//! let pixels = vec![0u8; 1280 * 720];
//! let found = tracker.step(&[("gate", GrayFrame::new(&pixels, 1280, 720)?)])?;
//! for (id, f) in &found["gate"] {
//!     println!("{id} {} {:.2} {:?}", f.label, f.score, f.bbox);
//! }
//! # Ok::<(), tamserve_boxes::Error>(())
//! ```

mod detector;
#[cfg(feature = "fold")]
pub mod fold;
pub mod image;
mod kalman;
mod tracker;

pub use detector::{Detector, DetectorOptions};
pub use tracker::{BoxTracker, LookEvery, TrackerOptions};

/// The classes of YOLOX's released models, in order (COCO's).
pub const COCO: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
];

/// A box as (left, top, right, bottom), in the frame's own pixels.
pub type BBox = [f32; 4];

/// What went wrong.
#[derive(Debug)]
pub enum Error {
    /// A setting or a frame that cannot be used.
    Invalid(String),
    /// OpenVINO could not be found, or failed.
    OpenVino(String),
    /// A file could not be read or written.
    Io(std::io::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Invalid(s) => write!(f, "{s}"),
            Error::OpenVino(s) => write!(f, "OpenVINO: {s}"),
            Error::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// One frame, borrowed: H x W gray pixels, row by row, as a decoder's Y plane is.
#[derive(Clone, Copy, Debug)]
pub struct GrayFrame<'a> {
    pub pixels: &'a [u8],
    pub width: usize,
    pub height: usize,
    /// Bytes from the start of one row to the next: `width`, unless the rows are padded.
    pub stride: usize,
}

impl<'a> GrayFrame<'a> {
    /// A frame whose rows are packed one after another.
    pub fn new(pixels: &'a [u8], width: usize, height: usize) -> Result<Self> {
        Self::with_stride(pixels, width, height, width)
    }

    /// A frame whose rows start `stride` bytes apart, as many decoders pad them.
    pub fn with_stride(pixels: &'a [u8], width: usize, height: usize, stride: usize) -> Result<Self> {
        if width == 0 || height == 0 || stride < width || pixels.len() < stride * (height - 1) + width {
            return Err(Error::Invalid(format!(
                "a {width} x {height} frame with rows {stride} bytes apart needs more than {} bytes",
                pixels.len()
            )));
        }
        Ok(GrayFrame { pixels, width, height, stride })
    }

    #[inline]
    pub(crate) fn row(&self, y: usize) -> &'a [u8] {
        &self.pixels[y * self.stride..y * self.stride + self.width]
    }
}

/// What one look found on one frame, the most confident first.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Finds {
    pub boxes: Vec<BBox>,
    pub scores: Vec<f32>,
    /// Each find's class, as an index into the detector's `names()`.
    pub labels: Vec<usize>,
}

impl Finds {
    pub fn len(&self) -> usize {
        self.boxes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.boxes.is_empty()
    }
}

/// One followed object on one frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Found {
    pub bbox: BBox,
    /// The confidence of the latest find that matched it.
    pub score: f32,
    pub label: String,
}

/// What finds objects on frames. [`Detector`] is YOLOX through OpenVINO; anything else
/// that gives the same answers will do, such as a stand-in for tests.
pub trait Detect {
    /// The class names, in the order `Finds::labels` indexes them.
    fn names(&self) -> &[String];
    /// One call for a batch of frames from any cameras: what each one shows.
    fn detect(&mut self, frames: &[GrayFrame]) -> Result<Vec<Finds>>;
}

/// The overlap (IoU) of two boxes.
#[inline]
pub(crate) fn iou(a: &BBox, b: &BBox) -> f32 {
    let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = w * h;
    let area = |x: &BBox| (x[2] - x[0]).max(0.0) * (x[3] - x[1]).max(0.0);
    inter / (area(a) + area(b) - inter).max(1e-9)
}
