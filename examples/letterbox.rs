//! Write a PGM's letterbox (size x size, as YOLOX takes it) as raw bytes: to compare with OpenCV.
use tamserve_boxes::{image, GrayFrame};
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).unwrap();
    let head = String::from_utf8_lossy(&data[..20]).to_string();
    let f: Vec<&str> = head.split_whitespace().take(4).collect();
    let (w, h): (usize, usize) = (f[1].parse().unwrap(), f[2].parse().unwrap());
    let px = &data[data.len() - w * h..];
    let mut out = vec![image::PAD; 416 * 416];
    let mut scratch = Vec::new();
    image::letterbox(&GrayFrame::new(px, w, h).unwrap(), 416, &mut out, &mut scratch);
    std::fs::write(&a[2], out.iter().map(|&v| v as u8).collect::<Vec<u8>>()).unwrap();
}
