//! YOLOX's ONNX file (3 channels, batch 1) as one that takes 1 channel and any batch.
//! Feature `fold`; the result runs with [`Detector`](crate::Detector) and OpenVINO alone.
//!
//! The first convolution's weights are summed over the colour channels. On a gray
//! picture, whose three channels hold the same image `g`, that convolution computes
//! `Σ_c W_c * g = (Σ_c W_c) * g`, so the sum changes nothing it computes. YOLOX starts
//! with a Focus layer, which cuts the picture into 4 interleaved patches and stacks
//! them, 3 channels each: its first convolution reads 12 channels, summed in groups of
//! 3 to 4. Reshapes that name batch 1 are made to keep the input's batch.

use std::path::Path;

use onnx_protobuf::{tensor_proto::DataType, type_proto, Message, ModelProto, TensorProto};

use crate::{Error, Result};

fn bad(s: impl Into<String>) -> Error {
    Error::Invalid(s.into())
}

fn floats(t: &TensorProto) -> Result<Vec<f32>> {
    if t.data_type != DataType::FLOAT as i32 {
        return Err(bad(format!("{} is not float32", t.name)));
    }
    if !t.raw_data.is_empty() {
        Ok(t.raw_data.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    } else {
        Ok(t.float_data.clone())
    }
}

fn int64s(t: &TensorProto) -> Option<Vec<i64>> {
    if t.data_type != DataType::INT64 as i32 {
        return None;
    }
    Some(if !t.raw_data.is_empty() {
        t.raw_data.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
    } else {
        t.int64_data.clone()
    })
}

fn set_int64s(t: &mut TensorProto, v: &[i64]) {
    t.int64_data.clear();
    t.raw_data = v.iter().flat_map(|x| x.to_le_bytes()).collect();
}

/// The dimensions of a value's tensor type.
fn dims(v: &mut onnx_protobuf::ValueInfoProto) -> Result<&mut Vec<onnx_protobuf::tensor_shape_proto::Dimension>> {
    let ty = v.type_.as_mut().ok_or_else(|| bad(format!("{} has no type", v.name)))?;
    match &mut ty.value {
        Some(type_proto::Value::TensorType(t)) => Ok(&mut t.shape.mut_or_insert_default().dim),
        _ => Err(bad(format!("{} is not a tensor", v.name))),
    }
}

/// Write the 1-channel, any-batch version of the YOLOX ONNX file at `src` to `dst`.
pub fn gray_model(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> Result<()> {
    let bytes = std::fs::read(src)?;
    let mut m = ModelProto::parse_from_bytes(&bytes).map_err(|e| bad(format!("not an ONNX model: {e}")))?;
    let g = m.graph.as_mut().ok_or_else(|| bad("the model has no graph"))?;
    let inits: std::collections::HashSet<String> = g.initializer.iter().map(|t| t.name.clone()).collect();
    let image = g.input.iter_mut().find(|i| !inits.contains(&i.name)).ok_or_else(|| bad("the model has no input"))?;
    let d = dims(image)?;
    if d.len() != 4 || d[1].dim_value() != 3 {
        return Err(bad("expected a 3-channel YOLOX model (N x 3 x H x W input)"));
    }
    d[0].set_dim_param("batch".into());
    d[1].set_dim_value(1);
    for out in g.output.iter_mut() {
        if let Some(first) = dims(out)?.first_mut() {
            first.set_dim_param("batch".into());
        }
    }

    let conv = g.node.iter().find(|n| n.op_type == "Conv").ok_or_else(|| bad("the model has no convolution"))?;
    let wname = conv.input.get(1).cloned().ok_or_else(|| bad("the first convolution has no weights"))?;
    let w = g.initializer.iter_mut().find(|t| t.name == wname).ok_or_else(|| bad("its weights are not stored in the model"))?;
    let shape = w.dims.clone();
    if shape.len() != 4 || shape[1] % 3 != 0 {
        return Err(bad(format!("the first convolution reads {} channels, not a multiple of 3", shape.get(1).unwrap_or(&0))));
    }
    let (out_c, in_c, k) = (shape[0] as usize, shape[1] as usize, (shape[2] * shape[3]) as usize);
    let v = floats(w)?;
    let mut folded = vec![0f32; out_c * in_c / 3 * k];
    for o in 0..out_c {
        for g3 in 0..in_c / 3 {
            for i in 0..k {
                // summed in float64, then rounded once, as tamserve.boxes.gray_model does
                let s: f64 = (0..3).map(|c| v[(o * in_c + g3 * 3 + c) * k + i] as f64).sum();
                folded[(o * in_c / 3 + g3) * k + i] = s as f32;
            }
        }
    }
    w.dims[1] = (in_c / 3) as i64;
    w.float_data.clear();
    w.raw_data = folded.iter().flat_map(|x| x.to_le_bytes()).collect();

    let reshapes: Vec<String> = g.node.iter().filter(|n| n.op_type == "Reshape").filter_map(|n| n.input.get(1).cloned()).collect();
    for t in g.initializer.iter_mut().filter(|t| reshapes.contains(&t.name)) {
        if let Some(mut v) = int64s(t) {
            if v.first() == Some(&1) {
                v[0] = 0; // ONNX: 0 copies the input's own size, here its batch
                set_int64s(t, &v);
            }
        }
    }
    g.value_info.clear(); // shapes inferred for batch 1 no longer hold
    std::fs::write(dst, m.write_to_bytes().map_err(|e| bad(format!("could not write the model: {e}")))?)?;
    Ok(())
}
