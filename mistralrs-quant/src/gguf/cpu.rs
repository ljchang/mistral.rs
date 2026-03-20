//! CPU/Metal implementation of indexed MoE forward for GGUF quantized weights.
//!
//! This dequantizes the weights and delegates to UnquantLinear's gather_forward.

use candle_core::{
    quantized::{QMatMul, QTensor},
    Result, Tensor,
};
use candle_nn::Linear;
use std::sync::Arc;

use crate::{QuantMethod, QuantMethodConfig, UnquantLinear};

/// Perform indexed MoE forward pass on a QTensor by dequantizing and using UnquantLinear.
///
/// # Arguments
/// * `qtensor` - The quantized weight tensor [num_experts, n, k]
/// * `x` - Input tensor [batch, topk_or_1, k]
/// * `ids` - Expert indices tensor [batch, topk]
///
/// # Returns
/// Output tensor [batch, topk, n]
pub fn qtensor_indexed_moe_forward(
    qtensor: &Arc<QTensor>,
    x: &Tensor,
    ids: &Tensor,
) -> Result<Tensor> {
    let device = x.device();

    // Dequantize the full expert tensor on CPU to avoid Metal buffer allocation
    // failure (e.g. [128, N, K] at f32 can exceed Metal's single buffer limit).
    let all_weights = qtensor.dequantize(&candle_core::Device::Cpu)?;
    let (_num_experts, out_features, _in_features) = all_weights.dims3()?;

    // Input is 3D: either [n, 1, h] (gate/up) or [n, k, h] (down, already per-expert)
    // ids is 2D: [n, k] where k = num_experts_per_tok
    let (num_tokens, num_experts_per_tok) = ids.dims2()?;
    let (_, x_mid, hidden_dim) = x.dims3()?;

    // Select only the active expert weights on CPU (e.g. 8 of 128 → tiny tensor)
    let flat_ids = ids.flatten_all()?.to_device(&candle_core::Device::Cpu)?;
    let selected_w = all_weights.index_select(&flat_ids, 0)?;
    drop(all_weights); // free the large CPU buffer

    // Move the small selected weights to the compute device (Metal)
    let selected_w = selected_w.to_device(device)?;

    // Expand input to [n*k, h]:
    //   [n, 1, h] → broadcast to [n, k, h] → reshape [n*k, h]
    //   [n, k, h] → reshape [n*k, h] directly
    let a_expanded = if x_mid == 1 {
        x.broadcast_as((num_tokens, num_experts_per_tok, hidden_dim))?
            .reshape((num_tokens * num_experts_per_tok, hidden_dim))?
    } else {
        x.reshape((num_tokens * num_experts_per_tok, hidden_dim))?
    };

    // Matmul on device: [n*k, 1, h] @ [n*k, h, out] → [n*k, out]
    let result = a_expanded
        .unsqueeze(1)?
        .matmul(&selected_w.transpose(1, 2)?)?
        .squeeze(1)?;

    result.reshape((num_tokens, num_experts_per_tok, out_features))
}

/// Perform indexed MoE forward pass on a QMatMul.
///
/// This is the main entry point for CPU/Metal GGUF quantized MoE forward.
///
/// # Arguments
/// * `qmatmul` - The quantized weight matrix
/// * `x` - Input tensor [batch, topk_or_1, k]
/// * `ids` - Expert indices tensor [batch, topk]
///
/// # Returns
/// Output tensor [batch, topk, n]
pub fn cpu_indexed_moe_forward(qmatmul: &QMatMul, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    match qmatmul {
        QMatMul::QTensor(qtensor) => qtensor_indexed_moe_forward(qtensor, x, ids),
        QMatMul::Tensor(t) | QMatMul::TensorF16(t) => {
            // For non-quantized tensors, use UnquantLinear directly
            let unquant =
                UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(t.clone(), None)))?;
            unquant.gather_forward(x, ids)
        }
    }
}
