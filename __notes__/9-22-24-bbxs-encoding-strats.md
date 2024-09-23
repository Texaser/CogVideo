# **Setting**

### Model

| Model | LoRA? |
| :---: | :---: |
| CogVideoX-5B I2V | ✅ |

### Data

| # Samples | Size [H, W] | FPS | # Frames | Skip-Frame Interval | 
| :---: | :---: | :---: | :---: | :---: |
| 4000 | [480, 720] | 8 | 49 | 3 |

### Training Parameters

##### Global

| # Steps | Batch Size |  Precision |
| :---: | :---: | :---: |
| 4000 | 1 | `bf16` |

##### Optimizer

| Optimizer | LR (Initial) | LR-Scheduler | betas | eps | Weight Decay |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `FusedEmaAdam` | 5e-4 | --- | `[0.9, 0.95]` | 1e-8 | 1e-4 |

### Conditions

| Text | First Frame | Bbox |
| :---: | :---: | :---: |
| ✅ | ✅ | ✅ |

---

# **Experiments**

| Bbox Encoding Method | Draw Bbox on Frame?| Subject Consistency | mIoU | Imaging Quality | FVD-VAE | FVD-Classifier |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| `None` | ❌ | --- | --- | --- | --- | --- |
| Uniform Noise Masks | ❌| --- | --- | --- | --- |  --- |
| Uniform Noise Masks | ✅  | --- | --- | --- | --- |  --- |
| Player-Specific Constant Masks | ❌ | --- | --- | --- | --- |  --- |
| Player-Specific Constant Masks | ✅ | --- | --- | --- | --- |  --- |
