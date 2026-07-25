---
title: "GPU Kernel Performance Benchmarks by NVIDIA"
source: "https://research.nvidia.com/benchmarks/sol-execbench/blog/submission-guide#collection-submissions"
author:
published:
created: 2026-07-16
description: "Benchmark your GPU kernels on real NVIDIA B200 hardware. Submit optimized CUDA or PyTorch code, get your SOL Score, and compete on the global leaderboard."
tags:
  - "clippings"
---
## Quick Start

The fastest way to make your first submission is to use the reference implementation directly:

1. Pick a kernel on the website.
2. Copy its reference implementation from the kernel page.
3. Upload it as a `.py` file.

The reference is a known-correct solution. It will score SOL Score 0.5 (matching the baseline). From there, optimize and resubmit.

## Kernel Submissions

Each submission targets a single kernel and a single GPU type. You can submit as **Public** (eligible for public leaderboards) or **Private** (visible only to you).

### Overview

The evaluator accepts three categories of submission: Python source files (`.py`), host-entry C++ files (`.cpp`, `.cc`, `.cxx`, `.c`), and JSON solution files (`.json`). CUDA `.cu` implementation files are supported inside multi-file archives (`.zip`, `.tar.gz`, `.tgz`) or JSON solutions.

For all formats, the evaluator calls your `run()` function positionally, using the input order from the kernel definition. The contract is argument order and count — argument names are not enforced.

- **Inputs** are materialized as `torch.Tensor` objects on a single CUDA device. Scalar inputs may also appear when the kernel definition includes scalar parameters. Shapes and dtypes follow the kernel definition for the current workload (a workload is one concrete input configuration drawn from the kernel's parameter space).
- **Outputs** must match the kernel definition in shape and dtype exactly, and must satisfy the workload tolerance checks (numerical error, output sanity). Outputs must be concrete `torch.Tensor` objects — the evaluator rejects tensor subclasses and lazy proxy objects.

### Python (.py)

Upload a single `.py` file with a top-level `run()` function. The evaluator auto-detects the language and calling convention. Supported languages: PyTorch, `torch.compile`, Triton, CuTe DSL, cuTile, cuDNN Frontend.

```python
import torch

def run(hidden_states, weight):
    EPS = 1e-5
    x = hidden_states.to(torch.float32)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    y = (x * inv_rms) * weight.to(torch.float32)
    return y.to(hidden_states.dtype)
```

If your solution requires multiple files, package them as a `.zip`, `.tar.gz`, or `.tgz` archive. The entry point must be `submission.py` with a top-level `run()`. Additional `.py` files can be imported from `submission.py`.

### C++/CUDA (.cpp entry point with optional.cu sources)

The `cuda_cpp` entry point must be a host-compiled `.cpp`, `.cc`, `.cxx`, or `.c` file containing the Torch/PyBind binding. A single `.cpp` file is sufficient for host-only Torch C++ code. Custom CUDA kernels should be submitted as an archive or JSON solution containing a `.cpp` entry point plus one or more `.cu` implementation files.

For example, package these files together:

```
solution.zip
├── binding.cpp
└── kernel.cu
```

`binding.cpp`:

```cpp
#include <torch/extension.h>

void launch_add(torch::Tensor A, torch::Tensor B, torch::Tensor C);

void run(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    launch_add(A, B, C);
}

PYBIND11_MODULE(benchmark_kernel, m) {
    m.def("run", &run);
}
```

`kernel.cu` contains the CUDA kernel and the `launch_add` implementation. The evaluator locates the archive entry point by looking for `PYBIND11_MODULE` in a host source file.

### JSON Solution (.json)

When you upload a `.py` or `.cpp` file, the evaluator infers the language, generates the build specification, and auto-detects the calling convention. This works well for simple cases, but limits your control over the compilation process. A `.json` submission follows the [solution schema](https://github.com/NVIDIA/SOL-ExecBench/blob/main/docs/solution.md) and removes these limitations — you can:

- **Declare the language explicitly** rather than relying on file-extension inference (e.g., distinguish `"triton"` from `"pytorch"` when both use `.py`).
- **Set compiler flags** such as `-O3`, `--use_fast_math`, or `--expt-relaxed-constexpr` via `compile_options`, which are additive to the evaluator's defaults.
- **Declare dependencies** like CUTLASS header paths or specific Triton versions, ensuring the build environment is configured correctly.
- **Bundle multiple source files** with explicit entry points, without relying on the evaluator's heuristic for locating the entry in archives.
- **Control destination passing style** explicitly, rather than depending on auto-detection.

This makes `.json` well-suited for C++/CUDA submissions that need precise build control, and for programmatic and agent-driven workflows.

The two required top-level fields are `spec` and `sources`. Fields like `name`, `definition`, `author`, and `target_hardware` are optional — the evaluation service sets them based on the submission context.

**`spec`** — build specification:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `languages` | array\[string\] | Yes | `"pytorch"`, `"triton"`, `"cuda_cpp"`, `"cutlass"`, `"cute_dsl"`, `"cutile"`, `"cudnn_frontend"`, `"cudnn"`, `"cublas"` |
| `entry_point` | string | Yes | `"filename::function_name"` (e.g., `"kernel.py::run"`, `"binding.cpp::run"`) |
| `destination_passing_style` | bool | No | Default `true`: the evaluator pre-allocates output tensors and passes them as the last positional arguments; your function writes in-place and returns nothing. Set to `false` for value-returning style. For single-file `.py` submissions this is auto-detected. |
| `dependencies` | array\[string\] | No | `"torch"`, `"triton >= 2.3"`, `"CUTLASS_3_7"`, `"cutlass"`, `"cublas"`, `"cudnn"` |
| `compile_options` | object | No | Extra compiler flags for C++/CUDA (additive to defaults): `{"cflags": [...], "cuda_cflags": [...], "ld_flags": [...]}` |

**`sources`** — array of `{path, content}` objects. The evaluator writes these files to the working directory. The file referenced by `spec.entry_point` must appear in `sources`. Paths must be relative (no leading `/` or `..`), with no duplicates.

**Python example:**

```json
{
  "spec": {
    "languages": ["pytorch"],
    "entry_point": "submission.py::run",
    "destination_passing_style": false,
    "dependencies": ["torch"]
  },
  "sources": [
    {
      "path": "submission.py",
      "content": "import torch\n\ndef run(hidden_states, weight):\n    EPS = 1e-5\n    x = hidden_states.to(torch.float32)\n    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)\n    return ((x * inv_rms) * weight.to(torch.float32)).to(hidden_states.dtype)"
    }
  ]
}
```

**Triton example** — with dependency declaration:

```json
{
  "spec": {
    "languages": ["triton"],
    "entry_point": "kernel.py::run",
    "destination_passing_style": true,
    "dependencies": ["torch", "triton >= 2.3"]
  },
  "sources": [
    {
      "path": "kernel.py",
      "content": "import torch\nimport triton\nimport triton.language as tl\n\n@triton.jit\ndef _rmsnorm_kernel(X, W, Y, stride, n_cols, eps, BLOCK: tl.constexpr):\n    row = tl.program_id(0)\n    cols = tl.arange(0, BLOCK)\n    mask = cols < n_cols\n    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)\n    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)\n    rms = tl.sqrt(tl.sum(x * x) / n_cols + eps)\n    tl.store(Y + row * stride + cols, (x / rms * w).to(Y.dtype.element_ty), mask=mask)\n\ndef run(hidden_states, weight, output):\n    n_rows, n_cols = hidden_states.shape\n    BLOCK = triton.next_power_of_2(n_cols)\n    _rmsnorm_kernel[(n_rows,)](hidden_states, weight, output, hidden_states.stride(0), n_cols, 1e-5, BLOCK=BLOCK)"
    }
  ]
}
```

**CUDA C++ example** — multi-file with compiler flags:

```json
{
  "spec": {
    "languages": ["cuda_cpp"],
    "entry_point": "binding.cpp::run",
    "destination_passing_style": true,
    "compile_options": {
      "cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]
    }
  },
  "sources": [
    {
      "path": "binding.cpp",
      "content": "#include <torch/extension.h>\n\nvoid launch_add(torch::Tensor A, torch::Tensor B, torch::Tensor C);\n\nvoid run(torch::Tensor A, torch::Tensor B, torch::Tensor C) { launch_add(A, B, C); }\n\nPYBIND11_MODULE(benchmark_kernel, m) { m.def(\"run\", &run); }"
    },
    {
      "path": "kernel.cu",
      "content": "#include <torch/extension.h>\n\n__global__ void add_kernel(const float* A, const float* B, float* C, int N) {\n    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n    if (idx < N) C[idx] = A[idx] + B[idx];\n}\n\nvoid launch_add(torch::Tensor A, torch::Tensor B, torch::Tensor C) {\n    int N = A.numel();\n    add_kernel<<<(N + 255) / 256, 256>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);\n}"
    }
  ]
}
```

For the complete specification, see the [solution schema documentation](https://github.com/NVIDIA/SOL-ExecBench/blob/main/docs/solution.md). Complete examples for each supported language are available on GitHub: [pytorch](https://github.com/NVIDIA/SOL-ExecBench/tree/main/examples/pytorch), [triton](https://github.com/NVIDIA/SOL-ExecBench/tree/main/examples/triton), [cuda\_cpp](https://github.com/NVIDIA/SOL-ExecBench/tree/main/examples/cuda_cpp), [cutlass](https://github.com/NVIDIA/SOL-ExecBench/tree/main/examples/cutlass), [cute\_dsl](https://github.com/NVIDIA/SOL-ExecBench/tree/main/examples/cute_dsl), [cutile](https://github.com/NVIDIA/SOL-ExecBench/tree/main/examples/cutile), [cudnn](https://github.com/NVIDIA/SOL-ExecBench/tree/main/examples/cudnn).

## Collection Submissions

A collection groups related kernels (e.g., L1 — Single Operations, L2 — Fused Operations, Quant — Quantized kernels). A collection submission evaluates one archive across every kernel in the collection. For each matched kernel directory, the evaluator extracts `submission.py` and launches a separate evaluation (a "child submission").

Example for the L1 collection:

```
l1-submission.zip
├── 001_attention_softmax_dropout_value_matmul_backward/
│   └── submission.py
├── 002_vae_conv3x3_groupnorm_silu_residual_fused/
│   └── submission.py
├── 003_lm_head_projection_with_logit_slicing/
│   └── submission.py
└── ...
```

Each `submission.py` must contain a top-level `run()` function, just like a single-kernel Python submission. Collection submissions only support the `.py` format — `.cu`, `.json`, and other formats are not supported in collection archives.

**Rules:**

- Directory names must exactly match kernel names in the collection (visible on the collection page).
- Each matched directory must contain `submission.py` with a top-level `run()`.
- Extra files and folders at the root are ignored.
- Missing kernel directories are skipped and contribute SOL Score 0 to the collection mean.
- Archive limits: 10 MB upload, 50 MB archive size, 100 MB extracted size. Symlinks, path traversal, and null bytes are rejected.

You can develop and test solutions on individual kernels first, then package them into a collection archive.

## Evaluation Environment

Submissions are evaluated on real NVIDIA B200 GPUs inside isolated containers. The evaluation harness enforces the following controls for reproducible timing:

- **Fixed GPU clocks** — GPU and DRAM frequencies are locked to eliminate dynamic frequency scaling.
- **Cold L2 cache** — The L2 cache is flushed before every timed iteration.
- **Input isolation** — Tensor arguments are cloned before each iteration so that in-place modifications do not carry over.
- **Timing protocol** — 10 warmup iterations, 50 timed iterations per trial, 3 trials. The reported runtime is the mean across trials.
- **Process isolation** — Each solution runs in a dedicated subprocess with a hard time limit.

### Prohibited Behavior

The harness detects and rejects submissions that attempt to manipulate timing or hide work:

- Monkey-patching timing functions
- Launching hidden CUDA streams outside the measured path
- Spawning background threads during evaluation
- Returning lazy proxy outputs instead of concrete tensors
- Producing degenerate outputs (`NaN`, `inf`, all-zeros) that don't match the reference

Any workload that fails correctness or is rejected receives SOL Score 0.

## Scoring

Each submission is scored using the **SOL Score**, which measures how close your kernel gets to the hardware's theoretical Speed of Light:

$$
S(t) = \frac{t_b - t_{\text{sol}}}{(t - t_{\text{sol}}) + (t_b - t_{\text{sol}})}
$$

Where $t$ is the measured submission latency, $t_b$ is the precomputed baseline latency (from a PyTorch reference implementation), and $t_{\text{sol}}$ is the precomputed SOL latency derived from hardware analysis.

| SOL Score | Meaning |
| --- | --- |
| 0 | Incorrect result or failed execution |
| < 0.5 | Slower than the baseline |
| 0.5 | Matches the baseline |
| \> 0.5 | Outperforms the baseline |
| → 1.0 | Approaches the hardware Speed of Light |

SOL Score uses precomputed latency tables attached to each kernel, not live reference measurements during your submission run. For a single kernel, the reported score is the arithmetic mean across workloads. For a collection, it is the arithmetic mean across kernels (missing or failed kernels contribute 0).

## Submission Policy

Every submission runs on real GPU hardware. To keep the queue fair and stable, SOL-ExecBench applies rate limits. All daily counters reset at UTC midnight. Child submissions spawned from a collection do not count toward the standalone kernel delay. Submissions that end in immediate system failure are revealed without delay and excluded from future delay calculation.

### Standalone Kernel Submissions Policy

After your first 5 submissions of the day, a result-pending delay is applied — results are computed immediately but held before being revealed to you.

| Submission # (today) | Pending Delay |
| --- | --- |
| 1–5 | None |
| 6 | 10 minutes |
| 7 | 20 minutes |
| 8 | 30 minutes |
| ... | +10 min each |
| 11+ | 60 minutes |

You may have at most 5 submissions in flight (queued, running, or pending reveal) at once. There is no daily cap on total submissions.

### Collection Submissions Policy

- Results have a fixed 6-hour result-pending delay.
- Each user may submit to the same collection at most 2 times per UTC day.
- An archive that fails validation before evaluation does not count against the daily limit.

## Further Resources

- [SOL-ExecBench GitHub repository](https://github.com/NVIDIA/SOL-ExecBench) — example submissions and local evaluation tooling
- [Definition schema](https://github.com/NVIDIA/SOL-ExecBench/blob/main/docs/definition.md) — kernel specification format
- [Workload schema](https://github.com/NVIDIA/SOL-ExecBench/blob/main/docs/workload.md) — input configuration format
- [Solution schema](https://github.com/NVIDIA/SOL-ExecBench/blob/main/docs/solution.md) — full JSON solution specification