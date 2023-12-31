import torch

import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
                 # NOTE: `constexpr` so it can be used as a shape value.
):
    pid = tl.program_id(axis=0)  
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y

    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)

def swish_torch(x: torch.Tensor):
    return x * torch.sigmoid(x)

@torch.jit.script
def swish_torch_opt(x: torch.Tensor):
    return x * torch.sigmoid(x)

def swish_torch_builtin(x: torch.Tensor):
    return torch.nn.SiLU()(x)

def add_triton(x: torch.Tensor, y: torch.Tensor, block):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=block) 
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[
            2 ** 26
        ],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='add performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add_triton(x, y, 2**10))

    gbps = lambda ms: 12 * size / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[
            2**26
        ],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='blocksize',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=[2**i for i in range(5,16)],  # Possible values for `line_arg`.
        line_names=[str(2**i) for i in range(5,16)],  # Label name for the lines.
        #styles=[('blue', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='add performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark_blocksize(size, blocksize):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: add_triton(x, y, blocksize))

    gbps = lambda ms: 12 * size / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


import mlflow
import pandas

def main():
    mlflow.start_run()

    # Options for running this example
    # 1. Uncomment either of these benchmarks 
    with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
        benchmark.run(show_plots=False, print_data=True)
        #benchmark_blocksize.run(show_plots=False, print_data=True)

    # 2. Uncomment this: for just calling the kernel; no benchmarking
    #x = torch.rand(size, device='cuda', dtype=torch.float32)
    #y = torch.rand(size, device='cuda', dtype=torch.float32)
    #triton_res = add_triton(x, y, 2**10)
    #torch_res = x + y
    #assert torch.allclose(triton_res, torch_res)
    #print("PASS")

    mlflow.end_run()

if __name__ == "__main__":
    main()
