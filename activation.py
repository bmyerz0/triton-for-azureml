import torch

import triton
import triton.language as tl


@triton.jit
def swish_kernel(
    x_ptr,  # *Pointer* to first input vector.
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
   
    output = x * tl.sigmoid(x)

    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


# do 8: just a small block size
# do 2048 (8KB)
# do 8096 (32 KB)
#  Shared memory size / SM = 96KB
# do 2**16 (128KB)
#  Register file size / SM = 256KB
# do 2**18 (512KB)
#  6144 KB (L2 cache size)
# do 2*21 (8000KB)

def swish_torch(x: torch.Tensor):
    return x * torch.sigmoid(x)

@torch.jit.script
def swish_torch_opt(x: torch.Tensor):
    return x * torch.sigmoid(x)

def swish_torch_builtin(x: torch.Tensor):
    return torch.nn.SiLU()(x)

def swish_triton(x: torch.Tensor, block):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    swish_kernel[grid](x, output, n_elements, BLOCK_SIZE=block) 
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
        line_vals=['triton', 'torch', 'torch.script.jit', 'torch builtin'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch', 'torch.script.jit', 'Torch builtin'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-'), ('red', '-'), ('orange', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='swish performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: swish_torch(x))
    if provider == 'torch.script.jit':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: swish_torch_opt(x))
    if provider == 'torch builtin':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: swish_torch_builtin(x))
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: swish_triton(x, 2**10))

    gbps = lambda ms: 8 * size / ms * 1e-6
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
        plot_name='swish performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark_blocksize(size, blocksize):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: swish_triton(x, blocksize))

    gbps = lambda ms: 8 * size / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


import mlflow
import pandas

def main():
    mlflow.start_run()

    torch.set_printoptions(threshold=10000)
    #benchmark.run(show_plots=True, print_data=True)

    with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
        benchmark_blocksize.run(show_plots=False, print_data=True)

    mlflow.end_run()

if __name__ == "__main__":
    main()