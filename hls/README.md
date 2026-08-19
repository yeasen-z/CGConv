# CGConv HLS-style prototype

This directory contains the following C++ components:

- `modules/conv2d_spmv`: the Center Glance sparse convolution used by CGConv;
- `modules/conv_mv`: a dense im2col/GEMM reference with a similar layout;
- `MsLib`: a separate collection of floating-point and integer neural-network
  operators retained from the original research repository.

The sparse convolution follows this pipeline:

```text
image -> im2col -> remove windows with zero centres -> compact GEMM
      -> restore window positions -> accumulate input channels + bias
```
