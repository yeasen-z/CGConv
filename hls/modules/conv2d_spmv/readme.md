# Important Top File
1. src/_top.h
实现了一个流程的函数,包括内存申请，计算和输出。
2. _demo.h
展示了conv2d_spmv函数的使用方式。


# Stream
1. img2col/ker2col: 将权重和特征图2col，获得col_img和col_ker(该变化是base on input channel的)
2. ci2spci: 将densen的特征图中的0去除掉，获得sp_ci(逐层进行的)
3. gemm: 将sp_ci和col_ker进行矩阵运算，获得sp_cr(逐层进行的)
4. spcr2cr: 将sp形式的结果还原为原本的普通col_res
5. col2res: 将col_res还原到普通的卷积中

整体特点是，针对输入特征图的每一个输入通道，进行col2img，然后和对应的col_ker矩阵乘（同样ker是将所有乘到该通道的卷积核获得的），然后输出获得的是，该通道输入映射到每一个输出通道上的数值，所有要在col2res中进行累加操作。

# Notes
1. 所有后缀带有_0的都是逐层进行的，即指考虑了矩阵运算本身，进行的是二维运算
   - gemm是逐层进行的，所以后缀了名为gemm_sp_0
2. 函数使用hls的代码风格，将输入和输出都采用地址传输，同时避免指针的指针作为参数传递
