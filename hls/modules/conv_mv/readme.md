# Top File
1. src/_top.h
实现了一个流程的函数,包括内存申请，计算和输出。
2. _demo.h
展示了conv2d_spmv函数的使用方式。


# Stream
1. img2col/ker2col: 将权重和特征图2col，获得col_img和col_ker
2. gemm: 将col_img和col_ker进行矩阵运算，获得col_res
3. col2res: 将col_res还原到普通的卷积中

# Notes
1. 该代码事通过spmv简化得到的，所以和传统的矩阵加速算法有一定的区别，传统方案的编写in future。
   - 关键的区别就是，传统的矩阵加速是将img全部平坦，所有通道一同展开。kernel同样进行展开
   - 然后计算得到的输出就是模型的某一层的输出，不需要进行本代码中的col2res中再进行累加。
