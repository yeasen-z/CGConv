#ifndef _KER2COL_H_
#define _KER2COL_H_

/*
如果想要单独执行，可以按照这个过程使用
*/
struct CONFIG_K2col{
    static const unsigned CHout=4;
    static const unsigned CHin=3;
    static const unsigned Kh=3;
    static const unsigned Kw=3;
    typedef float Dtype_f; 
};


/*
Kernel shape: (CHout, CHin,kH,kW) == [CHout][CHin][Kh][Kw]
在ker2col中，将Kh和Kw的维度展开，变成一个一维数组Khw
每个相同CHin，不同CHout的Khw一维数组进行组合，形成二维数组[CHout][Khw]
每一个二维数组[CHout][Khw]都可以进一步的组合，形成一个三维数组[CHin][CHout][Kh*Kw]
那么当进行matrix运算的时候，第一层的索引得到的就是能够进行一次矩阵乘的操作
*/
template <typename CONFIG_K2col>
void ker2col(
    typename CONFIG_K2col::Dtype_f W[],
    typename CONFIG_K2col::Dtype_f W_col[]
){
    for(int cout_n=0;cout_n<CONFIG_K2col::CHout;cout_n++)
    {
        for(int cin_n=0;cin_n<CONFIG_K2col::CHin;cin_n++)
        {

            for(int ii=0;ii<CONFIG_K2col::Kh;ii++)
            {
                for(int jj=0;jj<CONFIG_K2col::Kw;jj++)
                {
                    // [CHin][CHout][Kh*Kw]
                    W_col[cin_n*CONFIG_K2col::CHout*CONFIG_K2col::Kh*CONFIG_K2col::Kw+cout_n*CONFIG_K2col::Kh*CONFIG_K2col::Kw+ii*CONFIG_K2col::Kw+jj]=
                    W[cout_n*CONFIG_K2col::CHin*CONFIG_K2col::Kh*CONFIG_K2col::Kw+cin_n*CONFIG_K2col::Kh*CONFIG_K2col::Kw+ii*CONFIG_K2col::Kw+jj];
                }
            }

        }
    }

    return;
};

#endif