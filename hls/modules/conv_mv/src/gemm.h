#ifndef _GEMM_H_
#define _GEMM_H_

/*
如果想要单独执行，可以按照这个过程使用
*/
struct CONFIG_gemm{
    static const unsigned padding=1;
    static const unsigned CHout=4;
    static const unsigned CHin=3;
    static const unsigned Kh=3;
    static const unsigned Kw=3;
    static const unsigned Sw=1; //stride
    static const unsigned Sh=1; //stride
    static const unsigned Hin=8;
    static const unsigned Win=8;
    typedef float Dtype_f; 

    static const unsigned Pw=(padding==0?0:(Kw-1)/2); //padding
    static const unsigned Ph=(padding==0?0:(Kh-1)/2); //padding

    static const unsigned Hout=(Hin+2*Ph-Kh)/Sh+1;
    static const unsigned Wout=(Win+2*Pw-Kw)/Sw+1;

    static const unsigned col_ker_H=CHout;
    static const unsigned col_ker_W=Kh*Kw;
    static const unsigned col_img_H=Kw*Kh;
    static const unsigned col_img_W=Hout*Wout;

    static const unsigned col_res_H=CHout;
    static const unsigned col_res_W=Hout*Wout;
};

/*
两个矩阵的大小分别为
col_ker[CHout][Kh*Kw]
col_img[Kw*Kh][Hout*Wout]
进行的乘法为col_ker*col_img
res的大小为
res[CHout][Hout*Wout]
*/
template <typename CONFIG_gemm>
void gemm_0(
    typename CONFIG_gemm::Dtype_f col_ker[],
    typename CONFIG_gemm::Dtype_f col_img[],
    typename CONFIG_gemm::Dtype_f col_res[]
){

    // col_ker_H=CHout;
    // col_ker_W=Kh*Kw;
    // col_img_H=Kw*Kh;
    // col_img_W=Hout*Wout;
    // res_H=CHout;
    // res_W=Hout*Wout;
    
    // Initialize result array to zero
    for (int i = 0; i < CONFIG_gemm::col_res_H; i++) {
        for (int j = 0; j < CONFIG_gemm::col_res_W; j++) {
            col_res[i * CONFIG_gemm::col_res_W + j] = 0;
        }
    }

    // Perform the matrix multiplication
    for (int i = 0; i < CONFIG_gemm::col_res_H; i++) {
        for (int j = 0; j < CONFIG_gemm::col_res_W; j++) {

            for (int k = 0; k < CONFIG_gemm::col_ker_W; k++) {
                /*
                col_ker[CHout][Kh*Kw]
                col_img[Kw*Kh][Hout*Wout]
                */
                col_res[i * CONFIG_gemm::col_res_W + j] += col_ker[i * CONFIG_gemm::col_ker_W + k]
                                                *  col_img[k * CONFIG_gemm::col_img_W + j];

                // std::cout<<col_ker[i * CONFIG_gemm::col_ker_W + k]<<" "<<
                //             col_img[k * CONFIG_gemm::col_img_W + j]<<" "<<res[i * CONFIG_gemm::res_W + j]<<std::endl;
            }
            // std::cout<<col_res[i * CONFIG_gemm::col_res_W + j]<<" ";
        }
        // std::cout<<std::endl;
    }

    return;
};


#endif