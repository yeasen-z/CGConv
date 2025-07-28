#include "ker2col.h"
#include "img2col.h"
#include "gemm.h"
#include "col2res.h"

#ifndef _CONV_SPMV_H_
#define _CONV_SPMV_H_

/*
使用SPMV的卷积层包裹
- CHin, CHout, Hin, Win, Kh, Kw, Sh, Sw, if_pad
- 或者直接使用默认的构造函数，输入列表为空
*/
struct SET_CONV_SPMV{
    typedef float Dtype_f; 
    typedef float Dtype_w;
    typedef float Dtype_b;

    //特征图输入维度的描述，包括输出维度的通道数
    static const unsigned CHin=16,CHout=32;
    static const unsigned Hin=8,Win=8;

    //卷积层维度的描述
    static const unsigned Kh=3,Kw=3;
    
    //卷积细节的描述
    static const unsigned Sh=1,Sw=1;
    static const unsigned if_pad=1;
};

struct CONFIG_CONV_SPMV:SET_CONV_SPMV{
    //其他通过计算得到的参数
    static const unsigned Ph=(if_pad==0?0:(Kw-1)/2),Pw=(if_pad==0?0:(Kh-1)/2);//padding
    static const unsigned Hout=(Hin+2*Ph-Kh)/Sh+1, Wout=(Win+2*Pw-Kw)/Sw+1;//最终输出,输出维度为[CHout][Hout][Wout]
    static const unsigned col_img_H=Kw*Kh, col_img_W=Hout*Wout;//中间col_img变量，维度为[CHin][Kw*Kh][Hout*Wout]
    static const unsigned col_ker_H=CHout, col_ker_W=Kh*Kw;//中间col_ker变量，维度为[CHin][CHout][Kh*Kw]
    static const unsigned col_res_H=CHout, col_res_W=Hout*Wout;//中间res变量，维度为[CHout][Hout*Wout]

    static const unsigned win_middle=col_img_H/2+1;//滑动窗口的中心位置索引
    static const unsigned index_H=CHin;//对应的是Chin，每一个通道对应一个H的索引
    static const unsigned index_W=col_img_W;//对应的是滑动窗口对应在col_img下占有一列
};


template<typename CONFIG_CONV_SPMV>
void conv_mv(
    typename CONFIG_CONV_SPMV::Dtype_f feature_in[],
    typename CONFIG_CONV_SPMV::Dtype_w W[],
    typename CONFIG_CONV_SPMV::Dtype_b bias[],
    typename CONFIG_CONV_SPMV::Dtype_f feature_out[]
){

    typename CONFIG_CONV_SPMV::Dtype_f col_ker[CONFIG_CONV_SPMV::CHin][CONFIG_CONV_SPMV::col_ker_H][CONFIG_CONV_SPMV::col_ker_W]= {0};
    typename CONFIG_CONV_SPMV::Dtype_f col_img[CONFIG_CONV_SPMV::CHin][CONFIG_CONV_SPMV::col_img_H][CONFIG_CONV_SPMV::col_img_W]= {0};
    typename CONFIG_CONV_SPMV::Dtype_f col_res[CONFIG_CONV_SPMV::CHin][CONFIG_CONV_SPMV::col_res_H][CONFIG_CONV_SPMV::col_res_W]= {0};

    ker2col<CONFIG_CONV_SPMV>(W,col_ker[0][0]);

    img2col<CONFIG_CONV_SPMV>(feature_in, col_img[0][0]);

    for(int cin_n=0;cin_n<CONFIG_CONV_SPMV::CHin;cin_n++){
        gemm_0<CONFIG_CONV_SPMV>(col_ker[cin_n][0],col_img[cin_n][0],col_res[cin_n][0]);
    }
    col2res<CONFIG_CONV_SPMV>(col_res[0][0],bias,feature_out);

    return;
}


#endif