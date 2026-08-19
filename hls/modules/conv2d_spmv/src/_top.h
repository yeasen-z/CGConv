#include "ker2col.h"
#include "img2col.h"
#include "gemm.h"
#include "col2res.h"
#include "ci2spci.h"
#include "spcr2cr.h"

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
    static const unsigned Ph=(if_pad==0?0:(Kh-1)/2),Pw=(if_pad==0?0:(Kw-1)/2);//padding
    static const unsigned Hout=(Hin+2*Ph-Kh)/Sh+1, Wout=(Win+2*Pw-Kw)/Sw+1;//最终输出,输出维度为[CHout][Hout][Wout]
    static const unsigned col_img_H=Kw*Kh, col_img_W=Hout*Wout;//中间col_img变量，维度为[CHin][Kw*Kh][Hout*Wout]
    static const unsigned col_ker_H=CHout, col_ker_W=Kh*Kw;//中间col_ker变量，维度为[CHin][CHout][Kh*Kw]
    static const unsigned col_res_H=CHout, col_res_W=Hout*Wout;//中间res变量，维度为[CHout][Hout*Wout]

    static const unsigned win_middle=col_img_H/2;//滑动窗口的中心位置索引
    static const unsigned index_H=CHin;//对应的是Chin，每一个通道对应一个H的索引
    static const unsigned index_W=col_img_W;//对应的是滑动窗口对应在col_img下占有一列
};



template<typename CONFIG_CONV_SPMV>
void conv2d_spmv(
    typename CONFIG_CONV_SPMV::Dtype_f feature_in[],
    typename CONFIG_CONV_SPMV::Dtype_w W[],
    typename CONFIG_CONV_SPMV::Dtype_b bias[],
    typename CONFIG_CONV_SPMV::Dtype_f feature_out[]
){
    typename CONFIG_CONV_SPMV::Dtype_f 
        col_ker[CONFIG_CONV_SPMV::CHin][CONFIG_CONV_SPMV::col_ker_H][CONFIG_CONV_SPMV::col_ker_W]= {0};
    typename CONFIG_CONV_SPMV::Dtype_f 
        col_img[CONFIG_CONV_SPMV::CHin][CONFIG_CONV_SPMV::col_img_H][CONFIG_CONV_SPMV::col_img_W]= {0};

    typename CONFIG_CONV_SPMV::Dtype_f 
        sp_ci[CONFIG_CONV_SPMV::CHin][CONFIG_CONV_SPMV::col_img_H*CONFIG_CONV_SPMV::col_img_W]={0};
    
    int sp_L[CONFIG_CONV_SPMV::CHin]={0};
    int index[CONFIG_CONV_SPMV::index_H][CONFIG_CONV_SPMV::index_W]={0};

    typename CONFIG_CONV_SPMV::Dtype_f 
        sp_cr[CONFIG_CONV_SPMV::CHin][CONFIG_CONV_SPMV::col_res_H*CONFIG_CONV_SPMV::col_res_W]={0};

    typename CONFIG_CONV_SPMV::Dtype_f 
        col_res[CONFIG_CONV_SPMV::CHin][CONFIG_CONV_SPMV::col_res_H*CONFIG_CONV_SPMV::col_res_W]={0};


    ker2col<CONFIG_CONV_SPMV>(W,col_ker[0][0]);

    img2col<CONFIG_CONV_SPMV>(feature_in, col_img[0][0]);

    get_ci_index<CONFIG_CONV_SPMV>(col_img[0][0],index[0],sp_L);


    for(int cin_n=0;cin_n<CONFIG_CONV_SPMV::CHin;cin_n++){
        ci2spci_0<CONFIG_CONV_SPMV>(col_img[cin_n][0],sp_ci[cin_n],index[cin_n],sp_L[cin_n]);
        gemm_sp_0<CONFIG_CONV_SPMV>(col_ker[cin_n][0],sp_ci[cin_n],sp_cr[cin_n],sp_L[cin_n]);
    }

    spcr2cr<CONFIG_CONV_SPMV>(sp_cr[0],col_res[0],index[0],sp_L);

    col2res<CONFIG_CONV_SPMV>(col_res[0],bias,feature_out);

    return;
}

#endif
