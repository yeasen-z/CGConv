#include<iostream>
#include "src/_top.h"
#include "testdata/testdata_easy.h"
#include <iomanip>

#ifndef _DEMO_CONV_SPMV_H_
#define _DEMO_CONV_SPMV_H_

struct set_conv_spmv2:SET_CONV_SPMV{
    static const unsigned CHin=1,CHout=1;
    static const unsigned Hin=4,Win=4;
    static const unsigned Kh=3,Kw=3;
    static const unsigned Sh=1,Sw=1;
    static const unsigned if_pad=1;
};

struct config_spmv_2:set_conv_spmv2{
    //其他通过计算得到的参数
    static const unsigned Ph=(if_pad==0?0:(Kw-1)/2),Pw=(if_pad==0?0:(Kh-1)/2);//padding
    static const unsigned Hout=(Hin+2*Ph-Kh)/Sh+1, Wout=(Win+2*Pw-Kw)/Sw+1;//最终输出,输出维度为[CHout][Hout][Wout]
    static const unsigned col_img_H=Kw*Kh, col_img_W=Hout*Wout;//中间col_img变量，维度为[CHin][Kw*Kh][Hout*Wout]
    static const unsigned col_ker_H=CHout, col_ker_W=Kh*Kw;//中间col_ker变量，维度为[CHin][CHout][Kh*Kw]
    static const unsigned col_res_H=CHout, col_res_W=Hout*Wout;//中间res变量，维度为[CHout][Hout*Wout]

    static const unsigned win_middle=col_img_H/2;//滑动窗口的中心位置索引
    static const unsigned index_H=CHin;//对应的是Chin，每一个通道对应一个H的索引
    static const unsigned index_W=col_img_W;//对应的是滑动窗口对应在col_img下占有一列
};

void demo_conv_mv(){
    typename config_spmv_2::Dtype_f  res[config_spmv_2::CHout][config_spmv_2::Hout][config_spmv_2::Wout]={0};
    conv_mv<config_spmv_2>(image_CHW[0][0], kernel_OIHW[0][0][0], bias[0][0], res[0][0]);

    for(int cout_n=0;cout_n<config_spmv_2::CHout;cout_n++){
        for(int i=0;i<config_spmv_2::Hout;i++){
            for(int j=0;j<config_spmv_2::Wout;j++){
                std::cout<<res[cout_n][i][j]<<" ";
            }
            std::cout<<std::endl;
            for(int j=0;j<config_spmv_2::Wout;j++){
                std::cout<<res_test[cout_n][i][j]<<" ";
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }
    return;
}


#endif