#ifndef _SPCR2CR_H_
#define _SPCR2CR_H_

struct CONFIG_SPCR2CR{
    static const unsigned CHin=3;
    static const unsigned CHout=4;
    static const unsigned col_res_H=CHout;
    static const unsigned col_res_W=16;//Hout=4,Wout=4
    static const unsigned index_H=CHin;//对应的是Chin，每一个通道对应一个H的索引
    static const unsigned index_W=col_res_W;//对应的是滑动窗口对应在col_img下占有一列
    //sp_cr的维度为[CHin][CHout][sp_L_0]
    //col_res的维度为[CHin][CHout][Hout*Wout]
    //使用sp_L记录spci长度的数组大小,每一个通道对应一个spci
    //sp_L的维度为[CHin]
    //index的存储结构为：
    //index[CHin][col_res_W]
    //第一个索引下为输入特征图的通道
    //第二个索引为该通道下，col_img下该位置的展平滑动窗口，对应在spci下的位置，如果为0则记为-1
    //spci将保存中心不为0的展平窗口，并将其紧密排列，再将位置保存在index中
};

template<typename CONFIG_SPCR2CR>
void spcr2cr(
    typename CONFIG_SPCR2CR::Dtype_f sp_cr[],
    typename CONFIG_SPCR2CR::Dtype_f col_res[],
    int index[],
    int sp_L[]
){
    for(int i=0;i<CONFIG_SPCR2CR::CHin;i++){
        for(int j=0;j<CONFIG_SPCR2CR::col_res_W;j++){
            //对col_img的每一列进行遍历
            int compact_index = index[i*CONFIG_SPCR2CR::index_W+j];
            if(compact_index!=-1){
                //如果该位置能够在sp中找到
                for(int k=0;k<CONFIG_SPCR2CR::col_res_H;k++){
                    col_res[i*CONFIG_SPCR2CR::col_res_H*CONFIG_SPCR2CR::col_res_W
                            +k*CONFIG_SPCR2CR::col_res_W+j]
                        =sp_cr[i*CONFIG_SPCR2CR::col_res_H*CONFIG_SPCR2CR::col_res_W
                               +k*sp_L[i]+compact_index];
                }
            }else{
                //如果该位置不能在sp中找到
                for(int k=0;k<CONFIG_SPCR2CR::col_res_H;k++){
                    col_res[i*CONFIG_SPCR2CR::col_res_H*CONFIG_SPCR2CR::col_res_W
                            +k*CONFIG_SPCR2CR::col_res_W+j]=0;
                }
            }
        }
    }

    return;
}




#endif
