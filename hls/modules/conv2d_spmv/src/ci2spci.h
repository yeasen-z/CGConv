/*
将col_img转换为spci
col_img [CHin][col_img_H=Kh*Kw][col_img_W=Hout*Wout]
spci [col_img_H=Kh*Kw][sp_L_0]
其中sp_L_0为spci的长度，index为col_img中的位置对应到spci中的位置
需要通过get_ci_index函数得到index
*/
#ifndef _CI2SPCI_H_
#define _CI2SPCI_H_

struct CONFIG_CI2SPCI{
    static const unsigned CHin=3;
    static const unsigned col_img_H=9; //Kh=3,Kw=3
    static const unsigned col_img_W=16;//Hout=4,Wout=4
    static const unsigned win_middle=col_img_H/2;//滑动窗口的中心位置索引
    static const unsigned index_H=CHin;//对应的是Chin，每一个通道对应一个H的索引
    static const unsigned index_W=col_img_W;//对应的是滑动窗口对应在col_img下占有一列
    //使用sp_L记录spci长度的数组大小,每一个通道对应一个spci
    //index的存储结构为：
    //index[CHin][col_img_H]
    //第一个索引下为输入特征图的通道
    //第二个索引为该通道下，col_img下该位置的展平滑动窗口，对应在spci下的位置，如果为0则记为-1
    //spci将保存中心不为0的展平窗口，并将其紧密排列，再将位置保存在index中
};

template<typename CONFIG_CI2SPCI>
void get_ci_index(
    typename CONFIG_CI2SPCI::Dtype_f col_img[],
    int index[],
    int sp_L[]//初始化为全0
){
    // col_img [CHin][col_img_H=Kh*Kw][col_img_W=Hout*Wout]
    //index[CHin][col_img_H]
    for(int i=0;i<CONFIG_CI2SPCI::CHin;i++){
        for(int j=0;j<CONFIG_CI2SPCI::col_img_W;j++){
            //遍历每一个滑动窗口
            if(col_img[i*CONFIG_CI2SPCI::col_img_H*CONFIG_CI2SPCI::col_img_W
                        +CONFIG_CI2SPCI::win_middle*CONFIG_CI2SPCI::col_img_W
                        +j]!=0){
            // if(col_img[i][CONFIG_CI2SPCI::win_middle][j]!=0){
                //如果中心不为0
                index[i*CONFIG_CI2SPCI::col_img_H+j]=sp_L[i];
                sp_L[i]++;
            }else{
                index[i*CONFIG_CI2SPCI::col_img_H+j]=-1;
            }
        }
    }
    return;
}

template<typename CONFIG_CI2SPCI>
void ci2spci_0(
    typename CONFIG_CI2SPCI::Dtype_f col_img[],
    typename CONFIG_CI2SPCI::Dtype_f sp_ci[],
    int index[],
    int sp_L_0
){
    // col_img [col_img_H=Kh*Kw][col_img_W=Hout*Wout]
    // sp_ci [col_img_H=Kh*Kw][sp_L_0]

    for(int j=0;j<CONFIG_CI2SPCI::col_img_W;j++){
        //对col_img的每一列进行遍历
        if(index[j]!=-1){
            //如果该位置能够放在sp中
            for(int k=0;k<CONFIG_CI2SPCI::col_img_H;k++){
                sp_ci[k*sp_L_0+index[j]]
                    =col_img[k*CONFIG_CI2SPCI::col_img_W+j];
            }
        }else{
            continue;
        }
        // for(int k=0;k<CONFIG_CI2SPCI::col_img_H;k++){
        //     std::cout<<sp_ci[k*sp_L_0+index[j]]<<" ";
        // }
        // std::cout<<std::endl;
    }
    return;
}




#endif