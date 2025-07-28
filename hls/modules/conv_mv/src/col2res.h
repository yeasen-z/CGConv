#ifndef _COL2RES_H_
#define _COL2RES_H_

/*
如果想要单独执行，可以按照这个过程使用
*/
struct CONFIG_col2res{
    static const unsigned CHout=4;
    static const unsigned CHin=3;
    static const unsigned Hout=8;
    static const unsigned Wout=8;
    static const unsigned col_res_H=CHout;
    static const unsigned col_res_W=CHin*Hout*Wout;
    typedef float Dtype_f; 
};


template <typename CONFIG_col2res>
void col2res(
    typename CONFIG_col2res::Dtype_f col_res[],
    typename CONFIG_col2res::Dtype_f bias[],
    typename CONFIG_col2res::Dtype_f res[]
){
    // col_res [CHin][CHout][Hout*Wout]
    // res [CHout][Hout][Wout]
    // bias [CHout][Hout][Wout]

    for(int cout_n=0;cout_n<CONFIG_col2res::CHout;cout_n++)
    {
        for(int i=0;i<CONFIG_col2res::Hout;i++)
        {
            for(int j=0;j<CONFIG_col2res::Wout;j++)
            {
                for(int cin_n=0;cin_n<CONFIG_col2res::CHin;cin_n++){
                    res[cout_n*CONFIG_col2res::Hout*CONFIG_col2res::Wout+i*CONFIG_col2res::Wout+j]+=
                    col_res[cin_n*CONFIG_col2res::CHout*CONFIG_col2res::Hout*CONFIG_col2res::Wout+
                            cout_n*CONFIG_col2res::Hout*CONFIG_col2res::Wout+i*CONFIG_col2res::Wout+j];
                }
                // std::cout<<res[cout_n*CONFIG_col2res::Hout*CONFIG_col2res::Wout+i*CONFIG_col2res::Wout+j]<<" ";
                res[cout_n*CONFIG_col2res::Hout*CONFIG_col2res::Wout+i*CONFIG_col2res::Wout+j]
                    +=bias[cout_n*CONFIG_col2res::Hout*CONFIG_col2res::Wout+i*CONFIG_col2res::Wout+j];
            } 
            // std::cout<<std::endl;
        }
    }
    return;
}



#endif