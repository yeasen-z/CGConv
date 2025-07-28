// This file is used to define the Linear layer in Pytorch
#ifndef MNET_LINEAR_H_INCLUDED
#define MNET_LINEAR_H_INCLUDED

namespace mnet{


struct linear0_config{
    typedef float  Dtype_f;
    typedef float  Dtype_b;
    typedef float  Dtype_w;
    typedef float  Dtype_tmp;

    static const unsigned relu_en=0;
    static const unsigned Lin =10;
    static const unsigned Lout =10;
};


/*
	Use for Pytorch Linear
	Feature in shape: (Lin) == [Lin]
	Weight shape: (Lout,Lin) == [Lout][Lin]
    Bias shape: (Lout) == [Lout]
	Feature out shape:(Lout​) == [Lout]
*/
template <typename CONFIG_L>
void Linear(
        typename CONFIG_L::Dtype_f feature_in[],
        typename CONFIG_L::Dtype_w W[],
        typename CONFIG_L::Dtype_b bias[],
        typename CONFIG_L::Dtype_f feature_out[]
    )
{

    // feature_out[Lout]=SUM{feature_in[Lin]*W[Lout][Lin]}+b[Lout]
    for(int i=0;i<CONFIG_L::Lout;i++){
        typename CONFIG_L::Dtype_tmp sum=0;
        for(int j=0;j<CONFIG_L::Lin;j++){
            sum+=feature_in[j]*W[i*CONFIG_L::Lin+j];
        }
        sum+=bias[i];
        if (CONFIG_L::relu_en&&sum<0){
            sum=0;
        }
        feature_out[i]=sum;
    }
};

};

#endif // NET_LINEAR_H_INCLUDED
