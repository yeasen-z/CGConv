//INT Inference NOT tested

#ifndef MNET_INT_LINEAR_H_INCLUDED
#define MNET_INT_LINEAR_H_INCLUDED

namespace mnet_int{


struct linear0_int_config{
    typedef int  Dtype_f;
    typedef int  Dtype_b;
    typedef int  Dtype_w;
    typedef int  Dtype_tmp;

	static const int Delta_w = 256; 
	// quant factor of weight
	static const int Delta_i = 256; 
	// quant factor of feature_in and feature_out, actually is not used in function
	static const int Delta_b = Delta_w*Delta_i;  //Delta_b=Delta_w*Delta_i
	// quant factor of bias, actually is not used in function, only needed for do quantization

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
        sum=sum+bias[i];
        if (CONFIG_L::relu_en&&sum<0){
            sum=0;
        }
        feature_out[i]=(typename CONFIG_L::Dtype_f)sum/CONFIG_L::Delta_w;
    }
};

};

#endif // NET_LINEAR_H_INCLUDED
