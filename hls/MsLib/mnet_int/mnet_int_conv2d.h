//INT Inference tested

#ifndef MNET_INT_CONV2D_H_INCLUDED
#define MNET_INT_CONV2D_H_INCLUDED

namespace mnet_int{

struct conv2d0_config{
    typedef int  Dtype_f; //recommand to be int 8, could use ap_int<bit_f>
    typedef int  Dtype_w; //recommand to be int 8
    typedef int  Dtype_b; //recommand to be int 8

    typedef int Index_t;
    typedef int Dtype_tmp; //recommand to be int 32, bits long enough

	static const int Delta_w = 256; 
	// quant factor of weight
	static const int Delta_i = 256; 
	// quant factor of feature_in and feature_out, actually is not used in function
	static const int Delta_b = Delta_w*Delta_i;  //Delta_b=Delta_w*Delta_i
	// quant factor of bias, actually is not used in function, only needed for do quantization

    static const bool padding =0;
    static const bool relu_en=0;

    static const unsigned CHin=16;
    static const unsigned Hin =6;
    static const unsigned Win =6;
    static const unsigned Kw =3;
    static const unsigned Kh =3;
    static const unsigned Sw =1;
    static const unsigned Sh =1;
    static const unsigned pad_w= (padding==0?0:(Kw-1)/2);
    static const unsigned pad_h= (padding==0?0:(Kh-1)/2);
    static const unsigned CHout=1;
    static const unsigned Wout=(Win+2*pad_w-Kw)/Sw+1;
    static const unsigned Hout=(Hin+2*pad_h-Kh)/Sh+1;
};


/*
	Use for Pytorch Conv2d
	Feature in shape: (CHin,Hin,Win) == [CHin][Hin][Win]
	Kernel shape: (CHout, CHin,kH,kW) == [CHout][CHin][Kh][Kw]
	Feature out shape:(CHout,Hout,Wout​) == [CHout][Hout][Wout]
	Bias shape: (CHout,) == [CHout]
	Feature_out[CHout][Hout][Wout] = Feature_in[CHin][Hin][Win] * W[CHout][CHin][Kh][Kw] + bias[CHout]
	padding: 0:VALID, 1:SAME
	Delta_w: the quant factor of weight
	Delta_i: the quant factor of input
	Delta_b: the quant factor of bias
*/
template <typename CONFIG_C>
void Conv2d(
		typename CONFIG_C::Dtype_f feature_in[],
		typename CONFIG_C::Dtype_w W[],
		typename CONFIG_C::Dtype_b bias[],
		typename CONFIG_C::Dtype_f feature_out[]
	)
{
	for(int cout_n=0;cout_n<CONFIG_C::CHout;cout_n++)
	{
		for(int i=0;i<CONFIG_C::Hout;i++)
		{
			for(int j=0;j<CONFIG_C::Wout;j++)
			{
				typename CONFIG_C::Dtype_tmp sum=0;
				for(int ii=0;ii<CONFIG_C::Kh;ii++)
				{
					for(int jj=0;jj<CONFIG_C::Kw;jj++)
					{
						typename CONFIG_C::Index_t h=i*CONFIG_C::Sh-CONFIG_C::pad_h+ii;
						typename CONFIG_C::Index_t w=j*CONFIG_C::Sw-CONFIG_C::pad_w+jj;
						if(h>=0 && w>=0 && h<CONFIG_C::Hin && w<CONFIG_C::Win)
						{
							Input_Channel:
							for(int cin_n=0;cin_n<CONFIG_C::CHin;cin_n++)
							{
								//Feature in shape: (CHin,Hin,Win) == [CHin][Hin][Win]
								//Kernel shape: (CHout, CHin,kH,kW) == [CHout][CHin][Kh][Kw]
								//Feature out shape:(CHout,Hout,Wout​) == [CHout][Hout][Wout]
								//Feature_out[CHout][Hout][Wout] = Feature_in[CHin][Hin][Win] * W[CHout][CHin][Kh][Kw] + bias[CHout]
								typename CONFIG_C::Dtype_tmp tp=
								feature_in[cin_n*CONFIG_C::Hin*CONFIG_C::Win+h*CONFIG_C::Win+w]*
								W[cout_n*CONFIG_C::CHin*CONFIG_C::Kh*CONFIG_C::Kw+cin_n*CONFIG_C::Kh*CONFIG_C::Kw+ii*CONFIG_C::Kw+jj];
								sum+=tp;
							}
						}
					}
				}
				sum=sum+(typename CONFIG_C::Dtype_tmp)bias[cout_n];
				//here the total Delta of sum is Delta_w*Delta_i == Delta_b

				//std::cout<<sum<<std::endl;
				if(CONFIG_C::relu_en && sum<0)
					sum=0;
				//Feature out shape:(CHout,Hout,Wout​) == [CHout][Hout][Wout]
				feature_out[cout_n*CONFIG_C::Hout*CONFIG_C::Wout+i*CONFIG_C::Wout+j]=(typename CONFIG_C::Dtype_f)(sum/CONFIG_C::Delta_w);
				//std::cout<<feature_out[i*CONFIG_C::Wout*CONFIG_C::CHout+j*CONFIG_C::CHout+cout_n]<<std::endl;
			}
		}
	}
};

}
#endif // NET_CONV_H_INCLUDED


