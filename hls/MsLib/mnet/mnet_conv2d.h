// This file is used to define the Conv2D layer in Pytorch
#ifndef MNET_CONV2D_H_INCLUDED
#define MNET_CONV2D_H_INCLUDED

namespace mnet{


struct conv2d0_config{
    typedef float  Dtype_f;
    typedef float  Dtype_w;
    typedef float  Dtype_b;
    typedef int Index_t;
    typedef float Dtype_tmp;

    static const bool padding =0;
    static const bool relu_en=0; 

    static const unsigned CHin=16;
    static const unsigned Hin =6;
    static const unsigned Win =6;
    static const unsigned Kw =3;
    static const unsigned Kh =3;
    static const unsigned Sw =1;
    static const unsigned Sh =1;

	static const unsigned CHout=1;

	//settings defined by the values of up above
    static const unsigned pad_w= (padding==0?0:(Kw-1)/2);
    static const unsigned pad_h= (padding==0?0:(Kh-1)/2);
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
	padding: 0:VALID, 1:SAME (the choice in tensorflow)
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
				sum+=(typename CONFIG_C::Dtype_tmp)bias[cout_n];
				//std::cout<<sum<<std::endl;
				if(CONFIG_C::relu_en && sum<0)
					sum=0;
				//Feature out shape:(CHout,Hout,Wout​) == [CHout][Hout][Wout]
				feature_out[cout_n*CONFIG_C::Hout*CONFIG_C::Wout+i*CONFIG_C::Wout+j]=(typename CONFIG_C::Dtype_f)sum;
				//std::cout<<feature_out[i*CONFIG_C::Wout*CONFIG_C::CHout+j*CONFIG_C::CHout+cout_n]<<std::endl;
			}
		}
	}
};

};
#endif // NET_CONV_H_INCLUDED


