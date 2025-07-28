//INT Inference tested

/*
Transpose Conv is a Upsample Methods, other is ‘bilinear,Unpooling’
Transpose Conv is a special conv. 
First it add 0 by a ratio to expand image size. Then routate the kernel.
input args has a new one 'output_padding' to control the output size. 
Because for Conv2d, different input size will have same output size.
for example:
	A:	input_size=(Hin,Win)==(6,6), kernel_size=(3,3), strides=(2,2), padding=1, output_size=(6+2*1-3)/2+1=3
	B:	input_size=(Hin,Win)==(5,5), Kernel_size=(3,3), strides=(2,2), padding=1, output_size=(5+2*1-3)/2+1=3
So when we has a transposeConv with settings:
	intput_size=(3,3), Kernel_size=(3,3),strides=(2,2).padding=1
What size should the Function output?
So we need a new arg to control the output size. It is output_padding.
and set the output_padding is (strides-1) by default, which results in output_size=input_size*strides.
Change of the output_padding could change the output_size.
Note that output_padding is only used to find output shape, but does not actually add zero-padding to output.
*/
#ifndef MNET_INT_TRANSPOSE_CONV2D_H_INCLUDED
#define MNET_INT_TRANSPOSE_CONV2D_H_INCLUDED

namespace mnet_int{

struct transconv2d0_config{
    typedef int  Dtype_f;
    typedef int  Dtype_w;
    typedef int  Dtype_b;
    typedef int Index_t;
    typedef int Dtype_tmp;

    static const bool padding =1; //1 for default
    static const bool relu_en=0; 

	static const int Delta_w = 256; 
	// quant factor of weight
	static const int Delta_i = 256; 
	// quant factor of feature_in and feature_out, actually is not used in function
	static const int Delta_b = Delta_w*Delta_i;  //Delta_b=Delta_w*Delta_i
	// quant factor of bias, actually is not used in function, only needed for do quantization

    static const unsigned CHin=6;
    static const unsigned Hin =16;
    static const unsigned Win =16;
    static const unsigned Kw =5;
    static const unsigned Kh =5;
    static const unsigned Sw =2;
    static const unsigned Sh =2;

	static const unsigned CHout=6;

	//settings defined by the values of up above

    static const unsigned pad_w = (padding==0?0:(Kw-1)/2);
    static const unsigned pad_h = (padding==0?0:(Kh-1)/2);
	static const unsigned o_pad_w = Sw-1;//can change it for other output_size
	static const unsigned o_pad_h = Sh-1;

    static const unsigned Wout = (Win-1)*Sw-2*pad_w+Kw+o_pad_w;
    static const unsigned Hout = (Hin-1)*Sh-2*pad_h+Kh+o_pad_h;


	//for padding and output_padding
	//default padding_size if padding==1: pad_h=(Kh-1)/2,pad_w=(Kw-1)/2
	//default output_padding: output_padding_h=Sh-1, output_padding_w=Sw-1

	//with default padding and output_padding
	//Hout=(Hin-1)*Sh-2*pad_h+Kh+output_padding_h=(Hin-1)*Sh-2*((Kh-1)/2)+Kh+Sh-1=Hin*Sh-Sh-Kh+1+Kh+Sh-1=Hin*Sh;
	//Wout=(Win-1)*Sw-2*pad_w+Kw+output_padding_w=(Win-1)*Sw-2*((Kw-1)/2)+Kw+Sw-1=Win*Sw-Sw-Kw+1+Kw+Sw-1=Win*Sw;
	//It is output_size/input_size=strides.

};


/*
	Use for Pytorch Transpose Conv2d in "conv backward calculate".
	Feature in shape: (CHin,Hin,Win) == [CHin][Hin][Win]
	Kernel shape: (CHin, CHout,kH,kW) == [CHin][CHout][Kh][Kw]   ATTENTION!!!这里和Conv2d不同，这里输入通道数在前
	Feature out shape:(CHout,Hout,Wout​) == [CHout][Hout][Wout]   ATTENTION!!!需要提前全部置零
	Bias shape: (CHout,) == [CHout]
	Feature_out[CHout][Hout][Wout] = Feature_in[CHin][Hin][Win] * W[CHin][CHout][Kh][Kw] + bias[CHout]
*/
template <typename CONFIG_C>
void ConvTranspose2d_cb(
		typename CONFIG_C::Dtype_f feature_in[],
		typename CONFIG_C::Dtype_w W[],
		typename CONFIG_C::Dtype_b bias[],
		typename CONFIG_C::Dtype_f feature_out[]
	)
{
	for(int cout_n=0;cout_n<CONFIG_C::CHout;cout_n++)
	{
		for(int i=0;i<CONFIG_C::Hin;i++)
		{
			for(int j=0;j<CONFIG_C::Win;j++)
			{
				AddKernels:
				for(int ii=0;ii<CONFIG_C::Kh;ii++)
				{
					for(int jj=0;jj<CONFIG_C::Kw;jj++)
					{	
						typename CONFIG_C::Index_t h=i*CONFIG_C::Sh-CONFIG_C::pad_h+ii;
						typename CONFIG_C::Index_t w=j*CONFIG_C::Sw-CONFIG_C::pad_w+jj;
						if(h>=0 && w>=0 && h<CONFIG_C::Hout && w<CONFIG_C::Wout)
						{
							InputChannels:
							for (int cin_n=0;cin_n<CONFIG_C::CHin;cin_n++)
							{ 
								//feature_out[cout_n][i*Sh+ii][j*Sw+jj] += SUM W[cin_n][cout_n][ii][jj] * feature_in[cin_n][Hin][Win]
								feature_out[cout_n*CONFIG_C::Hout*CONFIG_C::Wout+h*CONFIG_C::Wout+w]
									+=W[cin_n*CONFIG_C::CHout*CONFIG_C::Kh*CONFIG_C::Kw+cout_n*CONFIG_C::Kh*CONFIG_C::Kw+ii*CONFIG_C::Kw+jj]
									*feature_in[cin_n*CONFIG_C::Hin*CONFIG_C::Win+i*CONFIG_C::Win+j];
							}
						}
					}
				}
			}
		}
		for(int ll=0;ll<CONFIG_C::Hout*CONFIG_C::Wout;ll++){
			feature_out[cout_n*CONFIG_C::Hout*CONFIG_C::Wout+ll]
				=(feature_out[cout_n*CONFIG_C::Hout*CONFIG_C::Wout+ll]+bias[cout_n])/CONFIG_C::Delta_w;
			if(CONFIG_C::relu_en && feature_out[cout_n*CONFIG_C::Hout*CONFIG_C::Wout+ll]<0){
				feature_out[cout_n*CONFIG_C::Hout*CONFIG_C::Wout+ll]=0;
			}
		}
	}
};




//void ConvTranspose2d_pp 还没测试好


};
#endif // NET_CONV_H_INCLUDED


