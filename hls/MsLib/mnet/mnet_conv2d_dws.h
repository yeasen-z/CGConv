//Depthwise Separable Convolution 
//compare to the traditional convolution, deepwise convolution has 2 steps:deepwise convolution and pointwise convolution
//the deepwise convolution is the same as the traditional convolution, and groups==input_channels
//the pointwise convolution is the convolution with 1x1 kernel.

#ifndef __MNET_DWSCONV_H__
#define __MNET_DWSCONV_H__

namespace mnet{

struct DWSconv2d0_config{
    struct base{
        static const unsigned CHin=3;
        static const unsigned Hin =6;
        static const unsigned Win =6;
        static const unsigned Kw =5;
        static const unsigned Kh =5;
        static const unsigned Sw =1;
        static const unsigned Sh =1;
        static const unsigned CHout=1;
        static const bool padding =0;
        static const bool relu_en=0;
    };

    struct DW{
        typedef float  Dtype_f;
        typedef float  Dtype_w;
        typedef float  Dtype_b;
        typedef int Index_t;
        typedef float Dtype_tmp;

        static const bool padding =base::padding;

        static const unsigned CHin=base::CHin;
        static const unsigned Hin =base::Hin;
        static const unsigned Win =base::Win;
        static const unsigned Kw =base::Kw;
        static const unsigned Kh =base::Kh;
        static const unsigned Sw =base::Sw;
        static const unsigned Sh =base::Sh;

        //settings defined by the values of up above
        static const unsigned CHout=CHin; 
        //这个注释掉是因为，减少一个自主设置的变量能使调用更加容易
        static const unsigned pad_w= (padding==0?0:(Kw-1)/2);
        static const unsigned pad_h= (padding==0?0:(Kh-1)/2);
        static const unsigned Wout=(Win+2*pad_w-Kw)/Sw+1;
        static const unsigned Hout=(Hin+2*pad_h-Kh)/Sh+1;
    };

    struct PW{
        typedef float  Dtype_f;
        typedef float  Dtype_w;
        typedef float  Dtype_b;
        typedef int Index_t;
        typedef float Dtype_tmp;

        static const bool relu_en=base::relu_en; 

        static const unsigned CHin= base::CHin;
        static const unsigned Hin = DW::Hout;
        static const unsigned Win = DW::Wout;

        static const unsigned CHout=base::CHout; 
        static const unsigned Wout=DW::Wout;
        static const unsigned Hout=DW::Hout;
    };
};

/*
DeepWise Conv
Chin = Chout
Feature in shape: (CHin, Hin, Win) == [CHin][Hin][Win]
Kernel shape: (CHout, 1, kH, kW) == [CHout][1][Kh][Kw]
Feature out shape:(CHout, Hout, Wout​) == [CHout][Hout][Wout]
Bias shape: (CHout, ) == [CHout]
Feature_out[group_i][hout][wout] = SUM: Feature_in[group_i][hin][win] * W[group_i][0][Kh][Kw] + bias[group_i] 
*/
template <typename CONFIG_C>
void DWConv2d(
		typename CONFIG_C::Dtype_f feature_in[],
		typename CONFIG_C::Dtype_w W[],
		typename CONFIG_C::Dtype_b bias[],
		typename CONFIG_C::Dtype_f feature_out[]
	)
{
    for(int cout_n=0;cout_n<CONFIG_C::CHin;cout_n++) 
    //这里使用CHin=CHout的条件，能够降低参数传递的复杂度，和cout_n<CONFIG_C::CHout等价
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
                            // Feature in shape: (CHin, Hin, Win) == [CHin][Hin][Win]
                            // Kernel shape: (CHout, 1, kH, kW) == [CHout][1][Kh][Kw]
                            // Feature out shape:(CHout, Hout, Wout​) == [CHout][Hout][Wout]
                            // Bias shape: (CHout, ) == [CHout]
                            // Feature_out[group_i][hout][wout] = SUM: Feature_in[group_i][hin][win] * W[group_i][0][Kh][Kw] + bias[group_i] 
                            typename CONFIG_C::Dtype_tmp tp=
                            feature_in[cout_n*CONFIG_C::Hin*CONFIG_C::Win+h*CONFIG_C::Win+w]*
                            W[cout_n* 1 *CONFIG_C::Kh*CONFIG_C::Kw+ 0 +ii*CONFIG_C::Kw+jj];
                            sum+=tp;
						
						}
					}
				}
				sum+=(typename CONFIG_C::Dtype_tmp)bias[cout_n];
				//Feature out shape:(CHout,Hout,Wout​) == [CHout][Hout][Wout]
				feature_out[cout_n*CONFIG_C::Hout*CONFIG_C::Wout+i*CONFIG_C::Wout+j]=(typename CONFIG_C::Dtype_f)sum;
			}
		}
	}
};


/*
PWConv2d is conv1x1
Feature in shape: (CHin,Hin,Win) == [CHin][Hin][Win]
Kernel shape: (CHout, CHin, 1 , 1) == [CHout][CHin][1][1]
Feature out shape:(CHout,Hout,Wout​) == [CHout][Hout][Wout]
Feature_out[CHout][Hout][Wout] = Feature_in[CHin][Hin][Win] * W[CHout][CHin][Kh][Kw] + bias[CHout]
*/
template <typename CONFIG_C>
void PWConv2d(
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

                for(int cin_n=0;cin_n<CONFIG_C::CHin;cin_n++)
                {
                    //Feature in shape: (CHin,Hin,Win) == [CHin][Hin][Win]
                    //Kernel shape: (CHout, CHin,kH,kW) == [CHout][CHin][1][1]
                    //Feature out shape:(CHout,Hout,Wout​) == [CHout][Hout][Wout]
                    //Feature_out[CHout][Hout][Wout] = Feature_in[CHin][Hin][Win] * W[CHout][CHin][Kh][Kw] + bias[CHout]
                    typename CONFIG_C::Dtype_tmp tp=
                    feature_in[cin_n*CONFIG_C::Hin*CONFIG_C::Win+i*CONFIG_C::Win+j]*W[cout_n*CONFIG_C::CHin  +  cin_n ];
                    sum+=tp;
                }
                
				sum+=(typename CONFIG_C::Dtype_tmp)bias[cout_n];
				if(CONFIG_C::relu_en && sum<0)
					sum=0;
				//Feature out shape:(CHout,Hout,Wout​) == [CHout][Hout][Wout]
				feature_out[cout_n*CONFIG_C::Hout*CONFIG_C::Wout+i*CONFIG_C::Wout+j]
                =(typename CONFIG_C::Dtype_f)sum;
			}
		}
	}
};


/*
    DeepWise Seperable Convolution==DeepWise conv + PiontWise conv
    PointWise is conv_1x1
    CONFIG_C is total, CONFIG_P is for PW
*/
template <typename CONFIG_C, typename CONFIG_P>
void DWSConv2d(
		typename CONFIG_C::Dtype_f feature_in[],
		typename CONFIG_C::Dtype_w W_DW[],
		typename CONFIG_C::Dtype_b bias_DW[],
        typename CONFIG_C::Dtype_w W_PW[],
		typename CONFIG_C::Dtype_b bias_PW[],
		typename CONFIG_C::Dtype_f feature_out[]
	)
{
    //设置config中的参数
    typename CONFIG_C::Dtype_f DeepWiseConv_out[CONFIG_C::CHin*CONFIG_C::Hout*CONFIG_C::Wout]={0};
    for (int i=0;i<CONFIG_C::CHin*CONFIG_C::Hout*CONFIG_C::Wout;i++){ DeepWiseConv_out[i]=0; }
    // std::cout<<"W_DW:"<<W_DW<<std::endl;
    DeepWiseConv:
    DWConv2d<CONFIG_C>(feature_in,W_DW,bias_DW,DeepWiseConv_out);
    
    PointWiseConv:
    PWConv2d<CONFIG_P>(DeepWiseConv_out,W_PW,bias_PW,feature_out);
};

/*
In this, 'DeepWiseConv_out' in DWSConv2d
*/
template <typename CONFIG_C, typename CONFIG_P>
void DWSConv2d_NoDWBuffer(
		typename CONFIG_C::Dtype_f feature_in[],
		typename CONFIG_C::Dtype_w W_DW[],
		typename CONFIG_C::Dtype_b bias_DW[],
        typename CONFIG_P::Dtype_w W_PW[],
		typename CONFIG_P::Dtype_b bias_PW[],
		typename CONFIG_P::Dtype_f feature_out[]
    )
{
    for(int i=0;i<CONFIG_C::Hout;i++)
    {
        for(int j=0;j<CONFIG_C::Wout;j++)
        {
            typename CONFIG_C::Dtype_tmp DW_out_ij[CONFIG_C::CHin]={0};
            for(int cin_n=0;cin_n<CONFIG_C::CHin;cin_n++){DW_out_ij[cin_n]=bias_DW[cin_n];}

            //here get the output of Hout,Wout,
            for(int ii=0;ii<CONFIG_C::Kh;ii++)
            {
                for(int jj=0;jj<CONFIG_C::Kw;jj++)
                {
                    typename CONFIG_C::Index_t h=i*CONFIG_C::Sh-CONFIG_C::pad_h+ii;
                    typename CONFIG_C::Index_t w=j*CONFIG_C::Sw-CONFIG_C::pad_w+jj;
                    if(h>=0 && w>=0 && h<CONFIG_C::Hin && w<CONFIG_C::Win)
                    {
                        //for each channel
                        for (int cin_n=0;cin_n<CONFIG_C::CHin;cin_n++)
                        {
                            DW_out_ij[cin_n]+=
                            feature_in[cin_n*CONFIG_C::Hin*CONFIG_C::Win+h*CONFIG_C::Win+w]*
                            W_DW[cin_n*1*CONFIG_C::Kh*CONFIG_C::Kw+ 0 +ii*CONFIG_C::Kw+jj];
                        }
                    }
                }
            }

            //then we compute the PW output
            for(int cout_n=0;cout_n<CONFIG_P::CHout;cout_n++){
                feature_out[cout_n*CONFIG_P::Hout*CONFIG_P::Wout+i*CONFIG_P::Wout+j]=bias_PW[cout_n];
                for (int cin_n=0;cin_n<CONFIG_P::CHin;cin_n++){
                    feature_out[cout_n*CONFIG_P::Hout*CONFIG_P::Wout+i*CONFIG_P::Wout+j]
                    +=DW_out_ij[cin_n]*W_PW[cout_n*CONFIG_P::CHin  +  cin_n];
                }

                if(CONFIG_P::relu_en && feature_out[cout_n*CONFIG_P::Hout*CONFIG_P::Wout+i*CONFIG_P::Wout+j]<0){
                    feature_out[cout_n*CONFIG_P::Hout*CONFIG_P::Wout+i*CONFIG_P::Wout+j]=0;
                }
            }
        }
    }
};



};
#endif // __MNET_DPWCONV_H__