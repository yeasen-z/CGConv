// This file is used to define the POOL2D layer in Pytorch
#ifndef MNET_POOL2D_H_INCLUDED
#define MNET_POOL2D_H_INCLUDED

namespace mnet{

struct pool2d0_config{
    typedef float  Dtype_f;
    typedef int Index_t;

    static const unsigned CHin=1;
    static const unsigned Hin =5;
    static const unsigned Win =5;
    static const unsigned Kw =3;
    static const unsigned Kh =3;
	static const unsigned mode =0;

    static const unsigned Sw =2;
    static const unsigned Sh =2;
    static const bool padding=0;

	//settings defined by the values of up above
    static const unsigned pad_w= (padding==0?0:(Kw-1)/2);
    static const unsigned pad_h= (padding==0?0:(Kh-1)/2);
    
    static const unsigned Wout=(Win+2*pad_w-Kw)/Sw+1;
    static const unsigned Hout=(Hin+2*pad_h-Kh)/Sh+1;
};

/*
	Use for Pytorch Pool2d without padding
	Feature in shape: (CHin,Hin,Win) == [CHin][Hin][Win]
	Kernel shape: (kH,kW) == [Kh][Kw]
	Feature out shape:(CHin,Hout,Wout​) == [CHin][Hout][Wout]
	Pooling mode: 0:MEAN, 1:MIN, 2:MAX
*/
template <typename CONFIG_P>
void Pool2d(
	typename CONFIG_P::Dtype_f feature_in[],
	typename CONFIG_P::Dtype_f feature_out[]
	)
{
	for(int c=0;c<CONFIG_P::CHin;c++)
		for(int i=0;i<CONFIG_P::Hout;i++)
			for(int j=0;j<CONFIG_P::Wout;j++)
			{
				typename CONFIG_P::Dtype_f sum;
				if(CONFIG_P::mode==0)
					sum=0;
				else
					if(CONFIG_P::mode==1)
						sum=99999999999999999;
					else
						sum=-99999999999999999;
				for(int ii=0;ii<CONFIG_P::Kh;ii++)
					for(int jj=0;jj<CONFIG_P::Kw;jj++)
					{
						// typename CONFIG_P::Index_t h=i*CONFIG_P::Kh+ii;
						typename CONFIG_P::Index_t h=i*CONFIG_P::Sh-CONFIG_P::pad_h+ii;
						// typename CONFIG_P::Index_t w=j*CONFIG_P::Kw+jj;
						typename CONFIG_P::Index_t w=j*CONFIG_P::Sw-CONFIG_P::pad_w+jj;
						if(h>=0 && w>=0 && h<CONFIG_P::Hin && w<CONFIG_P::Win)
						{
							switch(CONFIG_P::mode)
							{
								case 0:{
									// Feature in shape: (CHin,Hin,Win) == [CHin][Hin][Win]
									sum+=feature_in[c*CONFIG_P::Hin*CONFIG_P::Win+h*CONFIG_P::Win+w];
									break;}
								case 1:{
									// Feature in shape: (CHin,Hin,Win) == [CHin][Hin][Win]
									if (feature_in[c*CONFIG_P::Hin*CONFIG_P::Win+h*CONFIG_P::Win+w]<sum){
										sum=feature_in[c*CONFIG_P::Hin*CONFIG_P::Win+h*CONFIG_P::Win+w];
									}
									// sum=mmath::min_manual<typename CONFIG_P::Dtype_f,typename CONFIG_P::Dtype_f>
									// (sum,feature_in[c*CONFIG_P::Hin*CONFIG_P::Win+h*CONFIG_P::Win+w]);
									break;}
								case 2:{
									// Feature in shape: (CHin,Hin,Win) == [CHin][Hin][Win]
									if (feature_in[c*CONFIG_P::Hin*CONFIG_P::Win+h*CONFIG_P::Win+w]>sum){
										sum=feature_in[c*CONFIG_P::Hin*CONFIG_P::Win+h*CONFIG_P::Win+w];
									}
									// sum=mmath::max_manual<typename CONFIG_P::Dtype_f,typename CONFIG_P::Dtype_f>
									// (sum,feature_in[c*CONFIG_P::Hin*CONFIG_P::Win+h*CONFIG_P::Win+w]);
									break;}
								default:break;
							}
						}
					}
				if(CONFIG_P::mode==0)
					sum=sum/(CONFIG_P::Kw*CONFIG_P::Kh);
				// Feature out shape:(CHout,Hout,Wout​) == [CHout][Hout][Wout]
				feature_out[c*CONFIG_P::Hout*CONFIG_P::Wout+i*CONFIG_P::Wout+j]=sum;
			}
};

}; // namespace mnet

#endif // NET_POOL_H_INCLUDED
