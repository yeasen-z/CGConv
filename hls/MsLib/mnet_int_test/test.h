#include "../mnet_int/mnet_int_top.h"
#include "../mnet/mnet_top.h"
#include "cfg.h"

#include <cstdlib>
#include <iostream>
#include "math.h"

template<class in_put>
void InitialRandom(in_put arra[],int lsize){
	for(int i=0;i<lsize;i++){
		arra[i]=rand() % (99999 +1) / (float)(99999 +1);
	}
}

template<class in_put>
void InitialZero(in_put arra[],int lsize){
	for(int i=0;i<lsize;i++){
		arra[i]=0;
	}
}

void test_conv_int(){
    srand(1);
    typename convcfg::Dtype_f feature_in[convcfg::CHin][convcfg::Hin][convcfg::Win];
	typename convcfg::Dtype_w W[convcfg::CHout][convcfg::CHin][convcfg::Kh][convcfg::Kw];
	typename convcfg::Dtype_b bias[convcfg::CHout];
	typename convcfg::Dtype_f feature_out[convcfg::CHout][convcfg::Hout][convcfg::Wout];

    typename convintcfg::Dtype_f feature_in_int[convintcfg::CHin][convintcfg::Hin][convintcfg::Win];
	typename convintcfg::Dtype_w W_int[convintcfg::CHout][convintcfg::CHin][convintcfg::Kh][convintcfg::Kw];
	typename convintcfg::Dtype_b bias_int[convintcfg::CHout];
	typename convintcfg::Dtype_f feature_out_int[convintcfg::CHout][convintcfg::Hout][convintcfg::Wout];


	for(int i=0;i<convcfg::Hin;i++)
		for(int j=0;j<convcfg::Win;j++)
			for(int cin_n=0;cin_n<convcfg::CHin;cin_n++){
				feature_in[cin_n][i][j]=rand() % (99999 +1) / (float)(99999 +1);
                feature_in_int[cin_n][i][j]=feature_in[cin_n][i][j]*convintcfg::Delta_i;
				// std::cout<<feature_in[cin_n][i][j]<<std::endl;
				}

	for(int i=0;i<convcfg::Kh;i++)
		for(int j=0;j<convcfg::Kw;j++)
			for(int cin_n=0;cin_n<convcfg::CHin;cin_n++)
				for(int cout_n=0;cout_n<convcfg::CHout;cout_n++){
					W[cout_n][cin_n][i][j]=rand()% (99999 +1) / (float)(99999 +1);
                    W_int[cout_n][cin_n][i][j]=W[cout_n][cin_n][i][j]*convintcfg::Delta_w;
                    //std::cout<<W[i][j][cin_n][cout_n]<<std::endl;
                    }
	for(int cout_n=0;cout_n<convcfg::CHout;cout_n++){
		bias[cout_n]=rand()% (99999 +1) / (float)(99999 +1);
        bias_int[cout_n]=bias[cout_n]*convintcfg::Delta_b;
		//std::cout<<bias[cout_n]<<std::endl;
	}

	printf("OK\n");

	mnet::Conv2d<convcfg>(feature_in[0][0],W[0][0][0],bias,feature_out[0][0]);
    mnet_int::Conv2d<convintcfg>(feature_in_int[0][0],W_int[0][0][0],bias_int,feature_out_int[0][0]);

    float avg_err=0;
	for(int i=0;i<convcfg::Hout;i++)
		for(int j=0;j<convcfg::Wout;j++)
			for(int cout_n=0;cout_n<convcfg::CHout;cout_n++)
			{
			    // std::cout<<i<<" "<<j<<" "<<cout_n<<" FP32:"<<feature_out[cout_n][i][j]
                // <<" INT32:"<<feature_out_int[cout_n][i][j]<<" DeQuant:"<<feature_out_int[cout_n][i][j]/float(convintcfg::Delta_i)<<std::endl;
                avg_err+=abs(feature_out[cout_n][i][j]-feature_out_int[cout_n][i][j]/float(convintcfg::Delta_i));
			}

    std::cout<<"avg_err:"<<avg_err/float(convcfg::Hout*convcfg::Wout*convcfg::CHout)<<std::endl;

};

void test_transconv_int(){
    srand(0);
    typename transconvcfg::Dtype_f feature_in[transconvcfg::CHin][transconvcfg::Hin][transconvcfg::Win];
	typename transconvcfg::Dtype_w W[transconvcfg::CHout][transconvcfg::CHin][transconvcfg::Kh][transconvcfg::Kw];
	typename transconvcfg::Dtype_b bias[transconvcfg::CHout];
	typename transconvcfg::Dtype_f feature_out[transconvcfg::CHout][transconvcfg::Hout][transconvcfg::Wout];

    typename int_transconvcfg::Dtype_f feature_in_int[int_transconvcfg::CHin][int_transconvcfg::Hin][int_transconvcfg::Win];
	typename int_transconvcfg::Dtype_w W_int[int_transconvcfg::CHout][int_transconvcfg::CHin][int_transconvcfg::Kh][int_transconvcfg::Kw];
	typename int_transconvcfg::Dtype_b bias_int[int_transconvcfg::CHout];
	typename int_transconvcfg::Dtype_f feature_out_int[int_transconvcfg::CHout][int_transconvcfg::Hout][int_transconvcfg::Wout];

	for(int i=0;i<transconvcfg::Hin;i++)
		for(int j=0;j<transconvcfg::Win;j++)
			for(int cin_n=0;cin_n<transconvcfg::CHin;cin_n++){
				feature_in[cin_n][i][j]=rand() % (999 +1) / (float)(999 +1);
                feature_in_int[cin_n][i][j]=feature_in[cin_n][i][j]*int_transconvcfg::Delta_i;
				// std::cout<<feature_in[cin_n][i][j]<<std::endl;
				}

	for(int i=0;i<transconvcfg::Kh;i++)
		for(int j=0;j<transconvcfg::Kw;j++)
			for(int cin_n=0;cin_n<transconvcfg::CHin;cin_n++)
				for(int cout_n=0;cout_n<transconvcfg::CHout;cout_n++){
					W[cout_n][cin_n][i][j]=rand()% (999 +1) / (float)(999 +1);
                    W_int[cout_n][cin_n][i][j]=W[cout_n][cin_n][i][j]*int_transconvcfg::Delta_w;
                    //std::cout<<W[i][j][cin_n][cout_n]<<std::endl;
                    }
	for(int cout_n=0;cout_n<transconvcfg::CHout;cout_n++){
		bias[cout_n]=rand()% (999 +1) / (float)(999 +1);
        bias_int[cout_n]=bias[cout_n]*int_transconvcfg::Delta_b;
		// std::cout<<bias_int[cout_n]<<std::endl;
	}

	for(int i=0;i<transconvcfg::Hout;i++)
		for(int j=0;j<transconvcfg::Wout;j++)
			for(int cout_n=0;cout_n<transconvcfg::CHout;cout_n++){
				feature_out[cout_n][i][j]=0;
                feature_out_int[cout_n][i][j]=0;
				// std::cout<<feature_in[cin_n][i][j]<<std::endl;
				}
	
	mnet::ConvTranspose2d_cb<transconvcfg>(feature_in[0][0],W[0][0][0],bias,feature_out[0][0]);

    mnet_int::ConvTranspose2d_cb<int_transconvcfg>(feature_in_int[0][0],W_int[0][0][0],bias_int,feature_out_int[0][0]);

    double avg_err=0;
    for(int i=0;i<transconvcfg::Hout;i++)
        for(int j=0;j<transconvcfg::Wout;j++)
            for(int cout_n=0;cout_n<transconvcfg::CHout;cout_n++)
            {
                double err=abs(feature_out[cout_n][i][j]-feature_out_int[cout_n][i][j]/double(int_transconvcfg::Delta_i));
                std::cout<<i<<" "<<j<<" "<<cout_n<<" FP64:"<<feature_out[cout_n][i][j]
                <<" INT32:"<<feature_out_int[cout_n][i][j]
                <<" DeQuant:"<<feature_out_int[cout_n][i][j]/float(int_transconvcfg::Delta_i)
                <<" Err:"<<err
                <<std::endl;
                avg_err+=err;
                if(err<1e-10||err>1e10){
                    break;
                }
            }
    std::cout<<"avg_err:"<<avg_err/(int_transconvcfg::Hout*int_transconvcfg::Wout*int_transconvcfg::CHout)<<std::endl;

};

void test_dws_int(){

	typename dwscfg::DW::Dtype_w W_DW[dwscfg::DW::CHin][1][dwscfg::DW::Kh][dwscfg::DW::Kw];
	typename dwscfg::DW::Dtype_b bias_DW[dwscfg::DW::CHin];
	typename dwscfg::DW::Dtype_f feature_in[dwscfg::DW::CHin][dwscfg::DW::Hin][dwscfg::DW::Win];
	typename dwscfg::PW::Dtype_w W_PW[dwscfg::PW::CHin][dwscfg::PW::CHout][1][1];
	typename dwscfg::PW::Dtype_b bias_PW[dwscfg::PW::CHout];

	typename int_dwscfg::DW::Dtype_w W_DW_int[int_dwscfg::DW::CHin][1][int_dwscfg::DW::Kh][int_dwscfg::DW::Kw];
	typename int_dwscfg::DW::Dtype_b bias_DW_int[int_dwscfg::DW::CHin];
	typename int_dwscfg::DW::Dtype_f feature_in_int[int_dwscfg::DW::CHin][int_dwscfg::DW::Hin][int_dwscfg::DW::Win];
	typename int_dwscfg::PW::Dtype_w W_PW_int[int_dwscfg::PW::CHin][1][1][1];
	typename int_dwscfg::PW::Dtype_b bias_PW_int[int_dwscfg::PW::CHout];

	InitialRandom<dwscfg::DW::Dtype_w>(W_DW[0][0][0],dwscfg::DW::CHin*1*dwscfg::DW::Kh*dwscfg::DW::Kw);
	InitialRandom<dwscfg::DW::Dtype_b>(bias_DW,dwscfg::DW::CHin);
	InitialRandom<dwscfg::DW::Dtype_f>(feature_in[0][0],dwscfg::PW::CHin*dwscfg::DW::Hin*dwscfg::DW::Win);
	InitialRandom<dwscfg::PW::Dtype_w>(W_PW[0][0][0],dwscfg::PW::CHin*dwscfg::PW::CHout);
	InitialRandom<dwscfg::PW::Dtype_b>(bias_PW,dwscfg::PW::CHout);

	// InitialRandom<int_dwscfg::DW::Dtype_w>(W_DW_int[0][0][0],int_dwscfg::DW::CHin*1*int_dwscfg::DW::Kh*int_dwscfg::DW::Kw);
	// InitialRandom<int_dwscfg::DW::Dtype_b>(bias_DW_int,int_dwscfg::DW::CHin);
	// InitialRandom<int_dwscfg::DW::Dtype_f>(feature_in_int[0][0],int_dwscfg::PW::CHin*int_dwscfg::DW::Hin*int_dwscfg::DW::Win);
	// InitialRandom<int_dwscfg::PW::Dtype_w>(W_PW_int[0][0][0],int_dwscfg::PW::CHin*int_dwscfg::PW::CHout);
	// InitialRandom<int_dwscfg::PW::Dtype_b>(bias_PW_int,int_dwscfg::PW::CHout);
	
	typename dwscfg::PW::Dtype_f feature_out[dwscfg::PW::CHout][dwscfg::PW::Hout][dwscfg::PW::Wout];
	typename dwscfg::PW::Dtype_f feature_out1[dwscfg::PW::CHout][dwscfg::PW::Hout][dwscfg::PW::Wout];

	InitialZero<dwscfg::PW::Dtype_f>(feature_out[0][0],dwscfg::PW::CHout*dwscfg::PW::Hout*dwscfg::PW::Wout);
	InitialZero<dwscfg::PW::Dtype_f>(feature_out1[0][0],dwscfg::PW::CHout*dwscfg::PW::Hout*dwscfg::PW::Wout);

	mnet::DWSConv2d<dwscfg::DW, dwscfg::PW>(feature_in[0][0],W_DW[0][0][0],bias_DW,W_PW[0][0][0],bias_PW,feature_out[0][0]);
  	mnet::DWSConv2d_NoDWBuffer<dwscfg::DW, dwscfg::PW>(feature_in[0][0],W_DW[0][0][0],bias_DW,W_PW[0][0][0],bias_PW,feature_out1[0][0]);

	// for(int i=0;i<dwscfg::PW::Hout;i++){
	// 	for(int j=0;j<dwscfg::PW::Wout;j++){
	// 		for(int cout_n=0;cout_n<dwscfg::PW::CHout;cout_n++)
	// 		{
	// 		    std::cout<<i<<" "<<j<<" "<<cout_n<<" RAW:"<<feature_out[cout_n][i][j]<<" NoBuffer:"<<feature_out1[cout_n][i][j];
    //   		}
    //   		std::cout<<std::endl;
    // 	}
  	// }


	for(int i=0;i<int_dwscfg::DW::CHin*1*int_dwscfg::DW::Kh*int_dwscfg::DW::Kw;i++){W_DW_int[0][0][0][i]=int(W_DW[0][0][0][i]*int_dwscfg::DW::Delta_w);}
	for(int i=0;i<int_dwscfg::DW::CHin;i++){bias_DW_int[i]=int(bias_DW[i]*int_dwscfg::DW::Delta_b);}
	for(int i=0;i<int_dwscfg::PW::CHin*int_dwscfg::PW::CHout*1*1;i++){W_PW_int[0][0][0][i]=int(W_PW[0][0][0][i]*int_dwscfg::PW::Delta_w);}
	for(int i=0;i<int_dwscfg::PW::CHout;i++){bias_PW_int[i]=int(bias_PW[i]*int_dwscfg::PW::Delta_b);}
	for(int i=0;i<int_dwscfg::DW::CHin*int_dwscfg::DW::Hin*int_dwscfg::DW::Win;i++){feature_in_int[0][0][i]=int(feature_in[0][0][i]*int_dwscfg::DW::Delta_i);}

	typename int_dwscfg::PW::Dtype_f feature_out_int[int_dwscfg::PW::CHout][int_dwscfg::PW::Hout][int_dwscfg::PW::Wout];
	typename int_dwscfg::PW::Dtype_f feature_out1_int[int_dwscfg::PW::CHout][int_dwscfg::PW::Hout][int_dwscfg::PW::Wout];

	InitialZero<int_dwscfg::PW::Dtype_f>(feature_out_int[0][0],int_dwscfg::PW::CHout*int_dwscfg::PW::Hout*int_dwscfg::PW::Wout);
	InitialZero<int_dwscfg::PW::Dtype_f>(feature_out1_int[0][0],int_dwscfg::PW::CHout*int_dwscfg::PW::Hout*int_dwscfg::PW::Wout);

	mnet_int::DWSConv2d<int_dwscfg::DW, int_dwscfg::PW>(feature_in_int[0][0],W_DW_int[0][0][0],bias_DW_int,W_PW_int[0][0][0],bias_PW_int,feature_out_int[0][0]);
  	mnet_int::DWSConv2d_NoDWBuffer<int_dwscfg::DW, int_dwscfg::PW>(feature_in_int[0][0],W_DW_int[0][0][0],bias_DW_int,W_PW_int[0][0][0],bias_PW_int,feature_out1_int[0][0]);

	float err_Rawint=0;
	float err_NBint=0;
	int flg=0;
	for(int i=0;i<dwscfg::PW::Hout;i++){
		for(int j=0;j<dwscfg::PW::Wout;j++){
			int show_cout_n=0;
			std::cout<<i<<" "<<j<<" "<<show_cout_n<<std::endl
			<<" RAW:"<<feature_out[show_cout_n][i][j]<<" NoBuffer:"<<feature_out1[show_cout_n][i][j]<<std::endl
			<<" RAWINT:"<<feature_out_int[show_cout_n][i][j]
			<<" RAWFP:"<<feature_out_int[show_cout_n][i][j]/float(int_dwscfg::DW::Delta_i)
			<<" NoBufferINT:"<<feature_out1_int[show_cout_n][i][j]
			<<" NoBufferFP:"<<feature_out1_int[show_cout_n][i][j]/float(int_dwscfg::DW::Delta_i)<<std::endl;
			for(int cout_n=0;cout_n<dwscfg::PW::CHout;cout_n++)
			{
				flg++;
				err_Rawint+=abs(feature_out_int[cout_n][i][j]/float(int_dwscfg::DW::Delta_i)-feature_out[cout_n][i][j]);
				err_NBint+=abs(feature_out1_int[cout_n][i][j]/float(int_dwscfg::DW::Delta_i)-feature_out[cout_n][i][j]);
     		}
    	}
  	}
	std::cout<<"  RAW-INT-ERR:"<<err_Rawint/flg<<"  NB-INT-ERR:"<<err_NBint/flg<<std::endl;
};
