#ifndef MNET_TEST_H_INCLUDED
#define MNET_TEST_H_INCLUDED
// funcs for lib test

#include "mnet_test_config.h"
#include "data/transpose_conv_test_w.h"
#include "data/gdn_gamma_test.h"
#include "data/gdn_beta_test.h"

#include <fstream>
#include <iostream>


// exp
int test_exp()
{
    typename exp_config::input_t x[]={0.5,1,1.5};
    typename exp_config::output_t y[]={0,0,0};
    mmath::exp_Taylor<exp_config::input_t,exp_config::output_t,exp_config>(x,y);
    std::cout<<y[0]<<" "<<y[1]<<" "<<y[2]<<std::endl;
    return 0;
};

//pool_core
int test_pool()
{
    //test pool_core
	typename pool_config::Dtype_f feature_in[pool_config::CHin][pool_config::Hin][pool_config::Win];
	typename pool_config::Dtype_f feature_out[pool_config::CHin][pool_config::Hout][pool_config::Wout];

	for(int i=0;i<pool_config::Hin;i++)
		for(int j=0;j<pool_config::Win;j++)
			for(int cin_c=0;cin_c<pool_config::CHin;cin_c++){
				feature_in[cin_c][i][j]=cin_c*pool_config::Hin*pool_config::Win+i*pool_config::Win+j;
			}
    for(int i=0;i<pool_config::Hout;i++)
		for(int j=0;j<pool_config::Wout;j++)
			for(int cin_c=0;cin_c<pool_config::CHin;cin_c++){
				feature_out[cin_c][i][j]=0;
			}
	mnet::Pool2d<pool_config>(
			feature_in[0][0],feature_out[0][0]
		);//mode: 0:MEAN, 1:MIN, 2:MAX

	for(int i=0;i<pool_config::Hout;i++){
		for(int j=0;j<pool_config::Wout;j++){
			for(int cout_c=0;cout_c<pool_config::CHin;cout_c++)
			{
				std::cout<<i<<" "<<j<<" "<<cout_c<<" "<<feature_out[cout_c][i][j]<<std::endl;
			}
		}
	}
	return 0;
};

//conv_core
int test_conv(){
  typename conv_config::Dtype_f feature_in[conv_config::CHin][conv_config::Hin][conv_config::Win];
	typename conv_config::Dtype_w W[conv_config::CHout][conv_config::CHin][conv_config::Kh][conv_config::Kw];
	typename conv_config::Dtype_b bias[conv_config::CHout];
	typename conv_config::Dtype_f feature_out[conv_config::CHout][conv_config::Hout][conv_config::Wout];

	for(int i=0;i<conv_config::Hin;i++)
		for(int j=0;j<conv_config::Win;j++)
			for(int cin_n=0;cin_n<conv_config::CHin;cin_n++){
				feature_in[cin_n][i][j]=(cin_n*conv_config::Hin*conv_config::Win+i*conv_config::Win+j);
				//std::cout<<feature_in[i][j][cin_n]<<std::endl;
				}

	for(int i=0;i<conv_config::Kh;i++)
		for(int j=0;j<conv_config::Kw;j++)
			for(int cin_n=0;cin_n<conv_config::CHin;cin_n++)
				for(int cout_n=0;cout_n<conv_config::CHout;cout_n++){
					W[cout_n][cin_n][i][j]=(cout_n*conv_config::CHin*conv_config::Kh*conv_config::Kw+cin_n*conv_config::Kh*conv_config::Kw*conv_config::CHout+i*conv_config::Kw+j);
                    //std::cout<<W[i][j][cin_n][cout_n]<<std::endl;
                    }
	for(int cout_n=0;cout_n<conv_config::CHout;cout_n++){
		bias[cout_n]=0;
		//std::cout<<bias[cout_n]<<std::endl;
	}

	printf("OK\n");

	mnet::Conv2d<conv_config>(feature_in[0][0],W[0][0][0],bias,feature_out[0][0]);

	for(int i=0;i<conv_config::Hout;i++)
		for(int j=0;j<conv_config::Wout;j++)
			for(int cout_n=0;cout_n<conv_config::CHout;cout_n++)
			{
			    std::cout<<i<<" "<<j<<" "<<cout_n<<" "<<feature_out[cout_n][i][j]<<std::endl;
			}

	return 0;
};

//linear
int test_linear()
{
    typename linear_config::Dtype_f feature_in[linear_config::Lin];
    typename linear_config::Dtype_f feature_out[linear_config::Lout];
    typename linear_config::Dtype_w W[linear_config::Lout][linear_config::Lin];
    typename linear_config::Dtype_b bias[linear_config::Lout];

    for(int i=0;i<linear_config::Lin;i++){
        feature_in[i]=i;
        for(int j=0;j<linear_config::Lout;j++){
            feature_out[j]=0;
            //W[i][j]=i*linear_config::Lout+j;
            W[j][i]=i;
            std::cout<<i<<" "<<j<<" "<<W[j][i]<<std::endl;
            bias[j]=j;
        }
    }

    mnet::Linear<linear_config>(feature_in,bias,W[0],feature_out);

	for(int i=0;i<linear_config::Lout;i++){
        std::cout<<i<<" "<<feature_out[i]<<std::endl;
    }
    return 0;
};

int test_DW(){
	typename DWC_config::Dtype_w W[DWC_config::CHin][1][DWC_config::Kh][DWC_config::Kw]=
		{{{{0.1277, -0.1238,  0.0030, -0.1353, -0.1569},
          {0.1987, -0.0839,  0.0719, -0.1582, -0.1276},
          {0.1014, -0.1580, -0.0644, -0.0498, -0.0936},
          {-0.0137,  0.1569, -0.0072,  0.0193,  0.1263},
          {0.1681, -0.0423,  0.0057, -0.0680,  0.0127}}},


        {{{0.1829, -0.0134,  0.1886,  0.0140,  0.1224},
          {0.1961,  0.0739,  0.0221, -0.0795, -0.1134},
          {-0.0444, -0.1985, -0.1814,  0.1159,  0.0921},
          {0.1564, -0.0301, -0.0322, -0.0130,  0.1924},
          {-0.1025,  0.1213,  0.0084,  0.1398, -0.0657}}},


        {{{-0.0659,  0.0435,  0.1582,  0.1543,  0.1837},
          {-0.1668, -0.1499, -0.0853, -0.1562,  0.1440},
          {-0.0445,  0.0155, -0.1102,  0.0120,  0.0877},
          {0.0485,  0.0502,  0.0508, -0.1816,  0.1943},
          {0.0119, -0.0663, -0.0059, -0.1838, -0.1324}}}};
	typename DWC_config::Dtype_b bias[DWC_config::CHin]=
		{0.0974, 0.0388, 0.0243};


	typename DWC_config::Dtype_f feature_in[DWC_config::CHin][DWC_config::Hin][DWC_config::Win]=
		{{{1.,  2.,  3.,  4.,  5.,  6.},
        {6.,  8.,  9., 10., 11., 12.},
          {13., 14., 15., 16., 17., 18.},
          {19., 20., 21., 22., 23., 24.},
          {25., 26., 27., 28., 29., 30.},
          {31., 32., 33., 34., 35., 36.}},

		{{1.,  2.,  3.,  4.,  5.,  6.},
        {6.,  8.,  9., 10., 11., 12.},
          {13., 14., 15., 16., 17., 18.},
          {19., 20., 21., 22., 23., 24.},
          {25., 26., 27., 28., 29., 30.},
          {31., 32., 33., 34., 35., 36.}},

		{{1.,  2.,  3.,  4.,  5.,  6.},
        {6.,  8.,  9., 10., 11., 12.},
          {13., 14., 15., 16., 17., 18.},
          {19., 20., 21., 22., 23., 24.},
          {25., 26., 27., 28., 29., 30.},
          {31., 32., 33., 34., 35., 36.}}};


	typename DWC_config::Dtype_f feature_out[DWC_config::CHin][DWC_config::Hout][DWC_config::Wout]={0};

	mnet::DWConv2d<DWC_config>(feature_in[0][0],W[0][0][0],bias,feature_out[0][0]);

	for(int i=0;i<DWC_config::Hout;i++)
		for(int j=0;j<DWC_config::Wout;j++)
			for(int cout_n=0;cout_n<DWC_config::CHin;cout_n++)
			{
			    std::cout<<i<<" "<<j<<" "<<cout_n<<" "<<feature_out[cout_n][i][j]<<std::endl;
			}

return 0;
};


int test_PW(){
	typename PWC_config::Dtype_w W[PWC_config::CHin][1][PWC_config::Kh][PWC_config::Kw]=
		{{{0.3800}},

         {{-0.1239}},

         {{0.4761}}};

	typename PWC_config::Dtype_b bias[PWC_config::CHout]=
		{0.3939};


	typename PWC_config::Dtype_f feature_in[PWC_config::CHin][PWC_config::Hin][PWC_config::Win]=
		{{{0.3761,  0.2840},
          {-1.2975, -1.4606}},

         {{7.3538,  8.3020},
          {11.8793, 12.8143}},

         {{-8.3308, -8.6920},
          {-9.5982, -9.8585}}};



	typename PWC_config::Dtype_f feature_out[PWC_config::CHout][PWC_config::Hout][PWC_config::Wout];

	mnet::PWConv2d<PWC_config>(feature_in[0][0],W[0][0][0],bias,feature_out[0][0]);
	for(int i=0;i<PWC_config::Hout;i++)
		for(int j=0;j<PWC_config::Wout;j++)
			for(int cout_n=0;cout_n<PWC_config::CHout;cout_n++)
			{
			    std::cout<<i<<" "<<j<<" "<<cout_n<<" "<<feature_out[cout_n][i][j]<<std::endl;
			}
	return 0;
};


int test_DWS(){

	typename DWS_config::DW::Dtype_w W_DW[DWS_config::DW::CHin][1][DWS_config::DW::Kh][DWS_config::DW::Kw]=
		{{{{0.1277, -0.1238,  0.0030, -0.1353, -0.1569},
          {0.1987, -0.0839,  0.0719, -0.1582, -0.1276},
          {0.1014, -0.1580, -0.0644, -0.0498, -0.0936},
          {-0.0137,  0.1569, -0.0072,  0.0193,  0.1263},
          {0.1681, -0.0423,  0.0057, -0.0680,  0.0127}}},


        {{{0.1829, -0.0134,  0.1886,  0.0140,  0.1224},
          {0.1961,  0.0739,  0.0221, -0.0795, -0.1134},
          {-0.0444, -0.1985, -0.1814,  0.1159,  0.0921},
          {0.1564, -0.0301, -0.0322, -0.0130,  0.1924},
          {-0.1025,  0.1213,  0.0084,  0.1398, -0.0657}}},


        {{{-0.0659,  0.0435,  0.1582,  0.1543,  0.1837},
          {-0.1668, -0.1499, -0.0853, -0.1562,  0.1440},
          {-0.0445,  0.0155, -0.1102,  0.0120,  0.0877},
          {0.0485,  0.0502,  0.0508, -0.1816,  0.1943},
          {0.0119, -0.0663, -0.0059, -0.1838, -0.1324}}}};
	typename DWS_config::DW::Dtype_b bias_DW[DWS_config::DW::CHin]=
		{0.0974, 0.0388, 0.0243};


	typename DWS_config::DW::Dtype_f feature_in[DWS_config::DW::CHin][DWS_config::DW::Hin][DWS_config::DW::Win]=
		{{{1.,  2.,  3.,  4.,  5.,  6.},
        {6.,  8.,  9., 10., 11., 12.},
          {13., 14., 15., 16., 17., 18.},
          {19., 20., 21., 22., 23., 24.},
          {25., 26., 27., 28., 29., 30.},
          {31., 32., 33., 34., 35., 36.}},

		{{1.,  2.,  3.,  4.,  5.,  6.},
        {6.,  8.,  9., 10., 11., 12.},
          {13., 14., 15., 16., 17., 18.},
          {19., 20., 21., 22., 23., 24.},
          {25., 26., 27., 28., 29., 30.},
          {31., 32., 33., 34., 35., 36.}},

		{{1.,  2.,  3.,  4.,  5.,  6.},
        {6.,  8.,  9., 10., 11., 12.},
          {13., 14., 15., 16., 17., 18.},
          {19., 20., 21., 22., 23., 24.},
          {25., 26., 27., 28., 29., 30.},
          {31., 32., 33., 34., 35., 36.}}};


	typename DWS_config::PW::Dtype_w W_PW[DWS_config::PW::CHin][1][1][1]=
		{{{0.3800}},

         {{-0.1239}},

         {{0.4761}}};

	typename DWS_config::PW::Dtype_b bias_PW[DWS_config::PW::CHout]=
		{0.3939};

	typename DWS_config::PW::Dtype_f feature_out[DWS_config::PW::CHout][DWS_config::PW::Hout][DWS_config::PW::Wout];
	typename DWS_config::PW::Dtype_f feature_out1[DWS_config::PW::CHout][DWS_config::PW::Hout][DWS_config::PW::Wout];

	mnet::DWSConv2d<DWS_config::DW, DWS_config::PW>(feature_in[0][0],W_DW[0][0][0],bias_DW,W_PW[0][0][0],bias_PW,feature_out[0][0]);
  mnet::DWSConv2d_NoDWBuffer<DWS_config::DW, DWS_config::PW>(feature_in[0][0],W_DW[0][0][0],bias_DW,W_PW[0][0][0],bias_PW,feature_out1[0][0]);

	for(int i=0;i<DWS_config::PW::Hout;i++){
		for(int j=0;j<DWS_config::PW::Wout;j++){
			for(int cout_n=0;cout_n<DWS_config::PW::CHout;cout_n++)
			{
			    std::cout<<i<<" "<<j<<" "<<cout_n<<" "<<feature_out[cout_n][i][j]<<"   ";
          std::cout<<i<<" "<<j<<" "<<cout_n<<" "<<feature_out1[cout_n][i][j];
      }
      std::cout<<std::endl;
    }
  }
	return 0;
};


int test_transposeconv(){

	float in_test[128*4*4];
	for(int i=0;i<128*4*4;i++){in_test[i]=1.;}

	float out_read[128][8][8];
	for(int i=0;i<128*8*8;i++){out_read[0][0][i]=0.;}
	
	mnet::ConvTranspose2d_cb<transposeConv_cfg>(in_test,w,b,out_read[0][0]);

  for(int j=0;j<8;j++){
    for(int k=0;k<8;k++){
      std::cout<<out_read[0][j][k]<<" ";
    }
    std::cout<<std::endl;
  }
  std::cout<<std::endl;
  
	for(int i=0;i<128*8*8;i++){out_read[0][0][i]=0.;}

  mnet::ConvTranspose2d_pp_notfinish<transposeConv_cfg>(in_test,w,b,out_read[0][0]);

  for(int j=0;j<8;j++){
    for(int k=0;k<8;k++){
      std::cout<<out_read[0][j][k]<<" ";
    }
    std::cout<<std::endl;
  }

	return 0;
};



int test_transConv_small(){

  float in_test[1*4*4];
	for(int i=0;i<1*4*4;i++){in_test[i]=1.;}

	float out_read[1][8][8];
	for(int i=0;i<1*8*8;i++){out_read[0][0][i]=0.;}

  float w[1][1][5][5];
  for (int i=0;i<5;i++){
    for (int j=0;j<5;j++){
      w[0][0][i][j]=i*5+j;
    }
  }
  float b[1];
  for (int i=0;i<1;i++){b[i]=-20;}

	mnet::ConvTranspose2d_cb<transposeConv_cfg>(in_test,w[0][0][0],b,out_read[0][0]);

	for(int k=0;k<8;k++){
    for (int l=0;l<8;l++){
		  std::cout<<out_read[0][k][l]<<" ";
    }
    std::cout<<std::endl;
	}

	return 0;
};



#endif