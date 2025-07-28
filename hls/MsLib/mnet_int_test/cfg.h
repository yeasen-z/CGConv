#ifndef MNET_INT_TEST_CONFIG_H_INCLUDED
#define MNET_INT_TEST_CONFIG_H_INCLUDED
// set configs for mnet_tes.h

// #include "ap_int.h"
// #include "ap_fixed.h"
#include "../mnet_int/mnet_int_top.h"
#include "../mnet/mnet_top.h"
#include "../mmath/mmath_top.h"

struct convintcfg: mnet_int::conv2d0_config{
    typedef int  Dtype_f; //recommand to be int, could use ap_int<bit_f>
    typedef int  Dtype_w; //recommand to be int
    typedef int  Dtype_b; //recommand to be int

    typedef int Index_t;
    typedef int Dtype_tmp; //recommand to be int, bits long enough

	static const int Delta_w=2048; // quant factor of weight
	static const int Delta_i=2048; // quant factor of feature_in, actually is not used in function
	static const int Delta_b = Delta_w*Delta_i;; // quant factor of bias, actually is not used in function, only needed for do quantization

    static const bool padding =1;
    static const bool relu_en=0;

    static const unsigned CHin=3;
    static const unsigned Hin =32;
    static const unsigned Win =32;
    static const unsigned Kw =5;
    static const unsigned Kh =5;
    static const unsigned Sw =2;
    static const unsigned Sh =2;
    static const unsigned pad_w= (padding==0?0:(Kw-1)/2);
    static const unsigned pad_h= (padding==0?0:(Kh-1)/2);
    static const unsigned CHout=6;
    static const unsigned Wout=(Win+2*pad_w-Kw)/Sw+1;
    static const unsigned Hout=(Hin+2*pad_h-Kh)/Sh+1;
};

struct convcfg:mnet::conv2d0_config{
    typedef float  Dtype_f;
    typedef float  Dtype_w;
    typedef float  Dtype_b;

    typedef int Index_t;
    typedef float Dtype_tmp;

    static const bool padding =1;
    static const bool relu_en=0;

    static const unsigned CHin=3;
    static const unsigned Hin =32;
    static const unsigned Win =32;
    static const unsigned Kw =5;
    static const unsigned Kh =5;
    static const unsigned Sw =2;
    static const unsigned Sh =2;

    static const unsigned pad_w= (padding==0?0:(Kw-1)/2);
    static const unsigned pad_h= (padding==0?0:(Kh-1)/2);

    static const unsigned CHout=6;
    static const unsigned Wout=(Win+2*pad_w-Kw)/Sw+1;
    static const unsigned Hout=(Hin+2*pad_h-Kh)/Sh+1;
};

struct int_transconvcfg : mnet_int::transconv2d0_config{
    typedef int  Dtype_f;
    typedef int  Dtype_w;
    typedef int  Dtype_b;
    typedef int  Index_t;
    typedef int  Dtype_tmp;

    static const bool padding =1; //1 for default
    static const bool relu_en=0; 

	static const int Delta_w = 512; 
	// quant factor of weight
	static const int Delta_i = 512; 
	// quant factor of feature_in and feature_out, actually is not used in function
	static const int Delta_b = Delta_w*Delta_i;  //Delta_b=Delta_w*Delta_i
	// quant factor of bias, actually is not used in function, only needed for do quantization

    static const unsigned CHin=3;
    static const unsigned Hin =16;
    static const unsigned Win =16;
    static const unsigned Kw =5;
    static const unsigned Kh =5;
    static const unsigned Sw =2;
    static const unsigned Sh =2;

	static const unsigned CHout=3;
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

struct transconvcfg : mnet::transconv2d0_config{
    typedef float  Dtype_f;
    typedef float  Dtype_w;
    typedef float  Dtype_b;
    typedef int Index_t;
    typedef float Dtype_tmp;

    static const bool padding =1; //1 for default
    static const bool relu_en=0; 

    static const unsigned CHin=3;
    static const unsigned Hin =16;
    static const unsigned Win =16;
    static const unsigned Kw =5;
    static const unsigned Kh =5;
    static const unsigned Sw =2;
    static const unsigned Sh =2;

	static const unsigned CHout=3;

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

struct dwscfg : mnet::DWSconv2d0_config{
    struct base{
        static const unsigned CHin=3;
        static const unsigned Hin =256;
        static const unsigned Win =256;
        static const unsigned Kw =5;
        static const unsigned Kh =5;
        static const unsigned Sw =2;
        static const unsigned Sh =2;
        static const unsigned CHout=192;
        static const bool relu_en=0;
        static const bool padding =1;
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

struct int_dwscfg : mnet_int::conv2d0_config{
    struct base{
        static const unsigned CHin=3;
        static const unsigned Hin =256;
        static const unsigned Win =256;
        static const unsigned Kw =5;
        static const unsigned Kh =5;
        static const unsigned Sw =2;
        static const unsigned Sh =2;
        static const unsigned CHout=192;
        static const bool relu_en=0;
        static const bool padding =1;

        static const int Delta_w = 1024; 
        // quant factor of weight
        static const int Delta_i = 1024; 
        // quant factor of feature_in and feature_out, actually is not used in function
        static const int Delta_b = Delta_w*Delta_i;  //Delta_b=Delta_w*Delta_i
        // quant factor of bias, actually is not used in function, only needed for do quantization
    };

    struct DW{
        typedef int  Dtype_f;
        typedef int  Dtype_w;
        typedef int  Dtype_b;
        typedef int Index_t;
        typedef int Dtype_tmp;

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

        static const int Delta_w = base::Delta_w; 
        // quant factor of weight
        static const int Delta_i = base::Delta_i; 
        // quant factor of feature_in and feature_out, actually is not used in function
        static const int Delta_b = Delta_w*Delta_i;  //Delta_b=Delta_w*Delta_i
        // quant factor of bias, actually is not used in function, only needed for do quantization
    };

    struct PW{
        typedef int  Dtype_f;
        typedef int  Dtype_w;
        typedef int  Dtype_b;
        typedef int Index_t;
        typedef int Dtype_tmp;

        static const bool relu_en=base::relu_en; 
        static const unsigned CHin= base::CHin;
        static const unsigned Hin = DW::Hout;
        static const unsigned Win = DW::Wout;

        static const unsigned CHout=base::CHout; 
        static const unsigned Wout=DW::Wout;
        static const unsigned Hout=DW::Hout;

        static const int Delta_w = base::Delta_w; 
        // quant factor of weight
        static const int Delta_i = base::Delta_i; 
        // quant factor of feature_in and feature_out, actually is not used in function
        static const int Delta_b = Delta_w*Delta_i;  //Delta_b=Delta_w*Delta_i
        // quant factor of bias, actually is not used in function, only needed for do quantization
    };
};

#endif

