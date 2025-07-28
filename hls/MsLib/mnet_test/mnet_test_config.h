#ifndef MNET_TEST_CONFIG_H_INCLUDED
#define MNET_TEST_CONFIG_H_INCLUDED
// set configs for mnet_tes.h

// #include "ap_int.h"
// #include "ap_fixed.h"
#include "../mnet/mnet_top.h"
#include "../mmath/mmath_top.h"

#define ACT_TTL_BIT 16 // total bits
#define ACT_INT_BIT 8  // integer bits


// typedef ap_fixed<ACT_TTL_BIT, ACT_INT_BIT, AP_RND_CONV, AP_SAT> mnet_default_t;
typedef float mnet_default_t;

struct exp_config : mmath::math_config_default{

    typedef mnet_default_t input_t;
    typedef mnet_default_t output_t;

    static const int len=3;

};

struct sigmod_config : mmath::math_config_default{

    typedef mnet_default_t input_t;
    typedef mnet_default_t output_t;

    static const int len=3;

};

struct pool_config : mnet::pool2d0_config{
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
    static const bool relu_en=0;
    static const bool padding=0;

	//settings defined by the values of up above
    static const unsigned pad_w= (padding==0?0:(Kw-1)/2);
    static const unsigned pad_h= (padding==0?0:(Kh-1)/2);
    
    static const unsigned Wout=(Win+2*pad_w-Kw)/Sw+1;
    static const unsigned Hout=(Hin+2*pad_h-Kh)/Sh+1;
};

struct conv_config:mnet::conv2d0_config{
    typedef mnet_default_t  Dtype_f;
    typedef mnet_default_t  Dtype_w;
    typedef mnet_default_t  Dtype_b;

    typedef int Index_t;
    typedef mnet_default_t Dtype_tmp;

    static const bool padding =0;
    static const bool relu_en=0;

    static const unsigned CHin=1;
    static const unsigned Hin =3;
    static const unsigned Win =3;
    static const unsigned Kw =2;
    static const unsigned Kh =2;
    static const unsigned Sw =1;
    static const unsigned Sh =1;

    static const unsigned pad_w= (padding==0?0:(Kw-1)/2);
    static const unsigned pad_h= (padding==0?0:(Kh-1)/2);

    static const unsigned CHout=1;
    static const unsigned Wout=(Win+2*pad_w-Kw)/Sw+1;
    static const unsigned Hout=(Hin+2*pad_h-Kh)/Sh+1;
};

struct linear_config:mnet::linear0_config{
    typedef mnet_default_t  Dtype_f;
    typedef mnet_default_t  Dtype_b;
    typedef mnet_default_t  Dtype_w;
    typedef mnet_default_t  Dtype_tmp;

    static const unsigned Lin =5;
    static const unsigned Lout=2;
};

struct DWC_config{
    typedef float  Dtype_f;
    typedef float  Dtype_w;
    typedef float  Dtype_b;
    typedef int Index_t;
    typedef float Dtype_tmp;

    static const bool padding =0;
    static const bool relu_en=0; 

    static const unsigned CHin=3;
    static const unsigned Hin =6;
    static const unsigned Win =6;
    static const unsigned Kw =5;
    static const unsigned Kh =5;
    static const unsigned Sw =1;
    static const unsigned Sh =1;

	//settings defined by the values of up above
	static const unsigned CHout=CHin; 
    //这个注释掉是因为，减少一个自主设置的变量能使调用更加容易
    static const unsigned pad_w= (padding==0?0:(Kw-1)/2);
    static const unsigned pad_h= (padding==0?0:(Kh-1)/2);
    static const unsigned Wout=(Win+2*pad_w-Kw)/Sw+1;
    static const unsigned Hout=(Hin+2*pad_h-Kh)/Sh+1;
};

struct PWC_config{
    typedef float  Dtype_f;
    typedef float  Dtype_w;
    typedef float  Dtype_b;
    typedef int Index_t;
    typedef float Dtype_tmp;

    static const bool padding =0;
    static const bool relu_en=0; 

    static const unsigned CHin=3;
    static const unsigned Hin =6;
    static const unsigned Win =6;
    static const unsigned Kw =5;
    static const unsigned Kh =5;
    static const unsigned Sw =1;
    static const unsigned Sh =1;

	static const unsigned CHout=1; 

    static const unsigned pad_w= (padding==0?0:(Kw-1)/2);
    static const unsigned pad_h= (padding==0?0:(Kh-1)/2);
    static const unsigned Wout=(Win+2*pad_w-Kw)/Sw+1;
    static const unsigned Hout=(Hin+2*pad_h-Kh)/Sh+1;
};

struct DWS_config:mnet::DWSconv2d0_config{
    struct base{
        static const unsigned CHin=3;
        static const unsigned Hin =6;
        static const unsigned Win =6;
        static const unsigned Kw =5;
        static const unsigned Kh =5;
        static const unsigned Sw =1;
        static const unsigned Sh =1;
        static const unsigned CHout=1;
    };

    struct DW{
        typedef float  Dtype_f;
        typedef float  Dtype_w;
        typedef float  Dtype_b;
        typedef int Index_t;
        typedef float Dtype_tmp;

        static const bool padding =0;
        static const bool relu_en=0; 

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

        static const bool relu_en=0; 
        static const unsigned CHin= base::CHin;
        static const unsigned Hin = DW::Hout;
        static const unsigned Win = DW::Wout;

        static const unsigned CHout=base::CHout; 
        static const unsigned Wout=DW::Wout;
        static const unsigned Hout=DW::Hout;
    };
};

struct g_a_0_cfg:mnet::DWSconv2d0_config{
    struct base{
        static const unsigned CHin=3;
        static const unsigned Hin =256;
        static const unsigned Win =256;
        static const unsigned Kw =5;
        static const unsigned Kh =5;
        static const unsigned Sw =2;
        static const unsigned Sh =2;
        static const unsigned CHout=128;
    };

    struct DW{
        typedef float  Dtype_f;
        typedef float  Dtype_w;
        typedef float  Dtype_b;
        typedef int Index_t;
        typedef float Dtype_tmp;

        static const bool padding =1;
        static const bool relu_en=0; 

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

        static const bool relu_en=0; 
        static const unsigned CHin= base::CHin;
        static const unsigned Hin = DW::Hout;
        static const unsigned Win = DW::Wout;

        static const unsigned CHout=base::CHout; 
        static const unsigned Wout=DW::Wout;
        static const unsigned Hout=DW::Hout;
    };
};

struct transposeConv_cfg:mnet::transconv2d0_config{
    typedef float  Dtype_f;
    typedef float  Dtype_w;
    typedef float  Dtype_b;
    typedef int Index_t;
    typedef float Dtype_tmp;

    static const bool padding =1; //1 for default
    static const bool relu_en=0; 

    static const unsigned CHin=128;
    static const unsigned Hin =4;
    static const unsigned Win =4;
    static const unsigned Kw =5;
    static const unsigned Kh =5;
    static const unsigned Sw =2;
    static const unsigned Sh =2;

	static const unsigned CHout=128;

	//settings defined by the values of up above

    static const unsigned pad_w = (padding==0?0:(Kw-1)/2);
    static const unsigned pad_h = (padding==0?0:(Kh-1)/2);
	static const unsigned o_pad_w = Sw-1;//can change it for other output_size
	static const unsigned o_pad_h = Sh-1;

    static const unsigned Wout = (Win-1)*Sw-2*pad_w+Kw+o_pad_w;
    static const unsigned Hout = (Hin-1)*Sh-2*pad_h+Kh+o_pad_h;
};


struct gdn_cfg:mnet::gdn0_cfg{
    typedef float Dtype_f;
    typedef float Beta;
    typedef float Gamma;
    typedef float Dtype_tmp;

    static const unsigned CHin=128;
    static const unsigned Hin =64;
    static const unsigned Win =64;
};

#endif // _MNET__