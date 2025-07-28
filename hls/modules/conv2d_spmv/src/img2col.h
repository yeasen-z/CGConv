#ifndef _IMG2COL_H_
#define _IMG2COL_H_

/*
如果想要单独执行，可以按照这个过程使用
*/
struct CONFIG_Img2col{
    static const unsigned padding=1;
    static const unsigned CHout=1;
    static const unsigned CHin=1;
    static const unsigned Kh=3;
    static const unsigned Kw=3;
    static const unsigned Sw=1; //stride
    static const unsigned Sh=1; //stride
    static const unsigned Hin=8;
    static const unsigned Win=8;
    typedef float Dtype_f; 

    static const unsigned Pw=(padding==0?0:(Kw-1)/2); //padding
    static const unsigned Ph=(padding==0?0:(Kh-1)/2); //padding

    static const unsigned Hout=(Hin+2*Ph-Kh)/Sh+1;
    static const unsigned Wout=(Win+2*Pw-Kw)/Sw+1;

    static const unsigned col_img_H=Kw*Kh;
    static const unsigned col_img_W=Hout*Wout;
};


template <typename CONFIG_Img2col>
void do_padding(
    typename CONFIG_Img2col::Dtype_f img[],
    typename CONFIG_Img2col::Dtype_f img_paded[]
){
    // Manually initialize padded image with zeros
    for(int i = 0; i < CONFIG_Img2col::CHin; i++){
        for(int j = 0; j < CONFIG_Img2col::Hin + 2 * CONFIG_Img2col::Ph; j++){
            for(int k = 0; k < CONFIG_Img2col::Win + 2 * CONFIG_Img2col::Pw; k++){
                img_paded[i * (CONFIG_Img2col::Hin + 2 * CONFIG_Img2col::Ph) * (CONFIG_Img2col::Win + 2 * CONFIG_Img2col::Pw) + j * (CONFIG_Img2col::Win + 2 * CONFIG_Img2col::Pw) + k] = 0;
            }
        }
    }

    // Copy image to padded image
    for(int i = 0; i < CONFIG_Img2col::CHin; i++){
        for(int j = 0; j < CONFIG_Img2col::Hin; j++){
            for(int k = 0; k < CONFIG_Img2col::Win; k++){
                int pad_i = j + CONFIG_Img2col::Ph;
                int pad_j = k + CONFIG_Img2col::Pw;
                img_paded[i * (CONFIG_Img2col::Hin + 2 * CONFIG_Img2col::Ph) * (CONFIG_Img2col::Win + 2 * CONFIG_Img2col::Pw) + pad_i * (CONFIG_Img2col::Win + 2 * CONFIG_Img2col::Pw) + pad_j] =
                img[i * CONFIG_Img2col::Hin * CONFIG_Img2col::Win + j * CONFIG_Img2col::Win + k];
            }
        }
    }
}


/*
图片输入为CHin*H*W
卷积核为CHout*CHin*Kh*Kw
步长为S, padding为P

要将图片转化为可以直接矩阵相乘的形式，即[CHin][Kw*Kh][Hout*Wout]
即按照stride和padding以及卷积核得到小，将图片切割然后展开成为一列
*/
template <typename CONFIG_Img2col>
void img2col(
    typename CONFIG_Img2col::Dtype_f img[],
    typename CONFIG_Img2col::Dtype_f img_col[]
){
    int paded_H=CONFIG_Img2col::Hin+2*CONFIG_Img2col::Ph;
    int paded_W=CONFIG_Img2col::Win+2*CONFIG_Img2col::Pw;
    typename CONFIG_Img2col::Dtype_f img_paded[CONFIG_Img2col::CHin * paded_H * paded_W];
    do_padding<CONFIG_Img2col>(img, img_paded);

    for(int c=0;c<CONFIG_Img2col::CHin;c++){
        for(int i=0;i<CONFIG_Img2col::Hout;i++){
            for(int j=0;j<CONFIG_Img2col::Wout;j++){
                
                int h_00=i*CONFIG_Img2col::Sh;
                int w_00=j*CONFIG_Img2col::Sw;

                for(int m=0;m<CONFIG_Img2col::Kh;m++){
                    for(int n=0;n<CONFIG_Img2col::Kw;n++){

                        img_col[
                                c*CONFIG_Img2col::Kh*CONFIG_Img2col::Kw*CONFIG_Img2col::Hout*CONFIG_Img2col::Wout+
                                m*CONFIG_Img2col::Kw*CONFIG_Img2col::Hout*CONFIG_Img2col::Wout+
                                n*CONFIG_Img2col::Hout*CONFIG_Img2col::Wout+
                                i*CONFIG_Img2col::Wout+
                                j]=
                        img_paded[
                                c*paded_H*paded_W+
                                (h_00+m)*paded_W+
                                (w_00+n)];
                    }
                }    
            }
        }
    }

    return;
};
#endif