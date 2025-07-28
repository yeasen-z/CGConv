/*
Generalized Divisive Normalization layer.

Introduced in `"Density Modeling of Images Using a Generalized Normalization
Transformation" <https://arxiv.org/abs/1511.06281>`_,
by Balle Johannes, Valero Laparra, and Eero P. Simoncelli, (2016).

.. math::

    y[i] = \frac{x[i]}{\sqrt{\beta[i] + \sum_j(\gamma[j, i] * x[j]^2)}}

"""
*/
#ifndef MNET_GDN_H
#define MNET_GDN_H

#include "../mmath/mmath_top.h"

// Your code here
namespace mnet{

struct gdn0_cfg
{
    typedef float Dtype_f;
    typedef float Beta;
    typedef float Gamma;
    typedef float Dtype_tmp;

    static const unsigned CHin=16;
    static const unsigned Hin =6;
    static const unsigned Win =6;

    static const unsigned CHout=CHin;
    static const unsigned Wout=Win;
    static const unsigned Hout=Hin; 
};


/*
	Use for Pytorch GDN
	Feature in shape: (CHin,Hin,Win) == [CHin][Hin][Win]
	Feature out shape:(CHin,Hin,Win) == [CHin][Hin][Win]
	Beta shape: (CHin,) == [CHin]
    Gamma shape: (CHin, CHin) == [CHin][CHin]
    Norm[c][h][w] = sqrt(Beta[c] + sum_c(Feature_in[c][h][w]*Gamma[c]))
	Feature_out[c][h][w] = Feature_in[c][h][w]/Norm[c][h][w]
*/
template <typename CONFIG_C>
void GDN(
    typename CONFIG_C::Dtype_f feature_in[],
    typename CONFIG_C::Gamma G[],
    typename CONFIG_C::Beta B[],
    typename CONFIG_C::Dtype_f feature_out[]
    )
{
    typename CONFIG_C::Dtype_f norm[CONFIG_C::CHin*CONFIG_C::Hin*CONFIG_C::Win];
    for(int i=0; i<CONFIG_C::CHin*CONFIG_C::Hin*CONFIG_C::Win; i++){ norm[i] = 1; }

    for (int c=0; c < CONFIG_C::CHin; c++){
        for (int h=0; h<CONFIG_C::Hin; h++){
            for (int w=0; w<CONFIG_C::Win; w++){
                typename CONFIG_C::Dtype_f sum = B[c];

                for (int k=0; k<CONFIG_C::CHin; k++){
                    sum += G[c*CONFIG_C::CHin+k]*feature_in[k*CONFIG_C::Hin*CONFIG_C::Win+h*CONFIG_C::Hin+w]*feature_in[k*CONFIG_C::Hin*CONFIG_C::Win+h*CONFIG_C::Hin+w];
                }
                norm[c*CONFIG_C::CHin+h*CONFIG_C::Hin+w] = mmath::Sqrt_newton<typename CONFIG_C::Dtype_f, typename CONFIG_C::Dtype_f>(sum);
                // norm[c*CONFIG_C::CHin+h*CONFIG_C::Hin+w] = hls::sqrt(sum); //use these line if use ap_int,ap_fixed type
                // norm[c*CONFIG_C::CHin+h*CONFIG_C::Hin+w] = sum;
            }
        }
    }

    for (int c=0; c < CONFIG_C::CHin; c++){
        for (int h=0; h<CONFIG_C::Hin; h++){
            for (int w=0; w<CONFIG_C::Win; w++){
                feature_out[c*CONFIG_C::Hin*CONFIG_C::Win+h*CONFIG_C::Hin+w] = feature_in[c*CONFIG_C::Hin*CONFIG_C::Win+h*CONFIG_C::Hin+w] / norm[c*CONFIG_C::Hin*CONFIG_C::Win+h*CONFIG_C::Hin+w];
            }
        }
    }

};


template <typename CONFIG_C>
void GDN1(
    typename CONFIG_C::Dtype_f feature_in[],
    typename CONFIG_C::Gamma G[],
    typename CONFIG_C::Beta B[],
    typename CONFIG_C::Dtype_f feature_out[]
    )
{
    typename CONFIG_C::Dtype_f norm[CONFIG_C::CHin*CONFIG_C::Hin*CONFIG_C::Win];
    for(int i=0; i<CONFIG_C::CHin*CONFIG_C::Hin*CONFIG_C::Win; i++){ norm[i] = 1; }

    for (int c=0; c < CONFIG_C::CHin; c++){
        for (int h=0; h<CONFIG_C::Hin; h++){
            for (int w=0; w<CONFIG_C::Win; w++){
                typename CONFIG_C::Dtype_f sum = B[c];

                for (int k=0; k<CONFIG_C::CHin; k++){
                    sum += G[c*CONFIG_C::CHin+k]*mmath::abs_number(feature_in[k*CONFIG_C::Hin*CONFIG_C::Win+h*CONFIG_C::Hin+w]);
                }
                norm[c*CONFIG_C::CHin+h*CONFIG_C::Hin+w] = 1.0/(sum);
                // norm[c*CONFIG_C::CHin+h*CONFIG_C::Hin+w] = sum;
            }
        }
    }

    for (int c=0; c < CONFIG_C::CHin; c++){
        for (int h=0; h<CONFIG_C::Hin; h++){
            for (int w=0; w<CONFIG_C::Win; w++){
                feature_out[c*CONFIG_C::Hin*CONFIG_C::Win+h*CONFIG_C::Hin+w] = feature_in[c*CONFIG_C::Hin*CONFIG_C::Win+h*CONFIG_C::Hin+w] / norm[c*CONFIG_C::Hin*CONFIG_C::Win+h*CONFIG_C::Hin+w];
            }
        }
    }

};

};

#endif // MNET_BN_H