#ifndef _MMATH_FUNC_H_
#define _MMATH_FUNC_H_


struct math_config_default{
    typedef  double input_t;
	typedef  double output_t;
	typedef  double intermediate_t;
};

template<class input_t,class output_t>
output_t max_manual(input_t a, input_t b){
    if (a>b)
        return output_t(a);
    else
        return output_t(b);
};

template<class input_t,class output_t>
output_t min_manual(input_t a, input_t b){
    if (a>b)
        return output_t(b);
    else
        return output_t(a);
};

template <class input_t, class output_t, typename CONFIG_EXP>
void exp_Taylor(input_t x[], output_t y[]){
	//use the equation e^x=lim_{n\rightarrow \infty}(1+x/n)^n
	//which is the same as e^x=1+x/1+x^2/(2*3)+x^3/(3*4*5)+...
    for(int i=0;i<CONFIG_EXP::len;i++){
        typename CONFIG_EXP::intermediate_t tmp = 1.0 + typename CONFIG_EXP::intermediate_t(x[i]) / 1024;
        tmp *= tmp; tmp *= tmp; tmp *= tmp; tmp *= tmp; tmp *= tmp;
        tmp *= tmp; tmp *= tmp; tmp *= tmp; tmp *= tmp; tmp *= tmp;
        y[i]=output_t(tmp);
    }
};


template<class input_t,class output_t,typename CONFIG_SIGMOD>
void sigmoid(input_t x[],output_t y[])
{
    //std::cout<<x<<std::endl;
    float tmp[CONFIG_SIGMOD::len]={0};
    exp_Taylor<input_t,float,CONFIG_SIGMOD>(x,tmp);
    for (int i=0;i<CONFIG_SIGMOD::len;i++){
        y[i]=output_t(1/(1+1/tmp[i]));
    }
};


/*
for a single float, int and other dtypes which is not in ap_**
the input and output
*/
template<class input_t>
input_t abs_number(input_t x){
    input_t y;
    if (x<0)
        y=-x;
    else
        y=x;
    return y;
};


#endif // !_MATH_FUNC_H_
