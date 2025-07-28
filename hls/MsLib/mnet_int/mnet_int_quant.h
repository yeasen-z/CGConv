#ifndef MNET_INT_QUANT_H
#define MNET_INT_QUANT_H

namespace mnet_int{

// quant use a int Delta, which could cover the range of value.
// the MAX ≈ 2^(bits_l-1)/Delta

/*
Here, My quant has the feature:
- Delta is the pow of 2, which is needed
- MAX ≈ 2^(bits_l-1)/Delta
*/
template<class input_t, class output_t>
void quant(input_t fp_value[], output_t int_value[],int len, int Delta){
    for (int i=0;i<len;i++){
        int_value[i]=(output_t)(fp_value[i]*Delta); 
    }
};

/*
Here, My quant has the feature:
- Delta is the pow of 2, which is needed
- MAX ≈ 2^(bits_l-1)/Delta
*/
template<class input_t, class output_t>
void dequant(input_t int_value[], output_t fp_value[],int len, int Delta){
    for (int i=0;i<len;i++){
        fp_value[i]=((output_t)int_value[i])/(output_t)Delta; 
    }
};

};


#endif // MNET_INT_QUANT_H