/****************************************/  
/*Function: Sqrt                        */  
/****************************************/  
#ifndef _MMATH_SQRT_H_
#define _MMATH_SQRT_H_

// use for 32 bits int input, 16 bits int output
unsigned int Sqrt_int32_int16(unsigned long M)  
{  
    unsigned int N, i;  
    unsigned long tmp, ttp;   // 结果、循环计数  
    if (M == 0)               // 被开方数，开方结果也为0  
        return 0;  
    N = 0;  

    tmp = (M >> 30);          // 获取最高位：B[m-1]  
    M <<= 2;  
    if (tmp > 1)              // 最高位为1  
    {  
        N ++;                 // 结果当前位为1，否则为默认的0  
        tmp -= N;  
    }  

    for (i=15; i>0; i--)      // 求剩余的15位  
    {  
        N <<= 1;              // 左移一位  
        tmp <<= 2;  
        tmp += (M >> 30);     // 假设  

        ttp = N;  
        ttp = (ttp<<1)+1;  

        M <<= 2;  
        if (tmp >= ttp)       // 假设成立  
        {  
            tmp -= ttp;  
            N ++;  
        }  

    }  
    return N;  
}


// use for fixed float
template<class input_t,class output_t>
output_t Sqrt_newton(input_t x){
  // let initial guess to be 1
  input_t z = 1.0;
  for(int i = 1; i <= 10; i++){
    z -= (z*z - x) / (2*z); // MAGIC LINE!!
  }
  return output_t(z);
}


#endif 