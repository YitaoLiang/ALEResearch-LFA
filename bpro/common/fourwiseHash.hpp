//
//  fourwiseHash.h
//  
//
//  Created by Yitao Liang on 8/13/15.
//
//

#ifndef ____fourwiseHash__
#define ____fourwiseHash__

#include <stdio.h>
#include <random>

class FourwiseHash:{
    protected:
        int a,b,c,d;
        std::mt19937 randomNumberGenerator;
        int hashTableSize;
    public:
        FourwiseHash(int seed, int hashTableSize);
        ~FourwiseHash();
    int hash(long long index);
}


#endif /* defined(____fourwiseHash__) */
