//
//  fourwiseHash.cpp
//  
//
//  Created by Yitao Liang on 8/13/15.
//
//

#include "fourwiseHash.hpp"

FourwiseHash::FourwiseHash(int seed, int hashTableSize){
    randomNumberGenerator.seed(seed);
    a = randomNumberGenerator()%hashTableSize;
    b = randomNumberGenerator()%hashTableSize;
    c = randomNumberGenerator()%hashTableSize;
    d = randomNumberGenerator()%hashTableSize;
    this->hashTableSize = hashTableSize;
}

fourwiseHash::hash(long long index){
    long long r;
    index = index % hashTableSize;
    r = (a*index) + b;
    r = r % hashTableSize;
    r = (r+index)+c;
    r = r % hashTableSize;
    r = (r*index) + d;
    r = r % hashTableSize;
    
    if ((r & 0x1)==1){
        return 1;
    }else{
        return -1;
    }
}