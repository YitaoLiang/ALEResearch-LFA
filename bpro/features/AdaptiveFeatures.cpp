/****************************************************************************************
 ** Implementation of an adaptive feature representation
 **
 ** Author: Yitao Liang
 ***************************************************************************************/

#ifndef Adaptive_Features_H
#define Adaptive_Features_H
#include "AdaptiveFeatures.hpp"
#endif

#include <set>
#include <assert.h>
#include <algorithm>
#include <math.h>
#include <unordered_set>

using namespace std;

AdaptiveFeatures::AdaptiveFeatures(Parameters *param, const ALEScreen &screen){
     this->param = param;
     numColors   = param->getNumColors();
     numFeatuers = 0;
     constructBaseFeatures(screen);
}

AdaptiveFeatures::~AdaptiveFeautrs(){
}

void AdaptiveFeatures::constructBaseFeatures(){
    // initial base features: will be whether there is a color k pixel on the screen
    for (int i=0;i<numColors;++i){
        Feature a(++numFeatures,i,0,0,210,160); 
    }
}

void 

void AdaptiveFeatures::getActiveFeaturesIndices(){
    
}


