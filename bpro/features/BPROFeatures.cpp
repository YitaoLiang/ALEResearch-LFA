/****************************************************************************************
** Implementation of a variation of BASS Features, which has features to encode the 
**  relative position between tiles.
**
** REMARKS: - This implementation is basically Erik Talvitie's implementation, presented
**            in the AAAI'15 LGCVG Workshop.
**
** Author: Marlos C. Machado
***************************************************************************************/

#ifndef BPRO_FEATURES_H
#define BPRO_FEATURES_H
#include "BPROFeatures.hpp"
#endif
#ifndef BASIC_FEATURES_H
#define BASIC_FEATURES_H
#include "BasicFeatures.hpp"
#endif

#include <set>
#include <assert.h>
using namespace std;
//#include <tuple>
//#include <boost/tuple/tuple.hpp> //TODO: I have to remove this to not have to depend on boost

BlobBproFeatures::BlobBproFeatures(Parameters *param){
    this->param = param;
    numColumns  = param->getNumColumns();
	numRows     = param->getNumRows();
	numColors   = param->getNumColors();

	if(this->param->getSubtractBackground()){
        this->background = new Background(param);
    }

	//To get the total number of features:
	//TODO: Fix this!
    numBasicFeatures = this->param->getNumColumns() * this->param->getNumRows() * this->param->getNumColors();
	numRelativeFeatures = (2 * this->param->getNumColumns() - 1) * (2 * this->param->getNumRows() - 1) 
							* (1+this->param->getNumColors()) * this->param->getNumColors()/2;

}

BlobBproFeatures::~BlobBproFeatures(){}

int BlobBproFeatures::getBlobs(const ALEScreen &screen, ){
}

void BlobBproFeatures::addRelativeFeaturesIndices(const ALEScreen &screen){
    
}

void BlobBproFeatures::getActiveFeaturesIndices(const ALEScreen &screen, const ALERAM &ram, vector<long long>& features){
	int screenWidth = screen.width();
	int screenHeight = screen.height();
	int blockWidth = screenWidth / numColumns;
	int blockHeight = screenHeight / numRows;

	assert(features.size() == 0); //If the vector is not empty this can be a mess
    vector<vector<tuple<int,int> > > whichColors(numColors);

    //Before generating features we must check whether we can subtract the background:
    if(this->param->getSubtractBackground()){
        unsigned int sizeBackground = this->background->getWidth() * this->background->getHeight();
        assert(sizeBackground == screen.width()*screen.height());
    }

    //We first get the Basic features, keeping track of the next featureIndex vector:
    //We don't just use the Basic implementation because we need the whichColors information
	int featureIndex = getBasicFeaturesIndices(screen, blockWidth, blockHeight, whichColors, features);
	addRelativeFeaturesIndices(screen, featureIndex, whichColors, features);

	//Bias
	features.push_back(numBasicFeatures+numRelativeFeatures);
}

long long BlobBproFeatures::getNumberOfFeatures(){
    return numBasicFeatures + numRelativeFeatures + 1;
}

void BlobBproFeatures::resetBproExistence(vector<vector<bool> >& bproExistence, vector<tuple<int,int> >& changed){
    for (vector<tuple<int,int> >::iterator it = changed.begin();it!=changed.end();it++){
        bproExistence[get<0>(*it)][get<1>(*it)]=true;
    }
    changed.clear();
}