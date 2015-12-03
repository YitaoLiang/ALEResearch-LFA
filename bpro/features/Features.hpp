/****************************************************************************************
** Superclass of all other classes that define Features. It is required from classes that
** inherit from Features to implement the virtual methods getActiveFeaturesIndices and
** getNumberOfFeatures, which are operations specific to the kind of selected features.
** It already implements getCompleteFeatureVector, which can be used if someone needs the
** complete feature vector. This is not an efficient approach.
** 
** TODO: Right now this class only deals with binary features, it may be necessary to
**       further extend such approach.
**
** Author: Marlos C. Machado
***************************************************************************************/

#ifndef ALE_INTERFACE_H
#define ALE_INTERFACE_H
#include <ale_interface.hpp>
#endif

#ifndef FEATURES_H
#define FEATURES_H

using namespace std;

class Features{
	private:
		
	public:
		/**
 		* This method was created to allow optimizations in codes that use such features.
 		* Since several feature definitions generate sparse feature vectors it is much
 		* better to just iterate over the active features. This method allows it by
 		* providing a vector with the indices of active features. Recall we assume the
 		* features to be binary. Each implementation of a feature needs to instantiate this
 		* method.
 		*
 		* REMARKS: - It is necessary to provide both the screen and the ram as one may
 		* want to use both data to generate features.
 		*          - To avoid return huge vectors, this method is void and the appropriate
 		* vector is returned trough a parameter passed by reference
 		*
 		* TODO: If one intends to use non-binary features this class is not suitable.
 		*
 		* @param ALEScreen &screen is the current game screen that one may use to extract features.
 		* @param ALERAM &ram is the current game RAM that one may use to extract features.
 		* @param vector<int>& features an empy vector that will be filled with the requested information,
 		*        therefore it must be passed by reference.
 		* 
 		* @return nothing since one will receive the requested data by the last parameter, by reference.
 		*/
		virtual void getActiveFeaturesIndices(const ALEScreen &screen, const ALERAM &ram, vector<long long>& features) = 0;		
		/**
		* Destructor, not necessary in this class.
		*/
		virtual ~Features();
        virtual void promoteFeatures() = 0;
        virtual void updateDelta(float delta) = 0;
        //virtual void updateWeights(vector<vector<float> >& weights, float learningRate) =0;
        virtual void resetPromotionCriteria() = 0;
        virtual void resetActive() = 0;
        virtual long long getNumFeatures() = 0;
};
#endif
