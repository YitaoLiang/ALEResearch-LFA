/****************************************************************************************
** Superclass of all other classes that define Features. It is required from classes that
** inherit from Features to implement the virtual methods getCompleteFeatureVector and
** getNumberOfFeatures, which are operations specific to the kind of selected features.
** It already implements getActiveFeaturesIndices, which allow optimizations for the 
** learning algorithms.
**
** REMARKS: - All methods' high-level comments are in the .hpp file.
** 
** TODO: Right now this class only deals with binary features, it may be necessary to
**       further extend such approach.
**
** Author: Marlos C. Machado
***************************************************************************************/

#include "Features.hpp"

Features::~Features(){}
