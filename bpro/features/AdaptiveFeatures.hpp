/****************************************************************************************
** Implementation of a variation of BASS Features, which has features to encode the 
**  relative position between tiles.
**
** REMARKS: - This implementation is basically Erik Talvitie's implementation, presented
**            in the AAAI'15 LGCVG Workshop.
**
** Author: Yitao Liang
***************************************************************************************/
#include "Features.hpp"
#include "Background.hpp"

#include <tuple>
#include <set>
#include <unordered_map>

#ifndef Adaptive_Features_H
#define Adaptive_Features_H
struct Location{
    int x, y, resolutionX, resolutionY, color;
    Location(int c,int x,int y, int rx, int ry){
        this->x = x;
        this->y = y;
        this->resolutionX = rx;
        this->resolutionY = ry;
        this->color = c;
    }
    Location(){}
};

struct Feature{
    int featureIndex;
    Location location;
    vector<Location> offsets;
    bool extraOffset;
    int numAppearance;
    float sumDelta;
    bool active;
    vector<Feature*> children;
     Feature(int index, int color,int a, int b, int c, int d){
        this->featureIndex = index;
        this->location = Location(color,a,b,c,d);
        this->numAppearance = 0;
        this->sumDelta = 0;
        this->extraOffset = false;
    }
};

struct Disjoint_Set_Element{
    int rowUp, rowDown, columnLeft, columnRight;
    int size;
    int parent;
    int color;
};


using namespace std;

class AdaptiveFeatures : public Features::Features{
	private:
		Parameters *param;
        Background *background;
        int numColors, colorMultiplier;
        int numPromotions;
        long long promotionFrequency, numberOfFramesToPromotion;
        long long numFeatures;
        vector<vector<tuple<int,int> > > blobs;
        vector<Feature*> baseFeatures;
        unordered_map<long long, Feature*> indexToFeature;
        Feature* rootFeature;
    
        void constructBaseFeatures();
        void getBlobs(const ALEScreen& screen);
        void recursionToCheckFeatures(Feature*& current, vector<tuple<int,int> > possibleAnchorPositions, vector<long long>& activeFeatuers);
        void generateCandidateFeatures(Feature*& baseFeature);

        // helper functions to get blobs in the screen
        int neighborSize;
        vector<vector<vector<unsigned short> > >* fullNeighbors;
        vector<vector<vector<unsigned short> > >* extraNeighbors;
        void updateRepresentatiePixel(int& x, int& y, Disjoint_Set_Element* root, Disjoint_Set_Element* other);
	
    public:
		
		// Destructor, used to delete the background, which is allocated dynamically.
		~AdaptiveFeatures();
		/**
 		* TODO: COMMENT
 		* 
 		* @param Parameters *param, which gives access to the number of columns, number of rows,
 		*                   number of colors and the background information
 		* @return nothing, it is a constructor.
 		*/
		AdaptiveFeatures(Parameters *param);
		/**
 		* This method is the instantiation of the virtual method in the class Features (also check
 		* its documentation). It iterates over all tiles defined by the columns and rows and checks
 		* if each of the colors to be evaluated are present in the tile. 
 		* 
 		* REMARKS: - It is necessary to provide both the screen and the ram because of the superclass,
 		* despite the RAM being useless here. In fact a null pointer works just fine.
 		*          - To avoid return huge vectors, this method is void and the appropriate
 		* vector is returned trough a parameter passed by reference.
 		*          - This method was adapted from Sriram Srinivasan's code
 		* 
 		* @param ALEScreen &screen is the current game screen that one may use to extract features.
 		* @param ALERAM &ram is the current game RAM that one may use to extract features.
 		* @param vector<int>& features an empy vector that will be filled with the requested information,
 		*        therefore it must be passed by reference. It contain the active indices.
 		* @return nothing as one will receive the requested data by the last parameter, by reference.
 		*/
        virtual void getActiveFeaturesIndices(const ALEScreen &screen, const ALERAM &ram, vector<long long>& activeFeatures);
        void update(float delta);
        void promoteFeatures(long long frames);
        long long getNumFeatures();
        //void demoteFeature(long long index);
};
#endif