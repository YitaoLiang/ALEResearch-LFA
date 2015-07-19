/****************************************************************************************
** Implementation of a variation of BASS Features, which has features to encode the 
**  relative position between tiles.
**
** REMARKS: - This implementation is basically Erik Talvitie's implementation, presented
**            in the AAAI'15 LGCVG Workshop.
**
** Author: Marlos C. Machado
***************************************************************************************/

#ifndef Blob_BPRO_FEATURES_H
#define Blob_BPRO_FEATURES_H
#include "BlobBproFeatures.hpp"
#endif
#ifndef BASIC_FEATURES_H
#define BASIC_FEATURES_H
#include "BasicFeatures.hpp"
#endif

#include <set>
#include <assert.h>
#include <algorithm>
#include <math.h>
using namespace std;


BlobBproFeatures::BlobBproFeatures(Parameters *param){
    this->param = param;
    numColumns  = param->getNumColumns();
	numRows     = param->getNumRows();
	numColors   = param->getNumColors();

	if(this->param->getSubtractBackground()){
        this->background = new Background(param);
    }

	//To get the total number of features:
  
    numRelativeFeatures = (2 * 8 + 1) * (2 * 8 + 1)* (1+this->param->getNumColors()) * this->param->getNumColors()/2;
    
    changed.clear();
    bproExistence.resize(2*8+1);
    for (int i=0;i<2*8+1;i++){
        bproExistence[i].resize(2*8+1);
        for (int j=0;j<2*8+1;j++){
            bproExistence[i][j]=true;
        }
    }
}

BlobBproFeatures::~BlobBproFeatures(){}

void BlobBproFeatures::getBlobs(const ALEScreen &screen){
    int screenWidth = screen.width();
    int screenHeight = screen.height();
    
    
    vector<vector<int> > screenPixels(screenHeight,vector<int>(screenWidth,0));
   
     vector<tuple<int,int> > neighbors;
    for (int xDelta=-2;xDelta<0;xDelta++){
        for (int yDelta=-2;yDelta<=2;yDelta++){
            neighbors.push_back(make_tuple(xDelta,yDelta));
        }
    }
    neighbors.push_back(make_tuple(0,-2)); neighbors.push_back(make_tuple(0,-1));
    
    vector<Disjoint_Set_Element> disjoint_set;
    
    unordered_map<int,int> disjoint_set_indices;
    
    unordered_map<int,set<int> > blobIndices;
    
    
    for (int x=0;x<screenHeight;x++){
        for (int y=0;y<screenWidth;y++){
            set<tuple<int,int> > possibleBlobs;
            int color = screen.get(x,y);
            set<int> route;
            for (auto it=neighbors.begin();it!=neighbors.end();it++){
                int neighborX = get<0>(*it)+x;
                int neighborY = get<1>(*it)+y;
                if (neighborX>=0 && neighborY>=0 && neighborY<=screenWidth && screen.get(neighborX,neighborY)==color){
                    int posIndex = screenPixels[neighborX][neighborY];
                    route.insert(posIndex);
                    while (disjoint_set[posIndex].parent!=posIndex){
                        posIndex = disjoint_set[posIndex].parent;
                    }
                    possibleBlobs.insert(disjoint_set[posIndex].pos);
                }
            }
            
            tuple<int,int> pos(x,y);
            
            //case 1: it is the first pixel in this blob
            if (possibleBlobs.size()==0){
                addNewBlob(blobIndices, screenPixels, disjoint_set,disjoint_set_indices,pos,color/2);
                screenPixels[x][y]=disjoint_set.size()-1;
            
            //case 2: it belongs to some blob
            }else {
                possibleBlobs.insert(pos);
                tuple<int,int> representative;
                for (auto it=possibleBlobs.begin();it!=possibleBlobs.end();it++){
                    get<0>(representative) = min(get<0>(representative),get<0>(*it));
                    get<1>(representative) = min(get<1>(representative),get<1>(*it));
                }
                auto found = possibleBlobs.find(representative);
                if (found==possibleBlobs.end() || *found==pos){
                    addNewBlob(blobIndices, screenPixels, disjoint_set,disjoint_set_indices,representative,color/2);
                    screenPixels[x][y]=disjoint_set.size()-1;
                    mergeDisjointSet(route,disjoint_set,(int)disjoint_set.size()-1);
                }else{
                    screenPixels[x][y] = disjoint_set_indices[get<0>(*found)*1000+get<1>(*found)];
                    mergeDisjointSet(route,disjoint_set,disjoint_set_indices[get<0>(*found)*1000+get<1>(*found)]);
                }
            }
        }
    }
    for (auto it = blobIndices.begin();it!=blobIndices.end();it++){
        for (auto itt=it->second.begin();itt!=it->second.end();itt++){
            if (disjoint_set[*itt].parent==*itt){
                blobs[it->first].insert(disjoint_set[*itt].pos);
            }
        }
    }
}

void BlobBproFeatures::addRelativeFeaturesIndices(const ALEScreen &screen,vector<long long>& features){
    int numRowOffsets = 2*8 + 1;
    int numColumnOffsets = 2*8+ 1;
    for (int c1=0;c1<numColors;c1++){
        if (blobs[c1].size()>0){
            for (auto k=blobs[c1].begin();k!=blobs[c1].end();k++){
                for (auto h=blobs[c1].begin();h!=blobs[c1].end();h++){
                    int rowDelta = get<0>(*k)-get<0>(*h);
                    int columnDelta = get<1>(*k)-get<1>(*k);
                    rowDelta = getPowerTwoOffset(rowDelta);
                    columnDelta=getPowerTwoOffset(columnDelta);
                    bool newBproFeature = false;
                    if (rowDelta>0){
                        newBproFeature = true;
                    }else if (rowDelta==0 && columnDelta >=0){
                        newBproFeature = true;
                    }
                    rowDelta+=8;
                    columnDelta+=8;
                    if (newBproFeature && bproExistence[rowDelta][columnDelta]){
                        tuple<int,int> pos (rowDelta,columnDelta);
                        changed.push_back(pos);
                        bproExistence[rowDelta][columnDelta]=false;
                        features.push_back((numColors+numColors-c1+1)*c1/2*numRowOffsets*numColumnOffsets+rowDelta*numColumnOffsets+columnDelta);
                        
                    }
                }
            }
            resetBproExistence(bproExistence,changed);
        }
        
        for (int c2=c1+1;c2<numColors;c2++){
            if (blobs[c1].size()>0 && blobs[c2].size()>0){
                for (auto it1=blobs[c1].begin();it1!=blobs[c1].end();it1++){
                    for (auto it2=blobs[c2].begin();it2!=blobs[c2].end();it2++){
                        int rowDelta = getPowerTwoOffset(get<0>(*it1)-get<0>(*it2))+8;
                        int columnDelta = getPowerTwoOffset(get<1>(*it1)-get<1>(*it2))+8;
                        if (bproExistence[rowDelta][columnDelta]){
                            tuple<int,int> pos(rowDelta,columnDelta);
                            changed.push_back(pos);
                            bproExistence[rowDelta][columnDelta]=false;
                            features.push_back(numBasicFeatures+(numColors+numColors-c1+1)*c1/2*numRowOffsets*numColumnOffsets+(c2-c1)*numRowOffsets*numColumnOffsets+rowDelta*numColumnOffsets+columnDelta);
                        }
                    }
                }
            }
            resetBproExistence(bproExistence,changed);
        }
    }
}

void BlobBproFeatures::getActiveFeaturesIndices(const ALEScreen &screen, const ALERAM &ram, vector<long long>& features){
	
    blobs.clear();
    getBlobs(screen);
    addRelativeFeaturesIndices(screen,features);
}

long long BlobBproFeatures::getNumberOfFeatures(){
    return numRelativeFeatures + 1;
}

void BlobBproFeatures::resetBproExistence(vector<vector<bool> >& bproExistence, vector<tuple<int,int> >& changed){
    for (vector<tuple<int,int> >::iterator it = changed.begin();it!=changed.end();it++){
        bproExistence[get<0>(*it)][get<1>(*it)]=true;
    }
    changed.clear();
}

void BlobBproFeatures::addNewBlob(unordered_map<int,set<int> >& blobIndices,vector<vector<int> >& screenPixels, vector<Disjoint_Set_Element>& disjoint_set, unordered_map<int,int>&disjoint_set_indices, tuple<int,int>& pos, int color){
    blobIndices[color].insert(disjoint_set.size());
    Disjoint_Set_Element element;
    element.pos = pos;
    element.parent = disjoint_set.size();
    disjoint_set_indices[get<0>(pos)*1000+get<1>(pos)]=disjoint_set.size();
    disjoint_set.push_back(element);
}

void BlobBproFeatures::mergeDisjointSet(set<int>& route, vector<Disjoint_Set_Element>& disjoint_set, int representativeIndex){
    for (auto it=route.begin();it!=route.end();it++){
        int posIndex = *it;
        while (disjoint_set[posIndex].parent!=posIndex && disjoint_set[posIndex].parent!=representativeIndex){
            int temp = posIndex;
            posIndex = disjoint_set[posIndex].parent;
            disjoint_set[temp].parent = representativeIndex;
        }
    }
    
}

int BlobBproFeatures::getPowerTwoOffset(int rawDelta){
    int multiplier = 1;
    if (rawDelta<0){
        multiplier = -1;
    }
    rawDelta = abs(rawDelta)+1;
    return ceil(log2(rawDelta))*multiplier;
}