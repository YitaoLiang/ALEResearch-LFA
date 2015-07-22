/****************************************************************************************
** Implementation of a variation of BASS Features, which has features to encode the 
**  relative position between tiles.
**
** REMARKS: - This implementation is basically Erik Talvitie's implementation, presented
**            in the AAAI'15 LGCVG Workshop.
**
** Author: Marlos C. Machado
***************************************************************************************/

#ifndef Blob_TIME_FEATURES_H
#define Blob_TIME_FEATURES_H
#include "BlobTimeFeatures.hpp"
#endif

#include <set>
#include <assert.h>
#include <algorithm>
#include <math.h>
#include <unordered_set>
using namespace std;


BlobTimeFeatures::BlobTimeFeatures(Parameters *param){
    this->param = param;
    numColumns  = param->getNumColumns();
	numRows     = param->getNumRows();
	numColors   = param->getNumColors();
    colorMultiplier = 256 / numColors;

	if(this->param->getSubtractBackground()){
        this->background = new Background(param);
    }

	//To get the total number of features:
    numBasicFeaturesPart1 = numColors*(210/15)*(160/10); numBasicFeaturesPart2 =numColors*(210/3)*(160/2); numBasicFeaturesPart3 = numColors*(210/7)*(160/4);
    numBasicFeatures = numColors*(210/15)*(160/10) + numColors*(210/3)*(160/2) + numColors*(210/7)*(160/4);
    numRelativeFeatures = (2 * 8 + 1) * (2 * 8 + 1)* (1+this->param->getNumColors()) * this->param->getNumColors()/2;
    numTimeDimensionalOffsets = this->param->getNumColors() * this->param->getNumColors() *(2 * this->param->getNumColumns() - 1) * (2 * this->param->getNumRows() - 1) ;
    
    changed.clear();
    bproExistence.resize(2*8+1);
    for (int i=0;i<2*8+1;i++){
        bproExistence[i].resize(2*8+1);
        for (int j=0;j<2*8+1;j++){
            bproExistence[i][j]=true;
        }
    }
    
    for (int xDelta=-3;xDelta<0;xDelta++){
        for (int yDelta=-3;yDelta<=3;yDelta++){
            fullNeighbors.push_back(make_tuple(xDelta,yDelta));
        }
    }
    fullNeighbors.push_back(make_tuple(0,-3)); fullNeighbors.push_back(make_tuple(0,-2)); fullNeighbors.push_back(make_tuple(0,-1));
    
   extraNeighbors.push_back(make_tuple(0,-1)); extraNeighbors.push_back(make_tuple(-3,3)); extraNeighbors.push_back(make_tuple(-2,3));  extraNeighbors.push_back(make_tuple(-1,3));
    
    previousBlobs.clear();

}

BlobTimeFeatures::~BlobTimeFeatures(){}

void BlobTimeFeatures::getBlobs(const ALEScreen &screen){
    int screenWidth = screen.width();
    int screenHeight = screen.height();
    
    
    vector<vector<int> > screenPixels(screenHeight,vector<int>(screenWidth,-1));
    
    vector<Disjoint_Set_Element> disjoint_set;
    vector<unordered_set<int> > blobIndices(128,unordered_set<int>());
    vector<int> route;

    for (int x=0;x<screenHeight;x++){
        for (int y=0;y<screenWidth;y++){
            set<tuple<int,int> > possibleBlobs;
            int color = screen.get(x,y);
            vector<tuple<int,int> > neighbors;
            if (y>0 && color == screen.get(x,y-1)){
                neighbors = extraNeighbors;
            }else{
                neighbors = fullNeighbors;
            }
            for (auto it=neighbors.begin();it!=neighbors.end();++it){
                int neighborX = get<0>(*it)+x;
                int neighborY = get<1>(*it)+y;
                if (neighborX>=0 && neighborY>=0 && neighborY<screenWidth && screen.get(neighborX,neighborY)==color){
                    int posIndex = screenPixels[neighborX][neighborY];
                    
                    route.clear();
                    //get the true root
                    while (disjoint_set[posIndex].parent!=posIndex){
                        route.push_back(posIndex);
                        posIndex = disjoint_set[posIndex].parent;
                    }
                    
                    //maintain the disjoint_set
                    for (auto itt=route.begin();itt!=route.end();++itt){
                        disjoint_set[*itt].parent = posIndex;
                    }
                    
                    if (posIndex!=screenPixels[x][y]){
                        //case 1: current pixel does not belong to any blob
                        if (screenPixels[x][y]==-1){
                            screenPixels[x][y]=posIndex;
                            disjoint_set[posIndex].rowDown = x;
                            if (y < disjoint_set[posIndex].columnLeft){
                                disjoint_set[posIndex].columnLeft = y;
                            }else if (y > disjoint_set[posIndex].columnRight){
                                disjoint_set[posIndex].columnRight = y;
                            }
                            disjoint_set[posIndex].size+=1;
                        //case 2: current pixel belongs to two blobs
                        }else{
                            if (disjoint_set[posIndex].size>disjoint_set[screenPixels[x][y]].size){
                                disjoint_set[screenPixels[x][y]].parent = posIndex;
                                blobIndices[color/colorMultiplier].erase(screenPixels[x][y]);
                                screenPixels[x][y] = posIndex;
                                updateRepresentatiePixel(x,y,posIndex,screenPixels[x][y],disjoint_set);
                            }else{
                                disjoint_set[posIndex].parent = screenPixels[x][y];
                                blobIndices[color/colorMultiplier].erase(posIndex);
                                updateRepresentatiePixel(x,y,screenPixels[x][y],posIndex,disjoint_set);
                            }
                        }
                    }
                }
            }
            
            //current pixel is the first pixel for a new blob
            if (screenPixels[x][y]==-1){
                Disjoint_Set_Element element;
                element.columnLeft = y; element.columnRight = y;
                element.rowUp = x; element.rowDown = x;
                element.size = 1;
                element.parent = disjoint_set.size();
                screenPixels[x][y] = disjoint_set.size();
                //element.color = color / colorMultiplier;
                blobIndices[color/colorMultiplier].insert(disjoint_set.size());
                disjoint_set.push_back(element);
            }
            
        }
    }
    
    /*
    for (int index = 0;index<disjoint_set.size();index++){
        if (disjoint_set[index].parent==index){
            blobs[disjoint_set[index].color].push_back(make_tuple((disjoint_set[index].rowUp+disjoint_set[index].rowDown)/2,(disjoint_set[index].columnLeft+disjoint_set[index].columnRight)/2));
        }
    }*/
    
    //get all the blobs
    for (int color = 0;color<numColors;color++){
        for (auto index=blobIndices[color].begin();index!=blobIndices[color].end();++index){
            int x = (disjoint_set[*index].rowUp+disjoint_set[*index].rowDown)/2;
            int y = (disjoint_set[*index].columnLeft+disjoint_set[*index].columnRight)/2;
            blobs[color].push_back(make_tuple(x,y));
        }
    }
}

void BlobTimeFeatures::addRelativeFeaturesIndices(vector<long long>& features){
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
            resetBproExistence();
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
            resetBproExistence();
        }
    }
}

void BlobTimeFeatures::getBasicFeatures(vector<long long>& features,  unordered_map<int,vector<tuple<int,int> > >& blobs){
    for (auto it=blobs.begin();it!=blobs.end();++it){
        int color = it->first;
        long long base1 = color*14*16;
        long long base2 = numBasicFeaturesPart1+color*70*80;
        long long base3 = numBasicFeaturesPart1+numBasicFeaturesPart2+color*30*40;
        for (auto itt=it->second.begin();itt!=it->second.end();++itt){
            int x = get<0>(*itt);
            int y = get<1>(*itt);
            features.push_back(base1+(x/15)*16+(y/10));
            features.push_back(base2+(x/3)*80+(y/2));
            features.push_back(base3+(x/7)*40+(y/4));
        }
    }
}

void BlobTimeFeatures::getActiveFeaturesIndices(const ALEScreen &screen, const ALERAM &ram, vector<long long>& features){
	
    blobs.clear();
    getBlobs(screen);
    /*for (auto it=blobs.begin();it!=blobs.end();++it){
        cout<<it->first<<' '<<it->second.size()<<endl;
    }
    cout<<endl;*/
    getBasicFeatures(features,blobs);
    addRelativeFeaturesIndices(features);
    if (previousBlobs.size()>0){
        addTimeDimensionalOffsets(features);
    }
    previousBlobs = blobs;
}

void BlobTimeFeatures::addTimeDimensionalOffsets(vector<long long>& features){
    int numRowOffsets = 2*8 + 1;
    int numColumnOffsets = 2*8+ 1;
    for (int c1=0;c1<numColors;++c1){
        for (int c2=0;c2<numColors;++c2){
            if (previousBlobs[c1].size()>0 && blobs[c2].size()>0){
                for (auto it1=previousBlobs[c1].begin();it1 !=previousBlobs[c1].end();++it1){
                    for (auto it2=blobs[c2].begin();it2 != blobs[c2].end();++it2){
                        int rowDelta = getPowerTwoOffset(get<0>(*it1)-get<0>(*it2))+8;
                        int columnDelta = getPowerTwoOffset(get<1>(*it1)-get<1>(*it2))+8;
                        if (bproExistence[rowDelta][columnDelta]){
                            tuple<int,int> pos(rowDelta,columnDelta);
                            changed.push_back(pos);
                            bproExistence[rowDelta][columnDelta]=false;
                            features.push_back(numBasicFeatures+numRelativeFeatures+c1*numColors*numRowOffsets*numColumnOffsets+c2*numRowOffsets*numColumnOffsets+rowDelta*numColumnOffsets+columnDelta);
                        }
                    }
                }
                resetBproExistence();
            }
        }
    }

}

long long BlobTimeFeatures::getNumberOfFeatures(){
    return numBasicFeatures+numRelativeFeatures + numTimeDimensionalOffsets+1;
}

void BlobTimeFeatures::resetBproExistence(){
    for (vector<tuple<int,int> >::iterator it = changed.begin();it!=changed.end();it++){
        bproExistence[get<0>(*it)][get<1>(*it)]=true;
    }
    changed.clear();
}

int BlobTimeFeatures::getPowerTwoOffset(int rawDelta){
    int multiplier = 1;
    if (rawDelta<0){
        multiplier = -1;
    }
    rawDelta = abs(rawDelta)+1;
    return ceil(log2(rawDelta))*multiplier;
}

void BlobTimeFeatures::updateRepresentatiePixel(int& x, int& y, int& root, int& other,vector<Disjoint_Set_Element>& disjoint_set){
    disjoint_set[root].rowUp = min(disjoint_set[root].rowUp,disjoint_set[other].rowUp);
    disjoint_set[root].rowDown = x;
    disjoint_set[root].columnLeft = min(disjoint_set[root].columnLeft,disjoint_set[other].columnLeft);
    disjoint_set[root].columnRight = max(disjoint_set[root].columnRight,disjoint_set[other].columnRight);
    disjoint_set[root].size += disjoint_set[other].size;
}

void BlobTimeFeatures::clearCash(){
    previousBlobs.clear();
}