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
    //numColumns  = param->getNumColumns();
    //numRows     = param->getNumRows();
    numColors   = param->getNumColors();
    colorMultiplier =(int) log2(256 / numColors);
    
    if(this->param->getSubtractBackground()){
        this->background = new Background(param);
    }
    
    //To get the total number of features:
    resolutions.push_back(make_tuple(15,10)); resolutions.push_back(make_tuple(3,2)); resolutions.push_back(make_tuple(7,4));
    numBlocks.push_back(make_tuple(210/get<0>(resolutions[0]),160/get<1>(resolutions[0]))); numBlocks.push_back(make_tuple(210/get<0>(resolutions[1]),160/get<1>(resolutions[1]))); numBlocks.push_back(make_tuple(210/get<0>(resolutions[2]),160/get<1>(resolutions[2])));
    numOffsets.push_back(make_tuple(2 * get<0>(numBlocks[0])-1, 2*get<1>(numBlocks[0])-1));  numOffsets.push_back(make_tuple(2 * get<0>(numBlocks[1])-1, 2*get<1>(numBlocks[1])-1));  numOffsets.push_back(make_tuple(2 * get<0>(numBlocks[2])-1, 2*get<1>(numBlocks[2])-1));
    
    numBasicFeaturesPart1 = numColors*get<0>(numBlocks[0])*get<1>(numBlocks[0]); numBasicFeaturesPart2 =numColors*get<0>(numBlocks[1])*get<1>(numBlocks[1]); numBasicFeaturesPart3 = numColors*get<0>(numBlocks[2])*get<1>(numBlocks[2]);
    numBasicFeatures = numBasicFeaturesPart1 + numBasicFeaturesPart2 + numBasicFeaturesPart3;
    numRelativeFeatures = get<0>(numOffsets[0]) * get<1>(numOffsets[0])* (1+numColors) * numColors/2 + get<0>(numOffsets[1]) * get<1>(numOffsets[1]) * (1+numColors) * numColors/2 + get<0>(numOffsets[2]) * get<1>(numOffsets[2]) * (1+numColors) * numColors/2;
    numTimeDimensionalOffsets = get<0>(numOffsets[0]) * get<1>(numOffsets[0]) * numColors * numColors + get<0>(numOffsets[1]) * get<1>(numOffsets[1]) * numColors * numColors + get<0>(numOffsets[2]) * get<1>(numOffsets[2]) * numColors * numColors;
    
    baseBpro.push_back(numBasicFeatures); baseBpro.push_back(numBasicFeatures+get<0>(numOffsets[0]) * get<1>(numOffsets[0])* (1+numColors) * numColors/2); baseBpro.push_back(baseBpro.back()+get<0>(numOffsets[1]) * get<1>(numOffsets[1]) * (1+numColors) * numColors/2);
    baseTime.push_back(numBasicFeatures+numRelativeFeatures); baseTime.push_back(baseTime.back()+ get<0>(numOffsets[0]) * get<1>(numOffsets[0]) * numColors * numColors); baseTime.push_back(baseTime.back()+get<0>(numOffsets[1]) * get<1>(numOffsets[1]) * numColors * numColors);
    
    changedPart1.clear();
    bproExistencePart1.resize(get<0>(numOffsets[0]));
    for (int i=0;i<get<0>(numOffsets[0]);++i){
        bproExistencePart1[i].resize(get<1>(numOffsets[0]));
        for (int j=0;j<get<1>(numOffsets[1]);++j){
            bproExistencePart1[i][j]=true;
        }
    }
    
    changedPart2.clear();
    bproExistencePart2.resize(get<0>(numOffsets[1]));
    for (int i=0;i<get<0>(numOffsets[1]);++i){
        bproExistencePart2[i].resize(get<1>(numOffsets[1]));
        for (int j=0;j<get<1>(numOffsets[1]);++j){
            bproExistencePart2[i][j]=true;
        }
    }
    
    changedPart3.clear();
    bproExistencePart3.resize(get<0>(numOffsets[2]));
    for (int i=0;i<get<0>(numOffsets[2]);++i){
        bproExistencePart3[i].resize(get<1>(numOffsets[2]));
        for (int j=0;j<get<1>(numOffsets[2]);++j){
            bproExistencePart3[i][j]=true;
        }
    }
    
    changed.push_back(changedPart1); changed.push_back(changedPart2); changed.push_back(changedPart3);
    bproExistence.push_back(bproExistencePart1);  bproExistence.push_back(bproExistencePart2); bproExistence.push_back(bproExistencePart3);
    
    neighborSize = 3;
    fullNeighbors = new vector<tuple<int,int> >();
    for (int xDelta=-neighborSize;xDelta<0;++xDelta){
        for (int yDelta=-neighborSize;yDelta<=neighborSize;++yDelta){
            fullNeighbors->push_back(make_tuple(xDelta,yDelta));
        }
    }
    
    for (int yDelta=-neighborSize;yDelta<0;++yDelta){
        fullNeighbors->push_back(make_tuple(0,yDelta));
    }
    
    extraNeighbors = new vector<tuple<int,int> >();
    extraNeighbors->push_back(make_tuple(0,-1));
    for (int xDelta=-neighborSize;xDelta<0;++xDelta){
        extraNeighbors->push_back(make_tuple(xDelta,neighborSize));
    }
    previousBlobs.clear();
    
    
    
}

BlobTimeFeatures::~BlobTimeFeatures(){
    delete fullNeighbors;
    delete extraNeighbors;
}

void BlobTimeFeatures::getBlobs(const ALEScreen &screen){
    int screenWidth = screen.width();
    int screenHeight = screen.height();
    
    
    vector<int> screenPixels((screenHeight+neighborSize)*(screenWidth+2*neighborSize),-1);
    
    vector<Disjoint_Set_Element> disjoint_set;
    vector<unordered_set<int> > blobIndices(numColors,unordered_set<int>());
    vector<int> route;
    
    vector<tuple<int,int> >* neighbors;
    
    int width = screenWidth+2*neighborSize;
    
    for (int x=0;x<screenHeight;x++){
        for (int y=0;y<screenWidth;y++){
            int color = screen.get(x,y);
            color = color >>colorMultiplier;
            if (y>0 && color == screen.get(x,y-1)>>colorMultiplier){
                neighbors = extraNeighbors;
            }else{
                neighbors = fullNeighbors;
            }
            int currentIndex = (x+neighborSize)*width + (y+neighborSize);
            
            for (auto it=neighbors->begin();it!=neighbors->end();++it){
                int neighborX = get<0>(*it)+x;
                int neighborY = get<1>(*it)+y;
                
                int neighborRoot = screenPixels[(neighborX+neighborSize)*width+neighborY+neighborSize];
                if (neighborRoot!=-1 && color == disjoint_set[neighborRoot].color){
                    //get the true root
                    route.clear();
                    while (disjoint_set[neighborRoot].parent!=neighborRoot){
                        route.push_back(neighborRoot);
                        neighborRoot = disjoint_set[neighborRoot].parent;
                    }
                    
                    //maintain the disjoint_set
                    for (auto itt=route.begin();itt!=route.end();++itt){
                        disjoint_set[*itt].parent = neighborRoot;
                    }
                    
                    int currentRoot = screenPixels[currentIndex];
                    
                    if (neighborRoot!=currentRoot){
                        auto blobNeighbor  = &disjoint_set[neighborRoot];
                        //case 1: current pixel does not belong to any blob
                        if (currentRoot==-1){
                            screenPixels[currentIndex]=neighborRoot;
                            blobNeighbor->rowDown = x;
                            if (y < blobNeighbor->columnLeft){
                                blobNeighbor->columnLeft = y;
                            }else if (y > blobNeighbor->columnRight){
                                blobNeighbor->columnRight = y;
                            }
                            blobNeighbor->size+=1;
                            //case 2: current pixel belongs to two blobs
                        }else{
                            auto currentBlob = &disjoint_set[currentRoot];
                            if (blobNeighbor->size>currentBlob->size){
                                currentBlob->parent = neighborRoot;
                                blobIndices[color].erase(currentRoot);
                                updateRepresentatiePixel(x,y,blobNeighbor,currentBlob);
                                screenPixels[currentIndex] = neighborRoot;
                            }else{
                                blobNeighbor->parent = currentRoot;
                                blobIndices[color].erase(neighborRoot);
                                updateRepresentatiePixel(x,y,currentBlob,blobNeighbor);
                            }
                        }
                    }
                }
            }
            
            //current pixel is the first pixel for a new blob
            if (screenPixels[currentIndex]==-1){
                Disjoint_Set_Element element;
                element.columnLeft = y; element.columnRight = y;
                element.rowUp = x; element.rowDown = x;
                element.size = 1;
                element.parent = disjoint_set.size();
                screenPixels[currentIndex] = disjoint_set.size();
                element.color = color;
                blobIndices[color].insert(disjoint_set.size());
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
    for (int color = 0;color<numColors;++color){
        for (auto index=blobIndices[color].begin();index!=blobIndices[color].end();++index){
            int x = (disjoint_set[*index].rowUp+disjoint_set[*index].rowDown)/2;
            int y = (disjoint_set[*index].columnLeft+disjoint_set[*index].columnRight)/2;
            blobs[color].push_back(make_tuple(x,y));
        }
    }
}

void BlobTimeFeatures::addRelativeFeaturesIndices(vector<long long>& features){
    for (int c1=0;c1<numColors;++c1){
        if (blobs[c1].size()>0){
            for (auto k=blobs[c1].begin();k!=blobs[c1].end();++k){
                for (auto h=blobs[c1].begin();h!=blobs[c1].end();++h){
                    
                    for (int index=0;index<3;++index){
                        int rowDelta = (get<0>(*k)/get<0>(resolutions[index])-get<0>(*h))/get<0>(resolutions[index]);
                        int columnDelta = get<1>(*k)/get<1>(resolutions[index])-get<1>(*k)/get<1>(resolutions[index]);
                        bool newBproFeature = false;
                        if (rowDelta>0){
                            newBproFeature = true;
                        }else if (rowDelta==0 && columnDelta >=0){
                            newBproFeature = true;
                        }
                        rowDelta += get<0>(numBlocks[index])-1;
                        columnDelta += get<1>(numBlocks[index])-1;
                        if (newBproFeature && bproExistence[index][rowDelta][columnDelta]){
                            tuple<int,int> pos (rowDelta,columnDelta);
                            changed[index].push_back(pos);
                            bproExistence[index][rowDelta][columnDelta]=false;
                            features.push_back(baseBpro[index]+(numColors+numColors-c1+1)*c1/2*get<0>(numOffsets[index])*get<1>(numOffsets[index])+rowDelta*get<1>(numOffsets[index])+columnDelta);
                            
                        }
                    }
                    
                }
            }
            resetBproExistence();
        }
        for (int c2=c1+1;c2<numColors;++c2){
            if (blobs[c1].size()>0 && blobs[c2].size()>0){
                for (auto it1=blobs[c1].begin();it1!=blobs[c1].end();++it1){
                    for (auto it2=blobs[c2].begin();it2!=blobs[c2].end();++it2){
                        
                        for (int index=0;index<3;++index){
                            int rowDelta = get<0>(*it1)/get<0>(resolutions[index])-get<0>(*it2)/get<0>(resolutions[index])+get<0>(numBlocks[index])-1;
                            int columnDelta = get<1>(*it1)/get<1>(resolutions[index])-get<1>(*it2)/get<1>(resolutions[index])+get<1>(numBlocks[index])-1;
                            if (bproExistence[index][rowDelta][columnDelta]){
                                tuple<int,int> pos(rowDelta,columnDelta);
                                changed[index].push_back(pos);
                                bproExistence[index][rowDelta][columnDelta]=false;
                                features.push_back(baseBpro[index]+(numColors+numColors-c1+1)*c1/2*get<0>(numOffsets[index])*get<1>(numOffsets[index])+(c2-c1)*get<0>(numOffsets[index])*get<1>(numOffsets[index])+rowDelta*get<1>(numOffsets[index])+columnDelta);
                            }
                            
                        }
                    }
                }
                resetBproExistence();
            }
        }
    }
}

void BlobTimeFeatures::getBasicFeatures(vector<long long>& features,  unordered_map<int,vector<tuple<int,int> > >& blobs){
    for (auto it=blobs.begin();it!=blobs.end();++it){
        int color = it->first;
        long long base2 = numBasicFeaturesPart1;
        long long base3 = numBasicFeaturesPart1+numBasicFeaturesPart2;
        for (auto itt=it->second.begin();itt!=it->second.end();++itt){
            int x = get<0>(*itt);
            int y = get<1>(*itt);
            features.push_back(x/get<0>(resolutions[0])*get<1>(numBlocks[0])+y/get<1>(resolutions[0]));
            features.push_back(base2+x/get<0>(resolutions[1])*get<1>(numBlocks[1])+y/get<1>(resolutions[1]));
            features.push_back(base3+x/get<0>(resolutions[2])*get<1>(numBlocks[2])+y/get<1>(resolutions[2]));
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
    features.push_back(numBasicFeatures+numRelativeFeatures + numTimeDimensionalOffsets);
    previousBlobs = blobs;
}

void BlobTimeFeatures::addTimeDimensionalOffsets(vector<long long>& features){
    for (int c1=0;c1<numColors;++c1){
        for (int c2=0;c2<numColors;++c2){
            if (previousBlobs[c1].size()>0 && blobs[c2].size()>0){
                for (auto it1=previousBlobs[c1].begin();it1 !=previousBlobs[c1].end();++it1){
                    for (auto it2=blobs[c2].begin();it2 != blobs[c2].end();++it2){
                        
                        for (int index=0;index<3;++index){
                            int rowDelta =get<0>(*it1)/get<0>(resolutions[index])-get<0>(*it2)/get<0>(resolutions[index])+get<0>(numBlocks[index])-1;
                            int columnDelta = get<1>(*it1)/get<1>(resolutions[index])-get<1>(*it2)/get<1>(resolutions[index])+get<1>(numBlocks[index])-1;
                            if (bproExistence[index][rowDelta][columnDelta]){
                                tuple<int,int> pos(rowDelta,columnDelta);
                                changed[index].push_back(pos);
                                bproExistence[index][rowDelta][columnDelta]=false;
                                features.push_back(baseTime[index]+c1*numColors*get<0>(numOffsets[index])*get<1>(numOffsets[index])+c2*get<0>(numOffsets[index])*get<1>(numOffsets[index])+rowDelta*get<1>(numOffsets[index])+columnDelta);
                            }
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
    for (int index = 0; index<3;++index){
        for (vector<tuple<int,int> >::iterator it = changed[index].begin();it!=changed[index].end();++it){
            bproExistence[index][get<0>(*it)][get<1>(*it)]=true;
        }
        changed[index].clear();
    }
}

int BlobTimeFeatures::getPowerTwoOffset(int rawDelta){
    int multiplier = 1;
    if (rawDelta<0){
        multiplier = -1;
    }
    rawDelta = abs(rawDelta)+1;
    return ceil(log2(rawDelta))*multiplier;
}

void BlobTimeFeatures::updateRepresentatiePixel(int& x, int& y, Disjoint_Set_Element* root, Disjoint_Set_Element* other){
    root->rowUp = min(root->rowUp,other->rowUp);
    root->rowDown = x;
    root->columnLeft = min(root->columnLeft,other->columnLeft);
    root->columnRight = max(root->columnRight,other->columnRight);
    root->size += other->size;
}

void BlobTimeFeatures::clearCash(){
    previousBlobs.clear();
}