/****************************************************************************************
 ** Implementation of an adaptive feature representation
 **
 ** Author: Yitao Liang
 ***************************************************************************************/
#include <set>
#include <assert.h>
#include <algorithm>
#include <math.h>
#include <unordered_set>
#include <unordered_map>
#include <queue>

#include "AdaptiveFeatures.hpp"

using namespace std;

AdaptiveFeatures::AdaptiveFeatures(ALEInterface& ale, Parameters *param){
     this->param = param;
     numColors   = param->getNumColors();
     colorMultiplier =(int) log2(256 / numColors);
     numPromotions = param->getNumPromotions();
     numFeatures = 1;
    
    //Get the number of effective actions:
    if(param->isMinimalAction()){
       numActions = ale.getMinimalActionSet().size();
    }
    else{
        numActions = ale.getLegalActionSet().size();
    }
     
     constructBaseFeatures();
    
    neighborSize = param->getNeighborSize();
    vector<tuple<int,int> > fullNeighborOffsets;
    for (int xDelta=-neighborSize;xDelta<0;++xDelta){
        for (int yDelta=-neighborSize;yDelta<=neighborSize;++yDelta){
            fullNeighborOffsets.push_back(make_tuple(xDelta,yDelta));
        }
    }
    for (int yDelta=-neighborSize;yDelta<0;++yDelta){
        fullNeighborOffsets.push_back(make_tuple(0,yDelta));
    }
    vector<tuple<int,int> > extraNeighborOffsets;
    extraNeighborOffsets.push_back(make_tuple(0,-1));
    for (int xDelta=-neighborSize;xDelta<0;++xDelta){
        extraNeighborOffsets.push_back(make_tuple(xDelta,neighborSize));
    }
    fullNeighbors = new vector<vector<vector<unsigned short> > >(210);
    extraNeighbors = new vector<vector<vector<unsigned short> > >(210);
    for (unsigned short row=0; row<210;++row){
        fullNeighbors->at(row).resize(160);
        extraNeighbors->at(row).resize(160);
        for (unsigned short column=0; column<160;++column){
            for (auto it = fullNeighborOffsets.begin();it!=fullNeighborOffsets.end();++it){
                unsigned short neighborX = row + get<0>(*it);
                unsigned short neighborY = column + get<1>(*it);
                if (neighborX>=0 && neighborX<210 && neighborY>=0 && neighborY<160){
                    fullNeighbors->at(row)[column].push_back(neighborX*160+neighborY);
                }
            }
            
            for (auto it = extraNeighborOffsets.begin();it!=extraNeighborOffsets.end();++it){
                unsigned short neighborX = row + get<0>(*it);
                unsigned short neighborY = column + get<1>(*it);
                if (neighborX>=0 && neighborX<210 && neighborY>=0 && neighborY<160){
                    extraNeighbors->at(row)[column].push_back(neighborX*160+neighborY);
                }
            }
        }
    }
}

AdaptiveFeatures::~AdaptiveFeatures(){
    queue<Feature*> q;
    q.push(rootFeature);
    while(!q.empty()){
        Feature* f = q.front();
        q.pop();
        for (auto child:f->children){
            q.push(child);
        }
        delete f;
    }
    
    delete fullNeighbors;
    delete extraNeighbors;
}

void AdaptiveFeatures::constructBaseFeatures(){
    rootFeature = new Feature(0,0,0,0,210,160);
    rootFeature->active = true;
    rootFeature->previousActive = true;
    //initial base features: will be whether there is a color k pixel on the screen
    for (int i=0;i<numColors;++i){
        baseFeatures.push_back(new Feature(numFeatures++,i,0,0,210,160));
        rootFeature->children.push_back(baseFeatures[i]);
        generateCandidateFeatures(baseFeatures[i]);
    }
}

void AdaptiveFeatures::getActiveFeaturesIndices(const ALEScreen &screen, const ALERAM &ram, vector<long long>& activeFeatures){
    //get the information on the screen
    getBlobs(screen);

    //bias feature
    activeFeatures.push_back(0);
    
    for (int color=0;color<numColors;++color){
        baseFeatures[color]->previousActive = baseFeatures[color]->active;
        if (blobs[color].size()>0){
            baseFeatures[color]->active = true;
            activeFeatures.push_back(baseFeatures[color]->featureIndex);
            for (auto child:baseFeatures[color]->children){
                recursionToCheckFeatures(child,blobs[color],activeFeatures);
            }
        }else{
            baseFeatures[color]->active = false;
        }
    }
}

void AdaptiveFeatures::recursionToCheckFeatures(Feature*& current, vector<tuple<int,int> > possibleAnchorPositions, vector<long long>& activeFeatures){
    //check whether the feature is on
    vector<tuple<int,int> > anchors;
    current->previousActive = current->active;
    current->active = false;
    if (current->extraOffset){
        Location* offset = &(current->offsets.back());
        vector<tuple<int,int> >* possiblePositions = &blobs[offset->color];
        for (auto p:possibleAnchorPositions){
            int anchorX = get<0>(p); int anchorY = get<1>(p);
            auto s = make_tuple(min(209,anchorX+offset->resolutionX * offset->x),min(159,anchorY+offset->resolutionY * offset->y));
            auto l = make_tuple(min(209,anchorX+offset->resolutionX*(1+offset->x)-1),min(159,anchorY+offset->resolutionY*(1+offset->y)-1));
            auto fi = lower_bound(possiblePositions->begin(),possiblePositions->end(),s);
            if (fi!=possiblePositions->end()){
                for (auto i =fi;*i<=l;++i){
                    if (get<1>(*i)>=get<1>(s) && get<1>(*i)<=get<1>(l)){
                        current->active = true;
                        anchors.push_back(p);
                        break;
                    }
                }
            }
        }
    }else{
        //determine the range of current feature's location
        Location* location = &current->location;
        
        auto s = make_tuple(min(209,location->x*location->resolutionX),min(159,location->y * location->resolutionY));
        auto l = make_tuple(min(209,(location->x+1)*location->resolutionX-1),min(159,(1+location->y)*location->resolutionY-1));
        auto lb = lower_bound(possibleAnchorPositions.begin(),possibleAnchorPositions.end(),s);
        if (lb!=possibleAnchorPositions.end()){
            for (auto p = lb;*p<=l;++p){
                if (get<1>(*p)>=get<1>(s) && get<1>(*p)<=get<1>(l)){
                    anchors.push_back(*p);
                }
            }
            if (anchors.size()>0){
                current->active = true;
            }
        }
    }
    
    //check its children
    if (current->active){
        if (current->featureIndex){
            activeFeatures.push_back(current->featureIndex);
        }
        for  (int i=0;i<current->children.size();++i){
            recursionToCheckFeatures(current->children[i],anchors,activeFeatures);
        }
    }
}

void AdaptiveFeatures::promoteFeatures(){
    //features to promote
    vector<Feature*> bestCandidates(numPromotions,NULL);
    vector<float> bestCandidateValues(numPromotions,0.0);
    
    queue<Feature*> q;
    q.push(rootFeature);
    while (!q.empty()){
        Feature* f = q.front();
        q.pop();
        cout<<f->featureIndex<<' '<<f->sumDelta<<endl;
        //add its children to queue
        for (auto child:f->children){
            if (child->featureIndex==0){
                cout<<child->sumDelta<<' '<<f->sumDelta*child->sumDelta<<endl;
                //determine whether the candidate is worth promotion
                float v = -1;
                if (child->sumDelta * f->sumDelta <0){
                    v = fabs(child->sumDelta);
                }
                else if (fabs(child->sumDelta)>fabs(f->sumDelta)){
                    v = fabs(child->sumDelta) - fabs(f->sumDelta);
                }
                
                //rank the candidates' position in our promotion list
                if (v>=0){
                    int index = numPromotions;
                    while (index>0 && v>bestCandidateValues[index-1]){
                        --index;
                    }
                    for (int i=numPromotions-1;i>index;--i){
                        bestCandidates[i] = bestCandidates[i-1];
                        bestCandidateValues[i] = bestCandidateValues[i-1];
                    }
                    if (index<numPromotions){
                        bestCandidates[index] = f;
                        bestCandidateValues[index] = v;
                    }
                    
                }
            }else{
                q.push(child);
            }
        }
    }

    //promote the best candidates
    for (int i=0;i<numPromotions;++i){
        if (bestCandidates[i]){
            bestCandidates[i]->featureIndex = numFeatures;
            ++numFeatures;
            generateCandidateFeatures(bestCandidates[i]);
            ++s;
        }
    }
}

void AdaptiveFeatures::generateCandidateFeatures(Feature*& baseFeature){
    //refine its resolution
    Location* l = &(baseFeature->location);
    if (l->resolutionX>3 && l->resolutionY>3){
        int refinedResolutionX =  l->resolutionX/3;
        refinedResolutionX += (l->resolutionX%3==0)? 0:1;
        int refinedResolutionY =  l->resolutionY/3;
        refinedResolutionY += (l->resolutionY%3==0)? 0:1;
    
        for (int x = l->x * 3; x< l->x*3+3; ++x){
            for (int y = l->y*3; y< l->y*3+3; ++y){
                baseFeature->children.push_back(new Feature(0,l->color,x,y,refinedResolutionX,refinedResolutionY));
                baseFeature->children.back()->offsets = baseFeature->offsets;
            }
        }
    }
    
    //refine its offset resolution
    for (auto offset:baseFeature->offsets){
        if (offset.resolutionX>l->resolutionX){
            int refinedResolutionX =  offset.resolutionX/3;
            refinedResolutionX += (offset.resolutionX%3==0)? 0:1;
            int refinedResolutionY =  offset.resolutionY/3;
            refinedResolutionY += (offset.resolutionY%3==0)? 0:1;

            int negativeX = (-1)* (offset.x<0); int negativeY = (-1)*(offset.y<0);
            for (int x = offset.x * 3; x< offset.x*3+3;++x){
                for (int y = offset.y*3; y< offset.y*3+3;++y){
                    baseFeature->children.push_back(new Feature(0,l->color,l->x,l->y,l->resolutionX,l->resolutionY));
                    Feature* newF = baseFeature->children.back();
                    newF->offsets = baseFeature->offsets;
                    newF->offsets.push_back(Location(offset.color,x*negativeX,y*negativeY,refinedResolutionX,refinedResolutionY));
                    newF->extraOffset = true;
                }
            }
        }
    }
    
    //add offsets
    int rx = 210; int ry = 180;
    for (int color = 0; color<numColors;++color){
        for (int x=0;x<1;++x){
            for (int y=0;y<1;++y){
                bool alreadyAdded = false;
                for (auto offset:baseFeature->offsets){
                    if (offset.color==color && offset.resolutionX==rx && offset.resolutionY==ry && offset.x ==x && offset.y ==y){
                        alreadyAdded = true;
                        break;
                    }
                }
                if (!alreadyAdded){
                    baseFeature->children.push_back(new Feature(0,l->color,l->x,l->y,l->resolutionX,l->resolutionY));
                    Feature* newF = baseFeature->children.back();
                    newF-> offsets = baseFeature->offsets;
                    newF->offsets.push_back(Location(color,x,y,rx,ry));
                    newF->extraOffset = true;
                }
            }
        }
    }
}

void AdaptiveFeatures::updateDelta(float delta){
    queue<Feature*> q;
    q.push(rootFeature);
    
    while (!q.empty()){
        Feature* f = q.front();
        q.pop();
        if (f->previousActive){
            f->sumDelta+=delta;
            for (auto child:f->children){
                q.push(child);
            }
        }
    }
}

/*void AdaptiveFeatures::updateWeights(vector<vector<float> >& w,float learningRate){
    
    
    queue<Feature*> q;
    q.push(rootFeature);
    
    while (!q.empty()){
        Feature* f = q.front();
        q.pop();
        
        long long index = f->featureIndex;
        for (int a=0;a<numActions;++a){
            w[a][index]+=learningRate*f->sumDelta[a];
        }
        
        for (auto child:f->children){
            if (child->featureIndex!=0) {
                q.push(child);
            }
        }
    }
}*/

void AdaptiveFeatures::resetDelta(){
    queue<Feature*> q;
    q.push(rootFeature);
    
    while (!q.empty()){
        Feature* f = q.front();
        q.pop();
        
        f->sumDelta = 0.0;
        
        for (auto child:f->children){
            q.push(child);
        }
    }

}

long long AdaptiveFeatures::getNumFeatures(){
    return numFeatures;
}

/*void Adaptive::demote(long long index){
    feature* demotedF  = indexToFeature[index];
    demotedF->featureIndex = 0;
    demotedF->numAppearance = 0;
    demotedF->sumDelta = 0.0;
    indexToFeature.erase(index);

    //erase all candidates that is based on this demoted feature
    vector<int> childrenToKeep; 
    for (unsigned i = 0;i<demotedF->children.size();++i){
        if (demotedF->children[i]->children.size()>0){
            childrenToKeep.push_back(i);
        }
    }
    for (unsigned i=0;i<childrenToKeep.size();++i){
        demotedF->children[i] = demotedF->children[childrenToKeep[i]];
    }
    demotedF->children.resize(childrenToKeep.size());
    }*/


void AdaptiveFeatures::getBlobs(const ALEScreen &screen){
    int screenWidth = 160;
    int screenHeight = 210;
    
    
    vector<int> screenPixels(screenHeight*screenWidth,-1);
    
    vector<Disjoint_Set_Element> disjoint_set;
    vector<unordered_set<int> > blobIndices(numColors,unordered_set<int>());
    vector<int> route;
    
    vector<vector<vector<unsigned short> > >* neighbors;
    
    for (int x=0;x<screenHeight;x++){
        for (int y=0;y<screenWidth;y++){
            int color = screen.get(x,y);
            color = color >>colorMultiplier;
            if (y>0 && color == screen.get(x,y-1)>>colorMultiplier){
                neighbors = extraNeighbors;
            }else{
                neighbors = fullNeighbors;
            }
            unsigned short currentIndex = x*screenWidth + y;
            int currentRoot = screenPixels[currentIndex];
            
            for (auto it=neighbors->at(x).at(y).begin();it!=neighbors->at(x).at(y).end();++it){
                int neighborRoot = screenPixels[*it];
                if (color == disjoint_set[neighborRoot].color){
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
                    
                   
                    
                    if (neighborRoot!=currentRoot){
                        auto blobNeighbor  = &disjoint_set[neighborRoot];
                        //case 1: current pixel does not belong to any blob
                        if (currentRoot==-1){
                            currentRoot=neighborRoot;
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
                                currentRoot = neighborRoot;
                            }else{
                                blobNeighbor->parent = currentRoot;
                                blobIndices[color].erase(neighborRoot);
                                updateRepresentatiePixel(x,y,currentBlob,blobNeighbor);
                            }
                        }
                    }
                }
            }
            screenPixels[currentIndex] = currentRoot;
            
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
    
    //get all the blobs
    blobs.clear();
    blobs.resize(numColors);
    for (int color = 0;color<numColors;++color){
        for (auto index=blobIndices[color].begin();index!=blobIndices[color].end();++index){
            int x = (disjoint_set[*index].rowUp+disjoint_set[*index].rowDown)/2;
            int y = (disjoint_set[*index].columnLeft+disjoint_set[*index].columnRight)/2;
            blobs[color].push_back(make_tuple(x,y));
        }
        sort(blobs[color].begin(),blobs[color].end());
    }
}

void AdaptiveFeatures::updateRepresentatiePixel(int& x, int& y, Disjoint_Set_Element* root, Disjoint_Set_Element* other){
    root->rowUp = min(root->rowUp,other->rowUp);
    root->rowDown = x;
    root->columnLeft = min(root->columnLeft,other->columnLeft);
    root->columnRight = max(root->columnRight,other->columnRight);
    root->size += other->size;
}

void AdaptiveFeatures::resetActive(){
    queue<Feature*> q;
    q.push(rootFeature);
    
    while (!q.empty()){
        Feature* f = q.front();
        q.pop();
        
        f->active = false;
        f->previousActive = false;
        
        for (auto child:f->children){
            q.push(child);
        }
    }

}
