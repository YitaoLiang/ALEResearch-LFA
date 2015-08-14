/****************************************************************************************
 ** Implementation of Sarsa(lambda). It implements Fig. 8.8 (Linear, gradient-descent
 ** Sarsa(lambda)) from the book "R. Sutton and A. Barto; Reinforcement Learning: An
 ** Introduction. 1st edition. 1988."
 ** Some updates are made to make it more efficient, as not iterating over all features.
 **
 ** TODO: Make it as efficient as possible.
 **
 ** Author: Marlos C. Machado
 ***************************************************************************************/

#ifndef TIMER_H
#define TIMER_H
#include "../../../common/Timer.hpp"
#endif
#include "SarsaLearner.hpp"
#include <stdio.h>
#include <math.h>
#include <set>
using namespace std;

SarsaLearner::SarsaLearner(ALEInterface& ale, Features *features, Parameters *param,int seed) : RLLearner(ale, param,seed) {
    
    totalNumberFrames = 0.0;
    maxFeatVectorNorm = 1;
    saveThreshold =0;
    
    delta = 0.0;
    alpha = param->getAlpha();
    learningRate = alpha;
    lambda = param->getLambda();
    traceThreshold = param->getTraceThreshold();
    numFeatures = features->getNumberOfFeatures();
    toSaveCheckPoint = param->getToSaveCheckPoint();
    saveWeightsEveryXFrames = param->getFrequencySavingWeights();
    pathWeightsFileToLoad = param->getPathToWeightsFiles();
    randomNoOp = param->getRandomNoOp();
    noOpMax = param->getNoOpMax();
    numStepsPerAction = param->getNumStepsPerAction();
    hashTableSize = param->getHashTableSize();
    
    e = new float* [numActions];
    w = new float* [numActions];
    Q.resize(numActions); Qnext.resize(numActions);
    for(int i = 0; i < numActions; i++){
        //Initialize e, w, featureValues
        e[i] = new float [hashTableSize+1]; memset(e[i],0,(hashTableSize+1)*sizeof(float));
        w[i] = new float [hashTableSize+1]; memset(w[i],0,(hashTableSize+1)*sizeof(float));
        nonZeroElig.push_back(vector<long long>());
    }
    featureValues = new int [hashTableSize]; memset(featureValues,0,hashTableSize*sizeof(int));
    activeHashedFeatures = new bool[hashTableSize]; memset(activeHashedFeatures,0,hashTableSize*sizeof(bool));
    
    episodePassed = 0;
    if(toSaveCheckPoint){
        checkPointName = param->getCheckPointName();
        //load CheckPoint
        ifstream checkPointToLoad;
        string checkPointLoadName = checkPointName+"-checkPoint.txt";
        checkPointToLoad.open(checkPointLoadName.c_str());
        if (checkPointToLoad.is_open()){
            loadCheckPoint(checkPointToLoad);
            remove(checkPointLoadName.c_str());
        }
        saveThreshold = (totalNumberFrames/saveWeightsEveryXFrames)*saveWeightsEveryXFrames;
        ofstream learningConditionFile;
        nameForLearningCondition = checkPointName+"-learningCondition-Frames"+to_string(saveThreshold)+"-finished.txt";
        string previousNameForLearningCondition =checkPointName +"-learningCondition.txt";
        rename(previousNameForLearningCondition.c_str(), nameForLearningCondition.c_str());
        saveThreshold+=saveWeightsEveryXFrames;
        learningConditionFile.open(nameForLearningCondition, ios_base::app);
        learningConditionFile.close();
    }
    
    fourwiseHash.seed(seed+1,hashTableSize);
}

SarsaLearner::~SarsaLearner(){
    for (unsigned i=0;i<numActions;i++){
        delete[] e[i];
        delete[] w[i];
    }
    delete[] e;
    delete[] w;
    delete[] featureValues;
    delete[] activeHashedFeatures;
}

void SarsaLearner::updateQValues(vector<long long> &Features,vector<float>& QValues){
    unsigned long long featureSize = Features.size();
    for (unsigned a=0;a<numActions;++a){
        float sumQ = 0.0;
        for (unsigned long long i=0;i<featureSize;i++){
            sumQ+=w[a][Features[i]]*featureValues[Features[i]];
        }
        sumQ += w[a][hashTableSize];
        QValues[a]=sumQ;
    }
}

void SarsaLearner::updateAcumTrace(int action, vector<long long> &Features){
    //e <- gamma * lambda * e
    for(unsigned int a = 0; a < nonZeroElig.size(); a++){
        long long numNonZero = 0;
        for(unsigned int i = 0; i < nonZeroElig[a].size(); i++){
            long long idx = nonZeroElig[a][i];
            //To keep the trace sparse, if it is
            //less than a threshold it is zero-ed.
            e[a][idx] = gamma * lambda * e[a][idx];
            if(fabs(e[a][idx]) < traceThreshold){
                e[a][idx] = 0;
            }
            else{
                nonZeroElig[a][numNonZero] = idx;
                numNonZero++;
            }
        }
        nonZeroElig[a].resize(numNonZero);
    }
    
    //For all i in Fa:
    nonZeroElig[action].clear();
    for(unsigned int i = 0; i < F.size(); i++){
        long long idx = Features[i];
        e[action][idx] += featureValues[idx];
        if (fabs(e[action][idx])>=traceThreshold){
            nonZeroElig[action].push_back(idx);
        }else{
            e[action][idx] = 0;
        }
    }
    
    //update trace for bias feature
    e[action][hashTableSize]+=1;
    nonZeroElig[action].push_back(hashTableSize);
}

void SarsaLearner::sanityCheck(){
    for(int i = 0; i < numActions; i++){
        if(fabs(Q[i]) > 10e7 || Q[i] != Q[i] /*NaN*/){
            std::string oldName = checkPointName+"-Result-writing.txt";
            std::string newName = checkPointName+"-Result-finished.txt";
            std::ofstream resultFile;
            resultFile.open(oldName.c_str());
            resultFile<<"It seems your algorithm diverged!\n";
            printf("It seems your algorithm diverged!\n");
            resultFile.close();
            rename(oldName.c_str(),newName.c_str());
            exit(0);
        }
    }
}

//To do: we do not want to save weights that are zero
void SarsaLearner::saveCheckPoint(int episode, int totalNumberFrames, vector<float>& episodeResults,int& frequency,vector<int>& episodeFrames, vector<double>& episodeFps){
    ofstream learningConditionFile;
    string newNameForLearningCondition = checkPointName+"-learningCondition-Frames"+to_string(saveThreshold)+"-writing.txt";
    int renameReturnCode = rename(nameForLearningCondition.c_str(),newNameForLearningCondition.c_str());
    if (renameReturnCode == 0){
        nameForLearningCondition = newNameForLearningCondition;
        learningConditionFile.open(nameForLearningCondition.c_str(), ios_base::app);
        int numEpisode = episodeResults.size();
        for (int index = 0;index<numEpisode;index++){
            learningConditionFile <<"Episode "<<episode-numEpisode+1+index<<": "<<episodeResults[index]<<" points,  "<<episodeFrames[index]<<" frames,  "<<episodeFps[index]<<" fps."<<endl;
        }
        episodeResults.clear();
        episodeFrames.clear();
        episodeFps.clear();
        learningConditionFile.close();
        newNameForLearningCondition.replace(newNameForLearningCondition.end()-11,newNameForLearningCondition.end()-4,"finished");
        rename(nameForLearningCondition.c_str(),newNameForLearningCondition.c_str());
        nameForLearningCondition = newNameForLearningCondition;
    }
    
    //write parameters checkPoint
    string currentCheckPointName = checkPointName+"-checkPoint-Frames"+to_string(saveThreshold)+"-writing.txt";
    ofstream checkPointFile;
    checkPointFile.open(currentCheckPointName.c_str());
    checkPointFile<<(*agentRand)<<endl;
    checkPointFile<<totalNumberFrames<<endl;
    checkPointFile << episode<<endl;
    checkPointFile << firstReward<<endl;
    checkPointFile << maxFeatVectorNorm<<endl;
    vector<int> nonZeroWeights;
    
    for (unsigned i=0;i<=hashTableSize;i++){
        unsigned numNonZeroWeights = 0;
        for (unsigned a=0;a<numActions;++a){
            if (w[a][i]!=0){
                ++numNonZeroWeights;
            }
        }
        if (numNonZeroWeights>0){
            checkPointFile<<i<<' '<<numNonZeroWeights;
            for (unsigned a=0;a<numActions;++a){
                if (w[a][i]!=0){
                    checkPointFile<<' '<<a<<' '<<w[a][i];
                }
            }
            checkPointFile<<'\t';
        }
    }
    checkPointFile<<endl;
    checkPointFile.close();
    
    string previousVersionCheckPoint = checkPointName+"-checkPoint-Frames"+to_string(saveThreshold-saveWeightsEveryXFrames)+"-finished.txt";
    remove(previousVersionCheckPoint.c_str());
    string oldCheckPointName = currentCheckPointName;
    currentCheckPointName.replace(currentCheckPointName.end()-11,currentCheckPointName.end()-4,"finished");
    rename(oldCheckPointName.c_str(),currentCheckPointName.c_str());
    
}

void SarsaLearner::loadCheckPoint(ifstream& checkPointToLoad){
    checkPointToLoad >> (*agentRand);
    checkPointToLoad >> totalNumberFrames;
    while (totalNumberFrames<1000){
        checkPointToLoad >> totalNumberFrames;
    }
    checkPointToLoad >> episodePassed;
    checkPointToLoad >> firstReward;
    checkPointToLoad >> maxFeatVectorNorm;
    learningRate = alpha / float(maxFeatVectorNorm);
    int numNonZeroWeights;
    unsigned hashedFeatureIndex;
    float weight;
    int action;
    while (checkPointToLoad>>hashedFeatureIndex && checkPointToLoad>>numNonZeroWeights){
        for (unsigned i=0;i<numNonZeroWeights;++i){
            checkPointToLoad>>action; checkPointToLoad>>weight;
            w[action][hashedFeatureIndex]=weight;
        }
    }
    
    checkPointToLoad.close();
}

void SarsaLearner::learnPolicy(ALEInterface& ale, Features *features){
    
    struct timeval tvBegin, tvEnd, tvDiff;
    vector<float> reward;
    double elapsedTime;
    double cumReward = 0, prevCumReward = 0;
    sawFirstReward = 0; firstReward = 1.0;
    vector<float> episodeResults;
    vector<int> episodeFrames;
    vector<double> episodeFps;
    long long trueFeatureSize, trueFeatureNextSize;
    
    //Repeat (for each episode):
    //This is going to be interrupted by the ALE code since I set max_num_frames beforehand
    for(int episode = episodePassed+1; totalNumberFrames < totalNumberOfFramesToLearn; episode++){
        //random no-op
        unsigned int noOpNum = 0;
        if (randomNoOp){
            noOpNum = (*agentRand)()%(noOpMax)+1;
            for (int i=0;i<noOpNum;++i){
                ale.act(actions[0]);
            }
        }
        
        //We have to clean the traces every episode:
        for(unsigned int a = 0; a < nonZeroElig.size(); a++){
            for(unsigned long long i = 0; i < nonZeroElig[a].size(); i++){
                long long idx = nonZeroElig[a][i];
                e[a][idx] = 0.0;
            }
            nonZeroElig[a].clear();
        }
        
        F.clear();
        features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), F);
        trueFeatureSize = F.size();
        translateFeatures(F);
        updateQValues(F, Q);
        
        currentAction = epsilonGreedy(Q,episode);
        gettimeofday(&tvBegin, NULL);
        int lives = ale.lives();
        //Repeat(for each step of episode) until game is over:
        //This also stops when the maximum number of steps per episode is reached
        while(!ale.game_over()){
            reward.clear();
            reward.push_back(0.0);
            reward.push_back(0.0);
            updateQValues(F, Q);
            updateAcumTrace(currentAction, F);
          
            sanityCheck();
            //Take action, observe reward and next state:
            act(ale, currentAction, reward);
            cumReward  += reward[1];
            if(!ale.game_over()){
                //Obtain active features in the new state:
                Fnext.clear();
                features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), Fnext);
                trueFeatureNextSize = Fnext.size();
                translateFeatures(Fnext);
                updateQValues(Fnext, Qnext);     //Update Q-values for the new active features
                nextAction = epsilonGreedy(Qnext,episode);
            }
            else{
                nextAction = 0;
                for(unsigned int i = 0; i < Qnext.size(); i++){
                    Qnext[i] = 0;
                }
            }
            //To ensure the learning rate will never increase along
            //the time, Marc used such approach in his JAIR paper
            if (trueFeatureSize > maxFeatVectorNorm){
                maxFeatVectorNorm = trueFeatureSize;
                learningRate = alpha/maxFeatVectorNorm;
            }
            
            delta = reward[0] + gamma * Qnext[nextAction] - Q[currentAction];
            
            //Update weights vector:
            for(unsigned int a = 0; a < nonZeroElig.size(); a++){
                for(unsigned int i = 0; i < nonZeroElig[a].size(); i++){
                    long long idx = nonZeroElig[a][i];
                    w[a][idx] = w[a][idx] + learningRate * delta * e[a][idx];
                }
            }
            F = Fnext;
            currentAction = nextAction;
        }
        gettimeofday(&tvEnd, NULL);
        timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
        elapsedTime = double(tvDiff.tv_sec) + double(tvDiff.tv_usec)/1000000.0;
        
        double fps = double(ale.getEpisodeFrameNumber())/elapsedTime;
        printf("episode: %d,\t%.0f points,\tavg. return: %.1f,\t%d frames,\t%.0f fps\n",
               episode, cumReward - prevCumReward, (double)cumReward/(episode),
               ale.getEpisodeFrameNumber(), fps);
        episodeResults.push_back(cumReward-prevCumReward);
        episodeFrames.push_back(ale.getEpisodeFrameNumber());
        episodeFps.push_back(fps);
        totalNumberFrames += ale.getEpisodeFrameNumber()-noOpNum*numStepsPerAction;
        prevCumReward = cumReward;
        features->clearCash();
        ale.reset_game();
        if(toSaveCheckPoint && totalNumberFrames>saveThreshold){
            saveCheckPoint(episode,totalNumberFrames,episodeResults,saveWeightsEveryXFrames,episodeFrames,episodeFps);
            saveThreshold+=saveWeightsEveryXFrames;
        }
    }
}

void SarsaLearner::evaluatePolicy(ALEInterface& ale, Features *features){
    double reward = 0;
    double cumReward = 0;
    double prevCumReward = 0;
    struct timeval tvBegin, tvEnd, tvDiff;
    double elapsedTime;
    
    std::string oldName = checkPointName+"-Result-writing.txt";
    std::string newName = checkPointName+"-Result-finished.txt";
    std::ofstream resultFile;
    resultFile.open(oldName.c_str());
    
    //Repeat (for each episode):
    for(int episode = 1; episode < numEpisodesEval; episode++){
        //Repeat(for each step of episode) until game is over:
        gettimeofday(&tvBegin, NULL);
        //random no-op
        unsigned int noOpNum;
        if (randomNoOp){
            noOpNum = (*agentRand)()%(noOpMax)+1;
            for (int i=0;i<noOpNum;++i){
                ale.act(actions[0]);
            }
        }
        
        for(int step = 0; !ale.game_over() && step < episodeLength; step++){
            //Get state and features active on that state:
            F.clear();
            features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), F);
            translateFeatures(F);
            updateQValues(F, Q);       //Update Q-values for each possible action
            currentAction = epsilonGreedy(Q);
            //Take action, observe reward and next state:
            reward = ale.act(actions[currentAction]);
            cumReward  += reward;
        }
        gettimeofday(&tvEnd, NULL);
        timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
        elapsedTime = double(tvDiff.tv_sec) + double(tvDiff.tv_usec)/1000000.0;
        double fps = double(ale.getEpisodeFrameNumber())/elapsedTime;
        
        resultFile<<"Episode "<<episode<<": "<<cumReward-prevCumReward<<std::endl;
        printf("episode: %d,\t%.0f points,\tavg. return: %.1f,\t%d frames,\t%.0f fps\n",
               episode, (cumReward-prevCumReward), (double)cumReward/(episode), ale.getEpisodeFrameNumber(), fps);
        features->clearCash();
        ale.reset_game();
        prevCumReward = cumReward;
    }
    resultFile<<"Average: "<<(double)cumReward/numEpisodesEval<<std::endl;
    resultFile.close();
    rename(oldName.c_str(),newName.c_str());
    
}

void SarsaLearner::translateFeatures(vector<long long>& features){
    memset(activeHashedFeatures,0,hashTableSize*sizeof(bool));
    memset(featureValues,0,hashTableSize*sizeof(int));
    unsigned long long numActiveHashFeatures = 0;
    for (unsigned long long i=0;i<features.size();++i){
        int hashedFeatureIndex = features[i] % hashTableSize;
        if (fourwiseHash.hash(features[i])==1){
            featureValues[hashedFeatureIndex]+=1;
        }else{
            featureValues[hashedFeatureIndex]-=1;
        }
        if (!activeHashedFeatures[hashedFeatureIndex]){
            activeHashedFeatures[hashedFeatureIndex]=true;
            features[numActiveHashFeatures] = hashedFeatureIndex;
            ++numActiveHashFeatures;
        }
    }
    features.resize(numActiveHashFeatures);
}


