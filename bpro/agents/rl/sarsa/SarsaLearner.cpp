/****************************************************************************************
 ** Implementation of Sarsa(lambda). It implements Fig. 8.8 (Linear, gradient-descent
 ** Sarsa(lambda)) from the book "R. Sutton and A. Barto; Reinforcement Learning: An
 ** Introduction. 1st edition. 1988."
 ** Some updates are made to make it more efficient, as not iterating over all features.
 ** Some updates are made to make it work with adaptive (representation) features
 ** TODO: Make it as efficient as possible.
 **
 ** Author: YItao Liang
 ***************************************************************************************/

#include "../../../common/Timer.hpp"

#include "SarsaLearner.hpp"
#include <stdio.h>
#include <math.h>
#include <set>
#include <algorithm>
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
    toSaveCheckPoint = param->getToSaveCheckPoint();
    saveWeightsEveryXFrames = param->getFrequencySavingWeights();
    pathWeightsFileToLoad = param->getPathToWeightsFiles();
    randomNoOp = param->getRandomNoOp();
    noOpMax = param->getNoOpMax();
    numStepsPerAction = param->getNumStepsPerAction();
    
    for(int i = 0; i < numActions; i++){
        //Initialize Q;
        Q.push_back(0);
        Qnext.push_back(0);
        //Initialize w:
        w.push_back(vector<float>());
    }
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
}

SarsaLearner::~SarsaLearner(){}

void SarsaLearner::updateQValues(vector<long long> &activeFeatures, vector<float> &QValues){
    unsigned long long featureSize = activeFeatures.size();
    for(int a = 0; a < numActions; ++a){
        float sumW = 0;
        for(unsigned long long i = 0; i < featureSize; ++i){
            sumW = sumW + w[a][activeFeatures[i]];
        }
        QValues[a] = sumW;
    }
}

void SarsaLearner::sanityCheck(){
    for(int i = 0; i < numActions; i++){
        if(fabs(Q[i]) > 10e7 || Q[i] != Q[i] /*NaN*/){
            printf("It seems your algorithm diverged!\n");
            exit(0);
        }
    }
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
    
    features->updateWeights(w,learningRate);
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

        
        F.clear();
        features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), F);
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
            
            sanityCheck();
            //Take action, observe reward and next state:
            act(ale, currentAction, reward);
            cumReward  += reward[1];
            if(!ale.game_over()){
                //Obtain active features in the new state:
                Fnext.clear();
                features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), Fnext);
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
            if (F.size() > maxFeatVectorNorm){
                maxFeatVectorNorm = F.size();
                learningRate = alpha/maxFeatVectorNorm;
            }
            delta = reward[0] + gamma * Qnext[nextAction] - Q[currentAction];
            
            //updeta features' sum of delta
            features->updateDelta(delta,currentAction);
            
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
        ale.reset_game();
        
        //promote features
        if (features->promoteFeatures(totalNumberFrames)){
            features->updateWeights(w,learningRate);
        }
        
        //save checkPoint
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
        ale.reset_game();
        prevCumReward = cumReward;
    }
    resultFile<<"Average: "<<(double)cumReward/numEpisodesEval<<std::endl;
    resultFile.close();
    rename(oldName.c_str(),newName.c_str());
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
    long long numberOfFeaturesSeen;
    checkPointToLoad >> numberOfFeaturesSeen;
    checkPointToLoad.close();
}