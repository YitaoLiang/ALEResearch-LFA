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
using namespace std;

SarsaLearner::SarsaLearner(ALEInterface& ale, Features *features, Parameters *param,int seed) : RLLearner(ale, param,seed) {
    numRand = 0;
    currentEpisode = 1;
    totalNumberFrames = 0.0;
    
    delta = 0.0;
	alpha = param->getAlpha();
	lambda = param->getLambda();
	traceThreshold = param->getTraceThreshold();
	numFeatures = features->getNumberOfFeatures();
    toSaveCheckPoint = param->getToSaveCheckPoint();
    saveWeightsEveryXSteps = param->getFrequencySavingWeights();
	pathWeightsFileToLoad = param->getPathToWeightsFiles();
    featureSeen.resize(numActions);
	
	for(int i = 0; i < numActions; i++){
		//Initialize Q;
		Q.push_back(0);
		Qnext.push_back(0);
		//Initialize e:
		e.push_back(vector<float>(numFeatures, 0.0));
		w.push_back(vector<float>(numFeatures, 0.0));
		nonZeroElig.push_back(vector<int>());
	}

    
	if(toSaveCheckPoint){
        checkPointName = param->getCheckPointName();
    }

	/*if(param->getToLoadWeights()){
		loadWeights();
	}*/
}

SarsaLearner::~SarsaLearner(){}

void SarsaLearner::updateQValues(vector<int> &Features, vector<float> &QValues){
	for(int a = 0; a < numActions; a++){
		double sumW = 0;
		for(unsigned int i = 0; i < Features.size(); i++){
			sumW += w[a][Features[i]];
		}
		QValues[a] = sumW;
	}
}

void SarsaLearner::updateReplTrace(int action, vector<int> &Features){
	//e <- gamma * lambda * e
	for(unsigned int a = 0; a < nonZeroElig.size(); a++){
		int numNonZero = 0;
	 	for(unsigned int i = 0; i < nonZeroElig[a].size(); i++){
	 		int idx = nonZeroElig[a][i];
	 		//To keep the trace sparse, if it is
	 		//less than a threshold it is zero-ed.
			e[a][idx] = gamma * lambda * e[a][idx];
			if(e[a][idx] < traceThreshold){
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
	for(unsigned int i = 0; i < F.size(); i++){
		int idx = Features[i];
		//If the trace is zero it is not in the vector
		//of non-zeros, thus it needs to be added
		if(e[action][idx] == 0){
	       nonZeroElig[action].push_back(idx);
	    }
		e[action][idx] = 1;
	}
}

void SarsaLearner::updateAcumTrace(int action, vector<int> &Features){
	//e <- gamma * lambda * e
	for(unsigned int a = 0; a < nonZeroElig.size(); a++){
		int numNonZero = 0;
	 	for(unsigned int i = 0; i < nonZeroElig[a].size(); i++){
	 		int idx = nonZeroElig[a][i];
	 		//To keep the trace sparse, if it is
	 		//less than a threshold it is zero-ed.
			e[a][idx] = gamma * lambda * e[a][idx];
			if(e[a][idx] < traceThreshold){
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
	for(unsigned int i = 0; i < F.size(); i++){
		int idx = Features[i];
		//If the trace is zero it is not in the vector
		//of non-zeros, thus it needs to be added
		if(e[action][idx] == 0){
	       nonZeroElig[action].push_back(idx);
	    }
		e[action][idx] += 1;
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

//To do: we do not want to save weights that are zero
/void SarsaLearner::saveWeightsToFile(int episode, int totalNumberFramers, vector<float>& episodeResults,int& frequency){
    ofstream learningConditionFile;
    string newNameForLearningCondition = checkPointName+"-learningCondition-Episode"+to_string(episode)+"-writing.txt";
    int renameReturnCode = rename(nameForLearningCondition.c_str(),newNameForLearningCondition.c_str());
    if (renameReturnCode == 0){
        nameForLearningCondition = newNameForLearningCondition;
        learningConditionFile.open(nameForLearningCondition.c_str(), ios_base::app);
        for (int index = 0;index<numEpisode;index++){
            learningConditionFile <<"Episode "<<episode-numEpisode+1+index<<": "<<episodeResults[index]<<endl;
        }
        episodeResults.clear();
        learningConditionFile.close();
        newNameForLearningCondition.replace(newNameForLearningCondition.end()-11,newNameForLearningCondition.end()-4,"finished");
        rename(nameForLearningCondition.c_str(),newNameForLearningCondition.c_str());
        nameForLearningCondition = newNameForLearningCondition;
    }
    
    //write parameters checkPoint
    string checkPointName = checkPointName+"-checkPoint-Episode"+to_string(episode)+"-writing.txt";
    ofstream checkPointFile;
    checkPointFile.open(checkPointName.c_str());
    checkPointFile<<agentRand<<endl;
    checkPointFile<<totalNumberFrames<<endl;
    checkPointFile << episode<<endl;
    checkPointFile << firstReward<<endl;
    for (int a=0;a<featureSeen.size();a++){
        for (int index=0; index<featureSeen[a].size();index++){
            checkPointFile<<a<<" "<<featureSeen[a][index]<<" "<<w[a][featureSeen[a][index]]<<"\t";
        }
    }
    checkPointFile << endl;
    checkPointFile.close();
    string previousVersionCheckPoint = checkPointName+"-checkPoint-Episode"+to_string(episode-frequency)+"-finished.txt";
    remove(previousVersionCheckPoint.c_str());
    string oldCheckPointName = checkPointName;
    checkPointName.replace(checkPointName.end()-11,checkPointName.end()-4,"finished");
    rename(oldCheckPointName.c_str(),checkPointName.c_str());

}

//loadWeights need to be changed

void SarsaLearner::loadWeights(){
	string line;
	int nActions, nFeatures;
	int i, j;
	double value;

	std::ifstream weightsFile (pathWeightsFileToLoad.c_str());
	
	weightsFile >> nActions >> nFeatures;
	assert(nActions == numActions);
	assert(nFeatures == numFeatures);

	while(weightsFile >> i >> j >> value){
		w[i][j] = value;
	}
}

void SarsaLearner::learnPolicy(ALEInterface& ale, Features *features){
	
	struct timeval tvBegin, tvEnd, tvDiff;
	vector<float> reward;
	double elapsedTime;
	double cumReward = 0, prevCumReward = 0;
	unsigned int maxFeatVectorNorm = 1;
	sawFirstReward = 0; firstReward = 1.0;

	//Repeat (for each episode):
    int totalNumberFrames=0;
	//This is going to be interrupted by the ALE code since I set max_num_frames beforehand
	for(int episode = 1; episode <= 50; episode++){
		//We have to clean the traces every episode:
		for(unsigned int a = 0; a < nonZeroElig.size(); a++){
			for(unsigned int i = 0; i < nonZeroElig[a].size(); i++){
				int idx = nonZeroElig[a][i];
				e[a][idx] = 0.0;
			}
			nonZeroElig[a].clear();
		}
		F.clear();
		features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), F);
		updateQValues(F, Q);
		currentAction = epsilonGreedy(Q);
		//Repeat(for each step of episode) until game is over:
		gettimeofday(&tvBegin, NULL);
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
				nextAction = epsilonGreedy(Qnext);
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
			}

			delta = reward[0] + gamma * Qnext[nextAction] - Q[currentAction];

			updateReplTrace(currentAction, F);
			//Update weights vector:
			for(unsigned int a = 0; a < nonZeroElig.size(); a++){
				for(unsigned int i = 0; i < nonZeroElig[a].size(); i++){
					int idx = nonZeroElig[a][i];
                    if (w[a][idx]==0){
                        featureSeen[a].push_back(idx);
                    }
					w[a][idx] = w[a][idx] + (alpha/maxFeatVectorNorm) * delta * e[a][idx];
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
		totalNumberFrames += ale.getEpisodeFrameNumber();
		prevCumReward = cumReward;
		ale.reset_game();
		if(toSaveCheckPoint && episode%saveWeightsEveryXSteps == 0){
			saveWeightsToFile(episode,totalNumberFrames,episodeResults,saveWeightsEveryXSteps);
        }
	}/*
	if(toSaveWeightsAfterLearning){
		stringstream ss;
		ss << episode;
		saveWeightsToFile(ss.str());
        }*/
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
	for(int episode = 1; episode <= 10; episode++){
		//Repeat(for each step of episode) until game is over:
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
    resultFile<<"Average: "<<(double)cumReward/10<<std::endl;
    resultFile.close();
    rename(oldName.c_str(),newName.c_str());
    
}
