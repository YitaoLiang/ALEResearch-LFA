/****************************************************************************************
** Starting point for running Sarsa algorithm. Here the parameters are set, the algorithm
** is started, as well as the features used. In fact, in order to create a new learning
** algorithm, once its class is implementend, the main file just need to instantiate
** Parameters, the Learner and the type of Features to be used. This file is a good 
** example of how to do it. A parameters file example can be seen in ../conf/sarsa.cfg.
** This is an example for other people to use: Sarsa with Basic Features.
** 
** Author: Marlos C. Machado
***************************************************************************************/

#ifndef ALE_INTERFACE_H
#define ALE_INTERFACE_H
#include <ale_interface.hpp>
#endif

#include "agents/rl/sarsa/SarsaLearner.hpp"
#include "common/Parameters.hpp"
#include "features/AdaptiveFeatures.hpp"

void printBasicInfo(Parameters param){
	printf("Seed: %d\n", param.getSeed());
	printf("\nCommand Line Arguments:\nPath to Config. File: %s\nPath to ROM File: %s\nPath to Backg. File: %s\n", 
		param.getConfigPath().c_str(), param.getRomPath().c_str(), param.getPathToBackground().c_str());
	if(param.getSubtractBackground()){
		printf("\nBackground will be subtracted...\n");
	}
	printf("\nParameters read from Configuration File:\n");
	printf("alpha:   %f\ngamma:   %f\nepsilon: %f\nlambda:  %f\nep. length: %d\n\n", 
		param.getAlpha(), param.getGamma(), param.getEpsilon(), param.getLambda(), 
		param.getEpisodeLength());
}


int main(int argc, char** argv){
	//Reading parameters from file defined as input in the run command:
	Parameters param(argc, argv);
	srand(param.getSeed());
    
	//Reporting parameters read:
	printBasicInfo(param);

    ALEInterface ale(param.getDisplay());
	ale.setFloat("repeat_action_probability", 0.00);
	ale.setInt("random_seed", 2*param.getSeed());
	ale.setFloat("frame_skip", param.getNumStepsPerAction());
	ale.setInt("max_num_frames_per_episode", param.getEpisodeLength());
    ale.setBool("color_averaging", true);
	ale.loadROM(param.getRomPath().c_str());
    
    AdaptiveFeatures features(ale, &param);

	//Instantiating the learning algorithm:
	SarsaLearner sarsaLearner(ale, &features, &param, 2*param.getSeed()-1);
    //Learn a policy:
    sarsaLearner.learnPolicy(ale, &features);
    

    printf("\n\n== Evaluation without Learning == \n\n");
    sarsaLearner.evaluatePolicy(ale, &features);
	
    return 0;
}
