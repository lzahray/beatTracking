//g++ -I/Users/eita/boost_1_63_0 -I/Users/eita/Dropbox/Research/Tool/All/ HMM_ViterbiEstimation_v181023-2.cpp -o HMM_ViterbiEstimation-2
#include<fstream>
#include<iostream>
#include<cmath>
#include<string>
#include<sstream>
#include<vector>
#include<algorithm>
#include"stdio.h"
#include"stdlib.h"
#include"HMM_v181023.hpp"
using namespace std;

int main(int argc, char** argv){
	vector<int> v(100);
	vector<double> d(100);
	vector<string> s(100);
	stringstream ss;
	clock_t start, end;
	start = clock();

	if(argc!=6){
		cout<<"Error in usage: $./this param.txt selfProb noChordProb dataFile.txt result.txt"<<endl;
		return -1;
	}//endif
	string paramFile=string(argv[1]);
	double selfProb=atof(argv[2]);
	double noChordProb=atof(argv[3]);
	string dataFile=string(argv[4]);
	string resultFile=string(argv[5]);

	HMM model;
	model.ReadFile(paramFile);
	model.ModifyTrProb(selfProb,noChordProb);
	model.SetDataDim(25);
	model.ReadObsData(dataFile);
	model.Viterbi();
	model.WriteEstimation(resultFile);

//	end = clock(); cout<<"Elapsed time : "<<((double)(end - start) / CLOCKS_PER_SEC)<<" sec"<<endl; start=end;
	return 0;
}//end main
