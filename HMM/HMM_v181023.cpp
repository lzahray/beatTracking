//g++ -I/Users/eita/boost_1_63_0 -I/Users/eita/Dropbox/Research/Tool/All/ HMM_v181023.cpp -o HMM
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
#include"ProbabilityVisualizer_v180920.hpp"
using namespace std;

int main(int argc, char** argv){
	vector<int> v(100);
	vector<double> d(100);
	vector<string> s(100);
	stringstream ss;
	clock_t start, end;
	start = clock();

//	if(argc!=2){
//		cout<<"Error in usage: $./this numOfPatterns"<<endl;
//		return -1;
//	}//endif

	rand();

	HMM model(25);

//	model.WriteFile("param_HMM.txt");
	model.ReadFile("param_HMM_trained.txt");
//	model.WriteFile("param_HMM2.txt");

	VisualizeTrProb(model.trProb,"TrProb.svg");

/*
	Prob<int> prob;
	prob.Resize(25);
	ofstream ofs("ex_data.txt");
	for(int n=0;n<100;n+=1){
		prob.Randomize();
		for(int i=0;i<25;i+=1){
			ofs<<prob.P[i]<<"\t";
		}//endfor i
		ofs<<"\n";
	}//endfor n
	ofs.close();
*/

//	end = clock(); cout<<"Elapsed time : "<<((double)(end - start) / CLOCKS_PER_SEC)<<" sec"<<endl; start=end;
	return 0;
}//end main
