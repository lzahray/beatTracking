#ifndef HMM_HPP
#define HMM_HPP

#include<iostream>
#include<iomanip>
#include<fstream>
#include<string>
#include<sstream>
#include<vector>
#include<stdio.h>
#include<stdlib.h>
#include<cfloat>
#include<cmath>
#include<cassert>
#include<algorithm>
#include"BasicCalculation_v170122.hpp"

#define EPS (0.1)
//#define PRINTON_ true

using namespace std;

class HMM{
public:
	int nState;
//	int nSymb;
	int dataDim;//dimension of the feature vector

	Prob<int> iniProb;
	vector<Prob<int> > trProb;
//	vector<Prob<int> > outProb;

	vector<vector<double> > data;
	vector<int> latentData;

	HMM(int _nState=2){
		nState=_nState;
		RandomInit();
	}//end HMM
	~HMM(){
	}//end ~HMM

	void SetDataDim(int dataDim_){
		dataDim=dataDim_;
	}//end SetDataDim

	void RandomInit(){
		iniProb.Resize(nState);
		iniProb.Randomize();
		trProb.resize(nState);
		for(int i=0;i<nState;i+=1){
			trProb[i].Resize(nState);
			trProb[i].Randomize();
		}//endfor i
//		outProb.resize(nState);
//		for(int i=0;i<nState;i+=1){
//			outProb[i].Resize(nSymb);
//			outProb[i].Randomize();
//		}//endfor i
	}//end RandomInit

	void WriteFile(string filename){
		ofstream ofs(filename.c_str());
		ofs<<"//nState: "<<nState<<"\n";
//		ofs<<"//nSymb: "<<nSymb<<"\n";

		ofs<<"### Init Prob\n";
		for(int i=0;i<nState;i+=1){
			ofs<<iniProb.P[i]<<"\t";
		}//endfor i
		ofs<<"\n";

		ofs<<"### Transition Prob\n";
		for(int i=0;i<nState;i+=1){
			for(int ip=0;ip<nState;ip+=1){
				ofs<<trProb[i].P[ip]<<"\t";
			}//endfor ip
			ofs<<"\n";
		}//endfor i

//		ofs<<"### Output Prob\n";
//		for(int i=0;i<nState;i+=1){
//			for(int ip=0;ip<nSymb;ip+=1){
//				ofs<<outProb[i].P[ip]<<"\t";
//			}//endfor ip
//			ofs<<"\n";
//		}//endfor i

		ofs.close();
	}//end WriteFile

	void ReadFile(string filename){
		vector<int> v(100);
		vector<double> d(100);
		vector<string> s(100);
		stringstream ss;

		ifstream ifs(filename.c_str());
		ifs>>s[1]>>nState;
		getline(ifs,s[99]);
//		ifs>>s[1]>>nSymb;
//		getline(ifs,s[99]);

		RandomInit();

		getline(ifs,s[99]);//### Init Prob
		for(int i=0;i<nState;i+=1){
			ifs>>iniProb.P[i];
		}//endfor i
		getline(ifs,s[99]);
		iniProb.Normalize();

		getline(ifs,s[99]);//### Transition Prob
		for(int i=0;i<nState;i+=1){
			for(int ip=0;ip<nState;ip+=1){
				ifs>>trProb[i].P[ip];
			}//endfor ip
			getline(ifs,s[99]);
			trProb[i].Normalize();
		}//endfor i

//		getline(ifs,s[99]);//### Output Prob
//		for(int i=0;i<nState;i+=1){
//			for(int ip=0;ip<nSymb;ip+=1){
//				ifs>>outProb[i].P[ip];
//			}//endfor ip
//			getline(ifs,s[99]);
//			outProb[i].Normalize();
//		}//endfor i

		ifs.close();
	}//end ReadFile

	void ModifyTrProb(double selfTrProb,double noChordProb){
		for(int i=0;i<nState-1;i+=1){
			for(int ip=0;ip<nState;ip+=1){
				if(i==ip){trProb[i].P[ip]=selfTrProb;}
				if(ip==nState-1){trProb[i].P[ip]=noChordProb;}
			}//endfor ip
			trProb[i].Normalize();
		}//endfor i
	}//end ModifyTrProb

	void ReadObsData(string filename){
		vector<int> v(100);
		vector<double> d(100);
		vector<string> s(100);
		stringstream ss;
		data.clear();
		vector<double> vd(dataDim);
		ifstream ifs(filename.c_str());
		while(!ifs.eof()){
			for(int i=0;i<dataDim;i+=1){
				ifs>>vd[i];
			}//endfor i
			data.push_back(vd);
			getline(ifs,s[99]);
		}//endwhile
		ifs.close();
//cout<<data.size()<<"\t"<<latentData.size()<<endl;
	}//end ReadObsData

	void WriteEstimation(string filename){
		ofstream ofs(filename.c_str());
		for(int n=0;n<latentData.size();n+=1){
			ofs<<latentData[n]<<"\n";
		}//endfor n
		ofs.close();
	}//end WriteEstimation

	void Viterbi(){
		assert(data.size()>0);
		vector<double> LP;
		vector<vector<int> > amax;
		double logP;
		LP.resize(nState);
		amax.resize(data.size());

		for(int n=0;n<data.size();n+=1){
			amax[n].resize(nState);
			if(n==0){
				for(int i=0;i<nState;i+=1){
					LP[i]=iniProb.LP[i]+OutputLP(i,data[n]);
				}//endfor i
				continue;
			}//endif

			vector<double> preLP(LP);
			for(int i=0;i<nState;i+=1){
				LP[i]=preLP[0]+trProb[0].LP[i];
				amax[n][i]=0;
				for(int ip=0;ip<nState;ip+=1){
					logP=preLP[ip]+trProb[ip].LP[i];
					if(logP>LP[i]){
						LP[i]=logP;
						amax[n][i]=ip;
					}//endif
				}//endfor ip
				LP[i]+=OutputLP(i,data[n]);
			}//endfor i

		}//endfor n

		latentData.clear();
		latentData.resize(data.size());
		latentData[data.size()-1]=0;
		for(int i=0;i<nState;i+=1){
			if(LP[i]>LP[latentData[data.size()-1]]){latentData[data.size()-1]=i;}
		}//endfor i
		for(int n=data.size()-2;n>=0;n-=1){
			latentData[n]=amax[n+1][latentData[n+1]];
		}//endfor n
	}//end Viterbi

	double OutputLP(int state,vector<double> datapoint){//output probability for a datapoint = feature vector given a latent state
//		return 0;
		return log(datapoint[state]);
	}//end OutputLP

};//endclass HMM


#endif // HMM_HPP
