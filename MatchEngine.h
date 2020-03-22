
#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>

#define GPU_MATCHER 1 // matching on GPU : only BF matcher
#define CROSS_CHECK 1 // flag for one-to-one match query <-> train
#if !CROSS_CHECK
#define KNN 0
#define LOWE_RATIO .75f
#else
#define KNN 0
#endif
#define FLANN 0 //use FLANN instead of BF (cpu only)

using namespace std;
using namespace cv;
class MatchEngine
{
public:
	vector<DMatch> v_DMatch;
	vector<vector<DMatch>> vv_DMatch;
	float fRadius;
private:
	bool m_crossCheck;
	
#if GPU_MATCHER
	Ptr<cuda::DescriptorMatcher> matcher;
	cuda::GpuMat matchArrayGpu, matchArrayGpu1;
#else
	DescriptorMatcher* matcher;
#endif

public:
	MatchEngine(int normType = NORM_L2, bool crossCheck = false)
	{
#if GPU_MATCHER
		matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_L2);
#elif FLANN
		matcher = new FlannBasedMatcher;
#else
		m_crossCheck = crossCheck;
		matcher = new BFMatcher(normType, crossCheck);
#endif
	}
	template<typename T>void match(T queryDesc, T trainDesc, bool knn = false);
	template<>void match<cuda::GpuMat>(cuda::GpuMat queryDesc, cuda::GpuMat trainDesc, bool knn);

};

template<typename T>
void MatchEngine::match(T queryDesc, T trainDesc, bool knn)
{
	v_DMatch.resize(0); // vector of type DMatch
	vv_DMatch.resize(0); // vector of vector of DMatch
#if GPU_MATCHER
	cuda::GpuMat queryDesc_gpu, trainDesc_gpu;
	queryDesc_gpu.upload(queryDesc);
	trainDesc_gpu.upload(trainDesc);
	//const GpuMat& mask = GpuMat()
#if KNN
	//vector<vector<DMatch>>v_v_DMatch;
	matcher->knnMatchAsync(queryDesc_gpu, trainDesc_gpu, matchArrayGpu, 2);
	matcher->knnMatchConvert(matchArrayGpu, vv_DMatch);

	float thr_dl = LOWE_RATIO;
	for (int i = 0; i < vv_DMatch.size(); i++){
		float ratio = vv_DMatch[i][0].distance / vv_DMatch[i][1].distance;
		if (ratio < thr_dl)v_DMatch.push_back(vv_DMatch[i][0]);
	}
#else
	matcher->radiusMatchAsync(queryDesc_gpu, trainDesc_gpu, matchArrayGpu, fRadius );
	matcher->radiusMatchConvert(matchArrayGpu, vv_DMatch);

#endif
#else
	if (knn & !m_crossCheck){
		vector<vector<DMatch>>v_v_DMatch;
		matcher->knnMatch(queryDesc, trainDesc, v_v_DMatch, 2);
		//lowe ratio test
		float thr_dl = LOWE_RATIO;
		for (int i = 0; i < v_v_DMatch.size(); i++){
			float ratio = v_v_DMatch[i][0].distance / v_v_DMatch[i][1].distance;
			if (ratio < thr_dl)v_DMatch.push_back(v_v_DMatch[i][0]);
		}
	}
	else{
		//matcher->match(queryDesc, trainDesc, v_DMatch);
		matcher->radiusMatch(queryDesc, trainDesc, vv_DMatch, fRadius);
	}
#endif
}

#if GPU_MATCHER
template<> //specialization if descriptor already on gpu
void MatchEngine::match<cuda::GpuMat>(cuda::GpuMat queryDesc_gpu, cuda::GpuMat trainDesc_gpu, bool knn)
{
	v_DMatch.resize(0);
	vv_DMatch.resize(0);

#if KNN
	vector<vector<DMatch>>v_v_DMatch;
	v_v_DMatch.resize(0);

	matcher->knnMatchAsync(queryDesc_gpu, trainDesc_gpu, matchArrayGpu, 2);
	matcher->knnMatchConvert(matchArrayGpu, v_v_DMatch);
	float thr_dl = LOWE_RATIO;
	for (int i = 0; i < v_v_DMatch.size(); i++){
		float ratio = v_v_DMatch[i][0].distance / v_v_DMatch[i][1].distance;
		if (ratio < thr_dl)v_DMatch.push_back(v_v_DMatch[i][0]);
	}
#else //crossCheck
	
	matcher->matchAsync(queryDesc_gpu, trainDesc_gpu, matchArrayGpu);
	matcher->matchConvert(matchArrayGpu, v_DMatch);
	
#endif

}
#endif