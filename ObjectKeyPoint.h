
#pragma once

#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <numeric>
//#include "ObjectDetector.h"
#include "LBSP.h"

using namespace std;
using namespace cv;

#include <omp.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <assert.h>

// Added by Tanushri

#include <algorithm>

// Min Max C3861 error
using namespace std;
#include "C:\Program Files\MATLAB\R2019b\extern\include\matrix.h"
#include "C:\Program Files\MATLAB\R2019b\extern\include\mat.h"
#define THRESH 10
struct objectKeys
{
	cv::KeyPoint key;
	cv::Point2f dis_Cen, predC;
	vector<cv::Point2f> predCenters;
	float distance;
	float weight1;
	float proxFactor;
	int diffT1;
	cv::Mat descriptor;
	int index;
	int indi;
};

struct objectDetected
{
	size_t nXPos;
	size_t nYPos;
	size_t nWidth;
	size_t nHeight;
	float fDetScore;
	float fLBSPScore;
	float fColorScore;
	cv::Mat oLBSPDes;
	cv::Mat oColorHist;
	vector<objectKeys> voSIFTKeys;
	float fDesDistance;
	size_t nKeyPointMatches;
	float fNormKeyPointMatches;
	float fTotalProbScore;
	float fNormDetScore;
	float fNormColorScore;
	float fNormLBSPScore;
	float fNormDisScore;
	float fKeyVar;
	float fColorVar;
	float fLBSPVar;
	int indiKP;
	float fOverlap;
	Point2f objectCenter;
	Point2f diffC;
	Point2f predC;
	int diffWidth;
	int diffHeight;
};

class ObjectKeyPoint : public cv::KeyPoint{


public:

	//! full constructor
	ObjectKeyPoint();

	//! default destructor
	virtual ~ObjectKeyPoint();

	//! Filtered keypoints	
	std::vector<objectKeys> m_voFilteredKeyPoints, m_voFilteredKeyPointsTrackROI;

	//! KeyPoint Descriptors
	cv::Mat m_voDescriptors;

	//! KeyPoints in Face Model
	static void filteredKeyPoints(std::vector<objectKeys>& voKeyPoints, std::vector<objectKeys>& voFilteredKeyPoints, cv::Rect ROI, int contextFlag);

	//! Sort the face descriptor Matches
	std::vector<DMatch> sortMatches(vector<DMatch>& matches);

	//! Face Model Keypoints in Face Euclidean Model
	void encodeStructure(cv::Point2f center, cv::Mat image2, float& fUpdate);

	//! Face Keypoints voting for the object center in a Gaussian Matrix
	void voting(std::vector<objectKeys>& keyPointsROI, std::vector<objectKeys>& keyPointsImg, cv::Mat image2, cv::Rect ROI, cv::Point2f& previousCenter, cv::Point2f& predCenter, size_t & frameNum,
		float& fExpo, bool& flag);

	//! Updates the weights of keypoints in the model
	void update(cv::Point2f& predCenter, float& a, size_t & frame, float& fUpdate, float& fRateRemove, int& nStartFrame);

	//! Remove keypoints from the model that are not voting or badly voting
	void removeKeys(float& tWeight, cv::Point2f& predCenter);

	//! Add new keypoints to the model
	void addKeys(cv::Rect ROI, cv::Point2f& predCenter, float& fUpdate, float& fnewWeight);

	//! Apply Low's ratio test to filter keypoint matches
	static void filterMatches(vector<vector<DMatch>>& foundMatches, float& ratioTestTh);

	//! @overload@ Apply Low's ratio test to filter keypoint matches
	static void filterMatches(vector<DMatch>& foundMatches, float& ratioTestTh);

	//! Check whether matched keys lie in which part of the quadrant
	void keyQuadLocation(std::vector<objectKeys> key, cv::Rect ROI, cv::Point2f& predCenter, std::vector<objectKeys>& numberofKeysinFirstQuad, std::vector<objectKeys>& numberofKeysinSecQuad, std::vector<objectKeys>& numberofKeysinThirdQuad, std::vector<objectKeys>& numberofKeysinFourthQuad);

	//! Determines new keypoints that can be added to the face keypoint model
	void nonModelKeysinTrackROI(cv::Rect ROI, vector<objectKeys>& voKeyPoints);

	//! Determines Non model keypoints are present in the sample Face ROI
	void nonModelKeysinFaceROI(objectDetected& TBoxes, vector<objectKeys>& voKeyPoints);

	//! Read detection result from the Face detector txt file
	void readNPDResult(string& resultFile);

	//! Read Tracking initialization from GT file
	cv::Rect readGTCenter(string& gtFile);

	//! Computes LBSP descriptor
	void computeLBSPDes(cv::Mat oROI, cv::Mat& oLBSPDes);


	//! Calculated Gaussian weighted Color Histogram
	void calWeightedColor(cv::Mat oImage, cv::Mat& oHist, cv::Rect oROI);

	//! Sets the ROI for calculation of LBSP Descriptor
	void setLBSPROI(cv::Mat& oImage, cv::Mat& oImage2, Point2f& pt1, Point2f& pt2, cv::Rect& oLBSPROI);

	//! Adjusts ROI if it is out of the bounds of the image or very close to the boundary
	static void ROIAdjust(const cv::Mat& oSource, Point2f& pt1, Point2f& pt2);

	//! @overload@ Adjusts ROI if it is out of the bounds of the image or very close to the boundary
	static void ROIBoxAdjust(const cv::Mat& oSource, cv::Rect& oBox);

	//! Interface to read a MAT file in C++ code
	void matRead(const char *file, size_t& n_ObjSize);

	//! Sets the index of keypoints in model that got matched
	void setMatchingIndexKeysAM(vector<vector<DMatch>>& matchesR, size_t &nTotalMatches, vector<objectKeys>& voKeyPoints);

	//! @overload@ Sets the index of keypoints  in model and nonModel that got matched
	void setMatchingIndexKeysAM(vector<DMatch>& matchesR, size_t& nTotalMatches, vector<objectKeys>& voKeyPoints);

	//! normalize the distance of keypoints that are present in the proposals
	void normalizeMatches(vector<float>& sampleMatches, vector<float>& normSampleMatches, vector<objectKeys>& voSIFTSampleKeys);

	//! Computes SIFT descriptor for the selected set of KeyPoints
	cv::Mat computeDes(const cv::Mat& oInitImg);
	
	//! Computes Color and LBSP Model for the new template
	void updateFaceModel(objectDetected oBestBox, Mat& oFaceLBSPModel, Mat& oFaceColorModel, Mat& oFaceLBSPModelMat, Rect& oFaceLBSPModelRect, Mat oImage);

	void updatePartialFaceModel(objectDetected oBestBox, Mat& oFaceLBSPModel, Mat& oFaceColorModel, Mat& oFaceLBSPModelMat, Rect& oFaceLBSPModelRect, Mat oImage);

	//! Updates partially the Color histogram and the LBSP descriptor as the new model
	void updatePartialTemplate(cv::Mat& oFaceColorModel2Hist, cv::Mat& oFaceLBSPModel2, cv::Mat& oFaceColorModel, cv::Mat& oFaceLBSPModel, cv::Mat& oFaceLBSPModelMat);

	//! Computes the LBSP and the color Models
	void computeFaceObjectsROI(vector<objectDetected>& voNPDBox, Mat oLBSPROIModelMat, Mat oImage2, vector<objectKeys>& voKeyPoints, objectDetected& oBestBox);

	//! Matches LBSP and Color Descriptors with that of Face Model
	void matchFaceObjects(vector<objectDetected>& voNPDBox, cv::Mat oDescriptorLBSPModel, cv::Mat oColor1, cv::Point2f& predCenter, objectDetected& oBestBox);


	//! Scale change by using the keypoints
	void detectScaleChange(std::vector<Point2f>& voDist1, std::vector<Point2f>& voDist2, float& fScaleChange);

	//! Pairwise distance between keypoints
	void computePairDistance(std::vector<objectKeys>& keyPoint1, std::vector<objectKeys>& keyPoint2, vector<std::pair <float, int>>& voKeyWeightValues, std::vector<Point2f>& voDist1, std::vector<Point2f>& voDist2);

	//! get the weights of the features
	void getMinMaxWeightKeyPoint(vector<objectKeys>& keyPoint1, vector<objectKeys>& keyPoint2, vector<std::pair <float, int>>& voKeyWeightValues);

	//! Match LBSP descriptors
	inline static void matchLBSPAM(cv::Mat& oDesc1, cv::Mat& oDesc2, vector<int> vOutArray, float& fLBSPDesDiff)
	{
		if (oDesc1.rows == oDesc2.rows)
		{
			for (auto i = 0; i < oDesc1.rows; i++)
			{
				int nRes = hdist(oDesc1.at<ushort>(i, 0), oDesc2.at<ushort>(i, 0));
				vOutArray.push_back(nRes);
			}

			int sum = std::accumulate(vOutArray.begin(), vOutArray.end(), 0);
			fLBSPDesDiff = (float)sum / (16 * oDesc1.rows); // normalize the descriptor
		}

		else{

			fLBSPDesDiff = 0.0;
		}

	}


	inline struct compare_first_only {
		template<typename T1, typename T2>
		bool operator()(const std::pair<T1, T2>& p1, const std::pair<T1, T2>& p2) {
			return p1.first < p2.first;
		}
	};



	//! Match weighted Color Histogram
	inline static void matchColorAM(cv::Mat oImage1Hist, cv::Mat oImage2Hist, float& fCompareHS)
	{

		fCompareHS = norm(oImage1Hist, oImage2Hist, NORM_L2);

	}

	//! Checks whether the non-ROI LBSP descriptor matches with the size of the LBSP model 
	inline static void checkLBSPsizewithModel(cv::Mat& oNonModel, cv::Mat& oModel)
	{
		if (oModel.size() != oNonModel.size())
		{

			cv::resize(oNonModel, oNonModel, oModel.size());

		}

	}

};
