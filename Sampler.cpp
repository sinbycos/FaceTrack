#pragma once
#include "Sampler.h"


#define _USE_MATH_DEFINES
#include <cmath>
#include <numeric>
#include <iterator>
#include <vector>
#define DRAW 1
using namespace std;


Sampler::Sampler()
{
}


Sampler::~Sampler()
{
}

void Sampler::getProbPropBoxes(vector<objectDetected>& TBoxes, Mat oImage, float&q, float&m, float&g, size_t& nFrame, size_t& hIndex, size_t& mIndex, size_t& lIndex) {


	Mat oImageClone = oImage.clone();

	size_t largestIndex = TBoxes.size() - 1;

	vector<float> voColorVal, voLBSPVal;
	float fScoreBoxesKeysSum = 0.0, fScoreBoxesColorSum = 0.0, fScoreBoxesLBSPSum = 0.0;
	float fScoreBoxesKeysMean = 0.0, fScoreBoxesColorMean = 0.0, fScoreBoxesLBSPMean = 0.0;
	float fScoreBoxesKeysVar = 0.0, fScoreBoxesColorVar = 0.0, fScoreBoxesLBSPVar = 0.0;

	Rect testBoxRect;
	float valueColorMin, valueColorMax, valueLBSPMin, valueLBSPMax;

	for (auto i = 0; i < TBoxes.size(); ++i)

	{

		TBoxes[i].fColorScore = 1.0 - TBoxes[i].fColorScore;
		TBoxes[i].fLBSPScore = 1.0 - TBoxes[i].fLBSPScore;
		//	voColorVal.push_back(TBoxes[i].fColorScore);

		//	voLBSPVal.push_back(TBoxes[i].fLBSPScore);

	}


	/*std::sort(voColorVal.begin(), voColorVal.end());
	std::sort(voLBSPVal.begin(), voLBSPVal.end());

	valueColorMax = voColorVal[largestIndex];
	valueColorMin = voColorVal[0];
	valueLBSPMax = voLBSPVal[largestIndex];
	valueLBSPMin = voLBSPVal[0];


	for (auto i = 0; i < TBoxes.size(); ++i)

	{

	if ((valueColorMax - valueColorMin) != 0)
	{
	TBoxes[i].fNormColorScore = (TBoxes[i].fColorScore - valueColorMin) / (valueColorMax - valueColorMin);
	}


	if ((valueColorMax - valueColorMin) == 0)
	{
	TBoxes[i].fNormColorScore = 0.0;
	}


	if ((valueLBSPMax - valueLBSPMin) != 0)
	{
	TBoxes[i].fNormLBSPScore = (TBoxes[i].fLBSPScore - valueLBSPMin) / (valueLBSPMax - valueLBSPMin);
	}


	if ((valueLBSPMax - valueLBSPMin) == 0)
	{
	TBoxes[i].fNormLBSPScore = 0.0;
	}

	TBoxes[i].fTotalProbScore = q*TBoxes[i].fNormKeyPointMatches + m*TBoxes[i].fNormColorScore + g*TBoxes[i].fNormLBSPScore;

	}
	*/
	for (auto i = 0; i < TBoxes.size(); ++i)

	{
		fScoreBoxesKeysSum += TBoxes[i].fNormKeyPointMatches;
		fScoreBoxesColorSum += TBoxes[i].fNormColorScore;
		fScoreBoxesLBSPSum += TBoxes[i].fNormLBSPScore;
	}


	fScoreBoxesKeysMean = fScoreBoxesKeysSum / float(TBoxes.size()); // Mean
	fScoreBoxesColorMean = fScoreBoxesColorSum / float(TBoxes.size());
	fScoreBoxesLBSPMean = fScoreBoxesLBSPSum / float(TBoxes.size());
	vector<float> voKeyVar, voColorVar, voLBSPVar;

	for (auto i = 0; i < TBoxes.size(); ++i)
	{
		TBoxes[i].fKeyVar = (TBoxes[i].fNormKeyPointMatches - fScoreBoxesKeysMean)*(TBoxes[i].fNormKeyPointMatches - fScoreBoxesKeysMean);
		TBoxes[i].fColorVar = (TBoxes[i].fNormColorScore - fScoreBoxesColorMean)*(TBoxes[i].fNormColorScore - fScoreBoxesColorMean);
		TBoxes[i].fLBSPVar = (TBoxes[i].fNormLBSPScore - fScoreBoxesLBSPMean)*(TBoxes[i].fNormLBSPScore - fScoreBoxesLBSPMean);

		voKeyVar.push_back(TBoxes[i].fKeyVar);
		voColorVar.push_back(TBoxes[i].fColorVar);
		voLBSPVar.push_back(TBoxes[i].fLBSPVar);

	}


	std::sort(voKeyVar.begin(), voKeyVar.end());
	std::sort(voColorVar.begin(), voColorVar.end());
	std::sort(voLBSPVar.begin(), voLBSPVar.end());


	if (voKeyVar[largestIndex] == 0 || voColorVar[largestIndex] == 0 || voLBSPVar[largestIndex] == 0)
	{
		hIndex = 1;
		mIndex = 2;
		lIndex = 3;

	}

	else
	{
		getHighestVar(voKeyVar, voColorVar, voLBSPVar, largestIndex, hIndex, mIndex, lIndex);

	}




	for (auto i = 0; i < TBoxes.size(); ++i)
	{
		if (hIndex == 1 && mIndex == 2 && lIndex == 3)
		{
			TBoxes[i].fTotalProbScore = q*TBoxes[i].fNormKeyPointMatches + m*TBoxes[i].fNormColorScore + g*TBoxes[i].fNormLBSPScore;
		}

		if (hIndex == 1 && mIndex == 3 && lIndex == 2)
		{

			TBoxes[i].fTotalProbScore = q*TBoxes[i].fNormKeyPointMatches + g*TBoxes[i].fNormColorScore + m*TBoxes[i].fNormLBSPScore;
		}


		if (hIndex == 2 && mIndex == 3 && lIndex == 1)
		{
			TBoxes[i].fTotalProbScore = g*TBoxes[i].fNormKeyPointMatches + q*TBoxes[i].fNormColorScore + m*TBoxes[i].fNormLBSPScore;

		}

		if (hIndex == 2 && mIndex == 1 && lIndex == 3)
		{
			TBoxes[i].fTotalProbScore = m*TBoxes[i].fNormKeyPointMatches + q*TBoxes[i].fNormColorScore + g*TBoxes[i].fNormLBSPScore;

		}

		if (hIndex == 3 && mIndex == 1 && lIndex == 2)
		{
			TBoxes[i].fTotalProbScore = m*TBoxes[i].fNormKeyPointMatches + g*TBoxes[i].fNormColorScore + q*TBoxes[i].fNormLBSPScore;

		}

		if (hIndex == 3 && mIndex == 2 && lIndex == 1)
		{
			TBoxes[i].fTotalProbScore = g*TBoxes[i].fNormKeyPointMatches + m*TBoxes[i].fNormColorScore + q*TBoxes[i].fNormLBSPScore;

		}




#if DRAW == 13

		testBoxRect.x = TBoxes[i].nXPos;
		testBoxRect.y = TBoxes[i].nYPos;
		testBoxRect.width = TBoxes[i].nWidth;
		testBoxRect.height = TBoxes[i].nHeight;
		cv::rectangle(oImageClone, testBoxRect, cv::Scalar(255, 0, 0), 1, 8, 0);
		imshow("Proposals", oImageClone);
		cvWaitKey(10);



#endif		


	}

}



void Sampler::getHighestVar(vector<float>& voKeyVar, vector<float>& voColorVar, vector<float>& voLBSPVar, size_t& largestIndex, size_t& hIndex, size_t& mIndex, size_t& lIndex)
{

	if (voKeyVar[largestIndex] > voColorVar[largestIndex] && voKeyVar[largestIndex] > voLBSPVar[largestIndex])

	{
		hIndex = 1;
	}
	else if (voLBSPVar[largestIndex] > voKeyVar[largestIndex] && voLBSPVar[largestIndex] > voColorVar[largestIndex])
	{
		hIndex = 2;
	}

	else if (voColorVar[largestIndex] > voLBSPVar[largestIndex] && voColorVar[largestIndex] > voKeyVar[largestIndex])
	{
		hIndex = 3;
	}

	if (hIndex == 1)
	{
		if (voColorVar[largestIndex] > voLBSPVar[largestIndex])
		{
			mIndex = 2;
			lIndex = 3;
		}
		else
		{
			mIndex = 3;
			lIndex = 2;
		}

	}

	if (hIndex == 2)
	{
		if (voKeyVar[largestIndex] > voLBSPVar[largestIndex])
		{
			mIndex = 1;
			lIndex = 3;
		}
		else
		{
			mIndex = 3;
			lIndex = 1;
		}

	}


	if (hIndex == 3)
	{
		if (voColorVar[largestIndex] > voKeyVar[largestIndex])
		{
			mIndex = 2;
			lIndex = 1;
		}
		else
		{
			mIndex = 1;
			lIndex = 2;
		}

	}


}

void Sampler::sortProbBoxes(vector<objectDetected>& TBoxes)
{

	if (TBoxes.size() > 1)
	{

		for (auto i = 0; i < TBoxes.size(); ++i)

		{
			if (TBoxes[0].fTotalProbScore < (TBoxes[i].fTotalProbScore))

			{

				TBoxes[0] = TBoxes[i];
			}



		}
	}

}

void Sampler::nonModelMatchedKeysinSampleROI(std::vector<objectKeys>& voKeyPoints, int nTotalModelKeys, objectDetected& TBoxes, cv::Rect& ROI, Point2f& predCenter)
{
	for (std::vector<objectKeys>::iterator iter = voKeyPoints.begin(); iter != voKeyPoints.end(); ++iter)
	{


		if (iter->key.pt.x >= ROI.x && iter->key.pt.x <= (ROI.x + ROI.width) && iter->key.pt.y >= ROI.y && iter->key.pt.y <= ((ROI.y) + (ROI.height)))
		{
			if (iter->indi == 1 && iter->predC != Point2f(0, 0))
			{

				TBoxes.fDesDistance += iter->distance;
				TBoxes.nKeyPointMatches += 1;



				/*	if (iter->distance < 150.0 && TBoxes.diffC.x < 40.0 && TBoxes.diffC.y < 40.0)
				{

				TBoxes.fDesDistance += iter->distance;
				TBoxes.nKeyPointMatches += 1;

				}*/


			}


		}


	}
	if (TBoxes.nKeyPointMatches > 0)
	{
		TBoxes.fDesDistance = TBoxes.fDesDistance / float(nTotalModelKeys);
	}
	else
	{
		TBoxes.fDesDistance = 0.0;
	}


}

void Sampler::getBoxesProbScore(vector<objectDetected>& TBoxes, vector<objectKeys>& voFilteredKeyPoints, vector<objectKeys>& voKeyPoints, Mat oLBSPModelDesc, Mat oColorModelDesc, Mat oSIFTModelDes, Mat LBSPModelROIMat, Mat oImage, ObjectDetector dector, Rect& oPropBox, Point2f& predCenter, size_t& nNumMatches)
{

	for (auto i = 0; i < TBoxes.size(); ++i)

	{
		Rect oRectT;
		Point2f pt1, pt2;
		cv::Mat oFrameSampleBoxImage, oImageClone, oImageClone1;
		oImageClone = oImage.clone();
		oImageClone1 = oImage.clone();
		cv::Mat oSampleDes, oImage2;
		vector<KeyPoint> oSampleKeys;
		vector<vector<DMatch>> matches;
		oRectT.x = TBoxes[i].nXPos;
		oRectT.y = TBoxes[i].nYPos;
		oRectT.width = TBoxes[i].nWidth;
		oRectT.height = TBoxes[i].nHeight;

		const int nSizes[3] = { 16, 16, 16 };
		TBoxes[i].oColorHist = Mat::zeros(3, nSizes, CV_32FC1);
		TBoxes[i].fTotalProbScore = 0.0;
		vector<int> voOutput;

		pt1 = Point2f(oRectT.x, oRectT.y);
		pt2 = Point2f((oRectT.x + oRectT.width), (oRectT.y + (oRectT.height)));

		oKeyPointFunc.setLBSPROI(oImage, oFrameSampleBoxImage, pt1, pt2, oRectT);



		/*cv::Rect2d inter;
		inter = oRectT && oPropBox;
		*/
		TBoxes[i].nKeyPointMatches = 0;
		TBoxes[i].fDesDistance = 0.0;
		TBoxes[i].fColorScore = 0.0;
		TBoxes[i].fLBSPScore = 0.0;
		TBoxes[i].fNormColorScore = 0.0;
		TBoxes[i].fNormLBSPScore = 0.0;
		TBoxes[i].fNormDisScore = 0.0;
		TBoxes[i].objectCenter.x = TBoxes[i].nXPos + TBoxes[i].nWidth / 2.0;
		TBoxes[i].objectCenter.y = TBoxes[i].nYPos + TBoxes[i].nHeight / 2.0;

		/*	TBoxes[i].diffC.x = predCenter.x - TBoxes[i].objectCenter.x;
		TBoxes[i].diffC.y = predCenter.y - TBoxes[i].objectCenter.y;

		TBoxes[i].diffWidth = oPropBox.width - TBoxes[i].nWidth;
		TBoxes[i].diffHeight = oPropBox.height - TBoxes[i].nHeight;

		*/
		//TBoxes[i].fOverlap = inter.area() / float(oRectT.area() + oPropBox.area() - inter.area());


		//cv::rectangle(oImageClone, oRectT, cv::Scalar(0, 0, 255), 3, 8, 0);
		//cv::rectangle(oImageClone, oPropBox, cv::Scalar(255, 0, 0), 3, 8, 0);
		//imshow("RectT", oImageClone);
		//cvWaitKey(10);




		if (!oFrameSampleBoxImage.empty())
		{
			vector<Rect> rects;
			vector<float> scores;
			vector<int> index;

			nonModelMatchedKeysinSampleROI(voKeyPoints, voFilteredKeyPoints.size(), TBoxes[i], oRectT, predCenter); // How many non model keypoints that are matched lie in the proposal box

			if (TBoxes[i].nKeyPointMatches > 0 && nNumMatches > 0)
			{
				TBoxes[i].indiKP = 1;
				TBoxes[i].fNormDetScore = 0.0;
				oKeyPointFunc.checkLBSPsizewithModel(oFrameSampleBoxImage, LBSPModelROIMat);

				oKeyPointFunc.computeLBSPDes(oFrameSampleBoxImage, TBoxes[i].oLBSPDes);

				TBoxes[i].fNormKeyPointMatches = TBoxes[i].nKeyPointMatches / float(nNumMatches);


				oKeyPointFunc.calWeightedColor(oImage, TBoxes[i].oColorHist, oRectT);

				oKeyPointFunc.matchLBSPAM(oLBSPModelDesc, TBoxes[i].oLBSPDes, voOutput, TBoxes[i].fLBSPScore);

				oKeyPointFunc.matchColorAM(oColorModelDesc, TBoxes[i].oColorHist, TBoxes[i].fColorScore);


				cvtColor(oImageClone, oImageClone, CV_BGR2GRAY);

				cv::Mat oNew = oImageClone(oRectT);

				index = dector.detectFace(oNew, rects, scores);


				if (!scores.empty())
				{
					if (scores[0] > 20.0)
					{
						TBoxes[i].fDetScore = scores[0];
					}
					else
					{
						TBoxes[i].fDetScore = 0.0;
					}


				}
				else
				{
					TBoxes[i].fDetScore = 0.0;

				}

				index.clear();
				rects.clear();
				scores.clear();


			}

			else
			{
				TBoxes[i].indiKP = 0;

				TBoxes[i].fDesDistance = 0.0;

				TBoxes[i].fNormKeyPointMatches = 0.0;

				TBoxes[i].fNormDetScore = 0.0;

				TBoxes[i].fNormDisScore = 0.0;

				oKeyPointFunc.checkLBSPsizewithModel(oFrameSampleBoxImage, LBSPModelROIMat);

				oKeyPointFunc.computeLBSPDes(oFrameSampleBoxImage, TBoxes[i].oLBSPDes);

				oKeyPointFunc.calWeightedColor(oImage, TBoxes[i].oColorHist, oRectT);

				oKeyPointFunc.matchLBSPAM(oLBSPModelDesc, TBoxes[i].oLBSPDes, voOutput, TBoxes[i].fLBSPScore);

				oKeyPointFunc.matchColorAM(oColorModelDesc, TBoxes[i].oColorHist, TBoxes[i].fColorScore);



				cvtColor(oImage, oImage2, CV_BGR2GRAY);

				cv::Mat oNew = oImage2(oRectT);


				index = dector.detectFace(oNew, rects, scores);

				if (!scores.empty())
				{

					if (scores[0] > 20.0)
					{
						TBoxes[i].fDetScore = scores[0];
					}
					else
					{
						TBoxes[i].fDetScore = 0.0;
					}

				}
				else
				{
					TBoxes[i].fDetScore = 0.0;

				}


				index.clear();
				rects.clear();
				scores.clear();

			}
		}

		else
		{

			TBoxes[i].fDesDistance = 0.0;
			TBoxes[i].nKeyPointMatches = 0;
			TBoxes[i].fNormKeyPointMatches = 0.0;
			TBoxes[i].fColorScore = 0.0;
			TBoxes[i].fLBSPScore = 0.0;
			TBoxes[i].fNormColorScore = 0.0;
			TBoxes[i].fNormLBSPScore = 0.0;
			TBoxes[i].indiKP = -1;
			TBoxes[i].fDetScore = 0.0;
			TBoxes[i].fNormDetScore = 0.0;
			TBoxes[i].fNormDisScore = 0.0;
		}
	}

}



bool operator>(const objectDetected &ob1, const objectDetected &ob2)
{
	return ob1.fTotalProbScore > ob2.fTotalProbScore;
}



void Sampler::sortTotalScore(vector<objectDetected>& voFaceObjects)
{


	std::sort(voFaceObjects.begin(), voFaceObjects.end(), [](const objectDetected &ob1, const objectDetected &ob2)
	{
		return ob1.fTotalProbScore > ob2.fTotalProbScore;
	});

}

void Sampler::sortNumberMatches(vector<objectDetected>& voFaceObjects)
{
	std::sort(voFaceObjects.begin(), voFaceObjects.end(), [](const objectDetected &ob1, const objectDetected &ob2)
	{
		return ob1.nKeyPointMatches > ob2.nKeyPointMatches;
	});


}



void Sampler::getBestProp(vector<objectDetected>& TBoxes, vector<std::pair <float, size_t>>& voTBoxesValues, objectDetected& oBestPropBox)
{

	for (auto j = 0; j < TBoxes.size(); ++j)
	{
		std::pair <float, size_t >oTBoxesValues(TBoxes[j].fTotalProbScore, j);
		voTBoxesValues.push_back(oTBoxesValues);

	}


	size_t nLargestIndex = TBoxes.size() - 1;
	std::sort(voTBoxesValues.begin(), voTBoxesValues.end(), ObjectKeyPoint::compare_first_only());

	oBestPropBox.nXPos = TBoxes[voTBoxesValues[nLargestIndex].second].nXPos;
	oBestPropBox.nYPos = TBoxes[voTBoxesValues[nLargestIndex].second].nYPos;
	oBestPropBox.nWidth = TBoxes[voTBoxesValues[nLargestIndex].second].nWidth;
	oBestPropBox.nHeight = TBoxes[voTBoxesValues[nLargestIndex].second].nHeight;
	oBestPropBox.oLBSPDes = TBoxes[voTBoxesValues[nLargestIndex].second].oLBSPDes;
	oBestPropBox.oColorHist = TBoxes[voTBoxesValues[nLargestIndex].second].oColorHist;
	oBestPropBox.fColorScore = TBoxes[voTBoxesValues[nLargestIndex].second].fColorScore;
	oBestPropBox.fLBSPScore = TBoxes[voTBoxesValues[nLargestIndex].second].fLBSPScore;
	oBestPropBox.fTotalProbScore = TBoxes[voTBoxesValues[nLargestIndex].second].fTotalProbScore;
}


void Sampler::findstructure(mxArray* aPtr) {

	vector<objectDetected> proposals;


	if (mxGetClassID(aPtr) == mxSTRUCT_CLASS) {
		/*if (mxGetFieldNumber(aPtr, field) == -1) {
		printf("Field not found: %s\n", field);
		}
		else {
		analyzestructarray(aPtr, proposals);
		}*/

		analyzestructarray(aPtr, proposals);

	}
	else {
		printf("%s is not a structure\n", aPtr);
	}

	mxDestroyArray(aPtr);

}



//int Sampler::findstructure2(const char *arr) {
//
//	MATFile *mfPtr; /* MAT-file pointer */
//	mxArray *aPtr;  /* mxArray pointer */
//
//	
//
//	aPtr = matGetVariable(mfPtr, arr);
//	if (aPtr == NULL) {
//		printf("mxArray not found: %s\n", arr);
//		return(1);
//	}
//
//	if (mxGetClassID(aPtr) == mxSTRUCT_CLASS) {
//		if (mxGetFieldNumber(aPtr, field) == -1) {
//			printf("Field not found: %s\n", field);
//		}
//		else {
//			analyzestructarray(aPtr, field);
//		}
//	}
//	else {
//		printf("%s is not a structure\n", arr);
//	}
//	mxDestroyArray(aPtr);
//
//	if (matClose(mfPtr) != 0) {
//		printf("Error closing file %s\n", file);
//		return(1);
//	}
//	return(0);
//}


/* Find cell array ARR in MAT-file FILE. */
//int Sampler::findcell(const char *file, const char *arr) {
//
//	MATFile *mfPtr; /* MAT-file pointer */
//	mxArray *aPtr;  /* mxArray pointer */
//
//	mfPtr = matOpen(file, "r");
//	if (mfPtr == NULL) {
//		printf("Error opening file %s\n", file);
//		return(1);
//	}
//
//	aPtr = matGetVariable(mfPtr, arr);
//	if (aPtr == NULL) {
//		printf("mxArray not found: %s\n", arr);
//		return(1);
//	}
//
//	if (mxGetClassID(aPtr) == mxCELL_CLASS) {
//		analyzecellarray(aPtr);
//	}
//	else {
//		printf("%s is not a cell array\n", arr);
//	}
//	mxDestroyArray(aPtr);
//
//	if (matClose(mfPtr) != 0) {
//		printf("Error closing file %s\n", file);
//		return(1);
//	}
//	return(0);
//}

void Sampler::getRandomBoxes(cv::Rect& trackROI, vector<objectDetected>& voObjectDetected, cv::Point2f& predCenter, cv::Mat oImage2)
{

	vector<cv::Rect> testRects;
	Mat testImage2 = oImage2.clone();
	Mat testImage = oImage2.clone();
	float th = 0.1;
	Rect tBox1, tBox2, tBox3, tBox4, tBox5;

	tBox1.x = (predCenter.x - 1.0) - (trackROI.width / 2.0);

	tBox1.y = (predCenter.y - 1.0) - (trackROI.height / 2.0);

	tBox1.width = trackROI.width;
	tBox1.height = trackROI.height;



	tBox2.x = (predCenter.x + 1.0) - (trackROI.width / 2.0);

	tBox2.y = (predCenter.y + 1.0) - (trackROI.height / 2.0);

	tBox2.width = trackROI.width;
	tBox2.height = trackROI.height;


	tBox3.x = (predCenter.x - 2.0) - (trackROI.width / 2.0);

	tBox3.y = (predCenter.y - 2.0) - (trackROI.height / 2.0);

	tBox3.width = trackROI.width;
	tBox3.height = trackROI.height;



	tBox4.x = (predCenter.x + 2.0) - (trackROI.width / 2.0);

	tBox4.y = (predCenter.y + 2.0) - (trackROI.height / 2.0);

	tBox4.width = trackROI.width;
	tBox4.height = trackROI.height;


	testRects.push_back(tBox1);
	testRects.push_back(tBox2);
	testRects.push_back(tBox3);
	testRects.push_back(tBox4);

	for (auto j = 0; j < testRects.size(); ++j)
	{


		Rect2d testBoxRect, boxGT, inter, uni;
		float overlap;

		testBoxRect.x = testRects[j].x;
		testBoxRect.y = testRects[j].y;
		testBoxRect.width = testRects[j].width;
		testBoxRect.height = testRects[j].height;

		boxGT.x = trackROI.x;
		boxGT.y = trackROI.y;
		boxGT.width = trackROI.width;
		boxGT.height = trackROI.height;



		Point2f oCenterTestBox, diff;
		oCenterTestBox.x = testBoxRect.x + testBoxRect.width / 2.0;
		oCenterTestBox.y = testBoxRect.y + testBoxRect.height / 2.0;

		int diffXW, diffYH;
		diffXW = boxGT.width - testBoxRect.width;
		diffYH = boxGT.height - testBoxRect.height;
		diff.x = predCenter.x - oCenterTestBox.x;
		diff.y = predCenter.y - oCenterTestBox.y;


		inter = testBoxRect & boxGT;


		overlap = inter.area() / float(testBoxRect.area() + boxGT.area() - inter.area());


		//rectangle(testImage, testBoxRect, cv::Scalar(0, 255, 0), 1, 8, 0);
		//rectangle(testImage, boxGT, cv::Scalar(255, 0, 0), 1, 8, 0);


		//imshow("PROP", testImage);
		//cvWaitKey(10);

		//if (overlap > th && abs(diffXW) == 0 && abs(diffYH) == 0)
		if (overlap > th)

		{
			objectDetected testBox;

			testBox.nXPos = testBoxRect.x;
			testBox.nYPos = testBoxRect.y;
			testBox.nWidth = testBoxRect.width;
			testBox.nHeight = testBoxRect.height;
			testBox.objectCenter = Point2f(oCenterTestBox.x, oCenterTestBox.y);
			testBox.predC = Point2f(abs(diff.x), abs(diff.y));
			voObjectDetected.push_back(testBox);

		}


	}

}