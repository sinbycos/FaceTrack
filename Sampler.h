#pragma once
#define _USE_MATH_DEFINES
#include <cmath>

#include<opencv2\core\core.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\highgui\highgui.hpp>
#include <stdio.h>
#include "ObjectKeyPoint.h"
#include "ObjectDetector.h"
using namespace std;
using namespace cv;


#include <stdlib.h>
#include "mat.h"
#include <assert.h>


class Sampler
{

public:


	Sampler();

	virtual ~Sampler();

	ObjectKeyPoint oKeyPointFunc;

	void getProbPropBoxes(vector<objectDetected>& TBoxes, Mat oImage, float&q, float&m, float&g, size_t& nFrame, size_t& hIndex, size_t& mIndex, size_t& lIndex);

	void sortProbBoxes(vector<objectDetected>& TBoxes);

	void sortNumberMatches(vector<objectDetected>& voFaceObjects);

	void getBoxesProbScore(vector<objectDetected>& TBoxes, vector<objectKeys>& voFilteredKeyPoints, vector<objectKeys>& voKeyPoints, Mat LBSPModelDesc, Mat ColorModelDesc, Mat oSIFTModelDes, Mat LBSPModelROIMat, Mat oImage, ObjectDetector dector, Rect& oPropBox, Point2f& predCenter, size_t& nNumMatches);

	void findstructure(mxArray *ptr);

	void getHighestVar(vector<float>& voKeyVar, vector<float>& voColorVar, vector<float>& voLBSPVar, size_t& largestIndex, size_t& hIndex, size_t& mIndex, size_t& lIndex);
	//void findstructure2(mx)
	//int findcell(const char *file, const char *arr);

	//! Sorts the total probability score
	void sortTotalScore(vector<objectDetected>& voFaceObjects);

	//! Determines new keypoints that are lying the proposal boxes
	void nonModelMatchedKeysinSampleROI(std::vector<objectKeys>& voKeyPoints, int nTotalModelKeys, objectDetected& TBoxes, cv::Rect& ROI, Point2f& predCenter);

	//! Determines the proposal with the highest total prob score
	void getBestProp(vector<objectDetected>& TBoxes, vector<std::pair <float, size_t>>& voTBoxesValues, objectDetected& oBestPropBox);


	void getRandomBoxes(cv::Rect& trackROI, vector<objectDetected>& voObjectDetected, cv::Point2f& predCenter, cv::Mat oImage2);

	/* Analyze field FNAME in struct array SPTR. */
	inline static void analyzestructarray(const mxArray *sPtr, vector<objectDetected>& proposals)
	{
		mwSize nElements;       /* number of elements in array */
		mwIndex eIdx;           /* element index */
		const mxArray *fPtr;    /* field pointer */
		double *realPtr;        /* pointer to data */
		double total;           /* value to calculate */
		objectDetected pBox;
		total = 0;
		nElements = (mwSize)mxGetNumberOfElements(sPtr);
		for (eIdx = 0; eIdx < nElements; eIdx++) {
			/*fPtr = mxGetField(sPtr, eIdx, fName);
			if ((fPtr != NULL)
			&& (mxGetClassID(fPtr) == mxDOUBLE_CLASS)
			&& (!mxIsComplex(fPtr)))
			{
			realPtr = mxGetPr(fPtr);
			total = total + realPtr[0];
			}*/

			int row = (int)mxGetScalar(mxGetField(sPtr, eIdx, "row"));
			int col = (int)mxGetScalar(mxGetField(sPtr, eIdx, "col"));
			int height = (int)mxGetScalar(mxGetField(sPtr, eIdx, "height"));
			int width = (int)mxGetScalar(mxGetField(sPtr, eIdx, "width"));
			float score = (float)mxGetScalar(mxGetField(sPtr, eIdx, "score"));
			//pBox.nXPos = row;
			//pBox.nYPos = col;

			pBox.nXPos = col;
			pBox.nYPos = row;
			pBox.nWidth = width;
			pBox.nHeight = height;
			pBox.fDetScore = score;
			proposals.push_back(pBox);
		}
		//printf("Total for %s: %.2f\n", fName, total);
	}



	inline static void analyzecellarray(const mxArray *cPtr, vector<objectDetected>& propFromMat)
	{
		mwSize nCells;          /* number of cells in array */
		mwIndex cIdx;           /* cell index */
		const mxArray *ePtr;    /* element pointer */
		mxClassID category;     /* class ID */
								//const char *fName;

		nCells = (mwSize)mxGetNumberOfElements(cPtr);
		for (cIdx = 0; cIdx < nCells; cIdx++) {
			ePtr = mxGetCell(cPtr, cIdx);
			if (ePtr == NULL) {
				printf("Empty Cell\n");
				break;
			}
			category = mxGetClassID(ePtr);
			printf("%d: ", cIdx);
			switch (category) {
			case mxCHAR_CLASS:
				printf("string\n");
				/* see revord.c */
				break;
			case mxSTRUCT_CLASS:
				//printf("structure\n");
				/* see analyzestructure.c */
				analyzestructarray(ePtr, propFromMat);

				break;
			case mxCELL_CLASS:
				printf("cell\n");
				printf("{\n");
				analyzecellarray(ePtr, propFromMat);
				printf("}\n");
				break;
			case mxUNKNOWN_CLASS:
				printf("Unknown class\n");
				break;
			default:
				if (mxIsSparse(ePtr)) {
					printf("sparse array\n");
					/* see mxsetnzmax.c */
				}
				else {
					printf("numeric class\n");
				}
				break;
			}
		}
	}

};
