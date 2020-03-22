#include "opencv2/opencv.hpp"
#include "boost/lambda/lambda.hpp"
#include "boost/filesystem.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/algorithm/string/find_iterator.hpp"
#include "boost/regex.hpp"
#include <boost/algorithm/string.hpp>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>
#include <numeric>
#include <cmath>
#include<math.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"

#pragma once
#include "ObjectDetector.h"
#include "C:\Program Files\MATLAB\R2019b\extern\include\engine.h"
#include <assert.h>
#include <fstream>
#include <string>
#include "ObjectKeyPoint.h"
#include "LBSP.h"
#include "DistanceUtils.h"
#include "matlabfile.h"
#include "Sampler.h"
#include "C:\Program Files\MATLAB\R2019b\extern\include\mex.h"

#define DRAW 19
#define LB 0 
#define GT 10
#define SAVE 1
#define NPD 1
#define MAX_SAMPLES 100
#define SEARCH_RAD1 25
#define SEARCH_RAD2 20
#define OUTER_RAD 4
#define GPU_MATCHER 0
#define FTGEN 1
#define FTSCALE 0
#define AUTOD 1

using namespace std;
using namespace cv;
using namespace boost;
using namespace boost::filesystem;


const char* keys = {
	"{c||}"
	"{t||}"
	"{a||}"
	"{z||}"
	"{b||}"
	"{u||}"
	"{s||}"
	"{p||}"
	"{r||}"
	"{f||}"
	"{e||}"
	"{n||}"
	"{q||}"
	"{m||}"
	"{g||}"
	"{x||}"
	"{y||}"
	"{w||}"
	"{h||}"
	"{th||}"
	"{gpu||}"
};



#include <omp.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <algorithm>

BFMatcher bf(NORM_L2, true);

ObjectDetector oDAlg;


int main(int argc, char** argv)

{
	clock_t start = clock();
	double duration;
	vector<string> vidDir;
	vidDir.push_back("trellis");
	vidDir.push_back("boy");
	vidDir.push_back("dudek");
	vidDir.push_back("david2");
	vidDir.push_back("girl");
	vidDir.push_back("faceocc1");
	vidDir.push_back("mhyang");
	vidDir.push_back("soccer");
	vidDir.push_back("david");
	vidDir.push_back("freeman1");
	vidDir.push_back("freeman4");
	vidDir.push_back("freeman3");
	vidDir.push_back("faceocc2");
	vidDir.push_back("jumping");
	vidDir.push_back("fleetface");

	CommandLineParser parser(argc, argv, keys);

	size_t nContext = parser.get<int>("c");
	float fRadius = parser.get<float>("t");
	float fLearningRate = parser.get<float>("a");
	float fRateRemove = parser.get<float>("z");
	float fExpo = parser.get<float>("b");
	float fUpdate = parser.get<float>("u");
	string sSeqName = parser.get<String>("s");
	string sSeqPath = parser.get<String>("p");
	const string sResultDir = parser.get<String>("r");
	int nStartFrame = parser.get<int>("f");
	int nEndFrame = parser.get<int>("e");
	int nNz = parser.get<int>("n");
	int nX = parser.get<int>("x");
	int nY = parser.get<int>("y");
	int nW = parser.get<int>("w");
	int nH = parser.get<int>("h");
	float q = parser.get<float>("q");
	float m = parser.get<float>("m");
	float g = parser.get<float>("g");
	float ratioTh = parser.get<float>("th");
	float gpu = parser.get<float>("gpu");
	regex pattern("(.*\\.jpg)");
	float fThreshWeight = 0.1;
	float fnewWeight = 0.11;
	float fCompareColorHist = 0.0, fLBSPDesDiff = 0.0;

	Rect trackROI, oPropBest;
	objectDetected trackROIBox;
	Point2f oPropCenter;
	cv::Ptr<Feature2D> pSift = xfeatures2d::SIFT::create();
	int nLineNumber = 1;
	vector<std::string> accumulator;
	ObjectKeyPoint oKAlgModel;
	Mat oImage1, oDescriptor1, oDescriptorLBSPModel;
	const int nSizes[3] = { 16, 16, 16 };
	Mat oColorModelHist = Mat::zeros(3, nSizes, CV_32FC1);
	Mat oColorNonModelHist = Mat::zeros(3, nSizes, CV_32FC1);
	Mat oLBSPROIModelMat;
	vector<objectKeys> voKeyPoints1;
	vector<string>::iterator iter1;
	Rect ROI, oLBSPROIModelRect, oColorModelRect, oColorNonModelRect, oLBSPROINonModelRect;
	Point2f pt1, pt2;
	Point2f previousCenter, predCenter, center, diff, diffCenterFDT;
	size_t nFrame = 1;
	size_t nFrameCounter = 0;
	size_t hIndex = 0, mIndex = 0, lIndex = 0;
	std::vector<float> voAccScales;
	float fScaleChangeT = 1.0;
	float fScaleChangeTPlusOne = 1.0;
	Mat oImage3;
	ofstream file1;

	for (vector<string>::const_iterator i = vidDir.begin(); i != vidDir.end(); ++i)

	{
		start = std::clock();
		cout << *i << endl;

		sSeqPath = "C:\\TANUSHRI\\DATASETS\\data_seq\\" + *i + "\\img\\";

		sSeqName = sSeqName.substr(0, sSeqName.length() - 2);

		cout << "SEQUENCE" << "\t" << sSeqName << endl;


		if (nContext == 1)
		{
			file1.open(sResultDir + "\\" + sSeqName + "_1_FACETRACK_CONTEXT.txt ", ios::out);
		}


		if (nContext == 0)

		{

			file1.open(sResultDir + "\\" + sSeqName + "_1_FACETRACK_NO_CONTEXT.txt ", ios::out);

		}

		path target_path(sSeqPath);



		for (directory_iterator iter(target_path), end; iter != end; ++iter)
		{
			string sImgNum = iter->path().leaf().string();

			string imgFile = sSeqPath + sImgNum;

			if (regex_match(sImgNum, pattern))
			{
				accumulator.push_back(imgFile);
			}

		}


		if (sSeqName == "david")
		{
			nStartFrame = 1;
			nFrame = 1;
		}


		nFrame = nStartFrame;

		bool faceInit;
		faceInit = 0;

		for (iter1 = ((accumulator.begin() + (nStartFrame - 1))); iter1 != accumulator.end(); ++iter1)
		{
			oImage1 = imread(*iter1);

			Mat oFramePutText3 = oImage1.clone();


			////// Automatic Initialization

			vector<objectDetected> voFaceBox;

			oDAlg.Detect(iter1, voFaceBox);

			if (!voFaceBox.empty())
			{
				auto func = [](const objectDetected &ob1, const objectDetected &ob2)
				{
					return ob1.fDetScore > ob2.fDetScore;
				};


				std::sort(voFaceBox.begin(), voFaceBox.end(), func);

				for (auto i = 0; i < voFaceBox.size(); ++i)
				{


					ROI.x = voFaceBox[i].nXPos;
					ROI.y = voFaceBox[i].nYPos;
					ROI.width = voFaceBox[i].nWidth;
					ROI.height = voFaceBox[i].nHeight;


					cv::Mat oFramePutText1 = oImage1.clone();

					rectangle(oFramePutText1, ROI, cv::Scalar(255, 0, 0), 3, 8, 0);

					namedWindow("ROI", WINDOW_AUTOSIZE);
					moveWindow("ROI", 60, 200);
					imshow("ROI", oFramePutText1);
					cvWaitKey(10);
				}
			}

			if (faceInit == 0)
			{
				ROI.x = voFaceBox[0].nXPos;
				ROI.y = voFaceBox[0].nYPos;
				ROI.width = voFaceBox[0].nWidth;
				ROI.height = voFaceBox[0].nHeight;


				if (ROI.area() > 0)
				{
					faceInit = 1;

					////Model init
					std::vector<KeyPoint> voDetectedKeys;
					pSift->detect(oImage1, voDetectedKeys);
					assert(!voDetectedKeys.empty());
					voKeyPoints1.resize(voDetectedKeys.size());

					for (auto i = 0; i < voDetectedKeys.size(); ++i) {

						voKeyPoints1[i].key = voDetectedKeys[i];
						voKeyPoints1[i].descriptor = Mat(1, 128, CV_32FC1);
						voKeyPoints1[i].proxFactor = -1;
						voKeyPoints1[i].index = -1;
						voKeyPoints1[i].indi = -1;

					}


					center.x = ROI.x + ROI.width / 2;
					center.y = ROI.y + ROI.height / 2;
					pt1.x = ROI.x;
					pt1.y = ROI.y;


					if (nContext == 1)
					{

						pt2 = Point2f((center.x + ROI.width / 2), ((1.2*center.y) + (ROI.height / 2)));
					}

					if (nContext == 0)
					{
						pt2 = Point2f((center.x + ROI.width / 2), ((center.y) + (ROI.height / 2)));
					}

#if DRAW == 1

					cv::Mat oFramePutText1 = oImage1.clone();

					rectangle(oFramePutText1, pt1, pt2, cv::Scalar(255, 0, 0), 3, 8, 0);

					namedWindow("ROI", WINDOW_AUTOSIZE);
					moveWindow("ROI", 60, 200);
					imshow("ROI", oFramePutText1);
					cvWaitKey(10);

#endif


# if SAVE == 3
					namedWindow("TEST", WINDOW_AUTOSIZE);
					imwrite(".\\fdOut\\context.jpg", test1);
					imshow("TEST", test1);

					cvWaitKey(10);
#endif






# if SAVE == 1
					if (file1.is_open())

					{


						file1 << floor(pt1.x) << "," << floor(pt1.y) << "," << (ROI.width) << "," << (ROI.height) << endl;
					}




					if (!file1.is_open()){

						if (nContext == 1)
						{

							file1.open(sResultDir + "\\" + sSeqName + "_1_FACETRACK_CONTEXT.txt ", ios::out);

						}


						if (nContext == 0)

						{

							file1.open(sResultDir + "\\" + sSeqName + "_1_FACETRACK_NO_CONTEXT.txt ", ios::out);

						}

						file1 << ROI.x << "," << ROI.y << "," << ROI.width << "," << ROI.height << endl;

					}


#endif					





# if SAVE == 3
					if (file1.isOpened())
					{

						file1 << "{" << "frame " << frame;
						file1 << "X " << pt1.x;
						file1 << "Y " << pt1.y;
						file1 << "W " << ROI.width;
						file1 << "H " << ROI.height << "}";
					}

#endif
# if SAVE == 2
					if (file1.isOpened())
					{

						file1 << "{" << "frame " << frame;
						file1 << "CX " << center.x;
						file1 << "CY " << center.y;
						file1 << "ROIW " << ROI.width;
						file1 << "ROIH " << ROI.height << "}";
					}

#endif			

					oKAlgModel.filteredKeyPoints(voKeyPoints1, oKAlgModel.m_voFilteredKeyPoints, ROI, nContext);
					oDescriptor1 = oKAlgModel.computeDes(oImage1);
					Mat oImageCopy1, oLBSPColorImageClone1;

					oImageCopy1 = oImage1.clone();
					oLBSPColorImageClone1 = oImage1.clone();


					for (auto i = 0; i < oDescriptor1.rows; ++i)
					{


						oKAlgModel.m_voFilteredKeyPoints[i].descriptor = oDescriptor1.row(i).clone();

					}

					oKAlgModel.encodeStructure(center, oImageCopy1, fUpdate); //HOW MUCH KPS HAVE MOVED FROM CENTER AND THEIR DIRECTION TOWARDS THE CENTER

					oKAlgModel.setLBSPROI(oLBSPColorImageClone1, oLBSPROIModelMat, pt1, pt2, oLBSPROIModelRect);

					oKAlgModel.computeLBSPDes(oLBSPROIModelMat, oDescriptorLBSPModel);

					oColorModelRect = oLBSPROIModelRect;

					oKAlgModel.calWeightedColor(oLBSPColorImageClone1, oColorModelHist, oColorModelRect);
				}
			}

			if (iter1 > ((accumulator.begin() + (nStartFrame - 1))) && faceInit == 1) {



				size_t nNumMatches = 0;
				objectDetected oBestPropBox;

				nFrameCounter = nFrameCounter + 1;
				bool flag = false;
				bool fullFace = false;
				float oPropDetScore = 0.0;
				float oPropBestDetScore = 0.0;
				float oTrackROIDetScore = 0.0;
				float oFDBoxDetScore = 0.0;
				size_t oPropMatches = 0;



				if (nFrame == nStartFrame + 1)
				{
				
					oBestPropBox.nXPos = ROI.x;
					oBestPropBox.nYPos = ROI.y;
					oBestPropBox.nWidth = ROI.width;
					oBestPropBox.nHeight = ROI.height;
				}


				Mat oImage2 = imread(*iter1);

				Mat oFramePutText2 = oImage2.clone();
				Mat oFramePutText3 = oImage2.clone();
				Mat oLBSPColorImageClone2 = oImage2.clone();
				Mat oImageCopy2 = oImage2.clone();
				cv::Mat oImage2Clone = oImage2.clone();
				vector<KeyPoint> voDetectedKeys2;
				vector<objectKeys> voKeyPoints;
				vector<vector<DMatch>> matchesR;
				pSift->detect(oImage2, voDetectedKeys2);

				assert(!voDetectedKeys2.empty());

				voKeyPoints.resize(voDetectedKeys2.size());

				for (auto i = 0; i < voDetectedKeys2.size(); ++i) {

					cv::Mat testDes = cv::Mat(1, 128, CV_32FC1);

					voKeyPoints[i].key = voDetectedKeys2[i];

					voKeyPoints[i].descriptor = Mat(1, 128, CV_32FC1);

					voKeyPoints[i].index = -1;

					voKeyPoints[i].indi = -1;
				}


				std::ostringstream str;


				char frameString[10];
				char sym[2] = "#";
				itoa(nFrame, frameString, 10);
				strcat(frameString, sym);

				cv::putText(oFramePutText2, frameString, cv::Point(10, 20), FONT_HERSHEY_COMPLEX_SMALL, 1, CV_RGB(255, 0, 0));

				cv::putText(oFramePutText3, frameString, cv::Point(10, 20), FONT_HERSHEY_COMPLEX_SMALL, 1, CV_RGB(255, 0, 0));

				Mat oDescriptor2;


				pSift->compute(oImage2, voDetectedKeys2, oDescriptor2);



				for (auto i = 0; i < oDescriptor2.rows; ++i) {
					{


						voKeyPoints[i].descriptor = oDescriptor2.row(i).clone();

					}
				}


#if GPU_MATCHER == 1

				matchEngine.match(oDescriptor1, oDescriptor2, false);
#else

				bf.radiusMatch(oDescriptor1, oDescriptor2, matchesR, fRadius);
#endif


#if DRAW == 1

				vector<KeyPoint> voDrawKeysModel, voDrawKeysNonModel;
				cv::Mat img1, img2, out, out1, img3, img4;

				if (nFrame == nStartFrame + 1)
				{

					img1 = oImage1.clone();
					img3 = oImage1.clone();
				}

				else
				{
					img1 = oImage3.clone();
					img3 = oImage3.clone();

				}


				img2 = oImage2.clone();
				img4 = oImage2.clone();

				for (auto i = 0; i < oKAlgModel.m_voFilteredKeyPoints.size(); ++i){


					voDrawKeysModel.push_back(oKAlgModel.m_voFilteredKeyPoints[i].key);

				}


				namedWindow("MATCHING", WINDOW_AUTOSIZE);
				drawMatches(img3, voDrawKeysModel, img4, voDetectedKeys2, matchEngine.vv_DMatch, out1);
				imshow("RADMATCHING1", out1);
				cvWaitKey(10);



#endif

				
#if GPU_MATCHER == 1

				oKAlgModel.filterMatches(matchEngine.vv_DMatch, ratioTh);
				oKAlgModel.setMatchingIndexKeysAM(matchEngine.vv_DMatch, nNumMatches, voKeyPoints);

#else

				oKAlgModel.filterMatches(matchesR, ratioTh);

				oKAlgModel.setMatchingIndexKeysAM(matchesR, nNumMatches, voKeyPoints);
#endif

				oImage3 = oImage2.clone();

				if (nFrame == nStartFrame + 1)
				{

					previousCenter = center;
					trackROI = ROI;
					predCenter = center;
					oPropBest = ROI;
					oPropCenter = center;
				}


				vector<objectKeys> voKeyPointsROI;

				if (nNumMatches != 0)
				{
					oKAlgModel.voting(voKeyPointsROI, voKeyPoints, oFramePutText2, trackROI, previousCenter, predCenter,
						nFrame, fExpo, flag);

					if (flag || voKeyPointsROI.size() == 0)

					{
						if (nFrame != (nStartFrame + 1))
						{
							trackROI = oPropBest;
							predCenter = oPropCenter;
							oTrackROIDetScore = oPropBestDetScore;

						}

					}

					else
					{

						trackROI.x = predCenter.x - (oPropBest.width / 2.0);

						trackROI.y = predCenter.y - (oPropBest.height / 2.0);

						trackROI.width = oPropBest.width;
						trackROI.height = oPropBest.height;

					}

#if FTSCALE == 1

					std::vector<int> voIndexes; std::vector<Point2f> voDistance1, voDistance2;


					float fMeanScaleChange = 0.0;
					float fFinalScaleChange = 1.0;
					if (nNumMatches > 1)

					{
						Point2f pairedDist1, pairedDistNew1, pairedDist2, pairedDistNew2, pairedDist3, pairedDistNew3;

						vector<std::pair <float, int>> voKeyWeightValues;

						oKAlgModel.getMinMaxWeightKeyPoint(oKAlgModel.m_voFilteredKeyPoints, voKeyPoints, voKeyWeightValues);

						oKAlgModel.computePairDistance(oKAlgModel.m_voFilteredKeyPoints, voKeyPoints, voKeyWeightValues, voDistance1, voDistance2);

						oKAlgModel.detectScaleChange(voDistance1, voDistance2, fMeanScaleChange);

						if (fMeanScaleChange >= 0.9 && fMeanScaleChange < 1.1)
						{

							voAccScales.push_back(fMeanScaleChange);
						}

						else

						{
							voAccScales.push_back(1.0);
						}


						if (nFrameCounter == 11)
						{
							nFrameCounter = 0;


							fFinalScaleChange = std::accumulate(voAccScales.begin(), voAccScales.end(), 1.0, std::multiplies<float>());


							if (fFinalScaleChange > 0.9 && fFinalScaleChange < 1.06)
							{


								oPropBest.width = fFinalScaleChange*oPropBest.width;
								oPropBest.height = fFinalScaleChange*oPropBest.height;

								trackROI.x = predCenter.x - (oPropBest.width / 2.0);

								trackROI.y = predCenter.y - (oPropBest.height / 2.0);

								trackROI.width = oPropBest.width;
								trackROI.height = oPropBest.height;


							}


							voAccScales.clear();


						}

					}
#endif

				}

				else
				{
					flag = true;

				}


				oKAlgModel.ROIBoxAdjust(oImage2, trackROI);

				oKAlgModel.update(predCenter, fLearningRate, nFrame, fUpdate, fRateRemove, nStartFrame);


# if SAVE == 2
				if (file1.isOpened())
				{

					file1 << "{" << "frame " << frame;
					file1 << "CX " << predCenter.x;
					file1 << "CY " << predCenter.y;
					file1 << "ROIW " << trackROI.width;
					file1 << "ROIH " << trackROI.height << "}";
				}

#endif		


# if SAVE == 3
				if (file1.isOpened())

				{

					file1 << "{" << "frame " << frame;
					file1 << "X " << trackROI.x;
					file1 << "Y " << trackROI.y;
					file1 << "W" << trackROI.width;
					file1 << "H " << trackROI.height << "}";
				}

#endif					



				Rect oFDBox = { 0, 0, 0, 0 };
				Point2f oFDCenter;
				Rect trackROI1, trackROI2, trackROI3, trackROI4, trackROI5, trackROI6, trackROI7, trackROI8;
				vector<cv::Rect> gtTrackBoxes;
				std::vector<objectDetected> voObjectDetected; // get the proposals

				Sampler oSamples;

				if (!voFaceBox.empty())
				{

					oKAlgModel.computeFaceObjectsROI(voFaceBox, oLBSPROIModelMat, oImageCopy2, voKeyPoints, oBestPropBox);

					if (!voFaceBox.empty())
					{
						oKAlgModel.matchFaceObjects(voFaceBox, oDescriptorLBSPModel, oColorModelHist, predCenter, oBestPropBox);
					}


					for (auto k = 0; k < voFaceBox.size(); ++k)
					{
						Rect oFDBox1, oFDBox2, oFDBox3, oFDBox4;
						Mat testImage2 = oImage2.clone();
						oFDBox.x = voFaceBox[k].nXPos;
						oFDBox.y = voFaceBox[k].nYPos;
						oFDBox.width = voFaceBox[k].nWidth;
						oFDBox.height = voFaceBox[k].nHeight;
						oFDCenter.x = oFDBox.x + oFDBox.width / 2;
						oFDCenter.y = oFDBox.y + oFDBox.height / 2;




#if DRAW == 1

						Mat testImage = oImage2.clone();
						rectangle(testImage, oFDBox, cv::Scalar(0, 0, 255), 1, 8, 0);

						imshow("FD", testImage);
						//cout << nFrame << "\tFD" << "," << oFDBox.x << "," << oFDBox.y << "," << oFDBox.width << "," << oFDBox.height << endl;
						cvWaitKey(10);
#endif


					}
				}



#if FTGEN == 1	
				if (trackROI.area() != NULL)
				{
					vector<cv::Rect> testRects;

					oSamples.getRandomBoxes(trackROI, voObjectDetected, predCenter, oImage2);

				}
#endif


				trackROIBox.nXPos = trackROI.x;
				trackROIBox.nYPos = trackROI.y;
				trackROIBox.nWidth = trackROI.width;
				trackROIBox.nHeight = trackROI.height;




#if DRAW == 1
				Mat testImage = oImage2.clone();
				if (voObjectDetected.size() > 0)

				{
					for (auto l = 0; l < voObjectDetected.size(); ++l)
					{

						Rect testBoxRect;
						testBoxRect.x = voObjectDetected[l].nXPos;
						testBoxRect.y = voObjectDetected[l].nYPos;
						testBoxRect.width = voObjectDetected[l].nWidth;
						testBoxRect.height = voObjectDetected[l].nHeight;
						//rectangle(testImage, gtTrackBoxes[0], cv::Scalar(0, 0, 255), 1, 8, 0);
						rectangle(testImage, testBoxRect, cv::Scalar(0, 255, 0), 1, 8, 0);

					}
					imshow("PROP", testImage);
					cvWaitKey(10);
				}
#endif


				voObjectDetected.push_back(trackROIBox);


				if (!voFaceBox.empty())
				{
					for (auto l = 0; l < voFaceBox.size(); ++l)
					{

						if (voFaceBox[l].nWidth*voFaceBox[l].nHeight != NULL)
						{
							voObjectDetected.push_back(voFaceBox[l]);
						}

					}
				}


				Rect oProp = { 0, 0, 0, 0 };
				vector<std::pair <float, size_t>> voTBoxesValues;

				oSamples.getBoxesProbScore(voObjectDetected, oKAlgModel.m_voFilteredKeyPoints, voKeyPoints, oDescriptorLBSPModel, oColorModelHist, oDescriptor1, oLBSPROIModelMat, oImage2, oDAlg, oPropBest, predCenter, nNumMatches);
				oSamples.getProbPropBoxes(voObjectDetected, oImage2, q, m, g, nFrame, hIndex, mIndex, lIndex);
				oSamples.getBestProp(voObjectDetected, voTBoxesValues, oBestPropBox);

				size_t nLargestIndex = voObjectDetected.size() - 1;

				oProp.x = oBestPropBox.nXPos;
				oProp.y = oBestPropBox.nYPos;
				oProp.width = oBestPropBox.nWidth;
				oProp.height = oBestPropBox.nHeight;
				float oPropTotalScore = oBestPropBox.fTotalProbScore;

				cv::Mat testImageROI = oImage2.clone();
				cv::Mat testROI2 = oImage2.clone();
				cv::Mat testROI2Clone, testROI2Clone2, testROI2Clone3;

#if DRAW == 1
				imshow("ROI", testImageROI);
				cvWaitKey(10);
#endif

				if ((oProp.x >= 0 && oProp.x + oProp.width <= oImage2.cols &&  oProp.y >= 0 && oProp.y + oProp.height <= oImage2.rows))


				{

					pt1 = Point(oProp.x, oProp.y);


					pt2 = Point((oProp.x + oProp.width), (oProp.y + (oProp.height)));


					oPropCenter = Point2f((oProp.x + oProp.width / 2.0), (oProp.y + oProp.height / 2.0));


					diff = previousCenter - oPropCenter;

					
					if (abs(diff.x) < 33.0 && abs(diff.y) < 33.0)
					{
					
						predCenter = oPropCenter;
						previousCenter = oPropCenter;
						oPropBest = oProp;
						trackROI = oPropBest;

						char frameString[10];
						char sym[2] = "#";


						itoa(nFrame, frameString, 10);
						strcat(frameString, sym);

						cv::putText(oFramePutText2, frameString, cv::Point(10, 20), FONT_HERSHEY_COMPLEX_SMALL, 1, CV_RGB(255, 0, 0));
						rectangle(oFramePutText2, oPropBest, cv::Scalar(0, 255, 0), 4, 8, 0);



					}

					else
					{
						if (nFrame == nStartFrame + 1)

						{
							oPropBest = trackROI;
							oPropCenter = predCenter;
							oBestPropBox.nXPos = oPropBest.x;
							oBestPropBox.nYPos = oPropBest.y;
							oBestPropBox.nWidth = oPropBest.width;
							oBestPropBox.nHeight = oPropBest.height;
							oBestPropBox.fLBSPScore = 10.0;
							oBestPropBox.fColorScore = 10.0;
						}



						char frameString[10];
						char sym[2] = "#";


						itoa(nFrame, frameString, 10);
						strcat(frameString, sym);

						cv::putText(oFramePutText2, frameString, cv::Point(10, 20), FONT_HERSHEY_COMPLEX_SMALL, 1, CV_RGB(255, 0, 0));
						rectangle(oFramePutText2, oPropBest, cv::Scalar(0, 255, 0), 4, 8, 0);

					}


				}


				else{

					if (nFrame == nStartFrame + 1)

					{
						oPropBest = trackROI;
						oPropCenter = predCenter;
						oBestPropBox.nXPos = oPropBest.x;
						oBestPropBox.nYPos = oPropBest.y;
						oBestPropBox.nWidth = oPropBest.width;
						oBestPropBox.nHeight = oPropBest.height;
						oBestPropBox.fLBSPScore = 10.0;
						oBestPropBox.fColorScore = 10.0;

					}

					if (nFrame != 2)
					{

						oPropBest = oPropBest;
						oPropCenter = oPropCenter;
						oBestPropBox.nXPos = oPropBest.x;
						oBestPropBox.nYPos = oPropBest.y;
						oBestPropBox.nWidth = oPropBest.width;
						oBestPropBox.nHeight = oPropBest.height;
						oBestPropBox.fLBSPScore = 10.0;
						oBestPropBox.fColorScore = 10.0;
					}

					char frameString[10];
					char sym[2] = "#";
					itoa(nFrame, frameString, 10);
					strcat(frameString, sym);

					cv::putText(oFramePutText2, frameString, cv::Point(10, 20), FONT_HERSHEY_COMPLEX_SMALL, 1, CV_RGB(255, 0, 0));

					rectangle(oFramePutText2, oPropBest, cv::Scalar(0, 255, 0), 4, 8, 0);
					


				}

# if DRAW == 9
				namedWindow("OUTPUT", WINDOW_AUTOSIZE);

				imshow("OUTPUT", oFramePutText2);

				cvWaitKey(10);
# endif



# if SAVE == 1
				if (file1.is_open())

				{

					file1 << oPropBest.x << "," << oPropBest.y << "," << oPropBest.width << "," << oPropBest.height << endl;
				}



				if (!file1.is_open())
				{
					if (nContext == 1)
					{

						file1.open(sResultDir + "\\" + sSeqName + "_1_FACETRACK_CONTEXT.txt ", ios::out);

					}


					if (nContext == 0)

					{

						file1.open(sResultDir + "\\" + sSeqName + "_1_FACETRACK_NO_CONTEXT.txt ", ios::out);

					}

					file1 << oPropBest.x << "," << oPropBest.y << "," << oPropBest.width << "," << oPropBest.height << endl;
				}

#endif			


				if (oKAlgModel.m_voFilteredKeyPoints.size() > 500)
				{

					oKAlgModel.removeKeys(fThreshWeight, oPropCenter);

				}


				if (oBestPropBox.fLBSPScore <= 0.23 && oBestPropBox.fColorScore < 0.1)

				{

					oKAlgModel.nonModelKeysinTrackROI(oPropBest, voKeyPoints);

					oKAlgModel.addKeys(oPropBest, oPropCenter, fUpdate, fnewWeight);

					oKAlgModel.removeKeys(fThreshWeight, oPropCenter);

					oDescriptor1.create(oKAlgModel.m_voFilteredKeyPoints.size(), oDescriptor1.cols, CV_32FC1);

					for (auto i = 0; i < oKAlgModel.m_voFilteredKeyPoints.size(); ++i)
					{


						oKAlgModel.m_voFilteredKeyPoints[i].descriptor.copyTo(oDescriptor1.row(i));


					}

				}



				for (auto i = 0; i < oKAlgModel.m_voFilteredKeyPoints.size(); ++i) // As we have to use again for voting
				{


					oKAlgModel.m_voFilteredKeyPoints[i].index = -1;
					oKAlgModel.m_voFilteredKeyPoints[i].indi = -1;


				}

				oBestPropBox.fColorScore = 0.0;
				oBestPropBox.fLBSPScore = 0.0;

			}//accumulator

			nFrame = nFrame + 1;
		}
		
		duration = (std::clock() - start)  / (double)CLOCKS_PER_SEC;
		//cout << duration << endl;
	}

	file1.close();
	accumulator.clear();


	return 0;

}