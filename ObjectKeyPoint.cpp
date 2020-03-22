

#include "ObjectKeyPoint.h"
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include<numeric>
#include <boost/regex.hpp>
#include<regex>
#include <iostream>
#include <iterator>
#include <vector>


using namespace std;

using namespace cv;
using namespace cv::xfeatures2d;
using std::max_element;
#define e 2.71828
#define pi 3.14
#define DRAW 1
# define SAVE 2


#include <omp.h>
#include <math.h>
#include <vector>
#include <iostream>
// Added by Tanushri

#include <algorithm>

// Min Max C3861 error
using namespace std;

#include "C:\Program Files\MATLAB\R2019b\extern\include\matrix.h"
#include "C:\Program Files\MATLAB\R2019b\extern\include\mat.h"





ObjectKeyPoint::ObjectKeyPoint()
{

}


ObjectKeyPoint::~ObjectKeyPoint()
{
}





cv::Mat ObjectKeyPoint::computeDes(const cv::Mat& oInitImg) {

	vector<KeyPoint> keys;
	Ptr<Feature2D> pSift = xfeatures2d::SIFT::create();

	for (auto i = 0; i < m_voFilteredKeyPoints.size(); ++i){

		keys.push_back(m_voFilteredKeyPoints[i].key);
	}

	
	cv::Mat descriptor;
	
	pSift->compute(oInitImg, keys, descriptor);
	m_voDescriptors = descriptor;
	return m_voDescriptors;
}






void ObjectKeyPoint::filteredKeyPoints(std::vector<objectKeys>& voKeyPoints, std::vector<objectKeys>& voFilteredKeyPoints, cv::Rect ROI, int contextFlag){





	for (std::vector<objectKeys>::iterator iter = voKeyPoints.begin(); iter != voKeyPoints.end(); ++iter)
	{



		if (contextFlag == 1)

		{
			if (iter->key.pt.x >= ROI.x && iter->key.pt.x <= (ROI.x + ROI.width) && iter->key.pt.y >= ROI.y && iter->key.pt.y <= ((1.2*ROI.y) + (ROI.height)))
			{

				voFilteredKeyPoints.push_back((*iter));
			}

		}


		else
		{


			if (iter->key.pt.x >= ROI.x && iter->key.pt.x <= (ROI.x + ROI.width) && iter->key.pt.y >= ROI.y && iter->key.pt.y <= ((ROI.y) + (ROI.height)))
			{

				voFilteredKeyPoints.push_back((*iter));


			}
		}


	}


}




void ObjectKeyPoint::encodeStructure( cv::Point2f center, cv::Mat image1, float& fUpdate){

	
	for (unsigned int i = 0; i < m_voFilteredKeyPoints.size(); i++)
	{

		float x = 0, m = 0.01; float c = 0, y = 0;
		Point2f dis_cen_temp = center - m_voFilteredKeyPoints[i].key.pt;

		int dis_cen1 = (dis_cen_temp.x*dis_cen_temp.x) + (dis_cen_temp.y*dis_cen_temp.y);
		
		m_voFilteredKeyPoints[i].dis_Cen = dis_cen_temp; // X spatial constraint vector
		
		m_voFilteredKeyPoints[i].weight1 = std::max<float>((1 - abs(fUpdate*dis_cen1)), 0.5);

		

#if DRAW == 13

		cv::arrowedLine(image1, cvPoint(m_voFilteredKeyPoints[i].key.pt.x, m_voFilteredKeyPoints[i].key.pt.y),  cvPoint(center.x, center.y), CV_RGB(0,0,255), 2 , 8);
		//line( image1, cvPoint(center.x, center.y), cvPoint(SX, SY), CV_RGB(0,0,255), 2,8);

		//line( image1, cvPoint(m_voFilteredKeyPoints[i].key.pt.x , m_voFilteredKeyPoints[i].key.pt.y), cvPoint(center.x, center.y), Scalar(255,0,0), 2,8);

		circle( image1,
			center,
			0,
			Scalar( 0, 255, 0 ),
			2,
			8 );
		/*circle(image1,
		dis_cen_temp,
		0,
		Scalar( 255, 0, 0 ),
		8,
		10 ); *///delta x and delta y

		////line ( image1, cvPoint(m_voFilteredKeyPoints[i].dis_Cen.x + m_voFilteredKeyPoints[i].key.pt.x, m_voFilteredKeyPoints[i].dis_Cen.y + m_voFilteredKeyPoints[i].key.pt.y), cvPoint(center.x, center.y),
		//     Scalar( 0, 255, 0 ),
		//    2,
		//     8 );
		/*circle( image1, cvPoint(center.x - m_voFilteredKeyPoints[i].dis_Cen.x , center.y - m_voFilteredKeyPoints[i].dis_Cen.y),  0,
			Scalar( 0, 0, 255 ),
			8,
			10 ); //Distance vector
			circle( image1, cvPoint(m_voFilteredKeyPoints[i].key.pt.x + m_voFilteredKeyPoints[i].dis_Cen.x , m_voFilteredKeyPoints[i].key.pt.y + m_voFilteredKeyPoints[i].dis_Cen.y),  0,
			Scalar( 0, 0, 255 ),
			8,
			10 ); */


		// Spatial Constraint Distance vector
		namedWindow("APPEARANCE MODEL - PHASE 1", WINDOW_AUTOSIZE ); 
		moveWindow("APPEARANCE MODEL - PHASE 1", 200, 600);
		imshow("APPEARANCE MODEL - PHASE 1", image1);
		cvWaitKey(10);
		//imwrite(".\\fdCom\\faceContextPaper.jpg", image1);


#endif
		
	}

}


cv::Rect ObjectKeyPoint::readGTCenter(string& gtFile)
{
	
	size_t nStart = 0;
	size_t nInit = 0;
	size_t nTotal;
	int nRect[4] = {};
	ifstream groundFile;
	groundFile.open(gtFile.c_str(), ios::in);
	string sLine;
	int nLineNumber = 1;
	if (groundFile.is_open())
	{
		while (getline(groundFile, sLine))
		{
			if (sLine.find(" ") != string::npos)
			{
				/*cout << sLine << " " << nLineNumber2 << endl;
				cout << sLine << '\n';*/
			}

			
			else
			{

				if (nLineNumber == 1)
					break;
			}
		}



		groundFile.close();
	}

	else

	{
		cout << "Unable to open file";

	}





	nTotal = sLine.size();
	for (auto i = 0; i < 4; i++)
	{

		nStart = sLine.find(",", nInit); //find from position start
	/*	if (sLine.empty())
		{
			nStart = sLine.find(",", nInit);
		}*/
		string sX = sLine.substr(nInit, nStart); //from init pos till how many pos+count
		nRect[i] = stoi(sX);
		nInit = nStart + 1;
	}


	Rect ROI(nRect[0], nRect[1], nRect[2], nRect[3]);
	return ROI;
}


void ObjectKeyPoint::voting(std::vector<objectKeys>& keyPointsROI, std::vector<objectKeys>& m_voKeyPoints, cv::Mat image2, cv::Rect ROI, cv::Point2f& previousCenter, cv::Point2f& predCenter, size_t & frameNum, float& fExpo,
	bool& flag){


	Point2f Xc, diffCenter; float diff;

	// gaussian 5x5 pattern  based on fspecial('gaussian',[5 5], 6.0)

	float gaussVotingWindow[25] = { 0.0378, 0.0394, 0.0400, 0.0394, 0.0378,
		0.0394, 0.0411, 0.0417, 0.0411, 0.0394,
		0.0400, 0.0417, 0.0423, 0.0417, 0.0400,
		0.0394, 0.0411, 0.0417, 0.0411, 0.0394,
		0.0378, 0.0394, 0.0400, 0.0394, 0.0378 };


	cv::Mat gauss = cv::Mat(5, 5, CV_32FC1, gaussVotingWindow);
	cv::Mat voteMatrix = Mat::zeros(image2.rows, image2.cols, CV_32FC1);


	unsigned int bound = 20; //Voting does not happend outside the voting matrix

	float fGaussConst1;
	Point Loc;
	float xLocCenter;
	float yLocCenter;
	float fGaussNum;
	float fGaussDen;
	float fExp;


	for (unsigned int i = 0; i < m_voFilteredKeyPoints.size(); i++)
	{


		if (m_voFilteredKeyPoints[i].indi == 1)
		{

			keyPointsROI.push_back(m_voKeyPoints[m_voFilteredKeyPoints[i].index]); //non model keyppoints that have been matched with model keypoints 

			// VOTING BY ENCODED STRUCTURE
			Xc = m_voKeyPoints[m_voFilteredKeyPoints[i].index].key.pt + m_voFilteredKeyPoints[i].dis_Cen;
			m_voKeyPoints[m_voFilteredKeyPoints[i].index].predC = Xc;
			m_voFilteredKeyPoints[i].predC = Xc;
			m_voFilteredKeyPoints[i].predCenters.push_back(Xc);

			Point2f diff = previousCenter - Xc; //PENALIZE MATCHING IF THE CURRENT PREDICTION IS TOO FAR FROM THE PREVIOUS CENTER

			int diff2 = (diff.x*diff.x + diff.y*diff.y);
			float expoDiff = exp(-diff2 / fExpo);


#if DRAW == 13
			Mat testIm = image2.clone();
			cv::arrowedLine(testIm, cvPoint(m_voKeyPoints[m_voFilteredKeyPoints[i].index].key.pt.x, m_voKeyPoints[m_voFilteredKeyPoints[i].index].key.pt.y), cvPoint(Xc.x, Xc.y), Scalar(0, 0, 255), 2, 8);

			circle(testIm, cvPoint(Xc.x, Xc.y), 2,
				Scalar( 255, 0, 0 ),
				1,
				10 ); //Distance vector


			namedWindow("VOTING BY KEYPOINTS-PHASE 3", WINDOW_AUTOSIZE);
			moveWindow("VOTING BY KEYPOINTS-PHASE 3", 600, 400);

			imshow("VOTING BY KEYPOINTS-PHASE 3", testIm);
			cvWaitKey(10);

#endif

#if SAVE == 2



# endif


			Loc.x = int(Xc.x);
			Loc.y = int(Xc.y);


			if (Loc.x >= bound  && Loc.x <= (image2.cols - bound) && Loc.y >= bound && Loc.y <= (image2.rows - bound))
				/*if ((Loc.x - gauss.cols / 2) >= bound && (Loc.x + gauss.cols / 2) <= (image2.cols - bound) &&
					(Loc.y - gauss.rows / 2) >= bound && (Loc.y + gauss.rows / 2) <= (image2.rows - bound))*/
			{



				for (auto k = (Loc.x - gauss.cols / 2); k <= (Loc.x + gauss.cols / 2); k++)
				{
					for (auto l = (Loc.y - gauss.rows / 2); l <= (Loc.y + gauss.rows / 2); l++)

					{
						cv::Mat sub = voteMatrix(cv::Rect(k, l, gauss.cols, gauss.rows));

						float fValue = m_voFilteredKeyPoints[i].weight1*expoDiff;

						sub += m_voFilteredKeyPoints[i].weight1*gauss*expoDiff;

#if DRAW == 13


						//namedWindow("VOTING BY KEYPOINTS IILUSTRATION IN VOTING MATRIX", WINDOW_AUTOSIZE);

						//moveWindow("VOTING BY KEYPOINTS IILUSTRATION IN VOTING MATRIX", 10, 50);
						imshow("VOTING BY KEYPOINTS IILUSTRATION IN VOTING MATRIX", voteMatrix);
						cvWaitKey(10);
						//imwrite(".\\Res\\voting\\voting.jpg",voteMatrix);
# endif		


					}
				}
			}

		}

	}

	float maximum1 = 0.0;




	vector<std::pair <float, cv::Point2f>> voFCenterValues;


	for (unsigned int j = 0; j < voteMatrix.rows; ++j)
	{

		for (unsigned int i = 0; i < voteMatrix.cols; ++i)
		{

			if (voteMatrix.at<float>(j, i) >(maximum1) && voteMatrix.at<float>(j, i) > 0.0){
				cv::Point x;
				x.x = i;
				x.y = j;
				maximum1 = voteMatrix.at<float>(j, i);
				//fCenterValue = std::make_pair(float(maximum1), x);
				std::pair<float, cv::Point2f> fCenterValue(maximum1, x);
				voFCenterValues.push_back(fCenterValue);
			}


		}

	}


	
	if (voFCenterValues.size() > 0)

	{
		std::sort(voFCenterValues.begin(), voFCenterValues.end(), compare_first_only()); // Highest voting values with their pred Centers
		size_t largestIndex = voFCenterValues.size() - 1;


		if (voFCenterValues[largestIndex].first > 0.0)
		{


			predCenter.x = voFCenterValues[largestIndex].second.x;
			predCenter.y = voFCenterValues[largestIndex].second.y;
			
		}

		else
		{
			for (auto j = (largestIndex - 1); j > 0; --j)
			{
				if (voFCenterValues[j].first > 0.0)

				{

					predCenter.x = voFCenterValues[j].second.x;
					predCenter.y = voFCenterValues[j].second.y;
					
					break;
				}
			}
		}

		Point2f diff = predCenter - previousCenter;


	//	cout << abs(diff.x) << "&&" << abs(diff.y) << endl;


		if (abs(diff.x) > 35.0 || abs(diff.y) > 35.0) // if predicted center too far then do not update the position
		//if (abs(diff.x) > 27.0 || abs(diff.y) > 27.0)
		{
			
			flag = true;
			predCenter = previousCenter;
		}

			

	}

	else

	{
		flag = true;
		predCenter = previousCenter;
			
	}

}


void ObjectKeyPoint::update(cv::Point2f& predCenter, float& a, size_t & frame, float& fUpdate, float& fRateRemove, int& nStartFrame){


	for (unsigned int i = 0; i < m_voFilteredKeyPoints.size(); i++)
	{



		if ((m_voFilteredKeyPoints[i].indi == 1)){


			Point2f diffCenter = m_voFilteredKeyPoints[i].predC - predCenter;



			m_voFilteredKeyPoints[i].diffT1 = sqrt(diffCenter.x*diffCenter.x + diffCenter.y*diffCenter.y);


			m_voFilteredKeyPoints[i].proxFactor = std::max<float>((1 - abs(fUpdate*m_voFilteredKeyPoints[i].diffT1)), 0.0);

			if (m_voFilteredKeyPoints[i].proxFactor == 0.0)

			{

				m_voFilteredKeyPoints[i].weight1 = m_voFilteredKeyPoints[i].weight1 - (fRateRemove)*m_voFilteredKeyPoints[i].weight1; 
			}

			else

			{
				m_voFilteredKeyPoints[i].weight1 = (1 - a)*m_voFilteredKeyPoints[i].weight1 + a*m_voFilteredKeyPoints[i].proxFactor;

			}

		}




		else{
				m_voFilteredKeyPoints[i].weight1 = m_voFilteredKeyPoints[i].weight1 - (fRateRemove)*m_voFilteredKeyPoints[i].weight1;

		}



	}


}




//void ObjectKeyPoint::matchedAMKeyPoints(cv::Rect trackROI, cv::Point2f predCenter, std::vector<objectKeys> m_voKeyPoints)
//{
//	vector<objectKeys> keysInTrackROI;
//
//
//	Point2f A(trackROI.x, trackROI.y), B(predCenter.x, predCenter.y - trackROI.height / 2), O(predCenter.x, predCenter.y), H(predCenter.x - trackROI.width / 2, predCenter.y), C(predCenter.x + trackROI.width / 2, predCenter.y - trackROI.height / 2),
//
//		D(predCenter.x + trackROI.width / 2, predCenter.y), E(predCenter.x + trackROI.width / 2, predCenter.y + trackROI.height / 2), F(predCenter.x, predCenter.y + trackROI.height / 2),
//		G(predCenter.x - trackROI.width / 2, predCenter.y + trackROI.height / 2);
//
//	for (size_t s = 0; s < m_voKeyPoints.size(); ++s)
//	{
//
//		if (m_voKeyPoints[s].key.pt.x >= A.x && m_voKeyPoints[s].key.pt.x <= C.x && m_voKeyPoints[s].key.pt.y >= A.y && m_voKeyPoints[s].key.pt.y <= G.y)
//
//		{
//			keysInTrackROI.push_back(m_voKeyPoints[s]);
//
//			if (m_voKeyPoints[s].indi == 1)
//			{
//				matchedKeysInTrackROI.push_back(m_voKeyPoints[s]);
//			}
//
//		}
//	}
//
//
//
//}









void ObjectKeyPoint::keyQuadLocation(std::vector<objectKeys> key, cv::Rect trackROI, cv::Point2f& predCenter, std::vector<objectKeys> &numberofKeysinFirstQuad, std::vector<objectKeys> &numberofKeysinSecQuad, std::vector<objectKeys> &numberofKeysinThirdQuad, std::vector<objectKeys> & numberofKeysinFourthQuad){


	Point2f A(trackROI.x, trackROI.y), B(predCenter.x, predCenter.y - trackROI.height / 2), O(predCenter.x, predCenter.y), H(predCenter.x - trackROI.width / 2, predCenter.y), C(predCenter.x + trackROI.width / 2, predCenter.y - trackROI.height / 2),

		D(predCenter.x + trackROI.width / 2, predCenter.y), E(predCenter.x + trackROI.width / 2, predCenter.y + trackROI.height / 2), F(predCenter.x, predCenter.y + trackROI.height / 2),
		G(predCenter.x - trackROI.width / 2, predCenter.y + trackROI.height / 2);

	for (unsigned int i = 0; i < key.size(); i++)
	{


		if (key[i].key.pt.x >= A.x && key[i].key.pt.x <= B.x && key[i].key.pt.y >= A.y && key[i].key.pt.y <= H.y)



		{

			numberofKeysinFirstQuad.push_back(key[i]);

		}

		if (key[i].key.pt.x >= B.x && key[i].key.pt.x <= C.x && key[i].key.pt.y >= B.y && key[i].key.pt.y <= O.y)


		{

			numberofKeysinSecQuad.push_back(key[i]);

		}

		if (key[i].key.pt.x >= O.x && key[i].key.pt.x <= D.x && key[i].key.pt.y >= O.y && key[i].key.pt.y <= F.y)


		{

			numberofKeysinThirdQuad.push_back(key[i]);

		}


		if (key[i].key.pt.x >= H.x && key[i].key.pt.x <= O.x && key[i].key.pt.y >= H.y && key[i].key.pt.y <= G.y)


		{

			numberofKeysinFourthQuad.push_back(key[i]);

		}
	}
}


void::ObjectKeyPoint::nonModelKeysinTrackROI(cv::Rect ROI, vector<objectKeys>& voKeyPoints)
{
	

	for (std::vector<objectKeys>::iterator iter = voKeyPoints.begin(); iter != voKeyPoints.end(); ++iter)
	{


		if (iter->key.pt.x >= ROI.x && iter->key.pt.x <= (ROI.x + ROI.width) && iter->key.pt.y >= ROI.y && iter->key.pt.y <= ((ROI.y) + (ROI.height)))
		{
			
				m_voFilteredKeyPointsTrackROI.push_back(*iter);
			
		}


	}

}

void::ObjectKeyPoint::nonModelKeysinFaceROI(objectDetected& TBoxes, vector<objectKeys>& voKeyPoints)
{


	for (std::vector<objectKeys>::iterator iter = voKeyPoints.begin(); iter != voKeyPoints.end(); ++iter)
	{


		if (iter->key.pt.x >= TBoxes.nXPos && iter->key.pt.x <= (TBoxes.nXPos + TBoxes.nWidth) && iter->key.pt.y >= TBoxes.nYPos && iter->key.pt.y <= ((TBoxes.nYPos) + (TBoxes.nHeight)))
		{
			if (iter->indi == 1)
			{
				TBoxes.voSIFTKeys.push_back(*iter);
			}
		}


	}

}


void ObjectKeyPoint::addKeys(cv::Rect ROI, cv::Point2f& predCenter, float& fUpdate, float& fnewWeight){



	for (std::vector<objectKeys>::iterator iter = m_voFilteredKeyPointsTrackROI.begin(); iter != m_voFilteredKeyPointsTrackROI.end(); ++iter)


	{
		if (iter->indi != 1)
		{

			if (iter->predCenters.size() > 0)
			{

				cout << "here" << endl;
			}

			iter->dis_Cen = predCenter - iter->key.pt;




			Point2f dis_cen_temp = predCenter - iter->key.pt;

			float dis_cen1 = (dis_cen_temp.x*dis_cen_temp.x) + (dis_cen_temp.y*dis_cen_temp.y);

			iter->dis_Cen = dis_cen_temp; // X spatial constraint vector


			//iter->weight1 = fnewWeight;
			iter->weight1 = std::max<float>((1 - abs(0.002*dis_cen1)), 0.0);

			//cout << "ADD\t" << iter->weight1 << endl;

			m_voFilteredKeyPoints.push_back(*iter);


		}




	}

	m_voFilteredKeyPointsTrackROI.clear();


}









//	int totalAppKeys = m_voFilteredKeyPoints.size();
//		float MatchRatio = float(matchedKeysInTrackROI)/float(totalAppKeys);



/*if(matchedKeysInTrackROI > 0) {


	//see the position of kps in the quadrant if lying in half then do not add

	if(                      ((KeysinFirstQuad.size() + KeysinSecQuad.size()) > 0 && (KeysinFirstQuad.size() + KeysinSecQuad.size()) < matchedKeysInTrackROI/2 ) ||
	((KeysinFirstQuad.size() + KeysinThirdQuad.size() > 0 && (KeysinFirstQuad.size() + KeysinThirdQuad.size()) <  matchedKeysInTrackROI/2) ||

	(KeysinFirstQuad.size() + KeysinFourthQuad.size() > 0 && (KeysinFirstQuad.size() + KeysinFourthQuad.size()) <  matchedKeysInTrackROI/2) ))



	{


	cout<< "UPDATE FRAME" << frame << endl;
	for(unsigned int i = 0; i <  addedobjectKeys.size(); i ++)
	{

	m_voFilteredKeyPoints.push_back(addedobjectKeys[i]);

	}



	}


	*/


//}













void ObjectKeyPoint::removeKeys( float& tWeight, cv::Point2f& predCenter){


	for (unsigned int i = 0; i < m_voFilteredKeyPoints.size(); i++)
	{

		{
			
			if (m_voFilteredKeyPoints[i].weight1 <= tWeight ){

				m_voFilteredKeyPoints.erase(m_voFilteredKeyPoints.begin() + (i));

			}


		}

	}


}



void ObjectKeyPoint::setLBSPROI(cv::Mat& oImage, cv::Mat& oImage2, Point2f& pt1, Point2f& pt2, cv::Rect& oLBSPROI){

	if (pt1.x < 0)
	{
		pt1.x = 2;
	}

	if (pt1.x >= oImage.cols)
	{
		pt1.x = oImage.cols - 2;

	}

	if (pt2.y < 0)
	{
		pt2.y = 2;
	}

	if (pt2.y >= oImage.rows)
	{
		pt2.y = oImage.rows - 2;
	}

	int width = int(pt2.x - pt1.x);
	int height = int(pt2.y - pt1.y);

	if (pt1.x >= 0 && pt1.x + width <= oImage.cols &&  pt1.y >=0 && pt1.y + height <= oImage.rows)
	{
		oLBSPROI.x = int(pt1.x);
		oLBSPROI.y = int(pt1.y);
		oLBSPROI.width = width;
		oLBSPROI.height = height;
		oImage2 = oImage.clone();
		oImage2 = oImage2(cv::Rect(oLBSPROI.x, oLBSPROI.y, oLBSPROI.width, oLBSPROI.height));
		
	}

	
	
}






void ObjectKeyPoint::computeLBSPDes(cv::Mat oROI, cv::Mat& oLBSPDes)
{
	cv::Mat oGrey;
	cv::cvtColor(oROI, oGrey, CV_RGB2GRAY);


	ushort uLBSP;  vector<ushort> voLBSP;

	for (auto j = 2; j < (oGrey.rows - 2); j++)
	{
		for (auto i = 2; i < (oGrey.cols - 2); i++)

		{

			LBSP::computeGrayscaleDescriptor(oGrey, oGrey.at<uchar>(j, i), i, j, THRESH, uLBSP);

			voLBSP.push_back(uLBSP);

		}


	}



	oLBSPDes.create(voLBSP.size(), 1, CV_16UC1); //creation malloc row*times and then it will allocate the memory of the defined size


	for (auto i = voLBSP.begin(); i != voLBSP.end(); i++)
	{

		oLBSPDes.at<ushort>(std::distance(voLBSP.begin(), i), 0) = *i;

	}

	


}





vector<DMatch> ObjectKeyPoint::sortMatches(vector<DMatch>& matchDes){

	float sort;
	for (size_t i = 0; i < matchDes.size() - 1; i++)

	if (matchDes[i + 1].distance < matchDes[i].distance)


	{
		sort = matchDes[i].distance;
		matchDes[i].distance = matchDes[i + 1].distance;
		matchDes[i + 1].distance = sort;
	}

	return matchDes;

}


void ObjectKeyPoint::normalizeMatches(vector<float>& sampleMatches, vector<float>& normSampleMatches, vector<objectKeys>& voSIFTSampleKeys)
{
	normSampleMatches = sampleMatches;
	std::sort(normSampleMatches.begin(), normSampleMatches.end());
	int nLargestDisIndex = normSampleMatches.size() - 1;
	for (auto i = 0; i < voSIFTSampleKeys.size(); ++i)
	{
		if (voSIFTSampleKeys[i].indi == 1)
		{
			voSIFTSampleKeys[i].distance = voSIFTSampleKeys[i].distance / normSampleMatches[nLargestDisIndex];
		}

	}

}

void ObjectKeyPoint::filterMatches(vector<vector<DMatch>>& foundMatches, float& ratioTestTh ){


	for (size_t k = 0; k < foundMatches.size(); ++k)
	{

		if (foundMatches[k].size() > 1)
		{


			float ratio = foundMatches[k][0].distance / foundMatches[k][1].distance;

			//cout << ratio << endl;0.75 before 0.9
			if (ratio > ratioTestTh)
			{

				foundMatches[k].clear();
			}
			else
			{
				cv::DMatch match = foundMatches[k][0];

				foundMatches[k].clear();

				foundMatches[k].push_back(match);

			}
		}

	}
}

void ObjectKeyPoint::filterMatches(vector<DMatch>& foundMatches, float& ratioTestTh)
{
	for (size_t k = 0; k < foundMatches.size(); ++k)
	{

		if (foundMatches.size() > 1)
		{


			float ratio = foundMatches[0].distance / foundMatches[1].distance;
		//	cout << ratio << endl;

			if (ratio > ratioTestTh)
			{

				foundMatches.clear();
			}
			else
			{
				cv::DMatch match = foundMatches[k];

				foundMatches.clear();

				foundMatches.push_back(match);

			}
		}

	}

	}


void ObjectKeyPoint::setMatchingIndexKeysAM(vector<vector<DMatch>>& matchesR, size_t& nTotalMatches, vector<objectKeys>& voKeyPoints)
{


	for (size_t k = 0; k < matchesR.size(); ++k)
	{

		
			if (!matchesR[k].empty()){

				nTotalMatches += 1;
				cv::DMatch match = matchesR[k][0];

				if (match.queryIdx != -1 && match.queryIdx < m_voFilteredKeyPoints.size())

					if (match.trainIdx != -1 && match.trainIdx < voKeyPoints.size())


					{
						m_voFilteredKeyPoints.at(match.queryIdx).distance = match.distance;
						//cout << match.distance << endl;

						m_voFilteredKeyPoints.at(match.queryIdx).index = match.trainIdx;
						m_voFilteredKeyPoints.at(match.queryIdx).indi = 1;
						voKeyPoints[match.trainIdx].distance = match.distance;

						voKeyPoints[match.trainIdx].index = match.queryIdx;
						voKeyPoints[match.trainIdx].indi = 1;


					}


			}

	}
}


void ObjectKeyPoint::setMatchingIndexKeysAM(vector<DMatch>& matchesR, size_t& nTotalMatches, vector<objectKeys>& voKeyPoints)
{

	for (size_t k = 0; k < matchesR.size(); ++k)
	{

		{
			if (!matchesR.empty()){

				nTotalMatches += 1;
				cv::DMatch match = matchesR[k];

				if (match.queryIdx != -1 && match.queryIdx < m_voFilteredKeyPoints.size())

					if (match.trainIdx != -1 && match.trainIdx < voKeyPoints.size())


					{
						m_voFilteredKeyPoints.at(match.queryIdx).distance = match.distance;
						m_voFilteredKeyPoints.at(match.queryIdx).index = match.trainIdx;
						m_voFilteredKeyPoints.at(match.queryIdx).indi = 1;
						voKeyPoints[match.trainIdx].distance = match.distance;

						voKeyPoints[match.trainIdx].index = match.queryIdx;
						voKeyPoints[match.trainIdx].indi = 1;


					}


			}

		}

	}
}

void ObjectKeyPoint::ROIAdjust(const cv::Mat& oSource, Point2f & pt1, Point2f & pt2)
{

	if (pt1.x < 0)
	{
		pt1.x = 0;
	}

	if (pt2.x >= oSource.cols)
	{
		pt2.x = oSource.cols - 2;

	}

	if (pt1.y < 0)
	{
		pt2.y = 0;
	}

	if (pt2.y >= oSource.rows)
	{
		pt2.y = oSource.rows - 2;
	}

}



void ObjectKeyPoint::ROIBoxAdjust(const cv::Mat& oSource, cv::Rect& oBox)
{
	
	//assert(oBox.x >= 0);

	//assert(oBox.y >= 0);

	//assert(oBox.x + oBox.width <= oSource.cols);

	//assert(oBox.y + oBox.height <= oSource.rows);

	if (oBox.x < 0)
	{
		oBox.x = 0;
	}

	if (oBox.y < 0)
	{
		oBox.y = 0;
	}

	if (oBox.x + oBox.width > oSource.cols)
	{
		oBox.width = oSource.cols - oBox.x;
	}
	
	
	if (oBox.y + oBox.height > oSource.rows)
	{
		oBox.height = oSource.rows - oBox.y;
	}

	

	if (oBox.width >= oSource.cols)
	{
		oBox.width = oSource.cols;

	}

	

	if (oBox.height >= oSource.rows)
	{
		oBox.height = oSource.rows;
	}
	


	

}

void ObjectKeyPoint::calWeightedColor(cv::Mat oImage, cv::Mat& oHist, cv::Rect oROI){


	if (oROI.x >= 0 && oROI.x + oROI.width <= oImage.cols && oROI.height >= 0 && oROI.y + oROI.height <= oImage.rows)
	{
		Point oROICenterPixel;

		oROICenterPixel.x = (oROI.width) / 2;
		oROICenterPixel.y = (oROI.height) / 2;

		cv::Mat oTest = oImage.clone();

		oTest = oTest(oROI);


		float diagonal = sqrt(oROI.width*oROI.width + oROI.height*oROI.height);
		float sigma = diagonal / 6.0;

		float fGaussConst1 = 1.0 / (2 * pi*pow(sigma, 2));
		float fGaussConst2 = 2.0 * (pow(sigma, 2));
		
		const uchar* uImageData = oTest.data;
		const size_t stepRow = oTest.step.p[0];
		int nRedBinValue, nGreenBinValue, nBlueBinValue;

		int nRedBins = 16;
		int nGreenBins = 16;
		int nBlueBins = 16;
		int nRange = 256;

		//const int nSizes[3] = { nBlueBins, nGreenBins, nRedBins };

		//	oHist = Mat::zeros(3, nSizes, CV_32FC1);

		Point Loc;

		for (auto i = 0; i < oTest.rows; i++)
		{

			for (auto j = 0; j < oTest.cols; j++)
			{

				Loc.x = j;
				Loc.y = i;
				float xLocCenter = (Loc.x - oROICenterPixel.x)*(Loc.x - oROICenterPixel.x);
				float yLocCenter = (Loc.y - oROICenterPixel.y)*(Loc.y - oROICenterPixel.y);
				float fGaussNum = (xLocCenter + yLocCenter) / fGaussConst2;
				float fExp = exp(-(fGaussNum));
				fGaussNum = fGaussConst1*fExp;
				const uchar* uPixelValueBlue = uImageData + stepRow*(Loc.y) + 3 * (Loc.x) + 0;
				const uchar* uPixelValueGreen = uImageData + stepRow*(Loc.y) + 3 * (Loc.x) + 1;
				const uchar* uPixelValueRed = uImageData + stepRow*(Loc.y) + 3 * (Loc.x) + 2;

				nBlueBinValue = ((int)*uPixelValueBlue*nBlueBins) / nRange;
				nGreenBinValue = ((int)*uPixelValueGreen*nGreenBins) / nRange;
				nRedBinValue = ((int)*uPixelValueRed*nRedBins) / nRange;

				oHist.at<float>(nBlueBinValue, nGreenBinValue, nRedBinValue) += fGaussNum;

			}
		}

		//for (int i = 0; i < nBlueBins; i++){
		//	for (int j = 0; j < nGreenBins; j++){
		//		for (int k = 0; k < nRedBins; k++)
		//		{

		//			cout << oHist.at<float>(i, j, k) << endl;
		//		}
		//	}
		//}

	}
}



//void ObjectKeyPoint::calLBSPHist(vector<ushort>& gradientValue, cv::Mat& oDescriptor, cv::Mat& oLBSPHist, int& nKeys)
//{
//	int nBins = 256; //can reduce (in an interval)
//	int nchannels[] = { 0 }; // index for LBSP channel
//	int nHistSize = { nBins }; //no.of bins
//	int nBinValue;
//	int nGradValue;
//	float fRange[] = { 0, 65536 }; //range of LBSP values as in how many max values can be one
//	const float* fHistRange = { fRange };
//	int nGradientRange = 65536;
//	bool bUniform = true; bool bAccumulate = false;
//	oLBSPHist = Mat::zeros(nBins, 1, CV_16UC1);
//
//	for (auto i = 0; i < gradientValue.size(); i++)
//	{
//
//		cout << gradientValue[i] << endl;
//		nGradValue = (int)gradientValue[i];
//		nBinValue = (nGradValue*nBins) / nGradientRange;
//		oLBSPHist.at<ushort>(nBinValue, 0) += 1;
//
//	}
//
//}






void ObjectKeyPoint::matRead(const char *file, size_t& n_ObjSize)
{
	//open MAT file

	MATFile *pModelMatFile = matOpen(file, "r");

	if (pModelMatFile == NULL)
	{

		return;
	}


	mxArray *arr = matGetVariable(pModelMatFile, "Model");

	if (arr != NULL)
	{

		//copy data of MAT file to a varia
		mwSize num = mxGetNumberOfElements(arr);
		double *pr = mxGetPr(arr);
	}

	// cleanup
	mxDestroyArray(arr);
	matClose(pModelMatFile);
}


void ObjectKeyPoint::updateFaceModel(objectDetected oBestBox, Mat& oFaceLBSPModel, Mat& oFaceColorModel, Mat& oFaceLBSPModelMat, Rect& oFaceLBSPModelRect, Mat oImage)
{
	Point2f pt1, pt2;
	const int nSizes[3] = { 16, 16, 16 };
	
	//Mat oFaceColorModel2Hist = Mat::zeros(3, nSizes, CV_32FC1);
	//Mat oFaceLBSPModel2;
	Mat oImage2 = oImage.clone();
	oFaceLBSPModel = oBestBox.oLBSPDes;
	oFaceColorModel = oBestBox.oColorHist;
	oFaceLBSPModelRect.x = oBestBox.nXPos;
	oFaceLBSPModelRect.y = oBestBox.nYPos;
	oFaceLBSPModelRect.width = oBestBox.nWidth;
	oFaceLBSPModelRect.height = oBestBox.nHeight;
	ROIBoxAdjust(oImage2, oFaceLBSPModelRect);



	oFaceLBSPModelMat = oImage2(cv::Rect(oFaceLBSPModelRect.x, oFaceLBSPModelRect.y, oFaceLBSPModelRect.width, oFaceLBSPModelRect.height));
	
	//computeLBSPDes(oFaceLBSPModelMat, oFaceLBSPModel2);

	//calWeightedColor(oFaceLBSPModelMat, oFaceColorModel2, oFaceLBSPModelRect);


	//updatePartialTemplate(oFaceColorModel2Hist, oFaceLBSPModel2, oFaceColorModel, oFaceLBSPModel, oFaceLBSPModelMat);
}

void ObjectKeyPoint::updatePartialFaceModel(objectDetected oBestBox, Mat& oFaceLBSPModel, Mat& oFaceColorModel, Mat& oFaceLBSPModelMat, Rect& oFaceLBSPModelRect, Mat oImage)
{
	Point2f pt1, pt2;
	const int nSizes[3] = { 16, 16, 16 };

	Mat oFaceColorModel2Hist = Mat::zeros(3, nSizes, CV_32FC1);
	Mat oFaceLBSPModel2;
	Mat oImage2 = oImage.clone();
	oFaceLBSPModel2 = oBestBox.oLBSPDes;
	oFaceColorModel2Hist = oBestBox.oColorHist;
	oFaceLBSPModelRect.x = oBestBox.nXPos;
	oFaceLBSPModelRect.y = oBestBox.nYPos;
	oFaceLBSPModelRect.width = oBestBox.nWidth;
	oFaceLBSPModelRect.height = oBestBox.nHeight;
	ROIBoxAdjust(oImage2, oFaceLBSPModelRect);



	oFaceLBSPModelMat = oImage2(cv::Rect(oFaceLBSPModelRect.x, oFaceLBSPModelRect.y, oFaceLBSPModelRect.width, oFaceLBSPModelRect.height));

	//computeLBSPDes(oFaceLBSPModelMat, oFaceLBSPModel2);

	//calWeightedColor(oFaceLBSPModelMat, oFaceColorModel2, oFaceLBSPModelRect);

	
	updatePartialTemplate(oFaceColorModel2Hist, oFaceLBSPModel2, oFaceColorModel, oFaceLBSPModel, oFaceLBSPModelMat);


}



void ObjectKeyPoint::updatePartialTemplate(cv::Mat& oFaceColorModel2Hist, cv::Mat& oFaceLBSPModel2, cv::Mat& oFaceColorModel, cv::Mat& oFaceLBSPModel,cv::Mat& oFaceLBSPModelMat)
{

	
	const int nSizes2[3] = { 2, 2, 2 };
	//const int nSizes2[3] = { 6, 6, 6 };
	for (int i = 0; i<nSizes2[0]; i++) {
		for (int j = 0; j<nSizes2[1]; j++) {
			for (int k = 0; k<nSizes2[2]; k++) {

				oFaceColorModel.at<float>(i, j, k) = oFaceColorModel2Hist.at<float>(i, j, k);
				//cout << "Value(" << i << ", " << j << ", " << k << "): " << oFaceColorModel2Hist.at<double>(i, j, k) << "\n";
			}
		}
	}


	//oModel1.create(oModel2.rows, 1, CV_16UC1);

	for (auto i = 0; i < (0.1*oFaceLBSPModel2.rows); i++)
	{
		oFaceLBSPModel2.row(i).copyTo(oFaceLBSPModel.row(i));
		//cout << oDescriptorModel.at<ushort>(i, 0) << endl;
	}

	int test = 1;

}








void ObjectKeyPoint::computeFaceObjectsROI(vector<objectDetected>& voNPDBox, Mat oLBSPROIModelMat, Mat oImage2, vector<objectKeys>& voKeyPoints, objectDetected& oBestBox)
{
	const int nSizes[3] = { 16, 16, 16 };
	vector<objectDetected> voNPDBox2;
	Mat testImage = oImage2.clone();
	for (auto i = 0; i < voNPDBox.size(); ++i)
	{
		cv::Rect oFaceROIRect; cv::Mat oGrey, oSampleROIMat, oDescriptorLBSPFaceROI; vector<int> voOutput;
		oFaceROIRect.x = voNPDBox[i].nXPos;
		oFaceROIRect.y = voNPDBox[i].nYPos;
		oFaceROIRect.width = voNPDBox[i].nWidth;

		oFaceROIRect.height = voNPDBox[i].nHeight;
		
		voNPDBox[i].objectCenter.x = voNPDBox[i].nXPos + voNPDBox[i].nWidth / 2.0;
		voNPDBox[i].objectCenter.y = voNPDBox[i].nYPos + voNPDBox[i].nHeight / 2.0;

		

		/*if (diffWidth < 5.0 || diffHeight < 5.0)
		{
			cout << "Change" << endl;
			voNPDBox[i].nWidth = oBestBox.nWidth;
			voNPDBox[i].nHeight = oBestBox.nHeight;


		}*/
		nonModelKeysinFaceROI(voNPDBox[i], voKeyPoints); // How many non model keypoints that are matched lie in the proposal box
			if (oFaceROIRect.x >= 0 && oFaceROIRect.x + oFaceROIRect.width < oImage2.cols && oFaceROIRect.y >= 0 && oFaceROIRect.y + oFaceROIRect.height < oImage2.rows)
			{



/*
					rectangle(testImage, oFaceROIRect, cv::Scalar(0, 0, 255), 1, 8, 0);


					imshow("FD", testImage);
					cvWaitKey(10);*/
					

				oSampleROIMat = oImage2(oFaceROIRect);



				checkLBSPsizewithModel(oSampleROIMat, oLBSPROIModelMat);


				cv::cvtColor(oSampleROIMat, oGrey, CV_RGB2GRAY);

				ushort uLBSP2;  vector<ushort> voLBSP2;
				for (auto j = 2; j < (oGrey.rows - 2); j++)
				{

					for (auto i = 2; i < (oGrey.cols - 2); i++)

					{

						LBSP::computeGrayscaleDescriptor(oGrey, oGrey.at<uchar>(j, i), i, j, THRESH, uLBSP2);

						voLBSP2.push_back(uLBSP2);

					}


				}

				voNPDBox[i].oColorHist = Mat::zeros(3, nSizes, CV_32FC1);

				voNPDBox[i].oLBSPDes.create(voLBSP2.size(), 1, CV_16UC1);

				oDescriptorLBSPFaceROI.create(voLBSP2.size(), 1, CV_16UC1);

				for (auto i = voLBSP2.begin(); i != voLBSP2.end(); i++)
				{



					oDescriptorLBSPFaceROI.at<ushort>(std::distance(voLBSP2.begin(), i), 0) = *i;

				}

				voNPDBox[i].oLBSPDes = oDescriptorLBSPFaceROI.clone();

				voNPDBox2.push_back(voNPDBox[i]);
			}

		
		

	}
	voNPDBox.clear();
	voNPDBox = voNPDBox2;

}



void ObjectKeyPoint::matchFaceObjects(vector<objectDetected>& voNPDBox, cv::Mat oDescriptorLBSPModel, cv::Mat oColor1, cv::Point2f& predCenter, objectDetected& oBestBox )
{


	vector<int> voOutput; Rect roi;

	vector<objectDetected> voNPDBox2;

	for (auto i = 0; i < voNPDBox.size(); ++i)
	{

		matchLBSPAM(oDescriptorLBSPModel, voNPDBox[i].oLBSPDes, voOutput, voNPDBox[i].fLBSPScore);

		matchColorAM(oColor1, voNPDBox[i].oColorHist, voNPDBox[i].fColorScore);

		Point2f diff;
		diff.x = abs(voNPDBox[i].objectCenter.x - predCenter.x);
		diff.y = abs(voNPDBox[i].objectCenter.y - predCenter.y);
		float diffWidth, diffHeight;


		diffWidth = abs(voNPDBox[i].nWidth - oBestBox.nWidth);
		diffHeight = abs(voNPDBox[i].nHeight - oBestBox.nHeight);
	//	if (diff.x < 35.0 && diff.y < 35.0 && diffWidth < 21.2 && diffHeight < 21.2)
		//	if (diff.x < 35.0 && diff.y < 35.0)
				if (diff.x < 40.0 && diff.y < 40.0)
		{
			//cout << "W:"<< diff.x << "H:" << diff.y << endl; 
			voNPDBox2.push_back(voNPDBox[i]);
			
		}

	}
	
	voNPDBox.clear();
	voNPDBox = voNPDBox2;
}





void ObjectKeyPoint::getMinMaxWeightKeyPoint(vector<objectKeys>& keyPoint1, vector<objectKeys>& keyPoint2, vector<std::pair <float, int>>& voKeyWeightValues)
{
	

	for (auto i = 0; i < keyPoint1.size(); ++i)

	{
		if (keyPoint1[i].indi == 1)
		{
			std::pair <float, int> keyWeightValues(keyPoint1[i].weight1, i);

			voKeyWeightValues.push_back(keyWeightValues);
		}

	}

	std::sort(voKeyWeightValues.begin(), voKeyWeightValues.end(), compare_first_only());
}


void ObjectKeyPoint::detectScaleChange(std::vector<Point2f>& voDist1, std::vector<Point2f>& voDist2, float& fScaleChange)
{
	float fRatio;
	std::vector<float> voDisValues;



	for (auto i = 0; i < voDist1.size(); ++i)
	{

		float fDen = sqrt(voDist2[i].x*voDist2[i].x + voDist2[i].y*voDist2[i].y);

		if (fDen!= 0)
		{
			fRatio = sqrt(voDist1[i].x*voDist1[i].x + voDist1[i].y*voDist1[i].y) / fDen;
			voDisValues.push_back(fRatio);
		}
		else
		{
			voDisValues.push_back(fDen);
		}
	}



	if (voDisValues.size() > 1)
	{

		double sum = std::accumulate(voDisValues.begin(), voDisValues.end(), 0.0);
		fScaleChange = sum / voDisValues.size();
	}

	else
	{
		fScaleChange = 1.0;
	}


}


void ObjectKeyPoint::computePairDistance(std::vector<objectKeys>& keyPoint1, std::vector<objectKeys>& keyPoint2, vector<std::pair <float, int>>& voKeyWeightValues, std::vector<Point2f>& voDist1, std::vector<Point2f>& voDist2)
{
	Point2f pairedDist1, pairedDistNew1;

	

	size_t nLargestIndex = voKeyWeightValues.size() - 1;

	size_t nKeyIndex = voKeyWeightValues[nLargestIndex].second;


	for (auto k = 0; k < voKeyWeightValues.size() - 1; ++k)

		{
			/*if (k == (nLargestIndex+1))
			{
				break;
			}*/

			
				pairedDist1 = keyPoint1[nKeyIndex].key.pt - keyPoint1[voKeyWeightValues[k].second].key.pt;

				voDist1.push_back(pairedDist1);

			
				pairedDistNew1 = keyPoint2[keyPoint1[nKeyIndex].index].key.pt - keyPoint2[keyPoint1[voKeyWeightValues[k].second].index].key.pt;

				voDist2.push_back(pairedDistNew1);

			
		
		}

}