 
#pragma once
#include "boost/lambda/lambda.hpp"
#include "boost/filesystem.hpp"
#include "boost/algorithm/string.hpp"
#include "boost/algorithm/string/find_iterator.hpp"
#include "boost/regex.hpp"
#include <iostream>
#include <iterator>
#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>
#include<math.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "opencv2/features2d/features2d.hpp"

#include <stdio.h>
#include <fstream>
#include <string>
#include <vector>
#include "ObjectKeyPoint.h"
#include "C:\Program Files\MATLAB\R2019b\extern\include\engine.h"
#include "common.hpp"
#include "opencv2/core/core.hpp"

#include <cstdlib>

#pragma comment (lib, "libmat.lib")
#pragma comment (lib, "libmx.lib")

#pragma comment (lib, "libmex.lib")

#pragma comment (lib, "libeng.lib")


using std::vector;

using namespace std;
using namespace cv;
using namespace boost::filesystem;


class ObjectDetector {

public:
	
	//! full constructor
	ObjectDetector();
	
	
	//! default destructor
	virtual ~ObjectDetector();

	//! KeyPoint Object
	ObjectKeyPoint m_oObjectKeypoint;

	//! Get feature map of the face
	void GetPoints(int feaid, int *x1, int *y1, int *x2, int *y2);
	
	//!Load the trained Face model
	void LoadModel(string path);


public:
	/* \breif indicate how many stages the dector have */
	int stages;
	/* \breif vectors contain the model */
	vector<int> treeIndex;
	vector<int> feaIds, leftChilds, rightChilds;
	vector<unsigned char> cutpoints;
	vector<float> fits;
	vector<float> thresholds;
	int numBranchNodes;
	/* \breif save the points of feature id */
	vector< vector<int> > points1, points2;
	/* \breif vector contain point-feature map */
	vector<int> lpoints;
	vector<int> rpoints;
	/* \breif A feature map used for speed up calculate feature */
	cv::Mat ppNpdTable;

	/* \breif model template size */
	int DetectSize;
	
	//! Detects Face in an image
	void Detect(vector<std::string>::iterator iter1, vector<objectDetected>& FBoxes);

	//! Used in after the Detect member function. Returns the detected face and its score in an image
	vector<int> detectFace(cv::Mat img, vector<cv::Rect>&rects, vector<float>& scores);



	/* \breif nms Non - maximum suppression
		* the Nms algorithm result concerned score of areas
		*
		* \param rects     area of faces
		* \param scores    score of faces
		* \param Srect size of rects
		* \param overlap   overlap threshold
		* \param img  get size of origin img
		* \return          picked index*/
		
	
	vector<int> Nms(vector<cv::Rect>& rects, vector<float>& scores, vector<int>& Srect, float overlap, cv::Mat img);
	
	/* \breif function for Partation areas
	* From Predicate mat get a paration result
	*
	* \param predicate  The matrix marked cross areas
	* \param label  The vector marked classification label
	* return number of classfication
	*/
	int Partation(cv::Mat_<uchar>& predicate, vector<int>& label);
	/*
	* \breif Find classfication area parent
	*
	* \param parent  parent vector
	* \param x  current node
	*/
	int Find(vector<int>& parent, int x);
	/*
	* \breif Compute score
	* y = log(1+exp(x));
	*
	* \param scores  score vector
	* \param index  score index
	*/
	vector<float> Logistic(vector<float> scores, vector<int> index);

	
	//! Run EdgeBoxGenerator
	void execMatlab(Rect trackROI, objectDetected oFaceBox, int frameNumber, vector<std::string>::iterator iter1, vector<objectDetected>& propBoxes, vector<objectDetected>& propBoxesT, Engine* mEngine);

};


