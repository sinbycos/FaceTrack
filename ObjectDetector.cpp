/* Copyright (c) 2014, Tanushri Chakravorty (tanushri.chakravorty@polymtl.ca)
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
*  * Redistributions of source code must retain the above copyright notice,
*    this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of Ecole Polytechnique de Montreal nor the names of its
*    contributors may be used to endorse or promote products derived from
*    this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*/
#define DRAW 1

#define THRESH 10
#define  BUFSIZE 256

#include"ObjectDetector.h"
#include "Sampler.h"
#include <math.h>
#include <ctime>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <map>
#include <omp.h>


#define max(a, b)  (((a) > (b)) ? (a) : (b))
#define min(a, b)  (((a) < (b)) ? (a) : (b))

using namespace cv;

int pWinSize[] = { 24, 29, 35, 41, 50, 60, 72, 86, 103, 124, 149, 178, 214, 257, 308, 370, 444, 532, 639, 767, 920, 1104, 1325, 1590, 1908, 2290, 2747, 3297, 3956 };


ObjectDetector::ObjectDetector()
{

		const Options& opt = Options::GetInstance();
		stages = 0;

		ppNpdTable = Mat(256, 256, CV_8UC1);
		for (int i = 0; i < 256; i++)
		{
			for (int j = 0; j < 256; j++)
			{
				double fea = 0.5;
				if (i > 0 || j > 0) fea = double(i) / (double(i) + double(j));
				fea = floor(256 * fea);
				if (fea > 255) fea = 255;

				ppNpdTable.at<uchar>(i, j) = (unsigned char)fea;
			}
		}

		size_t numPixels = opt.objSize*opt.objSize;
		for (int i = 0; i < numPixels; i++)
		{
			for (int j = i + 1; j < numPixels; j++)
			{
				lpoints.push_back(i);
				rpoints.push_back(j);
			}
		}

		points1.resize(29);
		points2.resize(29);

	}




ObjectDetector::~ObjectDetector()
{
}

void ObjectDetector::GetPoints(int feaid, int *x1, int *y1, int *x2, int *y2){
	const Options& opt = Options::GetInstance();
	int lpoint = lpoints[feaid];
	int rpoint = rpoints[feaid];
	*y1 = lpoint / opt.objSize;
	*x1 = lpoint%opt.objSize;
	*y2 = rpoint / opt.objSize;
	*x2 = rpoint%opt.objSize;

}


void ObjectDetector::LoadModel(string path){
	FILE* file;
	if ((file = fopen(path.c_str(), "rb")) == NULL)
		return;

	fread(&DetectSize, sizeof(int), 1, file);
	fread(&stages, sizeof(int), 1, file);
	fread(&numBranchNodes, sizeof(int), 1, file);
	//printf("stages num :%d\n",stages);

	int *_tree = new int[stages];
	float *_threshold = new float[stages];
	fread(_tree, sizeof(int), stages, file);
	fread(_threshold, sizeof(float), stages, file);
	for (int i = 0; i<stages; i++){
		treeIndex.push_back(_tree[i]);
		thresholds.push_back(_threshold[i]);
	}
	delete[]_tree;
	delete[]_threshold;

	int *_feaId = new int[numBranchNodes];
	int *_leftChild = new int[numBranchNodes];
	int *_rightChild = new int[numBranchNodes];
	unsigned char* _cutpoint = new unsigned char[2 * numBranchNodes];
	fread(_feaId, sizeof(int), numBranchNodes, file);
	fread(_leftChild, sizeof(int), numBranchNodes, file);
	fread(_rightChild, sizeof(int), numBranchNodes, file);
	fread(_cutpoint, sizeof(unsigned char), 2 * numBranchNodes, file);
	for (int i = 0; i<numBranchNodes; i++){
		feaIds.push_back(_feaId[i]);
		leftChilds.push_back(_leftChild[i]);
		rightChilds.push_back(_rightChild[i]);
		cutpoints.push_back(_cutpoint[2 * i]);
		cutpoints.push_back(_cutpoint[2 * i + 1]);
		for (int j = 0; j<29; j++){
			int x1, y1, x2, y2;
			GetPoints(_feaId[i], &x1, &y1, &x2, &y2);
			float factor = (float)pWinSize[j] / (float)DetectSize;
			int p1x = x1*factor;
			int p1y = y1*factor;
			int p2x = x2*factor;
			int p2y = y2*factor;
			points1[j].push_back(p1y*pWinSize[j] + p1x);
			points2[j].push_back(p2y*pWinSize[j] + p2x);
		}
	}
	delete[]_feaId;
	delete[]_leftChild;
	delete[]_rightChild;
	delete[]_cutpoint;

	int numLeafNodes = numBranchNodes + stages;
	float *_fit = new float[numLeafNodes];
	fread(_fit, sizeof(float), numLeafNodes, file);
	for (int i = 0; i<numLeafNodes; i++){
		fits.push_back(_fit[i]);
	}
	delete[]_fit;

	fclose(file);
}

vector<int> ObjectDetector::detectFace(cv::Mat img,vector<cv::Rect>&rects, vector<float>& scores){

	const Options& opt = Options::GetInstance();
	LoadModel(opt.outFile);

	int minFace = 24;
	int maxFace = img.cols;
	int sizePatch = img.cols;

	omp_set_dynamic(1);
	
	int height = img.rows;
	int width = img.cols;
	int thresh = 0;
	const unsigned char *O = (unsigned char *)img.data;
	unsigned char *I = new unsigned char[width*height];
	int k = 0;
	for (int i = 0; i < width; i++){
		for (int j = 0; j < height; j++){
			I[k] = *(O + j*width + i);
			k++;
		}
	}

	for (auto i = 0; i < 29; i++)
	{
		if (sizePatch >= pWinSize[i])
		{
			thresh = i;
			continue;
		}
		else
		{
			break;
		}
	}

	thresh = thresh + 1;


	minFace = max(minFace, opt.objSize);
	maxFace = min(maxFace, min(height, width));

	vector<int> picked;
	if (min(height, width) < minFace)
	{
		return picked;
	}


	for (int k = 0; k < thresh; k++) // process each scale
	{


		if (pWinSize[k] < minFace) continue;
		else if (pWinSize[k] > maxFace) break;

		// determine the step of the sliding subwindow
		int winStep = (int)floor(pWinSize[k] * 0.1);

		if (pWinSize[k] > 40)

			winStep = (int)floor(pWinSize[k] * 0.05);

			//winStep = (int)floor(pWinSize[k] * 0.15);
		// calculate the offset values of each pixel in a subwindow
		// pre-determined offset of pixels in a subwindow
		vector<int> offset(pWinSize[k] * pWinSize[k]);
		int pp1 = 0, pp2 = 0, gap = height - pWinSize[k];

		for (int j = 0; j < pWinSize[k]; j++) // column coordinate
		{
			for (int i = 0; i < pWinSize[k]; i++) // row coordinate
			{
				offset[pp1++] = pp2++;
			}

			pp2 += gap;

		}


		int colMax = width - pWinSize[k] + 1;
		int rowMax = height - pWinSize[k] + 1;

		// process each subwindow
#pragma omp parallel for
		for (int c = 0; c < colMax; c += winStep) // slide in column
		{
			const unsigned char *pPixel = I + c * height;

			for (int r = 0; r < rowMax; r += winStep, pPixel += winStep) // slide in row
			{
				float _score = 0;
				int s;


				// test each tree classifier
				for (s = 0; s < stages; s++)
				{
					//	printf("stages val:%d\n", s);
					int node = treeIndex[s];

					// test the current tree classifier
					while (node > -1) // branch node
					{
						unsigned char p1 = pPixel[offset[points1[k][node]]];
						unsigned char p2 = pPixel[offset[points2[k][node]]];
						unsigned char fea = ppNpdTable.at<uchar>(p1, p2);

						if (fea < cutpoints[2 * node] || fea > cutpoints[2 * node + 1]) node = leftChilds[node];
						else node = rightChilds[node];

					}

					node = -node - 1;
					_score = _score + fits[node];

					if (_score < thresholds[s]){
						break; // negative samples
					}
				}

				if (s == stages) // a face detected
				{

					Rect roi(c, r, pWinSize[k], pWinSize[k]);


#pragma omp critical // modify the record by a single thread
					{
						rects.push_back(roi);
						scores.push_back(_score);
					}
				}
			}

		}

	}

	if (rects.size() > 0)
	{
		vector<int> Srect;
		picked = Nms(rects, scores, Srect, 0.5, img);

		int imgWidth = img.cols;
		int imgHeight = img.rows;


		//you should set the parameter by yourself
		for (int i = 0; i < picked.size(); i++){
			int idx = picked[i];
			int delta = floor(Srect[idx] * opt.enDelta);
			int y0 = max(rects[idx].y - floor(3.0 * delta), 0);
			int y1 = min(rects[idx].y + Srect[idx], imgHeight);
			int x0 = max(rects[idx].x + floor(0.25 * delta), 0);
			int x1 = min(rects[idx].x + Srect[idx] - floor(0.25 * delta), imgWidth);

			rects[idx].y = y0;
			rects[idx].x = x0;
			rects[idx].width = x1 - x0 + 1;
			rects[idx].height = y1 - y0 + 1;
		}


		delete[]I;
		return picked;
	}
	else
	{
		return{};
	}
}


void ObjectDetector::Detect(vector<std::string>::iterator iter1, vector<objectDetected>& FBoxes){
	Options& opt = Options::GetInstance();


	LoadModel(opt.outFile);

	string path = *iter1;

	Mat img = imread(path);

	// Mat img2 = img.clone();
	cvtColor(img, img, CV_BGR2GRAY);
	vector<Rect> rects;
	vector<float> scores;
	vector<int> index;
	objectDetected fBox;
	index = detectFace(img, rects, scores);

	

	/*rectangle(img2, rects[0], cv::Scalar(0, 255, 0), 1, 8, 0);
	rectangle(img2, rects2[0], cv::Scalar(255, 0, 0), 1, 8, 0);
	imshow("OUT", img2);
	cvWaitKey(10);
	*/
	if (index.size() > 0)
	{
		for (int i = 0; i < index.size(); i++){

			fBox.nXPos = rects[index[i]].x;
			fBox.nYPos = rects[index[i]].y;
			fBox.nWidth = rects[index[i]].width;
			fBox.nHeight = rects[index[i]].height;
			fBox.fDetScore = scores[index[i]];
			fBox.objectCenter = Point2f((fBox.nXPos + fBox.nWidth / 2.0), (fBox.nYPos + fBox.nHeight / 2.0));
			FBoxes.push_back(fBox);


		}
	}

}


bool operator<(const objectDetected &ob1, const objectDetected &ob2)
{
	return ob1.fDetScore > ob2.fDetScore;
}

void ObjectDetector::execMatlab(Rect trackROI, objectDetected oFaceBox, int frameNumber, vector<std::string>::iterator iter1, vector<objectDetected>& propBoxes, vector<objectDetected>& propBoxesT, Engine* mEngine)
{
	Sampler sFind;
	objectDetected oMatBox, oMatBoxT;



	const int arraySize1 = 4;
	const int arraySize2 = 5;
	const int arraySize3 = 1;
	double trackROIBox[arraySize1];
	double box[arraySize2];

	string path = *iter1;
	cv::Mat oImage = imread(*iter1);
	
	char myPath[1024];
	strcpy(myPath, path.c_str());

	double DROIBox[arraySize2];
	double TrackX, TrackY, TrackW, TrackH, DetX, DetY, DetW, DetH, Response;
	size_t size, sizeT;


	mxArray *TRACKROI = NULL, *DROI = NULL, *FRAME = NULL, *MYPATH = NULL;
	mxArray *Tx = NULL, *Ty = NULL, *Tw = NULL, *Th = NULL, *Dx = NULL, *Dy = NULL, *Dw = NULL, *Dh = NULL, *frame = NULL, *res = NULL, *resT = NULL, *total = NULL, *totalT = NULL;
	char buffer[BUFSIZE + 1];
	buffer[BUFSIZE] = '\0';
	engOutputBuffer(mEngine, buffer, BUFSIZE);



	trackROIBox[0] = trackROI.x;
	trackROIBox[1] = trackROI.y;
	trackROIBox[2] = trackROI.width;
	trackROIBox[3] = trackROI.height;



	DROIBox[0] = oFaceBox.nXPos;
	DROIBox[1] = oFaceBox.nYPos;
	DROIBox[2] = oFaceBox.nWidth;
	DROIBox[3] = oFaceBox.nHeight;
	DROIBox[4] = oFaceBox.fDetScore;



	if (trackROIBox[0] == 0 && trackROIBox[1] == 0 && trackROIBox[2] == 0 && trackROIBox[3] == 0 && DROIBox[0] == 0 &&
		DROIBox[1] == 0 && DROIBox[2] == 0 && DROIBox[3] == 0 && DROIBox[4] == 0)
	{

		exit;

	}




	if (trackROIBox[0] > 0 && trackROIBox[1] > 0 && trackROIBox[2] > 0 && trackROIBox[3] > 0)
	{
		TRACKROI = mxCreateDoubleMatrix(1, arraySize1, mxREAL);

		memcpy((void *)mxGetPr(TRACKROI), (void *)trackROIBox, sizeof(double)*arraySize1);

		engPutVariable(mEngine, "TrackROI", TRACKROI);
	}

	if (DROIBox[0] > 0 && DROIBox[1] > 0 && DROIBox[2] > 0 && DROIBox[3] > 0 && DROIBox[4] > 0)
	{

		DROI = mxCreateDoubleMatrix(1, arraySize2, mxREAL);

		memcpy((void *)mxGetPr(DROI), (void *)DROIBox, sizeof(double)*arraySize2);

		engPutVariable(mEngine, "DROI", DROI);
	}



	MYPATH = mxCreateString(myPath);

	engPutVariable(mEngine, "MYPATH", MYPATH);


	engEvalString(mEngine, "cd \ 'C:\\TANUSHRI\\MAT\\Final\\NPDFaceDetector\\' ");

	engEvalString(mEngine, "test");

	 Tx = engGetVariable(mEngine, "Tx");

	 Ty = engGetVariable(mEngine, "Ty");

	 Tw = engGetVariable(mEngine, "Tw");

	 Th = engGetVariable(mEngine, "Th");
	 TrackX = mxGetScalar(Tx);
	 TrackY = mxGetScalar(Ty);
	 TrackW = mxGetScalar(Tw);
	 TrackH = mxGetScalar(Th);

	 Dx = engGetVariable(mEngine, "Dx");

	 Dy = engGetVariable(mEngine, "Dy");

	 Dw = engGetVariable(mEngine, "Dw");

	 Dh = engGetVariable(mEngine, "Dh");

	 DetX = mxGetScalar(Dx);
	 DetY = mxGetScalar(Dy);
	 DetW = mxGetScalar(Dw);
	 DetH = mxGetScalar(Dh);


	resT = engGetVariable(mEngine, "resT");




	res = engGetVariable(mEngine, "res");

	assert(mxGetNumberOfElements != NULL);


	int numOfElements = mxGetNumberOfElements(res);

	assert(mxGetNumberOfElements != NULL);


	int numOET = mxGetNumberOfElements(resT);


	double *outPropsT = (double*)mxGetData(resT);
	double *outProps = (double*)mxGetData(res);




	int x = numOfElements / 5;
	//vector <int *> outData(num);
	//outData[0] = (int *) mxGetData(res);


	//sFind.analyzecellarray(RectBox, propBoxes);

	cv::Rect oRect1, oRect2;


	for (auto j = 0; j < x; ++j)
	{


		//	cout << outProps[j] << "\t" << outProps[j + x] << "\t" << outProps[j + 2 * x] << "\t" << outProps[j + 3 * x] << "\t" << outProps[j + 4 * x] << endl;
		if (outProps[j] == 0 && outProps[j + x] == 0 && outProps[j + 2 * x] == 0 && outProps[j + 3 * x] == 0 && outProps[j + 4 * x] == 0)
		{
			continue;
		}
		else
		{

			oMatBox.nXPos = outProps[j];

			oMatBox.nYPos = outProps[j + x];
			oMatBox.nWidth = outProps[j + 2 * x];
			oMatBox.nHeight = outProps[j + 3 * x];
			oMatBox.fDetScore = outProps[j + 4 * x];

			if (oMatBox.nXPos >= 0 && oMatBox.nXPos + oMatBox.nWidth <= oImage.cols &&  oMatBox.nYPos >= 0 && oMatBox.nYPos + oMatBox.nHeight <= oImage.rows)
			{

				propBoxes.push_back(oMatBox);
			}

			cv::Rect oRect1;

			for (auto i = 0; i < propBoxes.size(); ++i)
			{

				oRect1.x = oMatBox.nXPos;
				oRect1.y = oMatBox.nYPos;
				oRect1.width = oMatBox.nWidth;
				oRect1.height = oMatBox.nHeight;

				/*rectangle(oImage, oRect1, cv::Scalar(0, 255, 0), 1, 8, 0);

				imshow("fBox", oImage);
				cvWaitKey(10);*/
			}
		}
	}



	for (auto j = 0; j < x; ++j)

	{

		//	cout << outPropsT[j] << "\t" << outPropsT[j + x] << "\t" << outPropsT[j + 2 * x] << "\t" << outPropsT[j + 3 * x] << "\t" << outPropsT[j + 4 * x] << endl;
		if (outPropsT[j] == 0 && outPropsT[j + x] == 0 && outPropsT[j + 2 * x] == 0 && outPropsT[j + 3 * x] == 0 && outPropsT[j + 4 * x] == 0)
		{
			continue;
		}
		else
		{


			oMatBoxT.nXPos = outPropsT[j];
			oMatBoxT.nYPos = outPropsT[j + x];
			oMatBoxT.nWidth = outPropsT[j + 2 * x];
			oMatBoxT.nHeight = outPropsT[j + 3 * x];
			oMatBoxT.fDetScore = outPropsT[j + 4 * x];

			if (oMatBoxT.nXPos >= 0 && oMatBoxT.nXPos + oMatBoxT.nWidth <= oImage.cols &&  oMatBoxT.nYPos >= 0 && oMatBoxT.nYPos + oMatBoxT.nHeight <= oImage.rows)
			{
				propBoxesT.push_back(oMatBoxT);

			}

			cv::Rect oRect1;

			for (auto i = 0; i < propBoxesT.size(); ++i)
			{
				oRect1.x = oMatBoxT.nXPos;
				oRect1.y = oMatBoxT.nYPos;
				oRect1.width = oMatBoxT.nWidth;
				oRect1.height = oMatBoxT.nHeight;

				/*rectangle(oImage, oRect1, cv::Scalar(0, 0, 255), 1, 8, 0);

				imshow("fBoxT", oImage);
				cvWaitKey(10);*/
			}
		}
	}

	
	mxDestroyArray(TRACKROI);
	mxDestroyArray(Dx);
	mxDestroyArray(Dy);
	mxDestroyArray(Dw);
	mxDestroyArray(Dh);
	mxDestroyArray(frame);
	mxDestroyArray(MYPATH);
	mxDestroyArray(res);
	mxDestroyArray(DROI);
	mxDestroyArray(Tx);
	mxDestroyArray(Ty);
	mxDestroyArray(Tw);
	mxDestroyArray(Th);


}
	

vector<int> ObjectDetector::Nms(vector<Rect>& rects, vector<float>& scores, vector<int>& Srect, float overlap, Mat Img) {

	int numCandidates = rects.size();
	Mat_<uchar> predicate = Mat_<uchar>::eye(numCandidates, numCandidates);
	for (int i = 0; i < numCandidates; i++){
		for (int j = i + 1; j < numCandidates; j++){
			int h = min(rects[i].y + rects[i].height, rects[j].y + rects[j].height) - max(rects[i].y, rects[j].y);
			int w = min(rects[i].x + rects[i].width, rects[j].x + rects[j].width) - max(rects[i].x, rects[j].x);
			int s = max(h, 0)*max(w, 0);

			if ((float)s / (float)(rects[i].width*rects[i].height + rects[j].width*rects[j].height - s) >= overlap){
				predicate(i, j) = 1;
				predicate(j, i) = 1;
			}
		}
	}

	vector<int> label;

	int numLabels = Partation(predicate, label);

	vector<Rect> Rects;
	Srect.resize(numLabels);
	vector<int> neighbors;
	neighbors.resize(numLabels);
	vector<float> Score;
	Score.resize(numLabels);

	for (int i = 0; i < numLabels; i++){
		vector<int> index;
		for (int j = 0; j < numCandidates; j++){
			if (label[j] == i)
				index.push_back(j);
		}
		vector<float> weight;
		weight = Logistic(scores, index);
		float sumScore = 0;
		for (int j = 0; j < weight.size(); j++)
			sumScore += weight[j];
		Score[i] = sumScore;
		neighbors[i] = index.size();

		if (sumScore == 0){
			for (int j = 0; j < weight.size(); j++)
				weight[j] = 1 / sumScore;
		}
		else{
			for (int j = 0; j < weight.size(); j++)
				weight[j] = weight[j] / sumScore;
		}

		float size = 0;
		float col = 0;
		float row = 0;
		for (int j = 0; j < index.size(); j++){
			size += rects[index[j]].width*weight[j];
		}
		Srect[i] = (int)floor(size);
		for (int j = 0; j < index.size(); j++){
			col += (rects[index[j]].x + rects[index[j]].width / 2)*weight[j];
			row += (rects[index[j]].y + rects[index[j]].width / 2)*weight[j];
		}
		int x = floor(col - size / 2);
		int y = floor(row - size / 2);
		Rect roi(x, y, Srect[i], Srect[i]);
		Rects.push_back(roi);
	}


	predicate = Mat_<uchar>::zeros(numLabels, numLabels);

	for (int i = 0; i < numLabels; i++){
		for (int j = i + 1; j < numLabels; j++){
			int h = min(Rects[i].y + Rects[i].height, Rects[j].y + Rects[j].height) - max(Rects[i].y, Rects[j].y);
			int w = min(Rects[i].x + Rects[i].width, Rects[j].x + Rects[j].width) - max(Rects[i].x, Rects[j].x);
			int s = max(h, 0)*max(w, 0);

			if ((float)s / (float)(Rects[i].width*Rects[i].height) >= overlap || (float)s / (float)(Rects[j].width*Rects[j].height) >= overlap)
			{
				predicate(i, j) = 1;
				predicate(j, i) = 1;
			}
		}
	}

	vector<int> flag;
	flag.resize(numLabels);
	for (int i = 0; i < numLabels; i++)
		flag[i] = 1;

	for (int i = 0; i < numLabels; i++){
		vector<int> index;
		for (int j = 0; j < numLabels; j++){
			if (predicate(j, i) == 1)
				index.push_back(j);
		}
		if (index.size() == 0)
			continue;

		float s = 0;
		for (int j = 0; j<index.size(); j++){
			if (Score[index[j]]>s)
				s = Score[index[j]];
		}
		if (s > Score[i])
			flag[i] = 0;
	}

	vector<int> picked;
	for (int i = 0; i < numLabels; i++){
		if (flag[i]){
			picked.push_back(i);
		}
	}

	int height = Img.rows;
	int width = Img.cols;

	for (int i = 0; i < picked.size(); i++){
		int idx = picked[i];
		if (Rects[idx].x<0)
			Rects[idx].x = 0;

		if (Rects[idx].y<0)
			Rects[idx].y = 0;

		if (Rects[idx].y + Rects[idx].height>height)
			Rects[idx].height = height - Rects[idx].y;

		if (Rects[idx].x + Rects[idx].width>width)
			Rects[idx].width = width - Rects[idx].x;
	}

	rects = Rects;
	scores = Score;
	return picked;

}

vector<float> ObjectDetector::Logistic(vector<float> scores, vector<int> index){
	vector<float> Y;
	for (int i = 0; i<index.size(); i++){
		float tmp_Y = log(1 + exp(scores[index[i]]));
		if (isinf(tmp_Y))
			tmp_Y = scores[index[i]];
		Y.push_back(tmp_Y);
	}
	return Y;
}

int ObjectDetector::Partation(Mat_<uchar>& predicate, vector<int>& label){
	int N = predicate.cols;
	vector<int> parent;
	vector<int> rank;
	for (int i = 0; i<N; i++){
		parent.push_back(i);
		rank.push_back(0);
	}

	for (int i = 0; i<N; i++){
		for (int j = 0; j<N; j++){
			if (predicate(i, j) == 0)
				continue;
			int root_i = Find(parent, i);
			int root_j = Find(parent, j);

			if (root_j != root_i){
				if (rank[root_j] < rank[root_i])
					parent[root_j] = root_i;
				else if (rank[root_j] > rank[root_i])
					parent[root_i] = root_j;
				else{
					parent[root_j] = root_i;
					rank[root_i] = rank[root_i] + 1;
				}
			}
		}
	}

	int nGroups = 0;
	label.resize(N);
	for (int i = 0; i<N; i++){
		if (parent[i] == i){
			label[i] = nGroups;
			nGroups++;
		}
		else label[i] = -1;
	}

	for (int i = 0; i<N; i++){
		if (parent[i] == i)
			continue;
		int root_i = Find(parent, i);
		label[i] = label[root_i];
	}

	return nGroups;
}

int ObjectDetector::Find(vector<int>& parent, int x){
	int root = parent[x];
	if (root != x)
		root = Find(parent, root);
	return root;
}
