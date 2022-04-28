/***********************************************************************************************************************
* DISCLAIMER
* This software is supplied by Renesas Electronics Corporation and is only intended for use with Renesas products. No
* other uses are authorized. This software is owned by Renesas Electronics Corporation and is protected under all
* applicable laws, including copyright laws.
* THIS SOFTWARE IS PROVIDED "AS IS" AND RENESAS MAKES NO WARRANTIES REGARDING
* THIS SOFTWARE, WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. ALL SUCH WARRANTIES ARE EXPRESSLY DISCLAIMED. TO THE MAXIMUM
* EXTENT PERMITTED NOT PROHIBITED BY LAW, NEITHER RENESAS ELECTRONICS CORPORATION NOR ANY OF ITS AFFILIATED COMPANIES
* SHALL BE LIABLE FOR ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES FOR ANY REASON RELATED TO THIS
* SOFTWARE, EVEN IF RENESAS OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
* Renesas reserves the right, without notice, to make changes to this software and to discontinue the availability of
* this software. By using this software, you agree to the additional terms and conditions found by accessing the
* following link:
* http://www.renesas.com/disclaimer
*
* Copyright (C) 2021 Renesas Electronics Corporation. All rights reserved.
***********************************************************************************************************************/
/***********************************************************************************************************************
* File Name    : mtcnn.h
* Version      : 0.90
* Description  : RZ/V2L DRP-AI Sample Application for PyTorch ResNet USB Camera version
***********************************************************************************************************************/

/*****************************************
* Includes
******************************************/
#ifndef MTCNN_H
#define MTCNN_H


#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
using namespace std;
using namespace cv;

const float pnet_stride = 2;
const float pnet_cell_size = 12;
const int pnet_max_detect_num = 5000;

const float mean_val = 127.5f;
const float std_val = 0.0078125f;
//minibatch size
const int step_size = 128;


typedef struct FaceBox {
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
} FaceBox;
typedef struct FaceInfo {
    float bbox_reg[4];
    float landmark_reg[10];
    float landmark[10];
    FaceBox bbox;
} FaceInfo;

class MTCNN {
	public:
	    MTCNN(const string& proto_model_dir);
	    vector<FaceInfo> Detect_mtcnn(const cv::Mat& img, const int min_size, const float* threshold, const float factor, const int stage);
		//protected:
	    vector<FaceInfo> ProposalNet(const cv::Mat& img, int min_size, float threshold, float factor);
	    vector<FaceInfo> NextStage(const cv::Mat& image, vector<FaceInfo> &pre_stage_res, int input_w, int input_h, int stage_num, const float threshold);
	    void BBoxRegression(vector<FaceInfo>& bboxes);
	    void BBoxPadSquare(vector<FaceInfo>& bboxes, int width, int height);
	    void BBoxPad(vector<FaceInfo>& bboxes, int width, int height);
	    void GenerateBBox(Mat* confidence, Mat* reg_box, float scale, float thresh);
	    std::vector<FaceInfo> NMS(std::vector<FaceInfo>& bboxes, float thresh, char methodType);
	    float IoU(float xmin, float ymin, float xmax, float ymax, float xmin_, float ymin_, float xmax_, float ymax_, bool is_iom = false);
	public:
	    dnn::Net PNet_;
	    dnn::Net RNet_;
	    dnn::Net ONet_;

	    std::vector<FaceInfo> candidate_boxes_;
	    std::vector<FaceInfo> total_boxes_;
};

#endif