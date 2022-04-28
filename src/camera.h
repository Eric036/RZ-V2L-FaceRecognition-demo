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
* File Name    : camera.h
* Version      : 5.00
* Description  : RZ/V2L DRP-AI Sample Application for MMPose HRNet MIPI Camera version
***********************************************************************************************************************/

#ifndef CAMERA_H
#define CAMERA_H

#include <linux/renesas-v4l2-controls.h>    //[ISP] ADD
#include <linux/videodev2.h>
#include "define.h"

class Camera
{
    public:
        Camera();
        ~Camera();

        int8_t start_camera();
        int8_t capture_qbuf();
        unsigned long capture_image();
        int8_t close_camera();
        int8_t save_bin(std::string filename);

        unsigned char* get_img();
        int get_size();
        int get_w();
        void set_w(int w);
        int get_h();
        void set_h(int h);
        int get_c();
        void set_c(int c);

    private:
        std::string device;
        int camera_width;
        int camera_height;
        int camera_color;
        int m_fd;
        int imageLength;
        unsigned char *buffer[CAP_BUF_NUM];
        int udmabuf_file;

        struct v4l2_buffer buf_capture;

        int xioctl(int fd, int request, void *arg);
        int8_t start_capture();
        int8_t stop_capture();
        int8_t open_camera_device();
        int8_t init_camera_fmt();
        int8_t init_buffer();
};

#endif
