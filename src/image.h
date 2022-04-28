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
* File Name    : image.h
* Version      : 0.90
* Description  : RZ/V2L DRP-AI Sample Application for PyTorch ResNet USB Camera version
***********************************************************************************************************************/

#ifndef IMAGE_H
#define IMAGE_H

#include "define.h"
//#include "detect.h"

class Image
{
    public:
        Image();
        ~Image();

        unsigned char* img_buffer[WL_BUF_NUM];
        unsigned char get_buf_id();
        void write_string_rgb(std::string str, uint32_t x, uint32_t y, uint32_t h, uint32_t w, float size, int color);

        uint8_t udmabuf_fd;
        unsigned char buf_id = 0;
        uint32_t get_H();
        uint32_t get_W();
        uint32_t get_C();
        unsigned char at(int a);
        void set(int a, unsigned char val);

        uint8_t init(uint32_t w, uint32_t h, uint32_t c, uint32_t ow, uint32_t oh, uint32_t oc, uint32_t off);

        void convert_format();
        void convert_size();
        void convert_channels();
        void camera_to_image(const unsigned char* buffer, int size);
    private:


        /* Input Image (YUYV from V4L2) Information */
        uint32_t img_h;
        uint32_t img_w;
        uint32_t img_c;
        /* Output Image (BGRA for Wayland) Information */
        uint32_t out_h;
        uint32_t out_w;
        uint32_t out_c;

};

#endif
