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
* File Name    : image.cpp
* Version      : 0.90
* Description  : RZ/V2L DRP-AI Sample Application for PyTorch ResNet USB Camera version
***********************************************************************************************************************/

/*****************************************
* Includes
******************************************/
#include "image.h"
//#include "detect.h"
#include <opencv2/opencv.hpp>



Image::Image()
{

}


Image::~Image()
{
    for (int i = 0; i<WL_BUF_NUM; i++)
    {
        munmap(img_buffer[i], img_w * img_h * img_c);
    }
    close(udmabuf_fd);
}

/*****************************************
* Function Name : get_H
* Description   : Function to get the image height
*                 This function is NOT used currently.
* Arguments     : -
* Return value  : img_h = current image height
******************************************/
uint32_t Image::get_H()
{
    return img_h;
}


/*****************************************
* Function Name : get_W
* Description   : Function to get the image width
*                 This function is NOT used currently.
* Arguments     : -
* Return value  : img_w = current image width
******************************************/
uint32_t Image::get_W()
{
    return img_w;
}


/*****************************************
* Function Name : get_C
* Description   : Function to set the number of image channel
*                 This function is NOT used currently.
* Arguments     : c = new number of image channel to be set
* Return value  : -
******************************************/
uint32_t Image::get_C()
{
    return img_c;
}


/*****************************************
* Function Name : init
* Description   : Function to initialize img_buffer in Image class
*                 This application uses udmabuf in order to
*                 continuous memory area for DRP-AI input data
* Arguments     : w = input image width in YUYV
*                 h = input image height in YUYV
*                 c = input image channel in YUYV
*                 ow = output image width in BGRA to be displayed via Wayland
*                 oh = output image height in BGRA to be displayed via Wayland
*                 oc = output image channel in BGRA to be displayed via Wayland
* Return value  : 0 if succeeded
*                 not 0 otherwise
******************************************/
uint8_t Image::init(uint32_t w, uint32_t h, uint32_t c,
                    uint32_t ow, uint32_t oh, uint32_t oc, uint32_t offset)
{
    /*Initialize input image information */
    img_w = w;
    img_h = h;
    img_c = c;
    /*Initialize output image information*/
    out_w = ow;
    out_h = oh;
    out_c = oc;

    uint32_t out_size = out_w * out_h * out_c;
    udmabuf_fd = open("/dev/udmabuf0", O_RDWR );
    if (udmabuf_fd < 0)
    {
        return -1;
    }

    for (int i = 0; i<WL_BUF_NUM; i++)
    {
        //img_buffer[i] =(unsigned char*) mmap(NULL, out_size ,PROT_READ|PROT_WRITE, MAP_SHARED,  udmabuf_fd, UDMABUF_OFFSET + i*out_size);
        img_buffer[i] =(unsigned char*) mmap(NULL, out_size ,PROT_READ|PROT_WRITE, MAP_SHARED,  udmabuf_fd, offset + i*out_size);
        
        if (img_buffer[i] == MAP_FAILED)
        {
            return -1;
        }
        /* Write once to allocate physical memory to u-dma-buf virtual space.
        * Note: Do not use memset() for this.
        *       Because it does not work as expected. */
        {
            for(int j = 0 ; j < out_size; j++)
            {
                img_buffer[i][j] = 0;
            }
        }
    }
    return 0;
}


/*****************************************
* Function Name : write_string_rgb
* Description   : OpenCV putText() in RGB
* Arguments     : str = string to be drawn
*                 x = bottom left coordinate X of string to be drawn
*                 y = bottom left coordinate Y of string to be drawn
*                 scale = scale for letter size
*                 color = letter color must be in RGB, e.g. white = 0xFFFFFF
* Return Value  : -
******************************************/
void Image::write_string_rgb(std::string str, uint32_t x, uint32_t y, uint32_t h, uint32_t w, float scale, int color)
{
    unsigned char thickness = CHAR_THICKNESS;
    /*Extract RGB information*/
    unsigned char r = (color >> 16) & 0x0000FF;
    unsigned char g = (color >>  8) & 0x0000FF;
    unsigned char b = color & 0x0000FF;
    /*OpenCV image data is in BGRA */
    cv::Mat bgra_image(out_h, out_w, CV_8UC4, img_buffer[buf_id]);
    /*Color must be in BGR order*/
    cv::putText(bgra_image, str.c_str(), cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, g, 0), thickness);
    cv::rectangle(bgra_image, cv::Rect(x,y,w,h), cv::Scalar(b, 0, 0), 3, 1,0);

    //cv::putText(bgra_image, "00000000000", cv::Point(10,10), LINE_HEIGHT, CHAR_SCALE_LARGE, cv::Scalar(0, 0, 255), WHITE_DATA);
}


/*****************************************
* Function Name : convert_format
* Description   : Convert YUYV image to BGRA format
* Arguments     : -
* Return value  : -
******************************************/
void Image::convert_format()
{
    cv::Mat yuyv_image(img_h, img_w, CV_8UC2, img_buffer[buf_id]);
    cv::Mat bgra_image;
    cv::Mat out_image(img_h, img_w, CV_8UC4, img_buffer[buf_id]);

    cv::cvtColor(yuyv_image, bgra_image, cv::COLOR_YUV2BGRA_YUYV);
    memcpy(out_image.data, bgra_image.data, img_w * img_h * out_c);
}


/*****************************************
* Function Name : convert_size
* Description   : Scale up the input data (640x480) to the intermediate data (960x720) using OpenCV.
*                 To convert to the final output size (1280x720), fill the right margin of the
*                 intermediate data (960x720) with black.
* Arguments     : -
* Return value  : -
******************************************/
/* 
void Image::convert_size()
{
    cv::Mat org_image(img_h, img_w, CV_8UC4, img_buffer[buf_id]);
    cv::Mat resize_image;
    uint32_t resized_width = img_w * RESIZE_SCALE;
    // Resize: 640x480 to 960x720 //
    cv::resize(org_image, resize_image, cv::Size(), RESIZE_SCALE, RESIZE_SCALE);

    // Padding: 960x720 to 1280x720 //
    unsigned char *dst = img_buffer[buf_id];
    unsigned char *src = resize_image.data;
    for(int i = 0; i < out_h; i++)
    {
        memcpy(dst, src, resized_width * out_c);
        dst += (resized_width * out_c);
        src += (resized_width * out_c);
        for(int j = 0; j < (out_w - resized_width) * out_c; j+=out_c) {
            dst[j]   = 0; //B
            dst[j+1] = 0; //G
            dst[j+2] = 0; //R
            dst[j+3] = 0xff; //A
        }
        dst += ((out_w - resized_width) * out_c);
    }

}
*/

/**/
void Image::convert_size()
{
    cv::Mat org_image(img_h, img_w, CV_8UC4, img_buffer[buf_id]);
    cv::Mat resize_image;
    uint32_t resized_width = img_w * RESIZE_SCALE;
    // Resize: 640x480 to 960x720 //
    cv::resize(org_image, resize_image, cv::Size(), RESIZE_SCALE, RESIZE_SCALE);

    // Padding: 960x720 to 1280x720 //

    unsigned char *dst = img_buffer[buf_id];
    unsigned char *src = resize_image.data;
    for(int i = 0; i < out_h; i++)
    {
        memcpy(dst, src, resized_width * out_c);
        
        dst += (resized_width * out_c);
        src += (resized_width * out_c);
        /*
        for(int j = 0; j < (out_w - resized_width) * out_c; j+=out_c) {
            dst[j]   = 0; //B
            dst[j+1] = 0; //G
            dst[j+2] = 0; //R
            dst[j+3] = 0xff; //A
        }
        */
        dst += ((out_w - resized_width) * out_c);
        
    }


}


/*****************************************
* Function Name : convert_channels
* Description   : Function to copy the convert image channels hwc to chw
*                 
* Arguments     : 
*                 
* Return value  : none
******************************************/

/* */
void Image::convert_channels()
{
    cv::Mat dst_img(img_h, img_w, CV_8UC4, img_buffer[buf_id]);
    cv::Mat bgra_image;
    cv::Mat output_image(img_h, img_w, CV_8UC4, img_buffer[buf_id]);
    cv::cvtColor(dst_img, bgra_image, cv::COLOR_RGBA2RGB);  
    memcpy(output_image.data, bgra_image.data, img_w * img_h * out_c);                                                   
}


/*****************************************
* Function Name : camera_to_image
* Description   : Function to copy the external image buffer data to img_buffer
*                 This is only place where the buf_id is updated.
* Arguments     : buffer = buffer to copy the image data
*                 size = size of buffer
* Return value  : none
******************************************/
void Image::camera_to_image(const unsigned char* buffer, int size)
{
    /* Update buffer id */
    buf_id = ++buf_id % WL_BUF_NUM;
    memcpy(img_buffer[buf_id], buffer, sizeof(unsigned char)*size);
}


/*****************************************
* Function Name : at
* Description   : Get the value of img_buffer at index a.
*                 This function is NOT used currently.
* Arguments     : a = index of img_buffer
* Return Value  : value of img_buffer at index a
******************************************/
unsigned char Image::at(int a)
{
    return img_buffer[buf_id][a];
}

/*****************************************
* Function Name : set
* Description   : Set the value of img_buffer at index a.
*                 This function is NOT used currently.
* Arguments     : a = index of img_buffer
*                 val = new value to be set
* Return Value  : -
******************************************/
void Image::set(int a, unsigned char val)
{
    img_buffer[buf_id][a] = val;
    return;
}
/*****************************************
* Function Name : get_buf_id
* Description   : Get the value of the buf_id.
* Arguments     : -
* Return Value  : value of buf_id-
******************************************/
unsigned char Image::get_buf_id()
{
    return buf_id;
}
