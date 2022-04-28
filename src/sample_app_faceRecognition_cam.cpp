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
* File Name    : sample_app_resnet50_cam.cpp
* Version      : 0.90
* Description  : RZ/V2L DRP-AI Sample Application for PyTorch ResNet USB Camera version
***********************************************************************************************************************/

/*****************************************
* Includes
******************************************/
/*DRPAI Driver Header*/
#include <linux/drpai.h>
/*Definition of Macros & other variables*/
#include "define.h"
/*USB camera control*/
#include "camera.h"
/*Image control*/
#include "image.h"
/*Wayland control*/
#include "wayland.h"
/*face detection*/
#include "mtcnn.h"

//#include <opencv2/dnn.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <eigen3/Eigen/Dense> 
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Eigen>
#include <algorithm>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <termio.h>

/*****************************************
* Global Variables
******************************************/
/*Multithreading*/
static sem_t terminate_req_sem;
static pthread_t ai_inf_thread;
static pthread_t kbhit_thread;
static pthread_t capture_thread;
/*Flags*/
static std::atomic<uint8_t> inference_start (0);
static std::atomic<uint8_t> img_obj_ready   (0);
static std::atomic<uint8_t> capture_enabled (0);
static std::atomic<uint8_t> get_result_done (0);
static std::atomic<uint8_t> save_face_feature (0);
static std::atomic<uint8_t> update_feature_vectors (0);
static std::atomic<uint8_t> ingore_result_done (0);
static std::atomic<uint8_t> update_capture (0);

/*Global Variables*/
static uint32_t capture_address;
static Image img;
static Image show_img;

Camera* cap_infer = new Camera();

/*AI Inference for DRPAI*/
static st_addr_t drpai_address;
static std::string dir = drpai_prefix+"/";
static std::string address_file=dir+drpai_prefix+ "_addrmap_intm.txt";
static int drpai_fd = -1;
static uint8_t output_buf[NUM_CLASS*NUM_BYTE];
static float ai_time;
static Wayland wayland;

/*****************************************
* Function Name : GetFileNames
* Description   : Function to go through the files in the folder and get the name of the file
* NOTE          : This is just the simplest example to get the name of the file.
*                 This function does not have header check.
* Arguments     : path = the path where the file is located
*                 filenames  = name of file
*                 
* Return value  : 0 if succeeded
*                 not 0 otherwise
******************************************/
void GetFileNames(string path,vector<string>& filenames)
{
    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(path.c_str())))
        return;
    while((ptr = readdir(pDir))!=0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
            filenames.push_back(path + "/" + ptr->d_name);
    }
    closedir(pDir);
}




namespace FacePreprocess {

    inline cv::Mat meanAxis0(const cv::Mat& src)
    {
        int num = src.rows;
        int dim = src.cols;

        cv::Mat output(1, dim, CV_32F);
        for (int i = 0; i < dim; i++)
        {
            float sum = 0;
            for (int j = 0; j < num; j++)
            {
                sum += src.at<float>(j, i);
            }
            output.at<float>(0, i) = sum / num;
        }

        return output;
    }

    inline cv::Mat elementwiseMinus(const cv::Mat& A, const cv::Mat& B)
    {
        cv::Mat output(A.rows, A.cols, A.type());

        assert(B.cols == A.cols);
        if (B.cols == A.cols)
        {
            for (int i = 0; i < A.rows; i++)
            {
                for (int j = 0; j < B.cols; j++)
                {
                    output.at<float>(i, j) = A.at<float>(i, j) - B.at<float>(0, j);
                }
            }
        }
        return output;
    }


    inline cv::Mat varAxis0(const cv::Mat& src)
    {
        cv::Mat temp_ = elementwiseMinus(src, meanAxis0(src));
        cv::multiply(temp_, temp_, temp_);
        return meanAxis0(temp_);

    }

    inline int MatrixRank(cv::Mat M)
    {
        cv::Mat w, u, vt;
        cv::SVD::compute(M, w, u, vt);
        cv::Mat1b nonZeroSingularValues = w > 0.0001;
        int rank = countNonZero(nonZeroSingularValues);
        return rank;

    }

    //    References
    //    ----------
    //    .. [1] "Least-squares estimation of transformation parameters between two
    //    point patterns"
    //
    //    """
    inline cv::Mat similarTransform(cv::Mat src, cv::Mat dst) {
        int num = src.rows;
        int dim = src.cols;
        cv::Mat src_mean = meanAxis0(src);
        cv::Mat dst_mean = meanAxis0(dst);
        cv::Mat src_demean = elementwiseMinus(src, src_mean);
        cv::Mat dst_demean = elementwiseMinus(dst, dst_mean);
        cv::Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);
        cv::Mat d(dim, 1, CV_32F);
        d.setTo(1.0f);
        if (cv::determinant(A) < 0) {
            d.at<float>(dim - 1, 0) = -1;

        }
        cv::Mat T = cv::Mat::eye(dim + 1, dim + 1, CV_32F);
        cv::Mat U, S, V;
        cv::SVD::compute(A, S, U, V);
        // the SVD function in opencv differ from scipy .


        int rank = MatrixRank(A);
        if (rank == 0) {
            assert(rank == 0);

        }
        else if (rank == dim - 1) {
            if (cv::determinant(U) * cv::determinant(V) > 0) {
                T.rowRange(0, dim).colRange(0, dim) = U * V;
            }
            else {
                //            s = d[dim - 1]
                //            d[dim - 1] = -1
                //            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
                //            d[dim - 1] = s
                int s = d.at<float>(dim - 1, 0) = -1;
                d.at<float>(dim - 1, 0) = -1;

                T.rowRange(0, dim).colRange(0, dim) = U * V;
                cv::Mat diag_ = cv::Mat::diag(d);
                cv::Mat twp = diag_ * V; //np.dot(np.diag(d), V.T)
                cv::Mat B = cv::Mat::zeros(3, 3, CV_8UC1);
                cv::Mat C = B.diag(0);
                T.rowRange(0, dim).colRange(0, dim) = U * twp;
                d.at<float>(dim - 1, 0) = s;
            }
        }
        else {
            cv::Mat diag_ = cv::Mat::diag(d);
            cv::Mat twp = diag_ * V.t(); //np.dot(np.diag(d), V.T)
            cv::Mat res = U * twp; // U
            T.rowRange(0, dim).colRange(0, dim) = -U.t() * twp;
        }
        cv::Mat var_ = varAxis0(src_demean);
        float val = cv::sum(var_).val[0];
        cv::Mat res;
        cv::multiply(d, S, res);
        float scale = 1.0 / val * cv::sum(res).val[0];
        T.rowRange(0, dim).colRange(0, dim) = -T.rowRange(0, dim).colRange(0, dim).t();
        cv::Mat  temp1 = T.rowRange(0, dim).colRange(0, dim); // T[:dim, :dim]
        cv::Mat  temp2 = src_mean.t(); //src_mean.T
        cv::Mat  temp3 = temp1 * temp2; // np.dot(T[:dim, :dim], src_mean.T)
        cv::Mat temp4 = scale * temp3;
        T.rowRange(0, dim).colRange(dim, dim + 1) = -(temp4 - dst_mean.t());
        T.rowRange(0, dim).colRange(0, dim) *= scale;
        return T;
    }
}


/*****************************************
* Function Name : read_bmp
* Description   : Function to load BMP file into img_buffer
* NOTE          : This is just the simplest example to read Windows Bitmap v3 file.
*                 This function does not have header check.
* Arguments     : filename = name of BMP file to be read
*                 width  = BMP image width
*                 height = BMP image height
*                 channel = BMP image color channel
* Return value  : 0 if succeeded
*                 not 0 otherwise
******************************************/
int8_t read_bmp(string filename, unsigned int width, unsigned int height, unsigned int channel)
{
    int i, j;
    FILE *fp;

    /* Number of byte in single row */
    int line_width = width*3+width%4;
    /* Single row image data */
    unsigned char *bmp_line_data;

    /*  Read header for Windows Bitmap v3 file. */
    uint32_t headersize = FILEHEADERSIZE+INFOHEADERSIZE_W_V3;
    unsigned char header_buf[headersize];
    if ((fp = fopen(filename.c_str(), "rb"))==NULL)
    {
        return -1;
    }

    /* Read all header */
    if (!fread(header_buf, sizeof(unsigned char), headersize, fp))
    {
        fprintf(stderr, "[ERROR] image file fread failed \n");
        return -1;
    }

    if((bmp_line_data = (unsigned char*) malloc(sizeof(unsigned char)*line_width))==NULL)
    {
        free(bmp_line_data);
        fclose(fp);
        return -1;
    }



    for(i = height-1; i >= 0; i--)
    {
        if (!fread(bmp_line_data, sizeof(unsigned char), line_width, fp))
        {
            fprintf(stderr, "[ERROR] image file fread failed \n");
            return -1;
        }
        printf("read_bmp ******************************** reading \n");
        memcpy(show_img.img_buffer+i*width*channel, bmp_line_data, sizeof(unsigned char)*width*channel);
    }

    free(bmp_line_data);
    fclose(fp);
    return 0;
}


/*****************************************
* Function Name : timedifference_msec
* Description   : compute the time diffences in ms between two moments
* Arguments     : t0 = start time
*                 t1 = stop time
* Return value  : the time diffence in ms
******************************************/
static double timedifference_msec(struct timespec t0, struct timespec t1)
{
    return (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1000000.0;
}


/*****************************************
* Function Name : wait_join
* Description   : waits for a fixed amount of time for the thread to exit
* Arguments     : p_join_thread = thread that the function waits for to Exit
*                 join_time = the timeout time for the thread for exiting
* Return value  : 0 if successful
*                 not 0 otherwise
******************************************/
static int wait_join(pthread_t *p_join_thread, uint32_t join_time)
{
    int ret_err;
    struct timespec join_timeout;
    ret_err = clock_gettime(CLOCK_REALTIME, &join_timeout);
    if( ret_err == 0 )
    {
        join_timeout.tv_sec += join_time;
        ret_err = pthread_timedjoin_np(*p_join_thread, NULL, &join_timeout);
    }
    return ret_err;
}


/*****************************************
* Function Name : read_addrmap_txt
* Description   : Loads address and size of DRP-AI Object files into struct addr.
* Arguments     : addr_file = filename of addressmap file (from DRP-AI Object files)
* Return value  : 0 if succeeded
*                 not 0 otherwise
******************************************/
int8_t read_addrmap_txt(std::string addr_file)
{
    std::ifstream ifs(addr_file);
    std::string str;
    unsigned long l_addr;
    unsigned long l_size;
    std::string element, a, s;

    if (ifs.fail())
    {
        fprintf(stderr, "[ERROR] Adddress Map List open failed : %s\n", addr_file.c_str());
        return -1;
    }

    while (getline(ifs, str))
    {
        std::istringstream iss(str);
        iss >> element >> a >> s;
        l_addr = strtol(a.c_str(), NULL, 16);
        l_size = strtol(s.c_str(), NULL, 16);

        if (element == "drp_config")
        {
            drpai_address.drp_config_addr = l_addr;
            drpai_address.drp_config_size = l_size;
        }
        else if (element == "desc_aimac")
        {
            drpai_address.desc_aimac_addr = l_addr;
            drpai_address.desc_aimac_size = l_size;
        }
        else if (element == "desc_drp")
        {
            drpai_address.desc_drp_addr = l_addr;
            drpai_address.desc_drp_size = l_size;
        }
        else if (element == "drp_param")
        {
            drpai_address.drp_param_addr = l_addr;
            drpai_address.drp_param_size = l_size;
        }
        else if (element == "weight")
        {
            drpai_address.weight_addr = l_addr;
            drpai_address.weight_size = l_size;
        }
        else if (element == "data_in")
        {
            drpai_address.data_in_addr = l_addr;
            drpai_address.data_in_size = l_size;
        }
        else if (element == "data")
        {
            drpai_address.data_addr = l_addr;
            drpai_address.data_size = l_size;
        }
        else if (element == "data_out")
        {
            drpai_address.data_out_addr = l_addr;
            drpai_address.data_out_size = l_size;
        }
        else if (element == "work")
        {
            drpai_address.work_addr = l_addr;
            drpai_address.work_size = l_size;
        }
    }

    return 0;
}


/*****************************************
* Function Name : load_data_to_mem
* Description   : Loads a file to memory via DRP-AI Driver
* Arguments     : data = filename to be written to memory
*                 drpai_fd = file descriptor of DRP-AI Driver
*                 from = memory start address where the data is written
*                 size = data size to be written
* Return value  : 0 if succeeded
*                 not 0 otherwise
******************************************/
int8_t load_data_to_mem(std::string data, int drpai_fd,
                            unsigned long from, unsigned long size)
{
    int obj_fd;
    uint8_t buf[BUF_SIZE];
    drpai_data_t drpai_data;

    printf("Loading : %s\n", data.c_str());
    obj_fd = open(data.c_str(), O_RDONLY);
    if (obj_fd < 0)
    {
        fprintf(stderr, "[ERROR] open failed: %s\n", data.c_str());
        return -1;
    }

    drpai_data.address = from;
    drpai_data.size = size;

    if (0 != ioctl(drpai_fd, DRPAI_ASSIGN, &drpai_data))
    {
        fprintf(stderr, "[ERROR] DRPAI_ASSIGN failed \n");
        return -1;
    }

    for (int i = 0 ; i<(drpai_data.size/BUF_SIZE) ; i++)
    {
        if ( 0 > read(obj_fd, buf, BUF_SIZE))
        {
            fprintf(stderr, "[ERROR] read failed: %s\n", data.c_str());
            return -1;
        }
        if ( 0 > write(drpai_fd, buf,  BUF_SIZE))
        {
            fprintf(stderr, "[ERROR] DRP-AI Driver write failed \n");
            return -1;
        }
    }
    if ( 0 != (drpai_data.size % BUF_SIZE))
    {
        if ( 0 > read(obj_fd, buf, (drpai_data.size % BUF_SIZE)))
        {
            fprintf(stderr, "[ERROR] read failed: %s\n", data.c_str());
            return -1;
        }
        if( 0 > write(drpai_fd, buf, (drpai_data.size % BUF_SIZE)))
        {
            fprintf(stderr, "[ERROR] DRP-AI Driver write failed \n");
            return -1;
        }
    }
    close(obj_fd);
    return 0;
}


/*****************************************
* Function Name :  load_drpai_data
* Description   : Loads DRP-AI Object files to memory via DRP-AI Driver.
* Arguments     : drpai_fd = file descriptor of DRP-AI Driver
* Return value  : 0 if succeeded
*               : not 0 otherwise
******************************************/
int load_drpai_data(int drpai_fd)
{
    unsigned long addr, size;
    for ( int i = 0; i < 5; i++ )
    {
        switch (i)
        {
            case (INDEX_W):
                addr = drpai_address.weight_addr;
                size = drpai_address.weight_size;
                break;
            case (INDEX_C):
                addr = drpai_address.drp_config_addr;
                size = drpai_address.drp_config_size;
                break;
            case (INDEX_P):
                addr = drpai_address.drp_param_addr;
                size = drpai_address.drp_param_size;
                break;
            case (INDEX_A):
                addr = drpai_address.desc_aimac_addr;
                size = drpai_address.desc_aimac_size;
                break;
            case (INDEX_D):
                addr = drpai_address.desc_drp_addr;
                size = drpai_address.desc_drp_size;
                break;
            default:
                break;
        }

        if (0 != load_data_to_mem(drpai_file_path[i], drpai_fd, addr, size))
        {
            fprintf(stderr,"[ERROR] load_data_to_mem failed : %s\n",drpai_file_path[i].c_str());
            return -1;
        }
    }
    return 0;
}


/*****************************************
* Function Name     : load_label_file
* Description       : Load label list text file and return the label list that contains the label.
* Arguments         : label_file_name = filename of label list. must be in txt format
* Return value      : std::map<int, std::string> list = list contains labels
*                     empty if error occured
******************************************/
std::map<int, std::string> load_label_file(std::string label_file_name)
{
    int n = 0;
    std::map<int, std::string> list;
    std::ifstream infile(label_file_name);

    if (!infile.is_open())
    {
        return list;
    }

    std::string line;
    while (std::getline(infile,line))
    {
        list[n++] = line;
        if (infile.fail())
        {
            std::map<int, std::string> empty;
            return empty;
        }
    }

    return list;
}


/*****************************************
* Function Name     : save_result
* Description       : save face feature vector to file.
* Arguments         : floatarr = face feature vector. 
*                     input_img = image's name.
* Return value      : 0 if succeeded
*                     not 0 otherwise
******************************************/
int8_t save_result(float* floatarr, string input_img)
{
    FILE * outfile;
    string filename = face_vectors + '/' + input_img.substr(0, input_img.rfind(".")) + ".bin";
    outfile = fopen(filename.c_str(), "wb" );
    
    if (outfile == NULL)
    {
        printf("write %s failed\r\n", filename.c_str());
        return -1;
    }

    fwrite( &floatarr[0],  sizeof( float ), OUT_SIZE, outfile );
    fclose(outfile);
    printf("\nSave Vector Result to %s\n", filename.c_str());
    return 0;
}


/*****************************************
* Function Name     : read_result
* Description       : read face feature vector from file.
* Arguments         : filename = face feature vector file. 
*                     res[] = face feature array.
* Return value      : 0 if succeeded
*                     not 0 otherwise
******************************************/
int8_t read_result(string filename, float res[])
{
    FILE * infile;
    infile = fopen(filename.c_str(), "rb" );
    
    if (infile == NULL)
    {
        printf("Read %s failed\r\n", filename.c_str());
        return -1;
    }

    int len = fread( &res[0],  sizeof( float ), OUT_SIZE, infile );
    fclose(infile);
    printf("\nRead Vector Result from %s\n", filename.c_str());
    return 0;
}

/*****************************************
* Function Name : get_result
* Description   : Get DRP-AI Output from memory via DRP-AI Driver
* Arguments     : drpai_fd = File descriptor of DRP-AI Driver
*                 output_addr = memory start address of DRP-AI output
*                 output_size = output data size
* Return value  : 0 if succeeded
*                 not 0 otherwise
******************************************/
int8_t get_result(int drpai_fd, unsigned long output_addr, unsigned long output_size)
{
    drpai_data_t drpai_data;
    uint8_t buf[BUF_SIZE];
    drpai_data.address = output_addr;
    drpai_data.size = output_size;
    int i, j;

    /* Assign the memory address and size to be read */
    if (0 != ioctl(drpai_fd, DRPAI_ASSIGN, &drpai_data))
    {
        fprintf(stderr, "[ERROR] DRPAI_ASSIGN failed \n");
        return -1;
    }

    /* Read the memory via DRP-AI Driver and store the output to buffer */
    for (i = 0; i < (drpai_data.size/BUF_SIZE); i++)
    {
        if ( 0 > read(drpai_fd, buf, BUF_SIZE))
        {
            fprintf(stderr, "[ERROR] DRP-AI Driver read failed \n");
            return -1;
        }
        for (j = 0; j < BUF_SIZE; j++)
        {
            output_buf[BUF_SIZE * i + j] = buf[j];
        }
    }

    if ( 0 != (drpai_data.size % BUF_SIZE))
    {
        if ( 0 > read(drpai_fd, buf, (drpai_data.size % BUF_SIZE)))
        {
            fprintf(stderr, "[ERROR] DRP-AI Driver read failed \n");
            return -1;
        }
        for (int j = 0; j < (drpai_data.size % BUF_SIZE); j++)
        {
            output_buf[BUF_SIZE * (drpai_data.size/BUF_SIZE) + j] = buf[j];
        }
    }

    return 0;
}



/*****************************************
* Function Name : R_Inf_Thread
* Description   : Executes the DRP-AI inference thread
* Arguments     : threadid = thread identification
* Return value  : -
******************************************/

/* */
void *R_Inf_Thread(void *threadid)
{
    //Semaphore Variable//
    int inf_sem_check;
    //Variable for getting Inference output data//
    drpai_data_t drpai_data;
    //Inference Variables//
    fd_set rfds;
    struct timespec tv;
    int ret_drpai;
    int inf_status = 0;
    drpai_data_t proc[DRPAI_INDEX_NUM];

    drpai_status_t drpai_status;
    //Variable for Performance Measurement//
    static struct timespec start_time;
    static struct timespec inf_end_time;
    uint32_t bcount = 0;

    printf("Inference Thread Starting\n");

    proc[DRPAI_INDEX_INPUT].address = drpai_address.data_in_addr;
    proc[DRPAI_INDEX_INPUT].size = drpai_address.data_in_size;
    proc[DRPAI_INDEX_DRP_CFG].address = drpai_address.drp_config_addr;
    proc[DRPAI_INDEX_DRP_CFG].size = drpai_address.drp_config_size;
    proc[DRPAI_INDEX_DRP_PARAM].address = drpai_address.drp_param_addr;
    proc[DRPAI_INDEX_DRP_PARAM].size = drpai_address.drp_param_size;
    proc[DRPAI_INDEX_AIMAC_DESC].address = drpai_address.desc_aimac_addr;
    proc[DRPAI_INDEX_AIMAC_DESC].size = drpai_address.desc_aimac_size;
    proc[DRPAI_INDEX_DRP_DESC].address = drpai_address.desc_drp_addr;
    proc[DRPAI_INDEX_DRP_DESC].size = drpai_address.desc_drp_size;
    proc[DRPAI_INDEX_WEIGHT].address = drpai_address.weight_addr;
    proc[DRPAI_INDEX_WEIGHT].size = drpai_address.weight_size;
    proc[DRPAI_INDEX_OUTPUT].address = drpai_address.data_out_addr;
    proc[DRPAI_INDEX_OUTPUT].size = drpai_address.data_out_size;

    //DRP-AI OUtput Memory Preparation//
    drpai_data.address = drpai_address.data_out_addr;
    drpai_data.size = drpai_address.data_out_size;

    printf("Inference Loop Starting\n");
    //Inference Loop Start//
    int frame_count=0;
    while(1)
    {
        frame_count++;   
        while(1)
        {
            //printf("frame inference \n");
            //Gets the Termination request semaphore value, if different then 1 Termination was requested//
            //Checks if sem_getvalue is executed wihtout issue//
            if(sem_getvalue(&terminate_req_sem, &inf_sem_check) != 0)
            {
                fprintf(stderr, "[ERROR] Failed to Get Semaphore Value %d\n", errno);
                goto err;
            }
            //Checks the semaphore value//
            if(inf_sem_check != 1)
            {
                goto ai_inf_end;
            }
            //Checks if image frame from Capture Thread is ready.//
            if (inference_start.load())
            {
                inference_start.store(0);
                break;
            }
            //printf("get inference start loop\n");
            usleep(WAIT_TIME);
        }
        //Writes Data Input Physical Address & Size Directly to DRP-AI//
        //proc[DRPAI_INDEX_INPUT].address = (uintptr_t) capture_address;
        proc[DRPAI_INDEX_INPUT].address = (uintptr_t) UDMABUF_INFER;

        //Gets Inference Starting Time//
        if(timespec_get(&start_time, TIME_UTC) == 0)
        {
            fprintf(stderr, "[ERROR] Failed to Get Inference Start Time\n");
            goto err;         
        }

        if(frame_count < 100)
        {
            if (0 != ioctl(drpai_fd, DRPAI_START, &proc[0]))
            {
                //fprintf(stderr, "[ERROR] Failed to Start DRPAI Inference: %d\n", errno);
                //goto err;

                if( errno != EBUSY )
                {
                    fprintf(stderr, "[ERROR] Failed to Start DRPAI Inference: errno=%d\n", errno);
                    goto err;
                }
                else
                {
                    capture_enabled.store(1); /* Flag for Capture Thread. */
                    usleep(WAIT_TIME);
                    ingore_result_done.store(0);
                    printf("+ inference counter is %d\n",frame_count);
                    //update_capture.store(0);
                    img_obj_ready.store(0); /* Flag for Main Thread. */
                    inference_start.store(0);
                    //continue;
                }
            }

        }
        else
        {
            if (0 != ioctl(drpai_fd, DRPAI_START, &proc[0]))
            {
                //fprintf(stderr, "[ERROR] Failed to Start DRPAI Inference: %d\n", errno);
                //goto err;

                if( errno != EBUSY )
                {
                    fprintf(stderr, "[ERROR] Failed to Start DRPAI Inference: errno=%d\n", errno);
                    goto err;
                }
                else
                {
                    capture_enabled.store(1); /* Flag for Capture Thread. */
                    usleep(WAIT_TIME);
                    ingore_result_done.store(0);
                    printf("+ inference counter is %d\n",frame_count);
                    //update_capture.store(0);
                    img_obj_ready.store(0); /* Flag for Main Thread. */
                    inference_start.store(0);
                    continue;
                }
            }
        }

#if 0
        //Ready to kick the Inference & run//
        if (0 != ioctl(drpai_fd, DRPAI_START, &proc[0]))
        {
            //fprintf(stderr, "[ERROR] Failed to Start DRPAI Inference: %d\n", errno);
            //goto err;

            if( errno != EBUSY )
            {
                fprintf(stderr, "[ERROR] Failed to Start DRPAI Inference: errno=%d\n", errno);
                goto err;
            }
            else
            {
                capture_enabled.store(1); /* Flag for Capture Thread. */
                usleep(WAIT_TIME);
                ingore_result_done.store(0);
                printf("+ inference counter is %d\n",frame_count);
                //update_capture.store(0);
                img_obj_ready.store(0); /* Flag for Main Thread. */
                continue;
            }
        }


#endif
        printf(" inference loop\n");
        //Waits for the Inference to end//
        FD_ZERO(&rfds);
        FD_SET(drpai_fd, &rfds);
        tv.tv_sec = DRPAI_TIMEOUT;
        tv.tv_nsec = 0;
        //Waits for AI Inference to be done//
        ret_drpai = pselect(drpai_fd+1, &rfds, NULL, NULL, &tv, NULL);
        if(ret_drpai == 0)
        {
            fprintf(stderr, "[ERROR] DRPAI Inference pselect() Timeout: %d\n", errno);
            goto err;
        }
        else if(ret_drpai < 0)
        {
            fprintf(stderr, "[ERROR] DRPAI Inference pselect() Error: %d\n", errno);
            goto err;
        }
        else
        {
            //Do nothing//
        }
        //Gets AI Inference End Time//
        if(timespec_get(&inf_end_time, TIME_UTC) == 0)
        {
            fprintf(stderr, "[ERROR] Failed to Get Inference End Time\n");
            goto err;
        }
        //Checks if DRPAI Inference is ended without issue//
        inf_status = ioctl(drpai_fd, DRPAI_GET_STATUS, &drpai_status);
        if(inf_status == 0)
        {   

            //capture_enabled.store(1); // Flag for Capture Thread. //

            //Process to read the DRPAI output data.//
            get_result(drpai_fd, drpai_data.address, drpai_data.size);
            get_result_done.store(1); // Flag for Main Thread. //
            //Inference Time Result//
            //ai_time = (float)((timedifference_msec(start_time, inf_end_time)));
        }
        else
        {
            // inf_status != 0 //
            fprintf(stderr, "[ERROR] DRPAI Internal Error: %d\n", errno);
            goto err;
        }
    }
    //End of Inference Loop//

//Error Processing//
err:
    sem_trywait(&terminate_req_sem);
    goto ai_inf_end;

ai_inf_end:
    //To terminate the loop in Capture Thread.//
    capture_enabled.store(1);
    get_result_done.store(1); //To terminate Main Thread.//

    printf("AI Inference Thread Terminated\n");
    pthread_exit(NULL);
}


/*****************************************
* Function Name : R_Capture_Thread
* Description   : Executes the V4L2 capture with Capture thread.
* Arguments     : threadid = thread identification
* Return value  : -
******************************************/
void *R_Capture_Thread(void *threadid)
{
    Camera* capture = (Camera*) threadid;
    /*Semaphore Variable*/
    int capture_sem_check;
    /*First Loop Flag*/
    bool setting_loop = true;
    uint32_t capture_addr;
    uint8_t ret = 0;
    //int counter = 0;
    int counter = 10;
    unsigned char* img_buffer_;
    int frame_cnt = 0;
    //const int th_cnt = INF_FRAME_NUM;

    printf("Capture Thread Starting\n");
    while(1)
    {
        frame_cnt++;
        //if(0 == update_capture.load())
        //{
        //    update_capture.store(1);
        //    continue;
        //}
        //
        /*Gets the Termination request semaphore value, if different then 1 Termination was requested*/
        /*Checks if sem_getvalue is executed wihtout issue*/
        if(sem_getvalue(&terminate_req_sem, &capture_sem_check) != 0)
        {
            fprintf(stderr, "[ERROR] Failed to Get Semaphore Value %d\n", errno);
            goto err;
        }
        /*Checks the semaphore value*/
        if(capture_sem_check != 1)
        {
            goto capture_end;
        }

        /* Capture USB camera image and stop updating the capture buffer */
        capture_addr = capture->capture_image();
        if (capture_addr == 0)
        {
            fprintf(stderr, "[ERROR] Camera::capture_image failed\n");
            goto err;
        }
        else
        {
            /* Check the number of frame. */
            //if (counter++ % th_cnt == 0)
            if (counter == 0)
            {
                /* Store captured image data memory address to global variable. */
                capture_address = capture_addr;
                //inference_start.store(1); /* Flag for AI Inference Thread. */

                /*Wait until img_obj_ready flag is cleared by the Main Thread.*/
                while (img_obj_ready.load())
                {
                    usleep(WAIT_TIME);
                }
                /* Copy captured image to Image object. This will be used in Main Thread. */
                img_buffer_ = capture->get_img();
                img.camera_to_image(img_buffer_, capture->get_size());

                img_obj_ready.store(1); /* Flag for Main Thread. */

                /* Wait until capture_enabled is set by AI Inference Thread. */
                while (!capture_enabled.load())
                {
                    usleep(WAIT_TIME);
                }
                capture_enabled.store(0);
            }
            else
            {
                /* Skip first 10 frames */
                counter--;
			}
        }

        /* IMPORTANT: Place back the image buffer to the capture queue */
        if (0 != capture->capture_qbuf())
        {
            fprintf(stderr, "[ERROR] Camera::capture_qbuf failed\n");
            goto capture_end;
        }
    } /*End of Loop*/

err:
    sem_trywait(&terminate_req_sem);
    goto capture_end;

capture_end:
    /*To terminate the loop in AI Inference Thread.*/
    // inference_start.store(1);

    printf("Capture Thread Terminated\n");
    pthread_exit(NULL);
}


int scanKeyboard()
{
    int input;
    struct termios new_settings;
    struct termios stored_settings;
    tcgetattr(0,&stored_settings);
    new_settings = stored_settings;
    new_settings.c_lflag &= (~ICANON);
    new_settings.c_cc[VTIME] = 0;
    tcgetattr(0,&stored_settings);
    new_settings.c_cc[VMIN] = 1;
    tcsetattr(0,TCSANOW,&new_settings);
     
    input = getchar();
     
    tcsetattr(0,TCSANOW,&stored_settings);
    return input;
}


/*****************************************
* Function Name : R_Kbhit_Thread
* Description   : Executes the Keyboard hit thread (checks if enter key is hit)
* Arguments     : threadid = thread identification
* Return value  : -
******************************************/
void *R_Kbhit_Thread(void *threadid)
{
    int ret_fcntl;
    int semget_err;
    /*Semaphore Variable*/
    int kh_sem_check;
    int c;
    printf("Key Hit Thread Started\n");

    printf("************************************************\n");
    printf("* Press s or S key to save face feature vectors. *\n");
    printf("* Press q or Q key to quit. *\n");
    printf("************************************************\n");

    /*Set Standard Input to Non Blocking*/
    ret_fcntl = fcntl(0, F_SETFL, O_NONBLOCK);
    if(ret_fcntl == -1)
    {
        fprintf(stderr, "[ERROR] Failed to fctnl() %d\n", errno);
        goto err;
    }
    int key_c;
    //int key_c = waitKey(1);
    while(1)
    {


        /*Gets the Termination request semaphore value, if different then 1 Termination was requested*/
        /*Checks if sem_getvalue is executed wihtout issue*/
        if(sem_getvalue(&terminate_req_sem, &kh_sem_check) != 0)
        {
            fprintf(stderr, "[ERROR] Failed to Get Semaphore Value %d\n", errno);
            goto err;
        }
        /*Checks the semaphore value*/
        if(kh_sem_check != 1) /* Termination Request Detected on other threads */
        {
            goto key_hit_end;
        }

        key_c = scanKeyboard();

        if((key_c == 83) || (key_c == 115))
        {
            save_face_feature.store(1);

        }

        //if(getchar() != EOF)
        if((key_c == 81) || (key_c == 113))
        {
            /* When ENTER is pressed. */
            printf("Enter Detected\n");
            goto err;
        }
        else
        {
            /* When nothing is pressed. */
            usleep(WAIT_TIME);
        }
    }

err:
    sem_trywait(&terminate_req_sem);
    goto key_hit_end;
key_hit_end:
    printf("Key Hit Thread Terminated\n");
    pthread_exit(NULL);
}

/*****************************************
* Function Name : R_Main_Process
* Description   : Runs the main process loop
* Arguments     : -
* Return value  : 0 if succeeded
*                 not 0 otherwise
******************************************/
uint8_t R_Main_Process()
{
    /*Main Process Variables*/
    uint8_t main_ret = 0;
    /*Semaphore Related*/
    int sem_check;
    int semget_err;
    int counter = 1;
    std::string filename;
    uint32_t ai_inf_prev = 0;
    //std::stringstream stream;
    //std::string str;
    unsigned char img_buf_id;
    float * floatarr;
    int i_line=0;

    printf("Main Loop Starts\n");

    cv::Mat frame;

    MTCNN detector(face_detect);
    
    float factor = 0.709f;
    float threshold[3] = { 0.7f, 0.6f, 0.6f };
    int minSize = 60;

    //int update_feature = 0;


    float val;
    vector<string> file_name;
    
    string line;
    float fNumber = 0.0;
    
    string path = face_vectors;

    GetFileNames(path, file_name);


    int fileNum = 0;
    DIR* dir;   
    dir = opendir(face_vectors.c_str());
    struct dirent* ptr;
    while((ptr = readdir(dir)) != NULL)
    {
      if(ptr->d_name[0] == '.') {continue;}
      fileNum++;
    }
    closedir(dir);


    Eigen::MatrixXf test_matrix(fileNum, OUT_SIZE);
    Eigen::MatrixXf detect_matrix(1, OUT_SIZE);


    Eigen::VectorXf detect_vector(OUT_SIZE);

    Eigen::VectorXf temp_vector(OUT_SIZE);

    Eigen::VectorXf sort_res(fileNum, OUT_DIM);
    Eigen::VectorXf::Index maxRow, maxCol;



    string show_name[20];
    int face_vector_sum = 0;

    //map<int, string> label_file_map = load_label_file(labels);

    face_vector_sum = file_name.size();
    /*load face feature vectors*/
     for(int file_ = 0; file_ <file_name.size(); file_++)
    {
        int i = 0;

        float feature_from_file[OUT_SIZE];
        Eigen::VectorXf result(OUT_SIZE);
        read_result(file_name[file_], &feature_from_file[0]);
        
        for (i=0; i<OUT_SIZE; i++)
        {
            //test_matrix(file_, i) = feature_from_file[i];
            test_matrix(file_, i) = feature_from_file[i];
        }
    }




    int img_count=0;
    while(1)
    {
        img_count++;

        /*Checks if there is any Terminationrequest in the other threads*/
        semget_err = sem_getvalue(&terminate_req_sem, &sem_check);
        if(semget_err != 0)
        {
            fprintf(stderr, "[ERROR] Failed to get Semaphore Value\n");
            goto err;
        }
        /*Checks the Termination Request Semaphore Value*/
        if(sem_check != 1)
        {
            goto main_end;
        }



        if(update_feature_vectors.load())
        {
            file_name = {};
            GetFileNames(path, file_name);

            dir = opendir(face_vectors.c_str());

            fileNum = 0;
            while((ptr = readdir(dir)) != NULL)
            {
              if(ptr->d_name[0] == '.') {continue;}
              fileNum++;
            }
            closedir(dir);


            test_matrix.resize(fileNum, OUT_SIZE);

            sort_res.resize(fileNum, OUT_DIM);

            face_vector_sum = file_name.size();
            printf("fileNum is %d , face_vector_sum is %d \n",fileNum,face_vector_sum);

            /*load face feature vectors*/
             for(int file_ = 0; file_ <file_name.size(); file_++)
            {
                int i = 0;

                float feature_from_file[OUT_SIZE];
                Eigen::VectorXf result(OUT_SIZE);
                read_result(file_name[file_], &feature_from_file[0]);
                
                for (i=0; i<OUT_SIZE; i++)
                {
                    //test_matrix(file_, i) = feature_from_file[i];
                    test_matrix(file_, i) = feature_from_file[i];
                }
            }

            face_vector_sum = file_name.size();           
            update_feature_vectors.store(0);
        }
            


        /* Check img_obj_ready flag which is set in Capture Thread. */


        if (img_obj_ready.load())
        {
            img.convert_format();

            
            cv::Mat out_img(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC4, img.img_buffer[img.buf_id]);
            cv::cvtColor(out_img, frame, cv::COLOR_BGRA2BGR);

            double t = (double) cv::getTickCount();
            vector<FaceInfo> faceInfo = detector.Detect_mtcnn(frame, minSize, threshold, factor, 3);
            std::cout << "Detect" << " time is: " << (double) (cv::getTickCount() - t) / cv::getTickFrequency() << "s" << std::endl;


            if(faceInfo.size() >= 1)
            {
                double recog = (double) cv::getTickCount();
                for (int i = 0; i < faceInfo.size(); i++) 
                {                    
                    int x = (int) faceInfo[i].bbox.xmin;
                    int y = (int) faceInfo[i].bbox.ymin;
                    int w = (int) (faceInfo[i].bbox.xmax - faceInfo[i].bbox.xmin + 1);
                    int h = (int) (faceInfo[i].bbox.ymax - faceInfo[i].bbox.ymin + 1);

                    // face aligen
                    /* */
                    float face_5points[5][2] = { 
                        {faceInfo[i].landmark[0],faceInfo[i].landmark[1]},
                        {faceInfo[i].landmark[2],faceInfo[i].landmark[3]},
                        {faceInfo[i].landmark[4],faceInfo[i].landmark[5]},
                        {faceInfo[i].landmark[6],faceInfo[i].landmark[7]},
                        {faceInfo[i].landmark[8],faceInfo[i].landmark[9]},
                    };

                    
                    float default_96m[5][2] = {
                            {46.704175f, 51.696300f},
                            {101.028192f, 51.696300f},
                            {74.03885f, 71.736600f},
                            {51.721837f, 92.365500f},
                            {96.708595f, 92.365500f}
                    };

                    cv::Mat arcfaceSrc = cv::Mat(5, 2, CV_32FC1, default_96m);
                    memcpy(arcfaceSrc.data, default_96m, 2 * 5 * sizeof(float));

                    cv::Mat dst(5, 2, CV_32FC1, face_5points);
                    memcpy(dst.data, face_5points, 2 * 5 * sizeof(float));
                    cv::Mat M = FacePreprocess::similarTransform(dst, arcfaceSrc);

                    cv::Mat warpImg;
                    cv::warpPerspective(frame, warpImg, M, cv::Size(128, 128));

                    //cv::imshow("face", warpImg);



                    /* copy face image data to mac address for inference*/
                    memcpy(show_img.img_buffer[show_img.buf_id], warpImg.data, sizeof(unsigned char)*INFER_IMAGE_WIDTH*INFER_IMAGE_HEIGHT*IMAGE_CHANNEL_BGR);

                    cv::Mat after_infer_img(INFER_IMAGE_WIDTH, INFER_IMAGE_HEIGHT, CV_8UC3, show_img.img_buffer[show_img.buf_id]);


                    inference_start.store(1); /* Flag for AI Inference Thread. */

                    //cv::rectangle(frame, cv::Rect(x, y, w, h), cv::Scalar(255, 0, 0), 2);
                    
                    /*
                    // 5 points landmark                   
                    for (int k = 0; k < 10; k+=2)
                    {
                        circle(frame, cv::Point(faceInfo[i].landmark[k], faceInfo[i].landmark[k + 1]), 2, cv::Scalar(0, 0, 255), -1);
                    }
                    */

                    if(ingore_result_done.load())
                    {

                        while (!get_result_done.load())
                        {
                            usleep(WAIT_TIME);
                        }



                        get_result_done.store(0);
                        //The format of DRP-AI output is determined by pre/postprocessing yaml in DRP-AI Translator
                        floatarr = (float*)output_buf;      
                     
                        detect_matrix = Eigen::Map<Eigen::MatrixXf>(floatarr, OUT_DIM, OUT_SIZE);


                        if(save_face_feature.load())
                        {
                            save_result(floatarr,"person"+to_string(face_vector_sum+1)+".jpg");
                            update_feature_vectors.store(1);
                        }
                        save_face_feature.store(0);


                        sort_res.transpose() = (detect_matrix.normalized()*(test_matrix.normalized().transpose()))*2;


                        double max = sort_res.transpose().maxCoeff(&maxRow,&maxCol);


                        int pos = file_name[maxCol].find_last_of('/');

                        string s(file_name[maxCol].substr(pos + 1));


                        string name = s.substr(0, s.rfind("."));


                        if(sort_res(maxCol) < 0.45)
                            name = "unknown";

                        //cv::putText(frame, name.c_str(), cv::Point(x+3, y+20), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), CHAR_THICKNESS);
                        
                        show_name[i] = name;


                        //get_result_done.store(0);
                    }

                    
              
                    
                }
                std::cout << "Recognition" << " time is: " << (double) (cv::getTickCount() - recog) / cv::getTickFrequency() << "s" << std::endl; 


            }

            /*
            //Opencv show imagev
            static const std::string kWinName = "Deep learning object detection in OpenCV";
            cv::namedWindow(kWinName, cv::WINDOW_NORMAL);
            cv::imshow(kWinName, frame);
            cv::waitKey(30);
             */
            //cv::waitKey(500);

            img.convert_size();

            //Add face rectangle to wayland image
            for (int k = 0; k < faceInfo.size(); k++)
            {
                img.write_string_rgb(show_name[k], (int) faceInfo[k].bbox.xmin*RESIZE_SCALE, faceInfo[k].bbox.ymin*RESIZE_SCALE, (faceInfo[k].bbox.ymax*RESIZE_SCALE - faceInfo[k].bbox.ymin*RESIZE_SCALE + 1), (int) (faceInfo[k].bbox.xmax*RESIZE_SCALE - faceInfo[k].bbox.xmin*RESIZE_SCALE + 1), CHAR_THICKNESS, WHITE_DATA);
            }
            
            //Update Wayland
            img_buf_id = img.get_buf_id();
            wayland.commit(img_buf_id);


            capture_enabled.store(1);
            //img_obj_ready.store(1);
            ingore_result_done.store(1);

            img_obj_ready.store(0);
            counter++;

            
        }
        img_obj_ready.store(0);
        /*Wait for 1 TICK.*/
        usleep(WAIT_TIME);
    }

err:
    sem_trywait(&terminate_req_sem);
    main_ret = 1;
    goto main_end;
main_end:
    /*To terminate the loop in Capture Thread.*/
    capture_enabled.store(1);
    img_obj_ready.store(0);
    inference_start.store(0);
    printf("Main Process Terminated\n");
    return main_ret;
}


int main(int argc, char * argv[])
{
    uint8_t main_proc;
    uint8_t ret;
    /*Multithreading Variables*/
    int create_thread_ai = -1;
    int create_thread_key = -1;
    int create_thread_capture = -1;
    int sem_create = -1;

    printf("RZ/V2L DRP-AI Sample Application\n");
    printf("Model : PyTorch Face Recognition    | %s\n", drpai_prefix.c_str());
    printf("Input : USB Camera\n");

    /* Read DRP-AI Object files address and size */
    if (0 != read_addrmap_txt(drpai_address_file))
    {
        fprintf(stderr, "[ERROR] read_addrmap_txt failed : %s\n", drpai_address_file.c_str());
        return -1;
    }

    //DRP-AI Driver Open//
    drpai_fd = open("/dev/drpai0", O_RDWR);
    if (drpai_fd < 0)
    {
        fprintf(stderr, "[ERROR] Failed to Open DRP-AI Driver: %d\n", errno);
        return -1;
    }

    // Load DRP-AI Data from Filesystem to Memory via DRP-AI Driver //
    ret = load_drpai_data(drpai_fd);
    if (ret != 0)
    {
        fprintf(stderr, "[ERROR] Failed to load DRPAI Data\n");
        if (0 != close(drpai_fd))
        {
            fprintf(stderr, "[ERROR] Failed to Close DRPAI Driver: %d\n", errno);
        }
        return -1;
    }
    

    /* Create Camera Instance */
    Camera* capture = new Camera();

    /* Init and Start Camera */
    if (0 != capture->start_camera())
    {
        fprintf(stderr, "[ERROR] Camera::start_camera failed\n");
        delete capture;
        if (0 != close(drpai_fd))
        {
            fprintf(stderr, "[ERROR] Failed to Close DRPAI Driver: %d\n", errno);
        }
        return -1;
    }

    /*Initialize Image object.*/
    if (0 != img.init(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL_YUY2, OUTPUT_WIDTH, OUTPUT_HEIGHT, IMAGE_CHANNEL_BGRA, UDMABUF_OFFSET))
    {
        fprintf(stderr, "[ERROR] Image::init failed\n");
        if (0 != close(drpai_fd))
        {
            fprintf(stderr, "[ERROR] Failed to Close DRPAI Driver: %d\n", errno);
        }
        return -1;
    }
    //malloc image offset
    uint32_t off_set = OUTPUT_WIDTH * OUTPUT_HEIGHT * IMAGE_CHANNEL_BGRA * WL_BUF_NUM + UDMABUF_OFFSET;
    if (0 != show_img.init(INFER_IMAGE_WIDTH, INFER_IMAGE_HEIGHT, IMAGE_CHANNEL_YUY2, INFER_IMAGE_WIDTH, INFER_IMAGE_HEIGHT, IMAGE_CHANNEL_BGR, off_set))
    {
        fprintf(stderr, "[ERROR] Image::init failed\n");
        if (0 != close(drpai_fd))
        {
            fprintf(stderr, "[ERROR] Failed to Close DRPAI Driver: %d\n", errno);
        }
        return -1;
    }

    // Initialize waylad //
    /**/
    if(wayland.init(img.udmabuf_fd, OUTPUT_WIDTH, OUTPUT_HEIGHT, IMAGE_CHANNEL_BGRA))
    {
        fprintf(stderr, "[ERROR] Wayland::init failed\n");
        delete capture;
        return -1;
    }

    /*Termination Request Semaphore Initialization*/
    /*Initialized value at 1.*/
    sem_create = sem_init(&terminate_req_sem, 0, 1);
    if(sem_create != 0)
    {
        fprintf(stderr, "[ERROR] Failed to Initialize Termination Request Semaphore\n");
        goto main_end;
    }

    /*Key Hit Thread*/
    create_thread_key = pthread_create(&kbhit_thread, NULL, R_Kbhit_Thread, NULL);
    if(create_thread_key != 0)
    {
        fprintf(stderr, "[ERROR] Key Hit Thread Creation Failed\n");
        goto main_end;
    }

    /*AI Inference Thread*/
    /**/
    create_thread_ai = pthread_create(&ai_inf_thread, NULL, R_Inf_Thread, NULL);
    if(create_thread_ai != 0)
    {
        sem_trywait(&terminate_req_sem);
        fprintf(stderr, "[ERROR] AI Inference Thread Creation Failed\n");
        goto main_end;
    }

    /*Capture Thread*/
    create_thread_capture = pthread_create(&capture_thread, NULL, R_Capture_Thread, (void *) capture);
    if(create_thread_capture != 0)
    {
        sem_trywait(&terminate_req_sem);
        fprintf(stderr, "[ERROR] Capture Thread Creation Failed\n");
        goto main_end;
    }

    /*Main Processing*/
    main_proc = R_Main_Process();
    if(main_proc != 0)
    {
        fprintf(stderr, "[ERROR] Error during Main Process\n");
        goto main_end;
    }
    goto main_end;


main_end:
    if(create_thread_capture == 0)
    {
        if(wait_join(&capture_thread, CAPTURE_TIMEOUT) != 0)
        {
            fprintf(stderr, "[ERROR] Capture Thread Failed to Exit on time\n");
        }
    }
    /**/
    if(create_thread_ai == 0)
    {
        if(wait_join(&ai_inf_thread, AI_THREAD_TIMEOUT) != 0)
        {
            fprintf(stderr, "[ERROR] AI Inference Thread Failed to Exit on time\n");
        }
    }
    
    if(create_thread_key == 0)
    {
        if(wait_join(&kbhit_thread, KEY_THREAD_TIMEOUT) != 0)
        {
            fprintf(stderr, "[ERROR] Key Hit Thread Failed to Exit on time\n");
        }
    }

    /*Delete Terminate Request Semaphore.*/
    if(sem_create == 0)
    {
        sem_destroy(&terminate_req_sem);
    }

    /* Exit waylad */
    wayland.exit();

    /*Close USB Camera.*/
    if (0 != capture->close_camera())
    {
        fprintf(stderr, "[ERROR] Camera::close_camera failed\n");
    }

    delete capture;

    /*Close DRP-AI Driver.*/
    if(drpai_fd > 0)
    {
        if (0 != close(drpai_fd))
        {
            fprintf(stderr, "[ERROR] Failed to Close DRP-AI Driver: %d\n", errno);
        }
    }
    printf("Application End\n");
    return 0;
}
