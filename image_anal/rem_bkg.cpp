#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <string.h>
#include <iostream>
#include <opencv2/video.hpp>
using namespace cv;
using namespace std;

class cloudwear
{
    public:
    // members
    Mat nc_img1, img1, img1_gray;
    Mat dst, det_edges;
    string wn1;
    int edge_thres, low_thres, ratio, kernel_size;
    int max_low_thres, offset_x, offset_y;

    // background shit
    Mat frame, fgMaskMOG2;
    Ptr<BackgroundSubtractor> pMOG2;

    cloudwear()
    {
        wn1 = "normal_image";
        edge_thres = 1;
        low_thres = 99;
        max_low_thres = 100;
        ratio = 3;
        kernel_size = 3;
        offset_x = 20;
        offset_y = 600;
    }

    void read_image(string a)
    {
        nc_img1 = imread(a);
        Rect roi;
        roi.x = offset_x;
        roi.y = offset_y;
        roi.width = nc_img1.size().width - (offset_x*2);
        roi.height = nc_img1.size().height - (offset_y*2);
        img1 = nc_img1(roi);
        dst.create(img1.size(), img1.type());
        cvtColor(img1, img1_gray, CV_BGR2GRAY);
    }

    void display_image(Mat i1)
    {
        namedWindow("wn1", WINDOW_NORMAL);
        imshow("wn1", i1);
        waitKey(0);
    }

    void canny_threshold()
    {
        blur(img1_gray, det_edges, Size(3,3));
        //Canny(detected_edges, low_thres, low_thres*ratio, kernel_size);
        Canny(img1_gray, det_edges, low_thres, low_thres*ratio);
        dst = Scalar::all(0);

        img1.copyTo(dst, det_edges);
        display_image(dst);
    }

     void gray_thresh()
     {
        for(int i=0;i<img1_gray.rows;i++)
        {
            for(int j=0; j<img1_gray.cols;j++)
            {
                if(img1_gray.at<double>(i,j) >180) 
                    img1_gray.at<double>(i,j) =255;
            }
        }
     }

    //void process_images()
    //{

}ob;

int main(int argc, char** argv)
{
    string a;
    a = "blue_shirt.JPG";
    ob.read_image(a);
    ob.gray_thresh();
    ob.display_image(ob.img1_gray);
    //ob.canny_threshold();
}

