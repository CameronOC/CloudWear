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
    Mat nc_img1, img1, img1_gray, c_img1;
    Mat dst, det_edges;
    string wn1;
    int edge_thres, low_thres, ratio, kernel_size;
    int max_low_thres, offset_x, offset_y, res;
    bool bkg_col;

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
        res = 750;
        bkg_col = true;
    }

    void read_image(string a)
    {
        nc_img1 = imread(a);
        resize(nc_img1, c_img1, Size(res, res),INTER_LANCZOS4);
        //c_img1 = img1;
        Rect roi;
        roi.x = offset_x;
        roi.y = offset_y;
        //roi.width = nc_img1.size().width - (offset_x*2);
        //roi.height = nc_img1.size().height - (offset_y*2);
        //img1 = nc_img1(roi);
        img1 = c_img1;
        dst.create(img1.size(), img1.type());
        cvtColor(img1, img1_gray, COLOR_BGR2GRAY);
        //img1_gray = img1;
        //img1_gray = Scalar::all(255);
    }

    void display_image(Mat i1)
    {
        namedWindow("wn1", WINDOW_NORMAL);
        imshow("wn1", i1);
        waitKey(0);
    }

    void canny_threshold()
    {
        //blur(img1_gray, det_edges, Size(3,3));
        //Canny(detected_edges, low_thres, low_thres*ratio, kernel_size);
        //Canny(img1_gray, det_edges, low_thres, low_thres*ratio);
        dst = Scalar::all(255);
        det_edges = img1_gray;

        img1.copyTo(dst, det_edges);
        display_image(dst);
    }
    
    void gray_thresh()
    {
        int a = 0, b = 0; 
        blur(img1_gray, img1_gray, Size(3,3));
        //cout << (int)img1_gray.at<uchar>(100, 100)<<endl;
        for(int i=0;i<img1_gray.rows;i++)
        {
            for(int j=0; j<img1_gray.cols;j++)
            {
                //if((int)img1_gray.at<uchar>(i,j)>1)
                //{
                    a = a + (int)img1_gray.at<uchar>(i,j);
                    b++;
                //}
            }
        }
        a = a/b;

        for(int i=0;i<img1_gray.rows;i++)
        {
            for(int j=0; j<img1_gray.cols;j++)
            {
                //cout << img1_gray.at<double>(i, j)<<endl;
                if((int)img1_gray.at<uchar>(i,j) > a - 20 && bkg_col == true) 
                {
                    img1_gray.at<uchar>(i,j) = 0;                    
                }
                else if((int)img1_gray.at<uchar>(i,j) < 25 && bkg_col == false)
                {
                    img1_gray.at<uchar>(i,j) = 255;
                }
            }
        }
        cout << a << endl;
        //blur(img1_gray, img1_gray, Size(3,3));
        canny_threshold();
        //display_image(img1_gray);
    }

    void process_images()
    {
        SimpleBlobDetector detector;
         
        // Detect blobs.
        vector<KeyPoint> keypoints;
        detector.detect(img1_gray, keypoints);
    }

}ob;

int main(int argc, char** argv)
{
    string a;
    //a = "skinny_jeans_1.jpg";
    a = argv[1];
    ob.read_image(a);
    ob.gray_thresh();
    //ob.process_images();
    //ob.display_image(ob.c_img1);
    //ob.canny_threshold();
}

