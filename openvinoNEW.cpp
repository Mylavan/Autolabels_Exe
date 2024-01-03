#include <iostream>
#include <Windows.h>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types.hpp>

#include <ie/inference_engine.hpp>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#define AUTOLABEL_TEXTDISPLAY_MIN_COUNTOUR_AREA_THRESHOLD 500 //AutoLabel's individual anatomy segmentation contour area threshold to decided whether to show number or  text for the anatomy label to handle label display of anatomy with small area(<=500)
static int Level2_counter = 0; //For IVS QC Check we need to know till which case it goes , so maintaning the counter for Debugging purpose 
using namespace InferenceEngine;
enum EFetalHeartViewClasss
{
    INVALID_VIEW_ID = -1,
    FOUR_CHAMBER = 0,
    LVOT = 1,
    RVOT = 2,
    THREE_VT = 3,
    THREE_VV = 4,
    OTHER_HEART = 5,  // use this as m_MAX_VIEW_ID similar to MAX_CLASS_ID as used by Fetal Biometry 
    NON_HEART = 6
};
enum  STATUS_INFO
{
    IVS_FAILED = 0,
    VALID_ANGLE = 1,
    NEXT_LEVEL_CHECK = 2,
};
float HeartAxisRatio;
float ThoraxAxisRatio;

#define IM_SEGMENTATION_WIDTH            896    // default width of segmented frame
#define IM_SEGMENTATION_HEIGHT            640 // default height of segmented frame
#define IM_SEGMENTATION_CHANNEL            50  // max number of channel with segmented frame
using namespace cv;
using namespace std;
using namespace std::chrono;
int  PaddingTop = 0;
int PaddingBottom = 0;
int PaddingLeft = 0;
int PaddingRight = 0;
int Original_Input_Height = 0;
int Original_Input_Width = 0;
int outCroppedWidth = 0;
int outCroppedHeight = 0;
int outCroppedOriginY = 0;
int outCroppedOriginX = 0;
float ms = 0;
Point A_Store;  // Points for storing the start and end of each line 
Point B_Store;
Point C_Store;
Point D_Store;
vector <string> ImageNames;
#define IVS_QC_HEART_IS_ELLIPSE_THRESHOLD 0.7f
#define AngleCmpwith45(a,c) (c ? (a>45) : (a<45))
#define MajorMinorAngleCmp(a,b,c) (c ? (a>b) : (a<b))
template <typename T> class Vector2D
{
private:
    T x;
    T y;

public:
    explicit Vector2D(const T& x = 0, const T& y = 0) : x(x), y(y) {}
    Vector2D(const Vector2D<T>& src) : x(src.x), y(src.y) {}
    virtual ~Vector2D() {}

    // Accessors
    inline T X() const { return x; }
    inline T Y() const { return y; }
    inline T X(const T& x) { this->x = x; }
    inline T Y(const T& y) { this->y = y; }

    // Vector arithmetic
    inline Vector2D<T> operator-() const
    {
        return Vector2D<T>(-x, -y);
    }

    inline Vector2D<T> operator+() const
    {
        return Vector2D<T>(+x, +y);
    }

    inline Vector2D<T> operator+(const Vector2D<T>& v) const
    {
        return Vector2D<T>(x + v.x, y + v.y);
    }

    inline Vector2D<T> operator-(const Vector2D<T>& v) const
    {
        return Vector2D<T>(x - v.x, y - v.y);
    }

    inline Vector2D<T> operator*(const T& s) const
    {
        return Vector2D<T>(x * s, y * s);
    }

    // Dot product
    inline T operator*(const Vector2D<T>& v) const
    {
        return x * v.x + y * v.y;
    }

    // l-2 norm
    inline T norm() const { return sqrt(x * x + y * y); }

    // inner angle (radians)
    static T angle(const Vector2D<T>& v1, const Vector2D<T>& v2)
    {
        return acos((v1 * v2) / (v1.norm() * v2.norm()));
    }
};
void FillAutoLabelLabelIndexBasedOnFetalHeartView(EFetalHeartViewClasss inFetalHeartViewInRenderedImage, std::vector<int>& outVecSegmentedFrameIndicesOfInterest, std::vector<std::string>& outVecSegmentedLabelsOfInterest)
{
    outVecSegmentedFrameIndicesOfInterest.clear();
    outVecSegmentedLabelsOfInterest.clear();

    if (inFetalHeartViewInRenderedImage == EFetalHeartViewClasss::FOUR_CHAMBER)
    {
        // Mapping of Anatomies to Pixel Values :
        // ["Outer Thorax=1","Inner Thorax=2", "Heart Contour=3", "RA=4", "LA=5", "RV=6", "LV=7","Spine Triangle=8","Spine Body=9",  "dAorta=10","IV Septum=11", "Atrium Septum=12","AV Septum=13",  "Mitral Valve=14", "Tricuspid Valve=15",  "PV1 (RPV)=16", "PV2 (LPV)=17"]
        // Order of Labelling(from Bigger to smaller) : = { "RA", "LA", "RV", "LV", "Sp", "dAo", "IVS", "TV", "MV", "AS", "AVS", "PV1", "PV2" }
        // "Outer Thorax=1","Heart Contour=2" is meant for CTR
        // 0th class is background

        outVecSegmentedFrameIndicesOfInterest = { 4,5,6,7,8,10,11,15,14,12,13, };
        outVecSegmentedLabelsOfInterest = { "RA", "LA", "RV", "LV", "Sp", "dAo", "IVS", "TV", "MV", "AS", "AVS"};
        outVecSegmentedFrameIndicesOfInterest = { 4,5,6,7,8,10,11,12,13,14,15,16,17 };
        outVecSegmentedLabelsOfInterest = { "RA", "LA", "RV", "LV", "Sp", "dAo", "IVS",  "AS", "AVS", "MV", "TV","PV1", "PV2" };

    }
    else if (inFetalHeartViewInRenderedImage == EFetalHeartViewClasss::LVOT)
    {
        // Mapping of Anatomies to Pixel Values :
        //3VT: {"AA":"1", "DA" : "2", "SVC" : "3", "Tr" : "4", "Sp" : "5"}
        // 0th class is background
        //3vt doesnt really have any impact on order of processing since no overlapping anatomies

        outVecSegmentedFrameIndicesOfInterest = { 1,2,3,4,5,6,7,8 };
        outVecSegmentedLabelsOfInterest = { "LV", "LA", "RV", "RA","Sp","dAo", "Ao","AoValve" };


    }
    else if (inFetalHeartViewInRenderedImage == EFetalHeartViewClasss::RVOT)
    {
        // Mapping of Anatomies to Pixel Values :
        //3VT: {"AA":"1", "DA" : "2", "SVC" : "3", "Tr" : "4", "Sp" : "5"}
        // 0th class is background
        //3vt doesnt really have any impact on order of processing since no overlapping anatomies

        outVecSegmentedFrameIndicesOfInterest = { 1,2,3,4,5,6,7,8,9,10,11 };
        outVecSegmentedLabelsOfInterest = { "RV", "PA", "LPA", "RPA","A.Ao", "SVC", "D.Ao", "Spine", "Duct","P.Valve","RA" };
    }
    else if (inFetalHeartViewInRenderedImage == EFetalHeartViewClasss::THREE_VV)
    {
        // Mapping of Anatomies to Pixel Values :
        //3VT: {"AA":"1", "DA" : "2", "SVC" : "3", "Tr" : "4", "Sp" : "5"}
        // 0th class is background
        //3vt doesnt really have any impact on order of processing since no overlapping anatomies

        outVecSegmentedFrameIndicesOfInterest = { 1,2,3,4,5,7 };
        outVecSegmentedLabelsOfInterest = { "MPA", "DA", "dAo", "Ao","SVC","Sp" };
    }
    else if (inFetalHeartViewInRenderedImage == EFetalHeartViewClasss::THREE_VT)
    {
        // Mapping of Anatomies to Pixel Values :
        //3VT: {"AA":"1", "DA" : "2", "SVC" : "3", "Tr" : "4", "Sp" : "5"}
        // 0th class is background
        //3vt doesnt really have any impact on order of processing since no overlapping anatomies

        outVecSegmentedFrameIndicesOfInterest = { 1,2,3,4,5 };
        outVecSegmentedLabelsOfInterest = { "AA", "DA", "SVC", "Tr","Sp" };
    }

}
Mat loadImageandPreProcess(const std::string& filename, int sizeX = IM_SEGMENTATION_WIDTH, int sizeY = IM_SEGMENTATION_HEIGHT)
{
    cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cout << "No image found.";
    }
    int OriginalInputImageWidth = image.size().width;
    int OriginalInputImageHight = image.size().height;

    //closeup crop calculation
    cv::Rect rect = cv::boundingRect(image);

    outCroppedOriginX = rect.x;
    outCroppedOriginY = rect.y;
    outCroppedWidth = rect.width;
    outCroppedHeight = rect.height;

    cv::Mat croppedImage = image(rect);
    cv::Mat fCroppedImage;
    croppedImage.convertTo(fCroppedImage, CV_32FC1);


    //mean and standard deviation calculation
    cv::Scalar mean, stddev;
    cv::meanStdDev(fCroppedImage, mean, stddev);

    double dMean = mean[0];
    double dStdDev = stddev[0];



    //normalize image pxiel values using image mean & standard deviation
    // using formula : (img � img.mean() / img.std())
    fCroppedImage = (fCroppedImage - dMean) / (dStdDev + 1e-8);



    //old code 
    //cv::Mat fCroppedImageResized;
    //cv::resize(fCroppedImage, fCroppedImageResized, cv::Size(IM_SEGMENTATION_WIDTH, IM_SEGMENTATION_HEIGHT), cv::INTER_NEAREST);



    //New Code 
    int hinput = fCroppedImage.size().height;
    int winput = fCroppedImage.size().width;
    float  aspectRatio = 0;
    int Target_Height = IM_SEGMENTATION_HEIGHT;
    int Target_Width = IM_SEGMENTATION_WIDTH;
    int Resized_Height = 0;
    int Resized_Width = 0;
    //Equal 
    if (winput < hinput)
    {
        aspectRatio = (float)winput / hinput;
        std::cout << aspectRatio << std::endl;
        Resized_Height = Target_Height;
        Resized_Width = (float)aspectRatio * Resized_Height;
        if (Resized_Width > Target_Width)
        {
            Resized_Height = Resized_Height - ((Resized_Width - Target_Width) / aspectRatio);
            Resized_Width = aspectRatio * Resized_Height;
        }
    }
    else
    {
        aspectRatio = (float)hinput / winput;
        Resized_Width = Target_Width;
        Resized_Height = (float)(aspectRatio * Resized_Width);
        if (Resized_Height > Target_Height)
        {
            Resized_Width = Resized_Width - ((Resized_Height - Target_Height) / aspectRatio);
            Resized_Height = aspectRatio * Resized_Width;
        }
    }
    cv::Mat fCroppedImageResized;
    Original_Input_Height = OriginalInputImageHight;
    Original_Input_Width = OriginalInputImageWidth;
    cv::resize(fCroppedImage, fCroppedImageResized, cv::Size(Resized_Width, Resized_Height), cv::INTER_NEAREST);

    int DiffWidth = Target_Width - Resized_Width;
    int DiffHeight = Target_Height - Resized_Height;
    PaddingTop = DiffHeight / 2;
    PaddingBottom = DiffHeight / 2 + DiffHeight % 2;
    PaddingLeft = DiffWidth / 2;
    PaddingRight = DiffWidth / 2 + DiffWidth % 2;


    Mat PaddedImage;
    copyMakeBorder(fCroppedImageResized, PaddedImage, PaddingTop, PaddingBottom, PaddingLeft, PaddingRight, BORDER_CONSTANT, 0);

    //std::vector<float> vec;
    //int cn = 1;//RGBA , 4 channel
    //int iCount = 0;

    //const int inputNumChannel = 1;
    //const int inputH = IM_SEGMENTATION_HEIGHT;
    //const int inputW = IM_SEGMENTATION_WIDTH;

    //std::vector<float> vecR;

    //vecR.resize(inputH * inputW);


    //for (int i = 0; i < inputH; i++)
    //{
    //    for (int j = 0; j < inputW; j++)
    //    {
    //        float pixelValue = PaddedImage.at<float>(i, j);
    //        vecR[iCount] = pixelValue;
    //        iCount++;
    //    }
    //}
    //vector<float> input_tensor_values;
    //for (auto i = vecR.begin(); i != vecR.end(); ++i)
    //{
    //    input_tensor_values.push_back(*i);
    //}
    ////return input_tensor_values;
   // imwrite("NewpaddedImage.png", PaddedImage);
    return PaddedImage;
  
}
int getMaxPerimeterLengthContourId(vector <vector<cv::Point>> contours) {
    // select the contour with max points on the contour's perimeter
    int indexofMaxLengthContour = -1;
    size_t maxLengthContour = 0;

    for (size_t i = 0; i < contours.size(); i++)
    {
        vector<Point> vecPoint = contours[i];
        if (maxLengthContour < vecPoint.size())
        {
            maxLengthContour = vecPoint.size();
            indexofMaxLengthContour = i;
        }
    }
    return indexofMaxLengthContour;
}
int getMaxAreaContourId(vector <vector<cv::Point>> contours, double& outMaxArea) {

    // select the contour with max area 
    outMaxArea = 0;
    int maxAreaContourId = -1;
    for (size_t j = 0; j < contours.size(); j++) {
        double newArea = cv::contourArea(contours.at(j));
        if (newArea > outMaxArea) {
            outMaxArea = newArea;
            maxAreaContourId = j;
        }
    }
    return maxAreaContourId;
}
// Function to convert degree to radian
double convertDegreeToRadian(double degree)
{
#define M_PI       3.14159265358979323846   // pi
    return (degree * (double(M_PI) / double(180)));
}

void get_coords(double x, double y, double angle, int imwidth, int imheight, Point& outPoint1, Point& outPoint2)
{
    /*double x1_length = (x - imwidth) / cos(convertDegreeToRadian(angle));
    double	y1_length = (y - imheight) / sin(convertDegreeToRadian(angle));*/
    double	length = 1000;
    double	endx1 = x + length * cos(convertDegreeToRadian(angle));
    double	endy1 = y + length * sin(convertDegreeToRadian(angle));

    /*double x2_length = (x - imwidth) / cos(convertDegreeToRadian(angle + 180));
    double y2_length = (y - imheight) / sin(convertDegreeToRadian(angle + 180));*/
    double endx2 = x + length * cos(convertDegreeToRadian(angle + 180));
    double endy2 = y + length * sin(convertDegreeToRadian(angle + 180));
    outPoint1 = Point(int(endx1), int(endy1));
    outPoint2 = Point(int(endx2), int(endy2));

}

// Function to find two line intersection point 
Point lineLineIntersection(Point A, Point B, Point C, Point D)
{
    // Line AB represented as a1x + b1y = c1
    double a1 = B.y - A.y;
    double b1 = A.x - B.x;
    double c1 = a1 * (A.x) + b1 * (A.y);

    // Line CD represented as a2x + b2y = c2
    double a2 = D.y - C.y;
    double b2 = C.x - D.x;
    double c2 = a2 * (C.x) + b2 * (C.y);

    double determinant = a1 * b2 - a2 * b1;

    if (determinant == 0)
    {
        // The lines are parallel. This is simplified
        // by returning a pair of FLT_MAX
        return Point(INT_MAX, INT_MAX);
    }
    else
    {
        int x = (int)((b2 * c1 - b1 * c2) / determinant);
        int y = (int)((a1 * c2 - a2 * c1) / determinant);
        return Point(x, y);
    }
}


// Function to check if point with in the image dimension(width, height) boundary
bool is_pt_within_image_dimension(Point pt, int imwidth, int imheight)
{
    if (pt.x < imwidth && pt.x >= 0 && pt.y < imheight && pt.y >= 0)
        return true;
    else
        return false;
}

//Get two pts of line segment from ellipse centre and angle
bool get_fin_coords(double x, double y, double angle, int imwidth, int imheight, Point& outPoint1, Point& outPoint2)
{
    Point pt1Coords, pt2Coords;
    get_coords(x, y, angle, imwidth, imheight, pt1Coords, pt2Coords);

    Point pt1LineIntersection = lineLineIntersection(pt1Coords, pt2Coords, Point(0, 0), Point(imwidth - 1, 0));
    Point	pt2LineIntersection = lineLineIntersection(pt1Coords, pt2Coords, Point(0, 0), Point(0, imheight - 1));
    Point	pt3LineIntersection = lineLineIntersection(pt1Coords, pt2Coords, Point(imwidth - 1, 0), Point(imwidth - 1, imheight - 1));
    Point	pt4LineIntersection = lineLineIntersection(pt1Coords, pt2Coords, Point(0, imheight - 1), Point(imwidth - 1, imheight - 1));

    std::vector <Point> vecCheckPts = { pt1LineIntersection ,pt2LineIntersection ,pt3LineIntersection,pt4LineIntersection };
    std::vector <Point> vecValidPts;
    vecValidPts.clear();

    for (size_t i = 0; i < vecCheckPts.size(); i++)
    {
        Point ptCheck = vecCheckPts[i];
        if (is_pt_within_image_dimension(ptCheck, imwidth, imheight) == true)
        {
            bool bNotFound = true;
            for (size_t j = 0; j < vecValidPts.size(); j++)
            {
                if (ptCheck == vecValidPts[j])
                {
                    bNotFound = false;
                }
            }
            if (bNotFound)
                vecValidPts.push_back(ptCheck);

        }
    }
    bool bRet = false;
    if (vecValidPts.size() >= 2)
    {
        outPoint1 = vecValidPts[0];
        outPoint2 = vecValidPts[1];
        bRet = true;
    }


    return bRet;
}
void calculate_angle_for_line_intersection(Point inTwoLineIntersectionPt, Point inFirstLineOtherEndPt, Point inSecondLineOtherEndPt, float& outAngle)
{
    outAngle = 0;
    //std::cout << "//////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    //std::cout << "inTwoLineIntersectionPt  :: " << inTwoLineIntersectionPt << std::endl;
    //std::cout << "inFirstLineOtherEndPt  :: " << inFirstLineOtherEndPt << std::endl;
    //std::cout << "inSecondLineOtherEndPt  :: " << inSecondLineOtherEndPt << std::endl;
    Vector2D<double> p1(inTwoLineIntersectionPt.x, inTwoLineIntersectionPt.y);
    Vector2D<double> p2(inFirstLineOtherEndPt.x, inFirstLineOtherEndPt.y);
    Vector2D<double> p3(inSecondLineOtherEndPt.x, inSecondLineOtherEndPt.y);

    double ang_rad = Vector2D<double>::angle(p2 - p1, p3 - p1);
    double ang_deg = ang_rad * 180.0 / M_PI;

    std::cout << " ORIGINAL ANGLE WAS  : " << ang_deg << endl;
    if (ang_deg - 180 >= 0)
        outAngle = 360 - ang_deg;
    else
        outAngle = ang_deg;

    if (outAngle > 90)
        outAngle = 180 - outAngle;
}
double Calculate_Vector_Angle(Point L1stpnt, Point L1endpt, Point L2stpnt, Point L2endpt)
{
    //Calculating the Line Segment for L1 
    Point LineSeg1stpnt;
    LineSeg1stpnt.x = L1endpt.x - L1stpnt.x;
    LineSeg1stpnt.y = L1endpt.y - L1stpnt.y;

    //Calculating the Line Segment for L2
    Point LineSeg2stpnt;
    LineSeg2stpnt.x = L2endpt.x - L2stpnt.x;
    LineSeg2stpnt.y = L2endpt.y - L2stpnt.y;

    //calculate Dot Product 
    float DotProduct;
    DotProduct = LineSeg1stpnt.x * LineSeg2stpnt.x + LineSeg1stpnt.y * LineSeg2stpnt.y;

    //Calculate Magnitude of Vector 1 
    float MgnofLine1 = sqrtf((pow(LineSeg1stpnt.x, 2) + pow(LineSeg1stpnt.y, 2)));

    //Calculate Magnitude of Vector 1 
    float MgnofLine2 = sqrtf((pow(LineSeg2stpnt.x, 2) + pow(LineSeg2stpnt.y, 2)));
    //CosA = (Dot Product)/((Magnitude of Vector 1)*(Magnitude of Vector 2))
    double  cosA = DotProduct / (MgnofLine1 * MgnofLine2);
    //Finding Radian of  A from COSA 
    double  angleA = acos(cosA);
    //Converting Radian to Angle 
    double ang_deg = angleA * 180.0 / 3.14;

    return ang_deg;
}
bool getAnatomySegmentedMaskAndContour(int iInSegmentedFrameIndexOfAnatomy, std::string strInSegmentedLabelOfAnatomy, const int img_width_original, const int img_height_original, unsigned char* pInoriginalImage, unsigned char* pInFrameWithMaxPixelValueIndex, unsigned char* pOutFrameWithMaxPixelValueIndexOfAnatomy, std::vector<Point>& vecOutContourOfAnatomy)
{
    int SegmentedFrameIndexOfInnerThorax = 2;
    int SegmentedFrameIndexOfHeartContour = 3;

    std::string outSegmentedFrameFilePath = "";
    int imgoriginalSize = img_width_original * img_height_original;
    //Set any pixel value below max(255) to 0
    memset(pOutFrameWithMaxPixelValueIndexOfAnatomy, 0, imgoriginalSize);
    bool bIsChannelImagePresent = false;


    if (iInSegmentedFrameIndexOfAnatomy == SegmentedFrameIndexOfHeartContour) //specifically for Heart Contour
    {
        int SegmentedFrameIndexOfSpineTriangle = 8;
        int SegmentedFrameIndexOfSpineBody = 9;
        int SegmentedFrameIndexOfdAorta = 10;
        int SegmentedFrameIndexOfPV1 = 16;
        int SegmentedFrameIndexOfPV2 = 17;

        for (int i = 0; i < imgoriginalSize; i++)
        {
            if (pInFrameWithMaxPixelValueIndex[i] >= SegmentedFrameIndexOfHeartContour)
            {
                pOutFrameWithMaxPixelValueIndexOfAnatomy[i] = 255;

                if (pInFrameWithMaxPixelValueIndex[i] == SegmentedFrameIndexOfSpineBody ||
                    pInFrameWithMaxPixelValueIndex[i] == SegmentedFrameIndexOfSpineTriangle ||
                    pInFrameWithMaxPixelValueIndex[i] == SegmentedFrameIndexOfdAorta ||
                    pInFrameWithMaxPixelValueIndex[i] == SegmentedFrameIndexOfPV1 ||
                    pInFrameWithMaxPixelValueIndex[i] == SegmentedFrameIndexOfPV2)
                {
                    pOutFrameWithMaxPixelValueIndexOfAnatomy[i] = 0;
                }

                bIsChannelImagePresent = true;
            }
        }
    }
    else if (iInSegmentedFrameIndexOfAnatomy == SegmentedFrameIndexOfInnerThorax) //specifically for Inner thorax
    {
        int SegmentedFrameIndexOfOutterThorax = 1;
        int SegmentedFrameIndexOfSpineTriangle = 8;
        int SegmentedFrameIndexOfSpineBody = 9;

        for (int i = 0; i < imgoriginalSize; i++)
        {
            if (pInFrameWithMaxPixelValueIndex[i] == SegmentedFrameIndexOfOutterThorax || pInFrameWithMaxPixelValueIndex[i] == SegmentedFrameIndexOfSpineTriangle || pInFrameWithMaxPixelValueIndex[i] == SegmentedFrameIndexOfSpineBody)
            {
                pOutFrameWithMaxPixelValueIndexOfAnatomy[i] = 0;
            }
            else if (pInFrameWithMaxPixelValueIndex[i] >= SegmentedFrameIndexOfInnerThorax)
            {
                pOutFrameWithMaxPixelValueIndexOfAnatomy[i] = 255;
                bIsChannelImagePresent = true;
            }
        }
    }
    else
    {
        for (int i = 0; i < imgoriginalSize; i++)
        {
            if (pInFrameWithMaxPixelValueIndex[i] == iInSegmentedFrameIndexOfAnatomy)
            {
                pOutFrameWithMaxPixelValueIndexOfAnatomy[i] = 255;
                bIsChannelImagePresent = true;
            }
        }

    }

    if (!bIsChannelImagePresent)
    {
        return false;
    }

    //Save the  max pixel value containing each channel grayscale converted image
    //outSegmentedFrameFilePath = "CA_Maxed_SegmentedFrame_Channel-" + std::to_string(iInSegmentedFrameIndexOfAnatomy) + "_" + strInSegmentedLabelOfAnatomy + std::string(".pgm");
//	CPgm::write_uchar((char *)outSegmentedFrameFilePath.c_str(), pOutFrameWithMaxPixelValueIndexOfAnatomy, img_width_original, img_height_original, 0, 255);


    // detect the contours on the binary image using cv2.CHAIN_APPROX_SIMPLE
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    cv::Mat inputImgContour = cv::Mat(cv::Size(img_width_original, img_height_original), CV_8UC1, pOutFrameWithMaxPixelValueIndexOfAnatomy, cv::Mat::AUTO_STEP);
    cv::findContours(inputImgContour, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    int iIndexofSelectedContour = -1;

    bool bSelectContourBasesOnArea = true; //true: max area, false:max point on perimeter

    if (contours.size() == 0)
    {
        return false;
    }

    if (bSelectContourBasesOnArea)
    {
        double outMaxArea = 0;
        iIndexofSelectedContour = getMaxAreaContourId(contours, outMaxArea);
    }
    else
    {
        iIndexofSelectedContour = getMaxPerimeterLengthContourId(contours);
    }

    if (iIndexofSelectedContour != -1)
    {
        vector<vector<Point>> selectedContours;
        selectedContours.clear();

        selectedContours.push_back(contours[iIndexofSelectedContour]);
        vecOutContourOfAnatomy.clear();
        vecOutContourOfAnatomy = contours[iIndexofSelectedContour];




    }
    else
    {
        return false;
    }
    return true;
}
bool getAnatomySegmentedMaskAndContourforOuterThorax(const int img_width_original, const int img_height_original, unsigned char* pInoriginalImage, unsigned char* pInFrameWithMaxPixelValueIndex, unsigned char* pOutFrameWithMaxPixelValueIndexOfAnatomy, std::vector<Point>& vecOutContourOfAnatomy)
{
    std::string outSegmentedFrameFilePath = "";
    int imgoriginalSize = img_width_original * img_height_original;
    //Set any pixel value below max(255) to 0
    memset(pOutFrameWithMaxPixelValueIndexOfAnatomy, 0, imgoriginalSize);
    bool bIsChannelImagePresent = false;
    for (int iInSegmentedFrameIndexOfMaskAnatomy = 1; (iInSegmentedFrameIndexOfMaskAnatomy <= 17); iInSegmentedFrameIndexOfMaskAnatomy++)

        for (int i = 0; i < imgoriginalSize; i++)
        {
            if (pInFrameWithMaxPixelValueIndex[i] == iInSegmentedFrameIndexOfMaskAnatomy)
            {
                pOutFrameWithMaxPixelValueIndexOfAnatomy[i] = 255;
                bIsChannelImagePresent = true;
            }
        }

    if (!bIsChannelImagePresent)
    {
        return false;
    }

   
    // detect the contours on the binary image using cv2.CHAIN_APPROX_SIMPLE
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    cv::Mat inputImgContour = cv::Mat(cv::Size(img_width_original, img_height_original), CV_8UC1, pOutFrameWithMaxPixelValueIndexOfAnatomy, cv::Mat::AUTO_STEP);
    cv::findContours(inputImgContour, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    int iIndexofSelectedContour = -1;

    bool bSelectContourBasesOnArea = true; //true: max area, false:max point on perimeter

    if (contours.size() == 0)
    {
        return false;
    }

    if (bSelectContourBasesOnArea)
    {
        double outMaxArea = 0;
        iIndexofSelectedContour = getMaxAreaContourId(contours, outMaxArea);
    }
    else
    {
        iIndexofSelectedContour = getMaxPerimeterLengthContourId(contours);
    }

    if (iIndexofSelectedContour != -1)
    {
        vector<vector<Point>> selectedContours;
        selectedContours.clear();

        selectedContours.push_back(contours[iIndexofSelectedContour]);
        vecOutContourOfAnatomy.clear();
        vecOutContourOfAnatomy = contours[iIndexofSelectedContour];

    }
    else
    {
        return false;
    }
    return true;
}
bool getAnatomySegmentedMaskAndContourforPartialThoraxHDZoomed(const int img_width_original, const int img_height_original, unsigned char* pInoriginalImage, unsigned char* pInFrameWithMaxPixelValueIndex, unsigned char* pOutFrameWithMaxPixelValueIndexOfAnatomy, std::vector<Point>& vecOutContourOfAnatomy)
{
    std::string outSegmentedFrameFilePath = "";
    int imgoriginalSize = img_width_original * img_height_original;
    //Set any pixel value below max(255) to 0
    memset(pOutFrameWithMaxPixelValueIndexOfAnatomy, 0, imgoriginalSize);
    bool bIsChannelImagePresent = false;
    for (int iInSegmentedFrameIndexOfMaskAnatomy = 2; (iInSegmentedFrameIndexOfMaskAnatomy <= 17); iInSegmentedFrameIndexOfMaskAnatomy++)

        for (int i = 0; i < imgoriginalSize; i++)
        {
            if (pInFrameWithMaxPixelValueIndex[i] == iInSegmentedFrameIndexOfMaskAnatomy)
            {
                pOutFrameWithMaxPixelValueIndexOfAnatomy[i] = 255;
                bIsChannelImagePresent = true;
            }
        }

    if (!bIsChannelImagePresent)
    {
        return false;
    }


    // detect the contours on the binary image using cv2.CHAIN_APPROX_SIMPLE
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    cv::Mat inputImgContour = cv::Mat(cv::Size(img_width_original, img_height_original), CV_8UC1, pOutFrameWithMaxPixelValueIndexOfAnatomy, cv::Mat::AUTO_STEP);
    cv::findContours(inputImgContour, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    int iIndexofSelectedContour = -1;

    bool bSelectContourBasesOnArea = true; //true: max area, false:max point on perimeter

    if (contours.size() == 0)
    {
        return false;
    }

    if (bSelectContourBasesOnArea)
    {
        double outMaxArea = 0;
        iIndexofSelectedContour = getMaxAreaContourId(contours, outMaxArea);
    }
    else
    {
        iIndexofSelectedContour = getMaxPerimeterLengthContourId(contours);
    }

    if (iIndexofSelectedContour != -1)
    {
        vector<vector<Point>> selectedContours;
        selectedContours.clear();

        selectedContours.push_back(contours[iIndexofSelectedContour]);
        vecOutContourOfAnatomy.clear();
        vecOutContourOfAnatomy = contours[iIndexofSelectedContour];

    }
    else
    {
        return false;
    }
    return true;
}
bool checkPartialInnerThorax_CA_FanBeam(const int img_width_original, const int img_height_original, unsigned char* pInoriginalImage, std::string strInSegmentedLabelOfInnerThorax, unsigned char* pInFrameWithMaxPixelValueIndexOfNeededMask, std::vector<Point> vecInContourOfNeededMask)
{

    std::string  outSegmentedFrameFilePath;
    cv::Mat inputOriginalImg = cv::Mat(cv::Size(img_width_original, img_height_original), CV_8UC1, pInoriginalImage, cv::Mat::AUTO_STEP);
    cv::Mat image_copy = inputOriginalImg.clone();
    cvtColor(image_copy, image_copy, COLOR_GRAY2BGR, 3);
    cv::Mat inputimageColor = inputOriginalImg.clone();
    int imgoriginalSize = img_width_original * img_height_original;
    std::cout << "imgoriginalSize :: " << imgoriginalSize << std::endl;

    //Find the Pixel Values with less than 5 and make it 0 in the Original Input Image 
    unsigned char* pOutFrameWithMaxPixelValueIndexOfAnatomy = new unsigned char[imgoriginalSize];
    memset(pOutFrameWithMaxPixelValueIndexOfAnatomy, 0, imgoriginalSize);
    unsigned char* pInFrameWithMaxPixelValueIndex = (uint8_t*)inputOriginalImg.data;
    for (int i = 0; i < imgoriginalSize; i++)
    {
        if (pInFrameWithMaxPixelValueIndex[i] < 5)
        {
            pInFrameWithMaxPixelValueIndex[i] = 0;
        }
    }
    /*outSegmentedFrameFilePath = "CA_check" + std::string(".pgm");
    CPgm::write_uchar((char *)outSegmentedFrameFilePath.c_str(), pInFrameWithMaxPixelValueIndex, img_width_original, img_height_original, 0, 255);*/
    //Create a Kernel which will be used for Erroding 
    Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, Size(3, 3));
    Mat ErrodedImage;
    //Errosion 
    erode(inputimageColor, ErrodedImage, kernel, Point(-1, -1), 2);
    /*imwrite("Erroded_image.png", ErrodedImage);*/
    //Dilation
    Mat DilatedImage;
    dilate(ErrodedImage, DilatedImage, kernel);
    /*imwrite("Dilated_image.png", DilatedImage);*/
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    //cv::Mat inputImgContour = cv::Mat(cv::Size(img_width_original, img_height_original), CV_8UC1, pOutFrameWithMaxPixelValueIndexOfAnatomy, cv::Mat::AUTO_STEP);
    //Draw Contours on top of the Dilated Image 
    cv::findContours(DilatedImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<vector<Point>> selectedContours;
    selectedContours.clear();
    selectedContours.push_back(vecInContourOfNeededMask);
    //drawContours(inputimageColor, contours, -1, Scalar(186, 117, 255), 1);
    drawContours(DilatedImage, contours, -1, Scalar(186, 117, 255), 1);
    //imwrite("Contour_image1.png", inputimageColor);
    //imwrite("Contour_image2.png", image_copy);
    bool bSelectContourBasesOnArea = true;
    int iIndexofSelectedContour = -1;
    if (bSelectContourBasesOnArea)
    {
        double outMaxArea = 0;
        iIndexofSelectedContour = getMaxAreaContourId(contours, outMaxArea);
    }
    else
    {
        iIndexofSelectedContour = getMaxPerimeterLengthContourId(contours);
    }
    if (iIndexofSelectedContour != -1)
    {
        //Find the Moments and Centroid of all the island (which was formed from different Contours , we got from input image )
        vector<Moments> mu(contours.size());
        vector<Point> CentroidCollection;
        for (size_t i = 0; i < contours.size(); i++)
        {
            int cx = 0;
            int cy = 0;
            mu[i] = moments(contours[i]);
            if (mu[i].m00 != 0)
            {
                cx = int(mu[i].m10 / mu[i].m00);
                cy = int(mu[i].m01 / mu[i].m00);
                CentroidCollection.push_back(Point(cx, cy));
            }

        }
        cv::Mat LinesDrawnOnDilatedImage = DilatedImage.clone();
        //Draw a line from the Biggest Contour Island to all other Island Contour
        for (size_t i = 0; i < CentroidCollection.size(); i++)
        {
            Point StartingCentroid = CentroidCollection[iIndexofSelectedContour];
            line(LinesDrawnOnDilatedImage, StartingCentroid, CentroidCollection[i], Scalar(255, 0, 0), 1, LINE_8);
        }
        // imwrite("Lines_On_Dilated.png", LinesDrawnOnDilatedImage);
         //Find Contour on the Dilated image with Lines Drawn on it (Lines Connecting the Different Island )
        vector<vector<Point>>  DilatedImagecontours;
        cv::findContours(LinesDrawnOnDilatedImage, DilatedImagecontours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        // imwrite("Centroid_On_Dilated.png", LinesDrawnOnDilatedImage);
        iIndexofSelectedContour = -1;
        if (bSelectContourBasesOnArea)
        {
            double outMaxArea = 0;
            iIndexofSelectedContour = getMaxAreaContourId(DilatedImagecontours, outMaxArea);
        }
        else
        {
            iIndexofSelectedContour = getMaxPerimeterLengthContourId(DilatedImagecontours);
        }
        if (iIndexofSelectedContour != -1)
        {
            // create hull array for convex hull points based on DIlatedImages Contours
            vector< vector<Point> > hulls(DilatedImagecontours.size());
            for (size_t i = 0; i < DilatedImagecontours.size(); i++)
                cv::convexHull(Mat(DilatedImagecontours[i]), hulls[i]); //for Every DilatedContour We can Draw the Hull
            int iIndexofSelectedHull = -1;
            bool bSelectHullsBasesOnArea = true; //true: max area, false:max point on hulls perimeter

            if (hulls.size() == 0)
            {
                return false;
            }

            if (bSelectHullsBasesOnArea)
            {
                double outMaxArea = 0;
                iIndexofSelectedHull = getMaxAreaContourId(hulls, outMaxArea);
            }
            else
            {
                iIndexofSelectedHull = getMaxPerimeterLengthContourId(hulls);
            }
            if (iIndexofSelectedHull != -1)
            {

                bool bIsChannelImagePresent = false;

                unsigned char* pOutFrameWithMaxPixelValueTemp = new unsigned char[imgoriginalSize];
                memset(pOutFrameWithMaxPixelValueTemp, 0, imgoriginalSize);
                for (int i = 0; i < imgoriginalSize; i++)
                {
                    if (pInFrameWithMaxPixelValueIndexOfNeededMask[i] >= 1)
                    {
                        pOutFrameWithMaxPixelValueTemp[i] = 255;
                        bIsChannelImagePresent = true;
                    }
                }
                cv::Mat OuterThoraxMask = cv::Mat(cv::Size(img_width_original, img_height_original), CV_8UC1, pOutFrameWithMaxPixelValueTemp, cv::Mat::AUTO_STEP);
                imwrite("outerThoraxMask.png", OuterThoraxMask);
                cv::Mat BlankHull = Mat::zeros(cv::Size(img_width_original, img_height_original), CV_8UC1);
                vector< vector<Point> > LargestHull;
                LargestHull.push_back(hulls[iIndexofSelectedHull]);
                //	for(size_t j=0;j<hulls.size();j++)
                //drawContours(BlankHull, hulls, -1, Scalar(186, 117, 255), 1);
            //	imwrite("HullContour.png", BlankHull);

                drawContours(BlankHull, LargestHull, -1, Scalar(186, 117, 255), 5);
                // imwrite("HullContour_Final.png", BlankHull);


                cv::Mat imgFanbeamHullIntersectionWithInnerThoraxContour = Mat::zeros(cv::Size(img_width_original, img_height_original), CV_8UC1);
                bitwise_and(OuterThoraxMask, BlankHull, imgFanbeamHullIntersectionWithInnerThoraxContour);
                imwrite("Partial_Thorax_Region.png", imgFanbeamHullIntersectionWithInnerThoraxContour);
                if (cv::countNonZero(imgFanbeamHullIntersectionWithInnerThoraxContour) != 0)
                {
                    std::cout << " ***** Found Partial Inner Thorax : Inner Thorax Contour intersects with Image Fanbeam" << std::endl;
                    delete[]pOutFrameWithMaxPixelValueTemp;
                    delete[]pOutFrameWithMaxPixelValueIndexOfAnatomy;
                    return false;
                }
                delete[]pOutFrameWithMaxPixelValueTemp;
            }
            else
            {
                delete[]pOutFrameWithMaxPixelValueIndexOfAnatomy;
                return false;
            }

        }
        else
        {
            delete[]pOutFrameWithMaxPixelValueIndexOfAnatomy;
            return false;
        }
    }
    else
    {
        delete[]pOutFrameWithMaxPixelValueIndexOfAnatomy;
        return false;
    }

    //delete[] pInFrameWithMaxPixelValueIndex;
    delete[]pOutFrameWithMaxPixelValueIndexOfAnatomy;
    return true;
}
bool checkPartialInnerThorax_CA_HDZoomed(const int img_width_original, const int img_height_original, unsigned char* pInoriginalImage, std::string strInSegmentedLabelOfInnerThorax, unsigned char* pFrameWithMaxPixelValueIndexOfNeededMask, std::vector<Point> vecInContourOfNeededMask)
{
    //
    std::string  outSegmentedFrameFilePath;
    cv::Mat inputOriginalImg = cv::Mat(cv::Size(img_width_original, img_height_original), CV_8UC1, pInoriginalImage, cv::Mat::AUTO_STEP);
    cv::Mat image_copy = inputOriginalImg.clone();
    cvtColor(image_copy, image_copy, COLOR_GRAY2BGR, 3);

    vector<vector<Point>> selectedContours;
    selectedContours.clear();
    selectedContours.push_back(vecInContourOfNeededMask);
    cv::Mat imgBlankWithBoundaryLine = Mat::zeros(cv::Size(img_width_original, img_height_original), CV_8UC1);
    cv::line(imgBlankWithBoundaryLine, Point(0, 0), Point(img_width_original - 1, 0), cv::Scalar(255, 255, 255, 255), 1, 8, false);
    cv::line(imgBlankWithBoundaryLine, Point(img_width_original - 1, 0), Point(img_width_original - 1, img_height_original - 1), cv::Scalar(255, 255, 255, 255), 1, 8, false);
    cv::line(imgBlankWithBoundaryLine, Point(img_width_original - 1, img_height_original - 1), Point(0, img_height_original - 1), cv::Scalar(255, 255, 255, 255), 1, 8, false);
    cv::line(imgBlankWithBoundaryLine, Point(0, img_height_original - 1), Point(0, 0), cv::Scalar(255, 255, 255, 255), 1, 8, false);

    cv::line(image_copy, Point(0, 0), Point(img_width_original - 1, 0), cv::Scalar(255, 255, 255, 255), 1, 8, false);
    cv::line(image_copy, Point(img_width_original - 1, 0), Point(img_width_original - 1, img_height_original - 1), cv::Scalar(255, 255, 255, 255), 1, 8, false);
    cv::line(image_copy, Point(img_width_original - 1, img_height_original - 1), Point(0, img_height_original - 1), cv::Scalar(255, 255, 255, 255), 1, 8, false);
    cv::line(image_copy, Point(0, img_height_original - 1), Point(0, 0), cv::Scalar(255, 255, 255, 255), 1, 8, false);

    //  cv::imwrite(" CA_image_copy.png ", image_copy);

    cv::Mat imgBlankWithInnerThoraxContour = Mat::zeros(cv::Size(img_width_original, img_height_original), CV_8UC1);
    drawContours(imgBlankWithInnerThoraxContour, selectedContours, -1, cv::Scalar(255, 255, 255, 255), 1);

    cv::Mat imgBoundaryLineIntersectionWithInnerThoraxContour = Mat::zeros(cv::Size(img_width_original, img_height_original), CV_8UC1);
    bitwise_and(imgBlankWithBoundaryLine, imgBlankWithInnerThoraxContour, imgBoundaryLineIntersectionWithInnerThoraxContour);
    // cv::imwrite(" CA_boundryline.png ", imgBlankWithBoundaryLine);
    // cv::imwrite(" CA_Contour.png ", imgBlankWithInnerThoraxContour);


    if (cv::countNonZero(imgBoundaryLineIntersectionWithInnerThoraxContour) != 0)
    {
        std::cout << " ***** Found Partial Inner Thorax : Inner Thorax Contour intersects with Image Boundary" << std::endl;
        return false;
    }

    return true;
}
STATUS_INFO Choosing_Axis_With_Respectto_IVS_angle(vector <int> AnatomyIndex, vector <string> AnatomyString, const int img_width_original, const int img_height_original, unsigned char* pInoriginalImage, unsigned char* pInFrameWithMaxPixelValueIndex, unsigned char* pOutFrameWithMaxPixelValueIndexOfAnatomy, std::vector<Point>& vecOutContourOfAnatomy, Point ptEllipseMajorAxisLineStart, Point ptEllipseMajorAxisLineStop, Point ptEllipseMinorAxisLineStart, Point ptEllipseMinorAxisLineStop, Point ptIVSStartPoint, Point ptIVSEndPoint, bool isArgmax)
{
    Level2_counter++;
    //we need not to set to 0 , because we need to store multiple anatomy values in single image and find the countor of all anatomy clubbed together
    int imgoriginalSize = img_width_original * img_height_original;
    memset(pOutFrameWithMaxPixelValueIndexOfAnatomy, 0, imgoriginalSize);
    std::cout << "----------------------------IVS QC  Choosing_Axis_With_Respectto_IVS_angle------------------------------" << std::endl;

    for (size_t j = 0; j < AnatomyIndex.size(); j++)
    {
        int iInSegmentedFrameIndexOfAnatomy = AnatomyIndex[j];
        string strInSegmentedLabelOfAnatomy = AnatomyString[j];
        std::cout << "iInSegmentedFrameIndexOfAnatomy :: " << iInSegmentedFrameIndexOfAnatomy << std::endl;

        bool bIsChannelImagePresent = false;

        for (int i = 0; i < imgoriginalSize; i++)
        {

            if (pInFrameWithMaxPixelValueIndex[i] == iInSegmentedFrameIndexOfAnatomy)
            {
                pOutFrameWithMaxPixelValueIndexOfAnatomy[i] = 255;
                bIsChannelImagePresent = true;

            }

        }

    }
    cv::Mat inputOriginalImg = cv::Mat(cv::Size(img_width_original, img_height_original), CV_8UC1, pInoriginalImage, cv::Mat::AUTO_STEP);
    //1. major axis , minor axis and ivs line start and end point are passed via function call

    std::string outSegmentedFrameFilePath = "";

    //Save the  max pixel value containing each channel grayscale converted image
    //outSegmentedFrameFilePath = "Level-" + std::to_string(Level2_counter) + std::string("-Clubbed-") + std::string(".pgm");
    //CPgm::write_uchar((char *)outSegmentedFrameFilePath.c_str(), pOutFrameWithMaxPixelValueIndexOfAnatomy, img_width_original, img_height_original, 0, 255);




    //2. detect the contours on the binary image using cv2.CHAIN_APPROX_SIMPLE

    cv::Mat inputImgContour = cv::Mat(cv::Size(img_width_original, img_height_original), CV_8UC1, pOutFrameWithMaxPixelValueIndexOfAnatomy, cv::Mat::AUTO_STEP);
    cv::Mat final_contour = inputImgContour.clone();
    //   string final_contour_IN = "IVSQC_Antomy_Contour" + string("Level-") + std::to_string(Level2_counter) + string(".png");
     //  imwrite(final_contour_IN, final_contour);
    cv::Mat morphology = final_contour.clone();
    final_contour.release();
    Mat Post_Morphology_operation;

    //Morphing the anatomy , to combine all anatomy and form a big contour 
    Mat element = Mat(20, 20, CV_8UC1, cv::Scalar(1)); // 20*20 matrix with all pixel 1 
    morphologyEx(morphology, Post_Morphology_operation, cv::MORPH_CLOSE, element);
    morphology.release();
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    cv::findContours(Post_Morphology_operation, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    drawContours(Post_Morphology_operation, contours, -1, Scalar(0, 255, 0), 2);
    //string Final_Morp_Output = "IVSQC_Final_Morphed_contour" + string("Level-") + std::to_string(Level2_counter) + string(".png");
    //imwrite(Final_Morp_Output, Post_Morphology_operation);


    int iIndexofSelectedContour = -1;

    bool bSelectContourBasesOnArea = true; //true: max area, false:max point on perimeter

    if (contours.size() == 0)
    {
        return IVS_FAILED;
    }

    if (bSelectContourBasesOnArea)
    {
        double outMaxArea = 0;
        iIndexofSelectedContour = getMaxAreaContourId(contours, outMaxArea);
    }
    else
    {
        iIndexofSelectedContour = getMaxPerimeterLengthContourId(contours);
    }

    if (iIndexofSelectedContour != -1)
    {


        //3. calculate fitline through all the anatomies clubbed countor 
        cv::Vec4f line4f;
        cv::fitLine(Mat(contours[iIndexofSelectedContour]), line4f, cv::DIST_L2, 0, 0.01, 0.01);


        float vx = line4f[0];
        float vy = line4f[1];
        float x = line4f[2];
        float y = line4f[3];


        //x and y is point on the fitline 
        //what vx and vy ? Ans: x+ = x+vx*distance, y+ = y+vy*distance , x- = x-vx*distance, y- = y-vy*distance
        int lefty = (-x * vy / vx) + y;
        int righty = ((img_width_original - x) * vy / vx) + y;

        Point pt1 = Point(img_width_original - 1, righty);
        Point pt2 = Point(0, lefty);

        cv::line(Post_Morphology_operation, pt1, pt2, cv::Scalar(255, 0, 0), 1); //Line passing through all the given anatomy 
    //    string namee = "IVSQC_Final_Morphed_contour_with_Line" + string("Level-") + std::to_string(Level2_counter) + string(".png");
      //  imwrite(namee, Post_Morphology_operation);
        Post_Morphology_operation.release();


        //WE HAVE  MAJOR ,MINOR ,LINE PASSING THROUGH THE ANATOMIES (FROM THE INPUT VECTOR) , WE NEED TO DECIDE WHICH AXIS TO CHOOSE 
        //4 CHOOSING FINAL AXIS (BTWN MAJOR AXIS AND MINOR AXIS and Calculating the angle made with anatomy line and major and minor axis 

        //4.1.1 find intersection between MajorAxis of ellipse and Line passing through the given anatomy
        Point MajorAxisIntersectionClubbedAnatomyLine;
        MajorAxisIntersectionClubbedAnatomyLine = lineLineIntersection(ptEllipseMajorAxisLineStart, ptEllipseMajorAxisLineStop, pt1, pt2);

        if (MajorAxisIntersectionClubbedAnatomyLine.x == INT_MAX && MajorAxisIntersectionClubbedAnatomyLine.y == INT_MAX)
        {
            return IVS_FAILED;
        }
        std::cout << "-------------intersection between MajorAxis of ellipse and Line passing through the given anatomy FOUND-------------" << std::endl;
        //4.1.2 Find the angle between the Major Axis and IVS line 
        float ClubbedAnatomy_MAJOR_angle = 0;
        calculate_angle_for_line_intersection(MajorAxisIntersectionClubbedAnatomyLine, ptEllipseMajorAxisLineStop, pt2, ClubbedAnatomy_MAJOR_angle);
        //ClubbedAnatomy_MAJOR_angle = 180 - ClubbedAnatomy_MAJOR_angle;
        std::cout << " ClubbedAnatomy_MAJOR_angle ::" << ClubbedAnatomy_MAJOR_angle << std::endl;

        //4.1.3 find intersection between Minor Axis of ellipse and Line passing through the given anatomy
        Point MinorAxisIntersectionClubbedAnatomyLine;
        MinorAxisIntersectionClubbedAnatomyLine = lineLineIntersection(ptEllipseMinorAxisLineStart, ptEllipseMinorAxisLineStop, pt1, pt2);

        if (MinorAxisIntersectionClubbedAnatomyLine.x == INT_MAX && MinorAxisIntersectionClubbedAnatomyLine.y == INT_MAX)
        {
            return IVS_FAILED;
        }
        std::cout << "-------------intersection between Minor Axis of ellipse and Line passing through the given anatomy FOUND-------------" << std::endl;
        //4.1.4 Find the angle between the Minor Axis and IVS line 
        float ClubbedAnatomy_MINOR_angle = 0;
        calculate_angle_for_line_intersection(MinorAxisIntersectionClubbedAnatomyLine, ptEllipseMinorAxisLineStop, pt2, ClubbedAnatomy_MINOR_angle);
        //ClubbedAnatomy_MINOR_angle = 180 - ClubbedAnatomy_MINOR_angle;
        std::cout << " ClubbedAnatomy_MINOR_angle ::" << ClubbedAnatomy_MINOR_angle << std::endl;

        //std::string name1 = "MAJOR_MINOR_AXIS_CLUBBED_ANATOMY_LINE_IVSLINE_" + std::to_string(Level2_counter) + std::string(".png");
        //cv::Mat minor_major_axis1 = inputOriginalImg.clone();
        //cv::line(minor_major_axis1, ptEllipseMajorAxisLineStart, ptEllipseMajorAxisLineStop, cv::Scalar(255, 0, 0), 1, LINE_AA); //Major AXIS
        //cv::line(minor_major_axis1, ptEllipseMinorAxisLineStart, ptEllipseMinorAxisLineStop, cv::Scalar(255, 0, 0), 1, LINE_AA); //Minor Axis 
        //cv::line(minor_major_axis1, pt1, pt2, cv::Scalar(255, 0, 0), 1); //Line passing through all the given anatomy 
        //cv::line(minor_major_axis1, ptIVSStartPoint, ptIVSEndPoint, cv::Scalar(255, 255, 0), 1); //IVS Line
        //imwrite(name1, minor_major_axis1);


        //4.2 Find which axis to choose as final axis based on a condition
        //4.2.1 Major Axis is the final Axis 
        if ((MajorMinorAngleCmp(ClubbedAnatomy_MAJOR_angle, ClubbedAnatomy_MINOR_angle, isArgmax)) && (AngleCmpwith45(ClubbedAnatomy_MAJOR_angle, isArgmax)))
        {
            std::cout << "--------------- MAJOR AXIS IS THE FINAL AXIS ----------------- " << std::endl;

            //4.2.1 find intersection between Major Axis  and IVS Line
            Point MajorAxisIntersectionIVSLine;
            MajorAxisIntersectionIVSLine = lineLineIntersection(ptEllipseMajorAxisLineStart, ptEllipseMajorAxisLineStop, ptIVSStartPoint, ptIVSEndPoint);

            if (MajorAxisIntersectionIVSLine.x == INT_MAX && MajorAxisIntersectionIVSLine.y == INT_MAX)
            {
                return IVS_FAILED;
            }
            //4.2.2 Find the angle between the Minor Axis and IVS line 
            float MAJOR_IVS_angle = 0;
            calculate_angle_for_line_intersection(MajorAxisIntersectionIVSLine, ptEllipseMajorAxisLineStop, ptIVSEndPoint, MAJOR_IVS_angle);
            std::cout << " MAJOR_IVS_angle :: " << MAJOR_IVS_angle << std::endl;
            // std::string file = "MAJOR_IVS_LINE_INTERSECTION_" + std::to_string(Level2_counter) + std::string(".png");
            cv::Mat Major_IVS = inputOriginalImg.clone();
            cv::line(Major_IVS, ptEllipseMajorAxisLineStart, ptEllipseMajorAxisLineStop, cv::Scalar(255, 0, 0), 1, LINE_AA); //Major AXIS			
            cv::line(Major_IVS, ptIVSStartPoint, ptIVSEndPoint, cv::Scalar(255, 255, 0), 1); //IVS Line
            cv::line(Major_IVS, MajorAxisIntersectionIVSLine, MajorAxisIntersectionIVSLine, cv::Scalar(255, 0, 0), 5); //Line passing through all the given anatomy 
           // imwrite(file, Major_IVS);
            Major_IVS.release();

            if (MAJOR_IVS_angle < 40)
            {
                std::cout << " -----------------( MAJOR_IVS_angle < 40 ) Condition Passed -----------------" << std::endl;
                return VALID_ANGLE;
            }
            else
            {
                std::cout << " -----------------( MAJOR_IVS_angle < 40 ) Condition Failed -----------------" << std::endl;
                return NEXT_LEVEL_CHECK;
            }


        }
        //4.2.2 Minor Axis is the Final axis

        else if ((MajorMinorAngleCmp(ClubbedAnatomy_MINOR_angle, ClubbedAnatomy_MAJOR_angle, isArgmax)) && (AngleCmpwith45(ClubbedAnatomy_MINOR_angle, isArgmax)))
        {
            std::cout << " ----------------- MINOR AXIS IS THE FINAL AXIS -----------------" << std::endl;
            //4.2.1 find intersection between Minor Axis of ellipse and Line passing through the given anatomy
            Point MinorAxisIntersectionClubbedIVSLine;
            MinorAxisIntersectionClubbedIVSLine = lineLineIntersection(ptEllipseMinorAxisLineStart, ptEllipseMinorAxisLineStop, ptIVSStartPoint, ptIVSEndPoint);

            if (MinorAxisIntersectionClubbedIVSLine.x == INT_MAX && MinorAxisIntersectionClubbedIVSLine.y == INT_MAX)
            {
                return IVS_FAILED;
            }
            //4.2.2 Find the angle between the Minor Axis and IVS line 
            float MINOR_IVS_angle = 0;
            calculate_angle_for_line_intersection(MinorAxisIntersectionClubbedIVSLine, ptEllipseMinorAxisLineStop, ptIVSEndPoint, MINOR_IVS_angle);
            std::cout << " MINOR_IVS_angle :: " << MINOR_IVS_angle << std::endl;
            // std::string file = "MINOR_IVS_LINE_INTERSECTION_" + std::to_string(Level2_counter) + std::string(".png");
            cv::Mat Major_IVS = inputOriginalImg.clone();
            cv::line(Major_IVS, ptEllipseMinorAxisLineStart, ptEllipseMinorAxisLineStop, cv::Scalar(255, 0, 0), 1, LINE_AA); //Minor Axis 		
            cv::line(Major_IVS, ptIVSStartPoint, ptIVSEndPoint, cv::Scalar(255, 255, 0), 1); //IVS Line
            cv::line(Major_IVS, MinorAxisIntersectionClubbedIVSLine, MinorAxisIntersectionClubbedIVSLine, cv::Scalar(255, 0, 0), 5); //Line passing through all the given anatomy 
          //  imwrite(file, Major_IVS);
            Major_IVS.release();

            if (MINOR_IVS_angle < 40)
            {
                std::cout << " -----------------( MINOR_IVS_angle < 40 ) Condition Passed -----------------" << std::endl;
                return VALID_ANGLE;
            }
            else
            {
                std::cout << " -----------------( MINOR_IVS_angle < 40 ) Condition Failed -----------------" << std::endl;
                return NEXT_LEVEL_CHECK;
            }

        }
        //4.2.3 IVS Check failed 
        else
        {
            std::cout << " ----------------- MAJOR AND MINOR AXIS LINE ANGLE WITH IVS FAILED  -----------------" << std::endl;
            return IVS_FAILED;
        }
    }
    inputOriginalImg.release();
    return VALID_ANGLE;
}


//Function to check QC of segmented IVS anatomy contour
bool checkIVS_CA(const int img_width_original, const int img_height_original, unsigned char* pInoriginalImage, std::string strInSegmentedLabelOfHeartContour, unsigned char* pInFrameWithMaxPixelValueIndexOfHeartContour, std::vector<Point> vecInContourOfHeartContour, std::string strInSegmentedLabelOfIVS, unsigned char* pInFrameWithMaxPixelValueIndex, unsigned char* pInFrameWithMaxPixelValueIndexOfIVS, std::vector<Point> vecInContourOfIVS)
{
    cv::Mat inputOriginalImg = cv::Mat(cv::Size(img_width_original, img_height_original), CV_8UC1, pInoriginalImage, cv::Mat::AUTO_STEP);
    //int SegmentedFrameIndexOfHeartContour = 3;
    //int SegmentedFrameIndexOfIVS = 11;

    ////Save the  max pixel value containing each channel grayscale converted image
    //std::string  segmentedFrameFilePath = "CA_IVS_QC_Maxed_SegmentedFrame_Channel-" + std::to_string(SegmentedFrameIndexOfHeartContour) + "_" + strInSegmentedLabelOfHeartContour + std::string(".pgm");
    //CPgm::write_uchar((char *)segmentedFrameFilePath.c_str(), pInFrameWithMaxPixelValueIndexOfHeartContour, img_width_original, img_height_original, 0, 255);
    //segmentedFrameFilePath = "CA_IVS_QC_Maxed_SegmentedFrame_Channel-" + std::to_string(SegmentedFrameIndexOfIVS) + "_" + strInSegmentedLabelOfIVS + std::string(".pgm");
    //CPgm::write_uchar((char *)segmentedFrameFilePath.c_str(), pInFrameWithMaxPixelValueIndexOfIVS, img_width_original, img_height_original, 0, 255);

    //check if ellipse is close to ellipse or circle
    bool bIsHeartIsEllipse = false;
    RotatedRect ellipseofHeartContour;
    ellipseofHeartContour = fitEllipse(vecInContourOfHeartContour);


    ////Draw Heart Ellipse Major Axis line on input image 
    //std::string  outFrameFilePath = "CA_IVS_QC_Heart_Ellipse.png";
    //cv::Mat image_ellipse = inputOriginalImg.clone();
    //cv::ellipse(image_ellipse, ellipseofHeartContour, cv::Scalar(255, 255, 255, 255), 1);
    //imwrite(outFrameFilePath, image_ellipse);

    //ellipse parameters
    float x_center = ellipseofHeartContour.center.x;
    float y_center = ellipseofHeartContour.center.y;
    float majorAxis = ellipseofHeartContour.size.width;    // width >= height
    float minorAxis = ellipseofHeartContour.size.height;
    float angle_theta = ellipseofHeartContour.angle;       // in degrees

    std::cout << x_center << "," << y_center << std::endl;
    std::cout << majorAxis << "," << minorAxis << std::endl;
    std::cout << angle_theta << std::endl;


    //ratio of minorAxis & majorAxis
    float ratioMinorMajorAxis = min(majorAxis, minorAxis) / max(majorAxis, minorAxis);

    if (ratioMinorMajorAxis < IVS_QC_HEART_IS_ELLIPSE_THRESHOLD)
        bIsHeartIsEllipse = true;

    cv::Vec4f line4f;
    cv::fitLine(Mat(vecInContourOfIVS), line4f, cv::DIST_L2, 0, 0.01, 0.01);
    float vx = line4f[0];
    float vy = line4f[1];
    float x = line4f[2];
    float y = line4f[3];


    //x and y is point on the fitline 
    //what vx and vy ? Ans: x+ = x+vx*distance, y+ = y+vy*distance , x- = x-vx*distance, y- = y-vy*distance
    int lefty = (-x * vy / vx) + y;
    int righty = ((img_width_original - x) * vy / vx) + y;

    Point ptIVSStartPoint = Point(img_width_original - 1, righty);
    Point ptIVSEndPoint = Point(0, lefty);
    //if heart contour's fitEllipse boundary is of
    //LEVEL 1 CASE  
    if (bIsHeartIsEllipse) // ellipse shape
    {
        cout << " -----------------------------------------------HEART CONTOUR IS OF ELLIPSE SHAPE ----------------------------------------" << endl;
        Point ptEllipseMajorAxisLineStart, ptEllipseMajorAxisLineStop;
        bool bRet = get_fin_coords(x_center, y_center, angle_theta + 90, img_width_original, img_height_original, ptEllipseMajorAxisLineStart, ptEllipseMajorAxisLineStop);

        // USING THE BELOW CODE TO CHECK FOR THE ROTATED RECTANGLE ,MAJOR AND MINOR AXIS 
        //std::string  name = "ROTATED_RECTANGLE_AXIS.png";
        //cv::Mat minor_major_axis = inputOriginalImg.clone();
        //cv::line(minor_major_axis, ptEllipseMajorAxisLineStart, ptEllipseMajorAxisLineStop, cv::Scalar(255, 0, 0), 1, LINE_AA); //POINT OF INTERSECTION
        //cv::line(minor_major_axis, ptEllipseMinorAxisLineStart, ptEllipseMinorAxisLineStop, cv::Scalar(255, 0, 0), 1, LINE_AA); //POINT OF INTERSECTION
        //Point2f vertices[4];
        //ellipseofHeartContour.points(vertices);
        //for (int i = 0; i < 4; i++)
        //	line(minor_major_axis, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2);
        //imwrite(name, minor_major_axis);
        //1. calculate fitline through IVS anatomy


        if (bRet)
        {
            //2. find intersection between MajorAxis of ellipse and IVS fitline 
            Point MajorAxisIntersectionIVSLine;
            MajorAxisIntersectionIVSLine = lineLineIntersection(ptEllipseMajorAxisLineStart, ptEllipseMajorAxisLineStop, ptIVSStartPoint, ptIVSEndPoint);

            if (MajorAxisIntersectionIVSLine.x == INT_MAX && MajorAxisIntersectionIVSLine.y == INT_MAX)
            {
                return false;
            }
            //3. Find the angle between the Major Axis and IVS line 
            float IVS_MAJOR_angle = 0;
            calculate_angle_for_line_intersection(MajorAxisIntersectionIVSLine, ptEllipseMajorAxisLineStop, ptIVSEndPoint, IVS_MAJOR_angle);
            //Draw Heart Ellipse Major Axis line on input image
            cout << " -----------------------------------------------IVS QC LEVEL 1 ENTERED----------------------------------------" << endl;
            // std::string  segmentedFrameFilePath = "CA_IVS_QC_Heart_Ellipse_MajorAxis.png";
            cv::Mat image_major_axis = inputOriginalImg.clone();
            cv::line(image_major_axis, ptEllipseMajorAxisLineStart, ptEllipseMajorAxisLineStop, cv::Scalar(255, 0, 0), 5, LINE_AA); //MAJOR AXIS LINE
            cv::line(image_major_axis, ptIVSStartPoint, ptIVSEndPoint, cv::Scalar(255, 0, 0), 5, LINE_AA); // IVS LINE
            cv::line(image_major_axis, MajorAxisIntersectionIVSLine, MajorAxisIntersectionIVSLine, cv::Scalar(255, 0, 0), 5, LINE_AA); //POINT OF INTERSECTION
            //imwrite(segmentedFrameFilePath, image_major_axis);
            image_major_axis.release();
            //DEL
            if (IVS_MAJOR_angle < 40)
            {
                cout << " -------------------IVS_MAJOR_ANGLE < 40 ::  SUCCESS ----------------------" << endl;
                return true;
            }
            else
            {
                cout << " -------------------IVS_MAJOR_ANGLE > 40 ::  FAILED ----------------------" << endl;
                return false;
            }
        }
        else
        {
            std::cout << "IVS QC Check Failed.Reason: Heart contour ellipse's major axis line end points computation failed." << std::endl;
            return false;
        }
    }
    else //not ellipse shape  -- Level 2 Quality Check Begins 
    {
        cout << " -----------------------------------------------HEART CONTOUR IS APPROX CIRCLE SHAPE ----------------------------------------" << endl;
        //common variables 
        int imgoriginalSize = img_width_original * img_height_original;

        std::vector<Point> vecContourOfAnatomyinVector;
        Point ptEllipseMajorAxisLineStart, ptEllipseMajorAxisLineStop;
        bool bRet = get_fin_coords(x_center, y_center, angle_theta + 90, img_width_original, img_height_original, ptEllipseMajorAxisLineStart, ptEllipseMajorAxisLineStop);

        Point ptEllipseMinorAxisLineStart, ptEllipseMinorAxisLineStop;
        bool bRet1 = get_fin_coords(x_center, y_center, angle_theta, img_width_original, img_height_original, ptEllipseMinorAxisLineStart, ptEllipseMinorAxisLineStop);


        if (!bRet1 || !bRet)  // if we area unable to find the points , we return error 
        {
            std::cout << "IVS QC Check Failed.Reason: Heart contour ellipse's Major/Minor axis line end points computation failed." << std::endl;
            return false;
        }
        STATUS_INFO  Return_enum;
        // 1.Case 1 (ArgMax{Angle(axis, {4,5,12,13,14,15})},ivs)  
        vector<int>AnatomyLabelLevel2Case1 = { 4,5,12,13,14,15 };
        vector<string>AnatomyLabelstringLevel2Case1 = { "RA","LA","AS" ,"AVS" ,"MV" ,"TV" };
        //2. Case 2	(ArgMax{ Angle(axis,{ 6,7,14,15 }) }, ivs)
        vector<int>AnatomyLabelLevel2Case2 = { 6,7,14,15 };
        vector<string>AnatomyLabelstringLevel2Case2 = { "RV","LV","MV" ,"TV" };
        //3. Case 3 (ArgMin{ Angle(axis,{ 4,6,15 }) }, ivs)
        vector<int>AnatomyLabelLevel2Case3 = { 4,6,15, };
        vector<string>AnatomyLabelstringLevel2Case3 = { "RA" ,"RV","TV" };
        //4. Case 4	(ArgMin{ Angle(axis,{ 5,7,14 }) }, ivs)
        vector<int>AnatomyLabelLevel2Case4 = { 5,7,14, };
        vector<string>AnatomyLabelstringLevel2Case4 = { "LA","LV","MV" };

        bool isArgmax = true;
        unsigned char* pFrameWithMaxPixelValueIndexOfAnatomyinVector = new unsigned char[imgoriginalSize];
        //Level 2 Case 1 Check 
        cout << " -----------------------------------------------IVS QC LEVEL 2 CASE 1 ENTERED----------------------------------------" << endl;
        Return_enum = Choosing_Axis_With_Respectto_IVS_angle(AnatomyLabelLevel2Case1, AnatomyLabelstringLevel2Case1, img_width_original, img_height_original, pInoriginalImage, pInFrameWithMaxPixelValueIndex, pFrameWithMaxPixelValueIndexOfAnatomyinVector, vecContourOfAnatomyinVector, ptEllipseMajorAxisLineStart, ptEllipseMajorAxisLineStop, ptEllipseMinorAxisLineStart, ptEllipseMinorAxisLineStop, ptIVSStartPoint, ptIVSEndPoint, isArgmax);
        cout << " Return_enum  ::" << Return_enum << endl;

        //if the enum is check next case we go to level 2 case 2 
        if (Return_enum == NEXT_LEVEL_CHECK)
        {
            Return_enum = IVS_FAILED;
            isArgmax = true;
            //	unsigned char *pFrameWithMaxPixelValueIndexOfAnatomyinVectorL2 = new unsigned char[imgoriginalSize];
            cout << " -----------------------------------------------IVS QC LEVEL 2 CASE 2 ENTERED----------------------------------------" << endl;
            Return_enum = Choosing_Axis_With_Respectto_IVS_angle(AnatomyLabelLevel2Case2, AnatomyLabelstringLevel2Case2, img_width_original, img_height_original, pInoriginalImage, pInFrameWithMaxPixelValueIndex, pFrameWithMaxPixelValueIndexOfAnatomyinVector, vecContourOfAnatomyinVector, ptEllipseMajorAxisLineStart, ptEllipseMajorAxisLineStop, ptEllipseMinorAxisLineStart, ptEllipseMinorAxisLineStop, ptIVSStartPoint, ptIVSEndPoint, isArgmax);
            cout << " Return_enum  ::" << Return_enum << endl;

        }
        //if the enum is check next case we go to level 2 case 3
        if (Return_enum == NEXT_LEVEL_CHECK)
        {
            Return_enum = IVS_FAILED;
            isArgmax = false;
            //	unsigned char *pFrameWithMaxPixelValueIndexOfAnatomyinVectorL3 = new unsigned char[imgoriginalSize];
            cout << " -----------------------------------------------IVS QC LEVEL 2 CASE 3 ENTERED----------------------------------------" << endl;
            Return_enum = Choosing_Axis_With_Respectto_IVS_angle(AnatomyLabelLevel2Case3, AnatomyLabelstringLevel2Case3, img_width_original, img_height_original, pInoriginalImage, pInFrameWithMaxPixelValueIndex, pFrameWithMaxPixelValueIndexOfAnatomyinVector, vecContourOfAnatomyinVector, ptEllipseMajorAxisLineStart, ptEllipseMajorAxisLineStop, ptEllipseMinorAxisLineStart, ptEllipseMinorAxisLineStop, ptIVSStartPoint, ptIVSEndPoint, isArgmax);
            cout << " Return_enum  ::" << Return_enum << endl;

        }
        //if the enum is check next case we go to level 2 case 4
        if (Return_enum == NEXT_LEVEL_CHECK)
        {
            Return_enum = IVS_FAILED;
            isArgmax = false;
            //	unsigned char *pFrameWithMaxPixelValueIndexOfAnatomyinVectorL4 = new unsigned char[imgoriginalSize];
            cout << " -----------------------------------------------IVS QC LEVEL 2 CASE 4 ENTERED----------------------------------------" << endl;
            Return_enum = Choosing_Axis_With_Respectto_IVS_angle(AnatomyLabelLevel2Case4, AnatomyLabelstringLevel2Case4, img_width_original, img_height_original, pInoriginalImage, pInFrameWithMaxPixelValueIndex, pFrameWithMaxPixelValueIndexOfAnatomyinVector, vecContourOfAnatomyinVector, ptEllipseMajorAxisLineStart, ptEllipseMajorAxisLineStop, ptEllipseMinorAxisLineStart, ptEllipseMinorAxisLineStop, ptIVSStartPoint, ptIVSEndPoint, isArgmax);
            cout << " Return_enum  ::" << Return_enum << endl;

        }
        //LEVEL 2 ENDS HERE , IF RETURN IS STILL NEXT_LEVEL_CHECK THEN WE PROCCED WITH LEVEL 3 
        if (Return_enum == NEXT_LEVEL_CHECK)
        {
            cout << " -----------------------------------------------IVS QC LEVEL 3 ENTERED----------------------------------------" << endl;
            //LEVEL 3 QUALITY CHECK BEGINS		
            // Mapping of Anatomies to Pixel Values :
            // ["Outer Thorax=1","Inner Thorax=2", "Heart Contour=3", "RA=4", "LA=5", "RV=6", "LV=7","Spine Triangle=8","Spine Body=9",  "dAorta=10","IV Septum=11", "Atrium Septum=12","AV Septum=13",  "Mitral Valve=14", "Tricuspid Valve=15",  "PV1 (RPV)=16", "PV2 (LPV)=17"]
            // Order of Labelling(from Bigger to smaller) : = { "RA", "LA", "RV", "LV", "Sp", "dAo", "IVS", "TV", "MV", "AS", "AVS", "PV1", "PV2" }
            //IVS Contour is present and checked already , so we just need to check for MV and TV 
            bool isMVContourFound = true;
            bool isTVContourFound = true;

            //LEVEL 3.1 Contour of MV 		
            int iSegmentedFrameIndexOfMV = 14; //Mitral Valve=14
            std::string strSegmentedLabelOfMV = "MV";
            unsigned char* pFrameWithMaxPixelValueIndexOfMV = new unsigned char[imgoriginalSize];
            std::vector<Point> vecContourOfMV;
            if (!getAnatomySegmentedMaskAndContour(iSegmentedFrameIndexOfMV, strSegmentedLabelOfMV, img_width_original, img_height_original, pInoriginalImage, pInFrameWithMaxPixelValueIndex, pFrameWithMaxPixelValueIndexOfMV, vecContourOfMV))
            {
                isMVContourFound = false;
                std::cout << "------------------------------- IVS QC LEVEL 3 FAILED , MV (Mitral Valve)  ANATOMY CONTOUR WAS NOT FOUND ------------------------------- " << std::endl;

            }
            delete[] pFrameWithMaxPixelValueIndexOfMV;
            //LEVEL 3.2 Contour of MV 	
            int iSegmentedFrameIndexOfTV = 15; //Tricuspid Valve=15
            std::string strSegmentedLabelOfTV = "TV";
            unsigned char* pFrameWithMaxPixelValueIndexOfTV = new unsigned char[imgoriginalSize];
            std::vector<Point> vecContourOfTV;
            if (!getAnatomySegmentedMaskAndContour(iSegmentedFrameIndexOfTV, strSegmentedLabelOfTV, img_width_original, img_height_original, pInoriginalImage, pInFrameWithMaxPixelValueIndex, pFrameWithMaxPixelValueIndexOfTV, vecContourOfTV))
            {
                isTVContourFound = false;
                std::cout << "------------------------------- IVS QC LEVEL 3 FAILED , TV (Tricuspid Valve)  ANATOMY CONTOUR WAS NOT FOUND ------------------------------- " << std::endl;

            }
            delete[] pFrameWithMaxPixelValueIndexOfTV;
            Point ptMVStartPoint = { 0,0 };
            Point ptMVEndPoint = { 0,0 };
            if (isMVContourFound) //if conyour exist then only find the lines , start and end point 
            {
                cv::Vec4f lineMV;
                cv::fitLine(Mat(vecContourOfMV), lineMV, cv::DIST_L2, 0, 0.01, 0.01);
                float vx_mv = lineMV[0];
                float vy_mv = lineMV[1];
                float x_mv = lineMV[2];
                float y_mv = lineMV[3];

                int lefty_mv = (-x_mv * vy_mv / vx_mv) + y_mv;
                int righty_mv = ((img_width_original - x_mv) * vy_mv / vx_mv) + y_mv;

                ptMVStartPoint = Point(img_width_original - 1, righty_mv);
                ptMVEndPoint = Point(0, lefty_mv);
            }
            Point ptTVStartPoint = { 0,0 };
            Point ptTVEndPoint = { 0,0 };
            if (isTVContourFound) //if conyour exist then only find the lines , start and end point 
            {
                cv::Vec4f lineTV;
                cv::fitLine(Mat(vecContourOfTV), lineTV, cv::DIST_L2, 0, 0.01, 0.01);
                float vx_tv = lineTV[0];
                float vy_tv = lineTV[1];
                float x_tv = lineTV[2];
                float y_tv = lineTV[3];

                int lefty_tv = (-x_tv * vy_tv / vx_tv) + y_tv;
                int righty_tv = ((img_width_original - x_tv) * vy_tv / vx_tv) + y_tv;

                ptTVStartPoint = Point(img_width_original - 1, righty_tv);
                ptTVEndPoint = Point(0, lefty_tv);
            }

            //LEVEL 3  :: check Angle(MV,TV) ,Angle(IVS,TV) ,Angle (IVS,MV) 
            //if we are unable to compute any one Contour , then out of 3 condition above only 1 we can compute ,Say MV failed , then we can compute only Angle(IVS,TV) , So only this dictates the Result
            //If we are able to compute all 3 angles , then if any 2 angle falls in the given range , it is true .The Range for angles is mentoined  below  

            //LEVEL 3.1 if Both MV and TV contours exist we compute all 3 angles ( Angle(MV,TV) ,Angle(IVS,TV) ,Angle (IVS,MV) )
            if (isMVContourFound && isTVContourFound)
            {
                cout << " -----------------------------------------------ALL 3 ANATOMY (IVS , MV, TV ) CONTOUR FOUND ----------------------------------------" << endl;
                double Angle_MV_TV = Calculate_Vector_Angle(ptMVStartPoint, ptMVEndPoint, ptTVStartPoint, ptTVEndPoint);
                double Angle_IVS_TV = Calculate_Vector_Angle(ptIVSStartPoint, ptIVSEndPoint, ptTVStartPoint, ptTVEndPoint);
                double Angle_IVS_MV = Calculate_Vector_Angle(ptIVSStartPoint, ptIVSEndPoint, ptMVStartPoint, ptMVEndPoint);

                bool isAngle_MV_TV_withinRange = false;
                bool isAngle_IVS_TV_withinRange = false;
                bool isAngle_IVS_MV_withinRange = false;

                //Checking Angle_MV_TV to make it falls within valid range 
                if (135 <= Angle_MV_TV && Angle_MV_TV <= 180)
                {
                    std::cout << "----------------------------- ANGLE (MV_TV) IS WITHIN RANGE ----------------------------" << std::endl;
                    isAngle_MV_TV_withinRange = true;
                }
                //Checking Angle_IVS_TV to make it falls within valid range 
                if (60 <= Angle_IVS_TV && Angle_IVS_TV <= 135)
                {
                    std::cout << "----------------------------- ANGLE (IVS_TV) IS WITHIN RANGE ----------------------------" << std::endl;
                    isAngle_IVS_TV_withinRange = true;
                }
                //Checking Angle_IVS_MV to make it falls within valid range 
                if (60 <= Angle_IVS_MV && Angle_IVS_MV <= 135)
                {
                    std::cout << "----------------------------- ANGLE (IVS_MV) IS WITHIN RANGE ----------------------------" << std::endl;
                    isAngle_IVS_MV_withinRange = true;
                }
                //Returning TRUE ,if atleast 2 cases passes out of 3 cases 
                bool Atleast2CasePassed = isAngle_MV_TV_withinRange ? (isAngle_IVS_TV_withinRange || isAngle_IVS_MV_withinRange) : (isAngle_IVS_TV_withinRange && isAngle_IVS_MV_withinRange);
                std::cout << "----------------------------- LEVEL 3 ATLEAST 2 OUT OF 3 CASES PASSED ? " << Atleast2CasePassed << " ----------------------------" << std::endl;
                return Atleast2CasePassed;
            }
            //LEVEL 3.2 if MV contour doesn't exit we can only compute Angle(IVS,TV)
            else if (!isMVContourFound && isTVContourFound)
            {
                std::cout << " -----------------------------------------------ALL 2 ANATOMY (IVS , TV ) CONTOUR FOUND ----------------------------------------" << std::endl;
                double Angle_IVS_TV = Calculate_Vector_Angle(ptIVSStartPoint, ptIVSEndPoint, ptTVStartPoint, ptTVEndPoint);
                if (60 <= Angle_IVS_TV && Angle_IVS_TV <= 135)
                {
                    std::cout << "----------------------------- ANGLE (IVS_TV) IS WITHIN RANGE ----------------------------" << std::endl;
                    return true;
                }
                else
                {
                    std::cout << "----------------------------- ANGLE (IVS_TV) IS OUTOF RANGE ----------------------------" << std::endl;
                    return false;
                }
            }
            //LEVEL 3.3 if TV contour doesn't exit we can only compute Angle(IVS,MV)
            else if (isMVContourFound && !isTVContourFound)
            {
                std::cout << " -----------------------------------------------ALL 2 ANATOMY (IVS , MV ) CONTOUR FOUND ----------------------------------------" << std::endl;
                double Angle_IVS_MV = Calculate_Vector_Angle(ptIVSStartPoint, ptIVSEndPoint, ptMVStartPoint, ptMVEndPoint);
                if (60 <= Angle_IVS_MV && Angle_IVS_MV <= 135)
                {
                    std::cout << "----------------------------- ANGLE (IVS_MV) IS WITHIN RANGE ----------------------------" << std::endl;
                    return true;
                }
                else
                {
                    std::cout << "----------------------------- ANGLE (IVS_MV) IS OUTOF RANGE ----------------------------" << std::endl;
                    return false;
                }
            }
            //LEVEL 3.4 if Both TV and MV Contour doesn't exit we return false 
            else
            {
                std::cout << " -----------------------------------------------ALL ANATOMY (IVS , MV, TV ) CONTOUR FAILED  ----------------------------------------" << std::endl;
                return false;
            }
        }
        if (Return_enum == VALID_ANGLE)
        {
            std::cout << " -----------------------------------------------IVS QC PASSED----------------------------------------" << std::endl;
            return true;
        }
        if (Return_enum == IVS_FAILED)
        {
            cout << " -----------------------------------------------IVS QC FAILED----------------------------------------" << endl;
            return false;
        }

        delete[] pFrameWithMaxPixelValueIndexOfAnatomyinVector;
    }
    inputOriginalImg.release();
    return true;
}


void DrawDashedLine(cv::Mat& img, cv::Point pt1, cv::Point pt2,
    cv::Scalar color, int thickness, std::string style,
    int gap) {
    float dx = pt1.x - pt2.x;
    float dy = pt1.y - pt2.y;
    float dist = std::hypot(dx, dy);

    std::vector<cv::Point> pts;
    for (int i = 0; i < dist; i += gap) {
        float r = static_cast<float>(i / dist);
        int x = static_cast<int>((pt1.x * (1.0 - r) + pt2.x * r) + .5);
        int y = static_cast<int>((pt1.y * (1.0 - r) + pt2.y * r) + .5);
        pts.emplace_back(x, y);
    }

    int pts_size = pts.size();

    if (style == "dotted") {
        for (int i = 0; i < pts_size; ++i) {
            cv::circle(img, pts[i], thickness, color, -1);
        }
    }
    else {
        cv::Point s = pts[0];
        cv::Point e = pts[0];

        for (int i = 0; i < pts_size; ++i) {
            s = e;
            e = pts[i];
            if (i % 2 == 1) {
                cv::line(img, s, e, color, thickness);
            }
        }
    }
}

void calculate_cardiac_angle(Point inTwoLineIntersectionPt, Point inFirstLineOtherEndPt, Point inSecondLineOtherEndPt, float& outAngle)
{
    outAngle = 0;
    /*std::cout << "//////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "inTwoLineIntersectionPt  :: " << inTwoLineIntersectionPt << std::endl;
    std::cout << "inFirstLineOtherEndPt  :: " << inFirstLineOtherEndPt << std::endl;
    std::cout << "inSecondLineOtherEndPt  :: " << inSecondLineOtherEndPt << std::endl;*/
    Vector2D<double> p1(inTwoLineIntersectionPt.x, inTwoLineIntersectionPt.y);
    Vector2D<double> p2(inFirstLineOtherEndPt.x, inFirstLineOtherEndPt.y);
    Vector2D<double> p3(inSecondLineOtherEndPt.x, inSecondLineOtherEndPt.y);

    double ang_rad = Vector2D<double>::angle(p2 - p1, p3 - p1);
    double ang_deg = ang_rad * 180.0 / M_PI;

    if (ang_deg - 180 >= 0)
        outAngle = 360 - ang_deg;
    else
        outAngle = ang_deg;
}


bool find_perpendicular_line(Point inFirstFarthestVertexFromCentroidOfSpineBody, Point inSecondFarthestVertexFromCentroidOfSpineBody, Point inCentroidOfSpineBody_OneEndOfMidSaggitalLine, Point& outPerpendicularToBaseLine_OtherEndOfMidSaggitalLine)
{
    outPerpendicularToBaseLine_OtherEndOfMidSaggitalLine.x = 0;
    outPerpendicularToBaseLine_OtherEndOfMidSaggitalLine.y = 0;

    std::cout << "inFirstFarthestVertexFromCentroidOfSpineBody (x,y) : " << inFirstFarthestVertexFromCentroidOfSpineBody.x << " , " << inFirstFarthestVertexFromCentroidOfSpineBody.y << std::endl;
    std::cout << "inSecondFarthestVertexFromCentroidOfSpineBody (x,y) : " << inSecondFarthestVertexFromCentroidOfSpineBody.x << " , " << inSecondFarthestVertexFromCentroidOfSpineBody.y << std::endl;
    std::cout << "inCentroidOfSpineBody_OneEndOfMidSaggitalLine (x,y) : " << inCentroidOfSpineBody_OneEndOfMidSaggitalLine.x << " , " << inCentroidOfSpineBody_OneEndOfMidSaggitalLine.y << std::endl;

    //1. calculate slope of Base line (Line between two farthest inner thorax rect vertices) 
    //1.0 if vertical line slope is 0
    if ((inFirstFarthestVertexFromCentroidOfSpineBody.y - inSecondFarthestVertexFromCentroidOfSpineBody.y) == 0)
    {
        //1.0.0 calculate the point on the baseline, which is perpendicular line to centroid of the spine body
        outPerpendicularToBaseLine_OtherEndOfMidSaggitalLine.x = inCentroidOfSpineBody_OneEndOfMidSaggitalLine.x;
        outPerpendicularToBaseLine_OtherEndOfMidSaggitalLine.y = inFirstFarthestVertexFromCentroidOfSpineBody.y;
    }
    else if ((inFirstFarthestVertexFromCentroidOfSpineBody.x - inSecondFarthestVertexFromCentroidOfSpineBody.x) == 0)
    {
        //1.0.0 calculate the point on the baseline, which is perpendicular line to centroid of the spine body
        outPerpendicularToBaseLine_OtherEndOfMidSaggitalLine.x = inFirstFarthestVertexFromCentroidOfSpineBody.x;
        outPerpendicularToBaseLine_OtherEndOfMidSaggitalLine.y = inCentroidOfSpineBody_OneEndOfMidSaggitalLine.y;
    }
    //1.1 if vertical line slope is not 0, then calculate the slope of baseline
    else
    {
        //1.1.0 slope of Base line
        float fSlopeofBaseLine = ((float)(inFirstFarthestVertexFromCentroidOfSpineBody.y - inSecondFarthestVertexFromCentroidOfSpineBody.y)) / ((float)(inFirstFarthestVertexFromCentroidOfSpineBody.x - inSecondFarthestVertexFromCentroidOfSpineBody.x));
        std::cout << "fSlopeofBaseLine:" << fSlopeofBaseLine << std::endl;

        //1.1.1 slope of perpendicular line to BaseLine = negative reciprocal of slope of Base line
        float fSlopeofPerpndLineToBaseLine = -1 / fSlopeofBaseLine;

        std::cout << "fSlopeofPerpndLineToBaseLine:" << fSlopeofPerpndLineToBaseLine << std::endl;

        //1.1.2 calculate the point on the baseline, which is perpendicular line to centroid of the spine body
        outPerpendicularToBaseLine_OtherEndOfMidSaggitalLine.x = (((fSlopeofPerpndLineToBaseLine * inCentroidOfSpineBody_OneEndOfMidSaggitalLine.x) - (fSlopeofBaseLine * inFirstFarthestVertexFromCentroidOfSpineBody.x)) - (inCentroidOfSpineBody_OneEndOfMidSaggitalLine.y - inFirstFarthestVertexFromCentroidOfSpineBody.y)) / (fSlopeofPerpndLineToBaseLine - fSlopeofBaseLine);
        outPerpendicularToBaseLine_OtherEndOfMidSaggitalLine.y = (inFirstFarthestVertexFromCentroidOfSpineBody.y - (fSlopeofBaseLine * (inFirstFarthestVertexFromCentroidOfSpineBody.x - outPerpendicularToBaseLine_OtherEndOfMidSaggitalLine.x)));

        std::cout << "outPerpendicularToBaseLine_OtherEndOfMidSaggitalLine (x,y) : " << outPerpendicularToBaseLine_OtherEndOfMidSaggitalLine.x << " , " << outPerpendicularToBaseLine_OtherEndOfMidSaggitalLine.y << std::endl;
    }

    return true;
}


bool find_midsagittal_line(const int img_width_original, const int img_height_original, unsigned char* pInoriginalImage, std::string strInSegmentedLabelOfSpineBody, std::string strInSegmentedLabelOfInnerThorax, unsigned char* pInFrameWithMaxPixelValueIndexOfSpineBody, unsigned char* pInFrameWithMaxPixelValueIndexOfInnerThorax, std::vector<Point> vecInContourOfSpineBody, std::vector<Point> vecInContourOfInnerThorax, Point& outCentroidOfSpineBody_OneEndOfMidSaggitalLine, Point& outPerpendicularToBaseLine_OtherEndOfMidSaggitalLine)
{




    //calculate centroid of spine body contour
    cv::Moments contourMoment = cv::moments(vecInContourOfSpineBody);
    int	contourSpineBodyCentroidX = (int)(contourMoment.m10 / contourMoment.m00);
    int contourSpineBodyCentroidY = (int)(contourMoment.m01 / contourMoment.m00);

    outCentroidOfSpineBody_OneEndOfMidSaggitalLine = cv::Point(contourSpineBodyCentroidX, contourSpineBodyCentroidY);



    //1.calculate  rect vertices for inner thorax
    const int rectNumberofVertices = 4;
    cv::RotatedRect rect = cv::minAreaRect(vecInContourOfInnerThorax);
    Point2f verticesOfRect[rectNumberofVertices];
    Point verticesOfRectOfTypeInt[rectNumberofVertices];
    rect.points(verticesOfRect);

    //2.get the distance wise farthest two rect vertices of inner thorax from centroid of spine body

    //2.0 Caluclate distance of 4 rect vertices from centroid of spine body
    std::vector<std::pair<float, int>> vectDistBtwCentroidOfSpineBodyAndRectVerticesOfInnerThorax;
    for (int i = 0; i < rectNumberofVertices; i++)
    {
        verticesOfRectOfTypeInt[i].x = (int)verticesOfRect[i].x;
        verticesOfRectOfTypeInt[i].y = (int)verticesOfRect[i].y;

        float fDistBtwCentroidOfSpineBodyAndRectVertexOfInnerThorax = sqrt(pow(contourSpineBodyCentroidX - verticesOfRectOfTypeInt[i].x, 2) + pow(contourSpineBodyCentroidY - verticesOfRectOfTypeInt[i].y, 2));
        vectDistBtwCentroidOfSpineBodyAndRectVerticesOfInnerThorax.push_back(std::make_pair(fDistBtwCentroidOfSpineBodyAndRectVertexOfInnerThorax, i));
    }



    //2.1 Sort based on the rect vertice's distance from centroid of spine body
    sort(vectDistBtwCentroidOfSpineBodyAndRectVerticesOfInnerThorax.begin(), vectDistBtwCentroidOfSpineBodyAndRectVerticesOfInnerThorax.end());

    //2.2 Out of 4 vertices, pick last two vertices to get distance wise farthest vertices
    Point firstFarthestVertexFromCentroidOfSpineBody = verticesOfRectOfTypeInt[vectDistBtwCentroidOfSpineBodyAndRectVerticesOfInnerThorax[3].second];
    Point secondFarthestVertexFromCentroidOfSpineBody = verticesOfRectOfTypeInt[vectDistBtwCentroidOfSpineBodyAndRectVerticesOfInnerThorax[2].second];

    Point firstNearestVertexFromCentroidOfSpineBody = verticesOfRectOfTypeInt[vectDistBtwCentroidOfSpineBodyAndRectVerticesOfInnerThorax[0].second];
    Point secondNearestVertexFromCentroidOfSpineBody = verticesOfRectOfTypeInt[vectDistBtwCentroidOfSpineBodyAndRectVerticesOfInnerThorax[1].second];



    //3 find perpendicular line from Centroid Of SpineBody to the Baseline(Line between the two farthest vertices)
    if (!find_perpendicular_line(firstFarthestVertexFromCentroidOfSpineBody, secondFarthestVertexFromCentroidOfSpineBody, outCentroidOfSpineBody_OneEndOfMidSaggitalLine, outPerpendicularToBaseLine_OtherEndOfMidSaggitalLine))
    {
        //  return false;
    }

    return true;
}


bool find_cardiac_axis_line(const int img_width_original,
    const int img_height_original,
    unsigned char* pInoriginalImage,
    std::string strInSegmentedLabelOfIVS,
    std::string strInSegmentedLabelOfOutterThorax,
    unsigned char* pInFrameWithMaxPixelValueIndexOfIVS,
    unsigned char* pInFrameWithMaxPixelValueIndexOfOutterThorax,
    std::vector<Point> vecInContourOfIVS,
    std::vector<Point> vecInContourOfOutterThorax,
    Point inCentroidOfSpineBody_OneEndOfMidSaggitalLine,
    Point inPerpendicularToBaseLine_OtherEndOfMidSaggitalLine,
    Point& outOutterThoraxIntersectionCardiacAxisLine,
    Point& outMidSagittalIntersectionCardiacAxisLine)
{




    //1. calculate fitline through IVS anatomy
    cv::Vec4f line4f;
    cv::fitLine(Mat(vecInContourOfIVS), line4f, cv::DIST_L2, 0, 0.01, 0.01);

    float vx = line4f[0];
    float vy = line4f[1];
    float x = line4f[2];
    float y = line4f[3];


    //x and y is point on the fitline 
    //what vx and vy ? Ans: x+ = x+vx*distance, y+ = y+vy*distance , x- = x-vx*distance, y- = y-vy*distance

    std::cout << "vx:" << vx << std::endl;
    std::cout << "vy:" << vy << std::endl;
    std::cout << "x:" << x << std::endl;
    std::cout << "y:" << y << std::endl;

    int lefty = (-x * vy / vx) + y;
    int righty = ((img_width_original - x) * vy / vx) + y;

    Point pt1 = Point(img_width_original - 1, righty);
    Point pt2 = Point(0, lefty);



    //2. find intersection between IVS fitline and Mid sagittal line

    outMidSagittalIntersectionCardiacAxisLine = lineLineIntersection(inPerpendicularToBaseLine_OtherEndOfMidSaggitalLine, inCentroidOfSpineBody_OneEndOfMidSaggitalLine, pt1, pt2);

    if (outMidSagittalIntersectionCardiacAxisLine.x == INT_MAX && outMidSagittalIntersectionCardiacAxisLine.y == INT_MAX)
    {
        return false;
    }


    //3. Starting from intersection point , check  and select part of IVS fitline , which overlaps on IVS anatomy segment mask
    //3.0 Prepare IVS segment mask image
    cv::Mat imgSegmentedIVSMask = cv::Mat(cv::Size(img_width_original, img_height_original), CV_8UC1, pInFrameWithMaxPixelValueIndexOfIVS, cv::Mat::AUTO_STEP);
    //imwrite("IVSsegmentmask_before_erosion.png", imgSegmentedIVSMask);  NO NEED OF THIS LOGIC , THE MASK WAS CLEANED AND ITS A PROPER MASK WITHOUT ANY DISTORTION AND HENCE NO NEED FOR EROSION
    //Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, Size(3, 3));
    //erode(imgSegmentedIVSMask, imgSegmentedIVSMask, kernel);
    ///*dilate(imgSegmentedIVSMask, imgSegmentedIVSMask, erode_element);*/
    //imwrite("IVSsegmentmask_after_erosion.png", imgSegmentedIVSMask);

    //3.1 Chec whether IVSFitLine From Pt1 To MidSagittalIntersection overlaps on IVS segment mask image
    cv::Mat imgIVSFitLineFromPt1ToMidSagittalIntersection = Mat::zeros(cv::Size(img_width_original, img_height_original), CV_8UC1);
    cv::line(imgIVSFitLineFromPt1ToMidSagittalIntersection, pt1, outMidSagittalIntersectionCardiacAxisLine, cv::Scalar(255, 255, 255, 255), 1, 8, false);
    cv::Mat imgLineIntersection;
    bitwise_and(imgSegmentedIVSMask, imgIVSFitLineFromPt1ToMidSagittalIntersection, imgLineIntersection);
    Point ptChoosenBasedOnIVSAnatomyOverlap = Point(-1, -1);
    if (cv::countNonZero(imgLineIntersection) != 0)
    {
        ptChoosenBasedOnIVSAnatomyOverlap = pt1;
        std::cout << " IVSFitLine From Pt1 To MidSagittalIntersection overlaps on IVS anatomy" << std::endl;

    }

    //3.2 Chec whether IVSFitLine From Pt2 To MidSagittalIntersection overlaps on IVS segment mask image
    cv::Mat imgIVSFitLineFromPt2ToMidSagittalIntersection = Mat::zeros(cv::Size(img_width_original, img_height_original), CV_8UC1);
    cv::line(imgIVSFitLineFromPt2ToMidSagittalIntersection, pt2, outMidSagittalIntersectionCardiacAxisLine, cv::Scalar(255, 255, 255, 255), 1, 8, false);
    imgLineIntersection = Mat::zeros(cv::Size(img_width_original, img_height_original), CV_8UC1);
    bitwise_and(imgSegmentedIVSMask, imgIVSFitLineFromPt2ToMidSagittalIntersection, imgLineIntersection);

    if (cv::countNonZero(imgLineIntersection) != 0)
    {
        ptChoosenBasedOnIVSAnatomyOverlap = pt2;
        std::cout << " IVSFitLine From Pt2 To MidSagittalIntersection overlaps on IVS anatomy" << std::endl;

    }


    if (ptChoosenBasedOnIVSAnatomyOverlap.x == -1 || ptChoosenBasedOnIVSAnatomyOverlap.y == -1)
    {
        return false;
    }

    //4. find farthest(from Centroid of Spine Body) intersection point between choosen part  IVS fitline and Outter thorax segment mask image
    //4.0 Prepare Outter thorax segment mask image
    cv::Mat imgSegmentedOutterThoraxMask = cv::Mat(cv::Size(img_width_original, img_height_original), CV_8UC1, pInFrameWithMaxPixelValueIndexOfOutterThorax, cv::Mat::AUTO_STEP);
    //4.1 Check whether choosen part  IVS fitline  overlaps on Outter thorax segment mask image
    cv::Mat imgChoosenPartOfIVSFitLine = Mat::zeros(cv::Size(img_width_original, img_height_original), CV_8UC1);
    cv::line(imgChoosenPartOfIVSFitLine, ptChoosenBasedOnIVSAnatomyOverlap, outMidSagittalIntersectionCardiacAxisLine, cv::Scalar(255, 255, 255, 255), 1, 8, false);
    imgLineIntersection = Mat::zeros(cv::Size(img_width_original, img_height_original), CV_8UC1);
    bitwise_and(imgSegmentedOutterThoraxMask, imgChoosenPartOfIVSFitLine, imgLineIntersection);
    if (cv::countNonZero(imgLineIntersection) != 0)
    {
        std::cout << " IVS Fitline part, which is Choosen Based On IVS Anatomy Overlap, also overlaps on Outter thorax anatomy" << std::endl;
    }

    Mat nonZeroCoordinates;
    std::vector <Point> vecIVSFitLineIntersectionCoordinatesOnOuuterThorax;
    findNonZero(imgLineIntersection, nonZeroCoordinates);

    //if (nonZeroCoordinates.total() == 0)
    //{
    //    std::cout << " IVS Fitline part, which is Choosen Based On IVS Anatomy Overlap, doesnot overlaps on Outter thorax anatomy" << std::endl;
    //    return false;
    //}

    //4.2 get all the overlapping points
    Point ptIVSFitLineIntersectionCoordinateOnOutterThorax;
    for (size_t i = 0; i < nonZeroCoordinates.total(); i++) {
        ptIVSFitLineIntersectionCoordinateOnOutterThorax = Point(nonZeroCoordinates.at<Point>(i).x, nonZeroCoordinates.at<Point>(i).y);

        vecIVSFitLineIntersectionCoordinatesOnOuuterThorax.push_back(ptIVSFitLineIntersectionCoordinateOnOutterThorax);
    }

    if (vecIVSFitLineIntersectionCoordinatesOnOuuterThorax.size() == 0)
    {
        std::cout << " IVS Fitline part, which is Choosen Based On IVS Anatomy Overlap, doesnot overlaps on Outter thorax anatomy 5555 " << std::endl;
        //   return false;
    }


    //4.3 Caluclate distance of OutterThorax Intersection points from centroid of spine body
    std::vector<std::pair<float, int>> vectDistBtwCentroidOfSpineBodyAndOutterThoraxIntersectionPoints;
    for (size_t i = 0; i < vecIVSFitLineIntersectionCoordinatesOnOuuterThorax.size(); i++)
    {
        ptIVSFitLineIntersectionCoordinateOnOutterThorax = vecIVSFitLineIntersectionCoordinatesOnOuuterThorax[i];
        float fDistBtwCentroidOfSpineBodyAndOutterThoraxIntersectionPoint = sqrt(pow(inCentroidOfSpineBody_OneEndOfMidSaggitalLine.x - ptIVSFitLineIntersectionCoordinateOnOutterThorax.x, 2) + pow(inCentroidOfSpineBody_OneEndOfMidSaggitalLine.y - ptIVSFitLineIntersectionCoordinateOnOutterThorax.y, 2));

        vectDistBtwCentroidOfSpineBodyAndOutterThoraxIntersectionPoints.push_back(std::make_pair(fDistBtwCentroidOfSpineBodyAndOutterThoraxIntersectionPoint, i));
    }

    if (vectDistBtwCentroidOfSpineBodyAndOutterThoraxIntersectionPoints.size() == 0)
    {
        std::cout << " IVS Fitline part, which is Choosen Based On IVS Anatomy Overlap, doesnot overlaps on Outter thorax anatomy  66666" << std::endl;
        //return false;
    }

    //4.4 Sort based on the OutterThorax Intersection point's distance from centroid of spine body
    sort(vectDistBtwCentroidOfSpineBodyAndOutterThoraxIntersectionPoints.begin(), vectDistBtwCentroidOfSpineBodyAndOutterThoraxIntersectionPoints.end());

    //4.5 Pick last item from sorted vectr to get distance wise farthest OutterThorax Intersection point from centroid of spine body
    int itemIndexofPtFarthestOutterThoraxIVSFitlineIntersectionFromCentroidOfSpineBody = vectDistBtwCentroidOfSpineBodyAndOutterThoraxIntersectionPoints.size() - 1;
    Point ptFarthestOutterThoraxIVSFitlineIntersectionFromCentroidOfSpineBody = vecIVSFitLineIntersectionCoordinatesOnOuuterThorax[vectDistBtwCentroidOfSpineBodyAndOutterThoraxIntersectionPoints[itemIndexofPtFarthestOutterThoraxIVSFitlineIntersectionFromCentroidOfSpineBody].second];
    outOutterThoraxIntersectionCardiacAxisLine = ptFarthestOutterThoraxIVSFitlineIntersectionFromCentroidOfSpineBody;



    return true;
}

bool computeCardiacAxisMeasurement(const int img_width_original, const int img_height_original, unsigned char* pInoriginalImage, unsigned char* pInFrameWithMaxPixelValueIndex, string r)
{
    int imgoriginalSize = img_width_original * img_height_original;
    std::string strCardiacAxisResultFile = "./Extra/CA/" + r + ".png";
   
    std::vector<int> vec4CHSegmentedFrameIndicesOfCA = { 1,2,9,11 };
    std::vector<std::string> vec4CHSegmentedLabelsOfCA = { "OuterThorax","Inner Thorax","Spine Body","IV Septum" };

    //0. Set Cardia Axis Angle as "NA" value to dispalyed image's labels store for displaying it as one of the label
    //This done incase of failure to compute Cardia Axis Line & Angle
 
    std::string strLabelCardiacAxisAngle;

    //1. for finding the IVS anatomy segmented mask and max area contour 
    int iSegmentedFrameIndexOfIVS = vec4CHSegmentedFrameIndicesOfCA[3];
    std::string strSegmentedLabelOfIVS = vec4CHSegmentedLabelsOfCA[3];
    unsigned char* pFrameWithMaxPixelValueIndexOfIVS = new unsigned char[imgoriginalSize];
    std::vector<Point> vecContourOfIVS;

    if (!getAnatomySegmentedMaskAndContour(iSegmentedFrameIndexOfIVS, strSegmentedLabelOfIVS, img_width_original, img_height_original, pInoriginalImage, pInFrameWithMaxPixelValueIndex, pFrameWithMaxPixelValueIndexOfIVS, vecContourOfIVS))
    {
        std::cout << "CA measurement failed to compute.Reason:IVS anatomy segmented contour missing in the input image." << std::endl;
        std::wstring strPromptCAMeasurementFail;
        std::wstring strPromptCAMeasurementFailReason;
      
        delete[]pFrameWithMaxPixelValueIndexOfIVS;
        return false;
    }

    //2. for finding the Inner Thorax anatomy segmented mask and max area contour 
    int iSegmentedFrameIndexOfInnerThorax = vec4CHSegmentedFrameIndicesOfCA[1];
    std::string strSegmentedLabelOfInnerThorax = vec4CHSegmentedLabelsOfCA[1];
    unsigned char* pFrameWithMaxPixelValueIndexOfInnerThorax = new unsigned char[imgoriginalSize];
    std::vector<Point> vecContourOfInnerThorax;

    if (!getAnatomySegmentedMaskAndContour(iSegmentedFrameIndexOfInnerThorax, strSegmentedLabelOfInnerThorax, img_width_original, img_height_original, pInoriginalImage, pInFrameWithMaxPixelValueIndex, pFrameWithMaxPixelValueIndexOfInnerThorax, vecContourOfInnerThorax))
    {
        std::cout << "CA measurement failed to compute.Reason:Inner Thorax anatomy segmented contour missing in the input image." << std::endl;
        std::wstring strPromptCAMeasurementFail;
        std::wstring strPromptCAMeasurementFailReason;
     
        delete[]pFrameWithMaxPixelValueIndexOfIVS;
        delete[]pFrameWithMaxPixelValueIndexOfInnerThorax;
        return false;
    }


    //3. for finding the Outter Thorax anatomy segmented mask and max area contour 
    int iSegmentedFrameIndexOfOutterThorax = vec4CHSegmentedFrameIndicesOfCA[0];
    std::string strSegmentedLabelOfOutterThorax = vec4CHSegmentedLabelsOfCA[0];
    unsigned char* pFrameWithMaxPixelValueIndexOfOutterThorax = new unsigned char[imgoriginalSize];
    std::vector<Point> vecContourOfOutterThorax;

    if (!getAnatomySegmentedMaskAndContour(iSegmentedFrameIndexOfOutterThorax, strSegmentedLabelOfOutterThorax, img_width_original, img_height_original, pInoriginalImage, pInFrameWithMaxPixelValueIndex, pFrameWithMaxPixelValueIndexOfOutterThorax, vecContourOfOutterThorax))
    {
        std::cout << "CA measurement failed to compute.Reason:Outter Thorax anatomy segmented contour missing in the input image." << std::endl;
        std::wstring strPromptCAMeasurementFail;
        std::wstring strPromptCAMeasurementFailReason;
    
        delete[]pFrameWithMaxPixelValueIndexOfIVS;
        delete[]pFrameWithMaxPixelValueIndexOfInnerThorax;
        delete[]pFrameWithMaxPixelValueIndexOfOutterThorax;
        return false;
    }

    //4. for finding the Spine Body anatomy segmented mask and max area contour 
    int iSegmentedFrameIndexOfSpineBody = vec4CHSegmentedFrameIndicesOfCA[2];
    std::string strSegmentedLabelOfSpineBody = vec4CHSegmentedLabelsOfCA[2];
    unsigned char* pFrameWithMaxPixelValueIndexOfSpineBody = new unsigned char[imgoriginalSize];
    std::vector<Point> vecContourOfSpineBody;

    if (!getAnatomySegmentedMaskAndContour(iSegmentedFrameIndexOfSpineBody, strSegmentedLabelOfSpineBody, img_width_original, img_height_original, pInoriginalImage, pInFrameWithMaxPixelValueIndex, pFrameWithMaxPixelValueIndexOfSpineBody, vecContourOfSpineBody))
    {
        std::cout << "CA measurement failed to compute.Reason:Spine Body anatomy segmented contour missing in the input image." << std::endl;
        std::wstring strPromptCAMeasurementFail;
        std::wstring strPromptCAMeasurementFailReason;
     
        delete[]pFrameWithMaxPixelValueIndexOfIVS;
        delete[]pFrameWithMaxPixelValueIndexOfInnerThorax;
        delete[]pFrameWithMaxPixelValueIndexOfOutterThorax;
        delete[]pFrameWithMaxPixelValueIndexOfSpineBody;
        return false;
    }


    //5. find_midsagittal_line points(centroidOfSpineBody_OneEndOfMidSaggitalLine , perpendicularToBaseLine_OtherEndOfMidSaggitalLine)
    Point centroidOfSpineBody_OneEndOfMidSaggitalLine;
    Point perpendicularToBaseLine_OtherEndOfMidSaggitalLine;

    if (!find_midsagittal_line(img_width_original, img_height_original, pInoriginalImage, strSegmentedLabelOfSpineBody, strSegmentedLabelOfInnerThorax, pFrameWithMaxPixelValueIndexOfSpineBody, pFrameWithMaxPixelValueIndexOfInnerThorax, vecContourOfSpineBody, vecContourOfInnerThorax, centroidOfSpineBody_OneEndOfMidSaggitalLine, perpendicularToBaseLine_OtherEndOfMidSaggitalLine))
    {
        std::cout << "CA measurement failed to compute.Reason:Failed to compute mid saggital line in the input image." << std::endl;
        std::wstring strPromptCAMeasurementFail;
        std::wstring strPromptCAMeasurementFailReason;
  
        delete[]pFrameWithMaxPixelValueIndexOfIVS;
        delete[]pFrameWithMaxPixelValueIndexOfInnerThorax;
        delete[]pFrameWithMaxPixelValueIndexOfOutterThorax;
        delete[]pFrameWithMaxPixelValueIndexOfSpineBody;
        return false;
    }

    //5. find cardiac axis line points considering IVS
    Point outterThoraxIntersectionCardiacAxisLine;
    Point midSagittalIntersectionCardiacAxisLine;
    //Needed new Mask Generation, Because for OUTER THORAX computation we are going to make all Inner labels from 1 to 17 all white so that we can find contour.
    unsigned char* pFrameWithMaxPixelValueIndexOfOutterThoraxMakingallInnerLabelsWhite = new unsigned char[imgoriginalSize];
    std::vector<Point> vecContourOfOutterThoraxMakingallInnerLabelsWhite;
    if (!getAnatomySegmentedMaskAndContourforOuterThorax(img_width_original, img_height_original, pInoriginalImage, pInFrameWithMaxPixelValueIndex, pFrameWithMaxPixelValueIndexOfOutterThoraxMakingallInnerLabelsWhite, vecContourOfOutterThoraxMakingallInnerLabelsWhite))
    {
        std::cout << "CA measurement failed to compute.Reason:Outter Thorax anatomy segmented contour missing in the input image." << std::endl;
        std::wstring strPromptCAMeasurementFail;
        std::wstring strPromptCAMeasurementFailReason;
      
        delete[]pFrameWithMaxPixelValueIndexOfIVS;
        delete[]pFrameWithMaxPixelValueIndexOfInnerThorax;
        delete[]pFrameWithMaxPixelValueIndexOfOutterThorax;
        delete[]pFrameWithMaxPixelValueIndexOfOutterThoraxMakingallInnerLabelsWhite;
        delete[]pFrameWithMaxPixelValueIndexOfSpineBody;
        return false;
    }

    if (!find_cardiac_axis_line(img_width_original, img_height_original, pInoriginalImage, strSegmentedLabelOfIVS, strSegmentedLabelOfOutterThorax, pFrameWithMaxPixelValueIndexOfIVS, pFrameWithMaxPixelValueIndexOfOutterThoraxMakingallInnerLabelsWhite, vecContourOfIVS, vecContourOfOutterThorax, centroidOfSpineBody_OneEndOfMidSaggitalLine, perpendicularToBaseLine_OtherEndOfMidSaggitalLine, outterThoraxIntersectionCardiacAxisLine, midSagittalIntersectionCardiacAxisLine))
    {
        std::cout << "CA measurement failed to compute.Reason:Failed to compute cardiac axis line in the input image." << std::endl;
        std::wstring strPromptCAMeasurementFail;
        std::wstring strPromptCAMeasurementFailReason;
    
        delete[]pFrameWithMaxPixelValueIndexOfIVS;
        delete[]pFrameWithMaxPixelValueIndexOfInnerThorax;
        delete[]pFrameWithMaxPixelValueIndexOfOutterThorax;
        delete[]pFrameWithMaxPixelValueIndexOfOutterThoraxMakingallInnerLabelsWhite;
        delete[]pFrameWithMaxPixelValueIndexOfSpineBody;
        return false;
    }


    //5.  calculate the cardiac angle between intersecting  caridac axis line and mid sagittal line
    float cardiac_angle = 0;
    calculate_cardiac_angle(midSagittalIntersectionCardiacAxisLine, perpendicularToBaseLine_OtherEndOfMidSaggitalLine, outterThoraxIntersectionCardiacAxisLine, cardiac_angle);

    A_Store = perpendicularToBaseLine_OtherEndOfMidSaggitalLine;
    B_Store = centroidOfSpineBody_OneEndOfMidSaggitalLine;
    C_Store = outterThoraxIntersectionCardiacAxisLine;
    D_Store = midSagittalIntersectionCardiacAxisLine;
    std::ostringstream oss;
    oss << std::fixed << std::setfill('0') << std::setprecision(2) << cardiac_angle;


    std::string strCardiacAngle = oss.str();
    strCardiacAngle.append("deg"); //Add Degree text at the end
    std::cout << " ==>>> Cardiac Angle(CA) : " << strCardiacAngle << std::endl;




    //Temporary Display Support: Save image with Cardia Axis Line & Angle Result. This image will be used in AppUI for user display
    cv::Mat inputOriginalImg = cv::Mat(cv::Size(img_width_original, img_height_original), CV_8UC1, pInoriginalImage, cv::Mat::AUTO_STEP);
    cv::Mat show_image = inputOriginalImg.clone();
    cvtColor(show_image, show_image, COLOR_GRAY2BGR, 3);
    Point  pCADisplay = Point(20, img_height_original - 40);
    cv::putText(show_image, " CA : " + strCardiacAngle, pCADisplay, FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255, 255), 2, 8, false);
    DrawDashedLine(show_image, centroidOfSpineBody_OneEndOfMidSaggitalLine, perpendicularToBaseLine_OtherEndOfMidSaggitalLine, cv::Scalar(0, 255, 255, 255), 2, "dotted", 10);
    DrawDashedLine(show_image, midSagittalIntersectionCardiacAxisLine, outterThoraxIntersectionCardiacAxisLine, cv::Scalar(0, 255, 255, 255), 2, "dotted", 10);
    imwrite(strCardiacAxisResultFile, show_image);
    std::cout << "CA measurement computed successfully." << std::endl;
    delete[]pFrameWithMaxPixelValueIndexOfIVS;
    delete[]pFrameWithMaxPixelValueIndexOfInnerThorax;
    delete[]pFrameWithMaxPixelValueIndexOfOutterThorax;
    delete[]pFrameWithMaxPixelValueIndexOfOutterThoraxMakingallInnerLabelsWhite;
    delete[]pFrameWithMaxPixelValueIndexOfSpineBody;
    return true;
}

/// ////////////////////////////////////////////////////////////

void Autolabels(cv::Mat maskapplied_orginputimgsize_cvframeWithMaxPixelValueIndex,cv::Mat inputOriginalImg, EFetalHeartViewClasss inFetalHeartViewClasss, string r)
{
    unsigned char* pInFrameWithMaxPixelValueIndex = (uint8_t*)maskapplied_orginputimgsize_cvframeWithMaxPixelValueIndex.data;
    int OriginalInputImageWidth = inputOriginalImg.size().width;
    int OriginalInputImageHight = inputOriginalImg.size().height;
    int imgoriginalSize = OriginalInputImageWidth * OriginalInputImageHight;

 
    std::vector<int> vecSegmentedFrameIndicesOfInterest;
    std::vector<std::string> vecSegmentedLabelsOfInterest;

    FillAutoLabelLabelIndexBasedOnFetalHeartView(inFetalHeartViewClasss, vecSegmentedFrameIndicesOfInterest, vecSegmentedLabelsOfInterest);

    //#pragma omp parallel for

    cv::Mat image_copy = inputOriginalImg.clone();
    cvtColor(image_copy, image_copy, COLOR_GRAY2BGR, 3);

    int legendcounter = 0;
    std::cout << "APR : AutoLabelling : Start : Found below anatomy labels \n";
    for (size_t iIndex = 0; iIndex < vecSegmentedFrameIndicesOfInterest.size(); iIndex++)
    {
        int iSegmentedFrameIndexOfInterest = vecSegmentedFrameIndicesOfInterest[iIndex];
        // std::cout << vecSegmentedLabelsOfInterest[iIndex] << std::endl;
        unsigned char* pTempFrameWithMaxPixelValueIndex = new unsigned char[imgoriginalSize];

        //Set any pixel value below max(255) to 0
        memset(pTempFrameWithMaxPixelValueIndex, 0, imgoriginalSize);
        bool bIsChannelImagePresent = false;
        for (int i = 0; i < imgoriginalSize; i++)
        {
            if (pInFrameWithMaxPixelValueIndex[i] == iSegmentedFrameIndexOfInterest)
            {
                pTempFrameWithMaxPixelValueIndex[i] = 255;
                bIsChannelImagePresent = true;
            }
        }

        if (!bIsChannelImagePresent)
        {
            delete[] pTempFrameWithMaxPixelValueIndex;
            continue;
        }




        // detect the contours on the binary image using cv2.CHAIN_APPROX_SIMPLE
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        cv::Mat inputImgContour = cv::Mat(cv::Size(OriginalInputImageWidth, OriginalInputImageHight), CV_8UC1, pTempFrameWithMaxPixelValueIndex, cv::Mat::AUTO_STEP);
        cv::findContours(inputImgContour, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);

        int iIndexofSelectedContour = -1;


        bool bSelectContourBasesOnArea = true; //true: max area, false:max point on perimeter
        bool bShowIndex = false;

        if (bSelectContourBasesOnArea)
        {
            double outMaxArea = 0;
            iIndexofSelectedContour = getMaxAreaContourId(contours, outMaxArea);

            //Always show text instead of number+legend based label display for the following the 4CH specific labels:RA,LA,RV,LV,dAo,Spine.
            if (inFetalHeartViewClasss == EFetalHeartViewClasss::FOUR_CHAMBER &&
                (vecSegmentedLabelsOfInterest[iIndex] == "RA" ||
                    vecSegmentedLabelsOfInterest[iIndex] == "LA" ||
                    vecSegmentedLabelsOfInterest[iIndex] == "RV" ||
                    vecSegmentedLabelsOfInterest[iIndex] == "LV" ||
                    vecSegmentedLabelsOfInterest[iIndex] == "PV1" ||
                    vecSegmentedLabelsOfInterest[iIndex] == "PV2" ||
                    vecSegmentedLabelsOfInterest[iIndex] == "dAo" ||
                    vecSegmentedLabelsOfInterest[iIndex] == "Sp")
                )
            {
                bShowIndex = false;
            }
            //Show number + legend instead of text based label display for anatomy with countour area below AUTOLABEL_TEXTDISPLAY_MIN_COUNTOUR_AREA_THRESHOLD 
            else if (outMaxArea < AUTOLABEL_TEXTDISPLAY_MIN_COUNTOUR_AREA_THRESHOLD)
            {
                bShowIndex = true;
            }
        }
        else
        {
            iIndexofSelectedContour = getMaxPerimeterLengthContourId(contours);
        }

        if (iIndexofSelectedContour != -1)
        {
            vector<vector<Point>> selectedContours;
            selectedContours.clear();

            selectedContours.push_back(contours[iIndexofSelectedContour]);

            // draw contours on the original image
            //drawContours(image_copy, selectedContours, -1, Scalar(0, 255, 0), 1);

            //Centroid of the contour
            cv::Moments contourMoment = cv::moments(contours[iIndexofSelectedContour]);
            int	contourCentroidX = (int)(contourMoment.m10 / contourMoment.m00);
            int contourCentroidY = (int)(contourMoment.m01 / contourMoment.m00);

            cv::Point labelTextPoint = cv::Point(contourCentroidX, contourCentroidY);
            if (!bShowIndex)
                cv::putText(image_copy, vecSegmentedLabelsOfInterest[iIndex], labelTextPoint, FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0, 255), 1, 8, false);
            else
            {
                legendcounter++;
                if (inFetalHeartViewClasss == FOUR_CHAMBER)
                {
                    vecSegmentedFrameIndicesOfInterest[iIndex] = vecSegmentedFrameIndicesOfInterest[iIndex] - 10;
                }
                cv::putText(image_copy, to_string(vecSegmentedFrameIndicesOfInterest[iIndex]), labelTextPoint, FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0, 255), 1, 8, false);

                string temp = to_string(vecSegmentedFrameIndicesOfInterest[iIndex]) + " :: " + vecSegmentedLabelsOfInterest[iIndex];
                cv::putText(image_copy, temp, cv::Point(OriginalInputImageWidth - 100, OriginalInputImageHight - 20 * legendcounter), FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0, 255), 1, 8, false);
            }

            ////Renaming PV1 and PV2 TO PV just for display purpose , while the outVecSegmentedFrameIndicesOfInterest and outVecSegmentedLabelsOfInterest is not disturbed 
            //if (vecSegmentedLabelsOfInterest[iIndex] == "PV1" || vecSegmentedLabelsOfInterest[iIndex] == "PV2")
            //{
            //    vecSegmentedLabelsOfInterest[iIndex] = "PV";
            //}

            //save the AutoLabelling result to rids image

        }

        //outSegmentedFrameFilePath = std::string("labelled_output.png");
        imwrite("labelled_output.png", image_copy);
        cv::imwrite("./Extra/Autolabels/" + r + ".png", image_copy);
        delete[] pTempFrameWithMaxPixelValueIndex;
    }



    image_copy.release();
 

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool postprocessSegmentationResult_CA(cv::Mat maskapplied_orginputimgsize_cvframeWithMaxPixelValueIndex, cv::Mat inputOriginalImg, const EFetalHeartViewClasss inFetalHeartViewClasss, string r)
{
 

    unsigned char* pInFrameWithMaxPixelValueIndex = (uint8_t*)maskapplied_orginputimgsize_cvframeWithMaxPixelValueIndex.data;
    int img_width_original = inputOriginalImg.size().width;
    int img_height_original = inputOriginalImg.size().height;
    int imgoriginalSize = img_width_original * img_height_original;
    unsigned char* pInoriginalImage = (unsigned char*)inputOriginalImg.data;
   

    std::vector<int> vecSegmentedFrameIndicesOfInterest;
    std::vector<std::string> vecSegmentedLabelsOfInterest;

    //FillAutoLabelLabelIndexBasedOnFetalHeartView(inFetalHeartViewClasss, vecSegmentedFrameIndicesOfInterest, vecSegmentedLabelsOfInterest);

    //#pragma omp parallel for





    /* ****************** Start: 4CH view specific : Cardiac Thoracic Ratio(CTR) & Cardia Axis(CA) Angle measurements computation ******** */
    if (inFetalHeartViewClasss == EFetalHeartViewClasss::FOUR_CHAMBER)
    {
        std::cout << " _________________EFetalHeartViewClasss::FOUR_CHAMBER __________________" << std::endl;


        /* ****************** Start:Check partial inner thorax anatomy ****************** */
        //1. find the Inner Thorax anatomy segmented mask and max area contour 
        std::vector<int> vec4CHSegmentedFrameIndicesForPartialThoraxCheck = { 2 };
        std::vector<std::string> vec4CHSegmentedLabelsForPartialThoraxCheck = { "Inner Thorax" };

        int iSegmentedFrameIndexOfInnerThorax = vec4CHSegmentedFrameIndicesForPartialThoraxCheck[0];
        std::string strSegmentedLabelOfInnerThorax = vec4CHSegmentedLabelsForPartialThoraxCheck[0];
        unsigned char* pFrameWithMaxPixelValueIndexOfInnerThorax = new unsigned char[imgoriginalSize];
        std::vector<Point> vecContourOfInnerThorax;
        //Needed new Mask Generation, Because for OUTER THORAX computation we are going to make all Inner labels from 1 to 17 all white so that we can find contour.
        unsigned char* pFrameWithMaxPixelValueIndexOfOutterThoraxMakingallInnerLabelsWhite = new unsigned char[imgoriginalSize];
        std::vector<Point> vecContourOfOutterThoraxMakingallInnerLabelsWhite;
        //Instead of Putting Constant Default Caliper we are Getting the caliper based on thorax , we are trying to fit the CA line within Thorax ,So that Line is not Too big for Unzoomed image
        if (!getAnatomySegmentedMaskAndContourforOuterThorax(img_width_original, img_height_original, pInoriginalImage, pInFrameWithMaxPixelValueIndex, pFrameWithMaxPixelValueIndexOfOutterThoraxMakingallInnerLabelsWhite, vecContourOfOutterThoraxMakingallInnerLabelsWhite))
        {
            std::cout << "CA measurement failed to compute.Reason:Outter Thorax anatomy segmented contour missing in the input image." << std::endl;
           delete[]pFrameWithMaxPixelValueIndexOfOutterThoraxMakingallInnerLabelsWhite;
            return false;
        }
        cv::Mat imgOuterThoraxImage = cv::Mat(cv::Size(img_width_original, img_height_original), CV_8UC1, pFrameWithMaxPixelValueIndexOfOutterThoraxMakingallInnerLabelsWhite, cv::Mat::AUTO_STEP);
        cv::Rect rect = cv::boundingRect(imgOuterThoraxImage);
        A_Store.x = rect.x + rect.width / 2;
        A_Store.y = rect.y;
        B_Store.x = rect.x + rect.width / 2;
        B_Store.y = rect.y + rect.height;
        C_Store.x = rect.x + rect.width / 2;
        C_Store.y = rect.y;
        D_Store.x = rect.x + rect.width / 2;
        D_Store.y = rect.y + rect.height / 2;
        if (!getAnatomySegmentedMaskAndContour(iSegmentedFrameIndexOfInnerThorax, strSegmentedLabelOfInnerThorax, img_width_original, img_height_original, pInoriginalImage, pInFrameWithMaxPixelValueIndex, pFrameWithMaxPixelValueIndexOfInnerThorax, vecContourOfInnerThorax))
        {
            delete[]pFrameWithMaxPixelValueIndexOfInnerThorax;
            return true;
        }

        //2. Check partial inner thorax anatomy i.e. in the acquired image,whether anatomy countour is intersecting with image fanbeam sector and image boundary.If found crossing, do not proceed with measurement computation.
        if (!checkPartialInnerThorax_CA_FanBeam(img_width_original, img_height_original, pInoriginalImage, strSegmentedLabelOfInnerThorax, pInFrameWithMaxPixelValueIndex, vecContourOfInnerThorax))
        {
            std::cout << "CA and CTR measurements failed to compute.Reason:Partial Inner Thorax found in the input image." << std::endl;
            std::wstring strPromptCAAndCTRMeasurementFail;
            std::wstring strPromptCAAndCTRMeasurementFailReason;
          

            //delete[]pFrameWithMaxPixelValueIndexOfInnerThorax;
            //return true;
        }
        //Partial Thorax for Check for HDZOOMED Images -------Starts-------
        unsigned char* pFrameWithMaxPixelValueIndexOfNeededMask = new unsigned char[imgoriginalSize];
        std::vector<Point> vecContourOfNeededMask;
        //Make the pixel values for Mask Index greater than 1 as 255 
        if (!getAnatomySegmentedMaskAndContourforPartialThoraxHDZoomed(img_width_original, img_height_original, pInoriginalImage, pInFrameWithMaxPixelValueIndex, pFrameWithMaxPixelValueIndexOfNeededMask, vecContourOfNeededMask))
        {
            delete[]pFrameWithMaxPixelValueIndexOfNeededMask;
            return true;
        }

        //2. Check partial inner thorax anatomy i.e. in the acquired image,whether anatomy countour is intersecting with image fanbeam sector and image boundary.If found crossing, do not proceed with measurement computation.
        if (!checkPartialInnerThorax_CA_HDZoomed(img_width_original, img_height_original, pInoriginalImage, strSegmentedLabelOfInnerThorax, pFrameWithMaxPixelValueIndexOfNeededMask, vecContourOfNeededMask))
        {
            std::cout << "CA and CTR measurements failed to compute.Reason:Partial Inner Thorax found in the input image." << std::endl;
            std::wstring strPromptCAAndCTRMeasurementFail;
            std::wstring strPromptCAAndCTRMeasurementFailReason;
    

            // delete[]pFrameWithMaxPixelValueIndexOfNeededMask;
             // return true;
        }

        //Partial Thorax for Check for HDOOMED Images  -----End---------
        delete[]pFrameWithMaxPixelValueIndexOfNeededMask;
        delete[]pFrameWithMaxPixelValueIndexOfInnerThorax;
        /* ****************** End:Check partial inner thorax anatomy ****************** */

        /* ****************** Start: IVS anatomy based QC check  ****************** */
        //1. find the Heart anatomy segmented mask and max area contour 
        std::vector<int> vec4CHSegmentedFrameIndicesForIVSCheck = { 3,11 };
        std::vector<std::string> vec4CHSegmentedLabelsForIVSCheck = { "HeartContour" , "IVS" };
        int iSegmentedFrameIndexOfHeart = vec4CHSegmentedFrameIndicesForIVSCheck[0];
        std::string strSegmentedLabelOfHeart = vec4CHSegmentedLabelsForIVSCheck[0];
        unsigned char* pFrameWithMaxPixelValueIndexOfHeart = new unsigned char[imgoriginalSize];
        std::vector<Point> vecContourOfHeart;

        if (!getAnatomySegmentedMaskAndContour(iSegmentedFrameIndexOfHeart, strSegmentedLabelOfHeart, img_width_original, img_height_original, pInoriginalImage, pInFrameWithMaxPixelValueIndex, pFrameWithMaxPixelValueIndexOfHeart, vecContourOfHeart))
        {
            delete[]pFrameWithMaxPixelValueIndexOfHeart;
            return true;
        }

        //2. find the IVS anatomy segmented mask and max area contour 
        int iSegmentedFrameIndexOfIVS = vec4CHSegmentedFrameIndicesForIVSCheck[1];
        std::string strSegmentedLabelOfIVS = vec4CHSegmentedLabelsForIVSCheck[1];
        unsigned char* pFrameWithMaxPixelValueIndexOfIVS = new unsigned char[imgoriginalSize];
        std::vector<Point> vecContourOfIVS;

        if (!getAnatomySegmentedMaskAndContour(iSegmentedFrameIndexOfIVS, strSegmentedLabelOfIVS, img_width_original, img_height_original, pInoriginalImage, pInFrameWithMaxPixelValueIndex, pFrameWithMaxPixelValueIndexOfIVS, vecContourOfIVS))
        {
            delete[]pFrameWithMaxPixelValueIndexOfIVS;
            delete[]pFrameWithMaxPixelValueIndexOfHeart;
            return true;
        }

        //2. Check IVS anatomy.If found any QC failure, do not proceed with measurement computation.
        if (!checkIVS_CA(img_width_original, img_height_original, pInoriginalImage, strSegmentedLabelOfHeart, pFrameWithMaxPixelValueIndexOfHeart, vecContourOfHeart, strSegmentedLabelOfIVS, pInFrameWithMaxPixelValueIndex, pFrameWithMaxPixelValueIndexOfIVS, vecContourOfIVS))
        {
            std::cout << "CA and CTR measurements failed to compute.Reason:IVS QC check failed for the input image." << std::endl;
            std::wstring strPromptCAAndCTRMeasurementFail;
            std::wstring strPromptCAAndCTRMeasurementFailReason;
          

            delete[]pFrameWithMaxPixelValueIndexOfIVS;
            delete[]pFrameWithMaxPixelValueIndexOfHeart;
            return true;
        }
        delete[]pFrameWithMaxPixelValueIndexOfIVS;
        delete[]pFrameWithMaxPixelValueIndexOfHeart;
        /* ****************** End:IVS anatomy based QC check ****************** */


        /* ****** Start: Compute Cardiac Axis Measurement ******** */
        std::cout << "APR :4CH CA MEASUREMENT STARTS ..................\n";
        if (!computeCardiacAxisMeasurement(img_width_original, img_height_original, pInoriginalImage, pInFrameWithMaxPixelValueIndex,r))
        {
            std::cout << "APR :4CH CA MEASUREMENT RETURNED ERROR ..................\n";
        }
        std::cout << "APR :4CH CA MEASUREMENT ENDS ..................\n";
        /* ****** Start: Cardiac Thoracic Ratio Measurement ******** */
        std::cout << "APR :4CH CT RATIO MEASUREMENTS LABEL STARTS ..................\n";
        std::vector<int> vec4CHSegmentedFrameIndicesOfCTRatio = { 1,3 };
        std::vector<std::string> vec4CHSegmentedLabelsOfCTRatio = { "OuterThorax ","HeartContour" };

        //	cv::Mat inputOriginalImg = cv::Mat(cv::Size(img_width_original, img_height_original), CV_8UC1, pInoriginalImage, cv::Mat::AUTO_STEP);
        //cv::imwrite("TEST.png", inputOriginalImg);
        float Hperimeter = 0;
        float Tperimeter = 0;
        for (size_t iIndex = 0; iIndex < vec4CHSegmentedFrameIndicesOfCTRatio.size(); iIndex++)
        {
            int i4CHSegmentedFrameIndexOfCTRatio = vec4CHSegmentedFrameIndicesOfCTRatio[iIndex];

            unsigned char* pTempFrameWithMaxPixelValueIndex = new unsigned char[imgoriginalSize];

            //Set any pixel value below max(255) to 0
            memset(pTempFrameWithMaxPixelValueIndex, 0, imgoriginalSize);
            bool bIsChannelImagePresent = false;
            for (int i = 0; i < imgoriginalSize; i++)
            {
                if (i4CHSegmentedFrameIndexOfCTRatio == 3) //based on discussion with Abhijit these if case is added for Finding the "whole" Heart Contour 
                {
                    if (pInFrameWithMaxPixelValueIndex[i] >= 3)
                    {
                        pTempFrameWithMaxPixelValueIndex[i] = 255;
                        if (pInFrameWithMaxPixelValueIndex[i] == 9 || pInFrameWithMaxPixelValueIndex[i] == 10 || pInFrameWithMaxPixelValueIndex[i] == 8 || pInFrameWithMaxPixelValueIndex[i] == 16 || pInFrameWithMaxPixelValueIndex[i] == 17)
                        {
                            pTempFrameWithMaxPixelValueIndex[i] = 0;
                        }
                        bIsChannelImagePresent = true;
                    }

                }
                if (i4CHSegmentedFrameIndexOfCTRatio == 1) //based on discussion with Abhijit these if case is added for Finding the "whole" Heart Contour 
                {
                    if (pInFrameWithMaxPixelValueIndex[i] >= 1)
                    {
                        pTempFrameWithMaxPixelValueIndex[i] = 255;
                        bIsChannelImagePresent = true;
                    }

                }
            }


            if (!bIsChannelImagePresent)
            {
                delete[] pTempFrameWithMaxPixelValueIndex;
                continue;
            }

         

            // detect the contours on the binary image using cv2.CHAIN_APPROX_SIMPLE
            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;
            cv::Mat inputImgContour = cv::Mat(cv::Size(img_width_original, img_height_original), CV_8UC1, pTempFrameWithMaxPixelValueIndex, cv::Mat::AUTO_STEP);
            cv::findContours(inputImgContour, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
            RNG rng(12345);
            // find the contour with max points
            size_t iMaxLengthContour = 0;
            int iIndexofMaxLengthContour = -1;
            vector<RotatedRect> minRect(contours.size());
            vector<RotatedRect> minEllipse(contours.size());
            vector<Point2f>centers(contours.size());
            vector<Point2f>centers_for_ellipse(contours.size());
            vector<float>radius(contours.size());
            RotatedRect RectTest;
            vector<float>angle(contours.size());
            for (size_t i = 0; i < contours.size(); i++)
            {
                vector<Point> vecPoint = contours[i];
                if (iMaxLengthContour < vecPoint.size())
                {
                    iMaxLengthContour = vecPoint.size();
                    iIndexofMaxLengthContour = i;
                }

            }

            vector<vector<Point>> maxLengthContour;

            if (iIndexofMaxLengthContour != -1)
            {

                maxLengthContour.clear();

                maxLengthContour.push_back(contours[iIndexofMaxLengthContour]);



                //Centroid of the contour
                cv::Moments contourMoment = cv::moments(contours[iIndexofMaxLengthContour]);
                int	contourCentroidX = (int)(contourMoment.m10 / contourMoment.m00);
                int contourCentroidY = (int)(contourMoment.m01 / contourMoment.m00);

                cv::Point labelTextPoint = cv::Point(contourCentroidX, contourCentroidY);

                //save the AutoLabelling result to rids image
                for (size_t i = 0; i < maxLengthContour.size(); i++)
                {
                    Scalar color = cv::Scalar(0, 255, 255, 255); //Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

                    minRect[i] = minAreaRect(maxLengthContour[i]);
                    minEnclosingCircle(maxLengthContour[i], centers[i], radius[i]);

                    Size2f size;

                    if (maxLengthContour[i].size() > 5)
                    {
                        minEllipse[i] = fitEllipse(maxLengthContour[i]);
                        centers_for_ellipse[i] = (minEllipse[i].center);

                        float Long_Radius = (minEllipse[i].size).width;
                        float Short_Radius = (minEllipse[i].size).height;
                        angle[i] = minEllipse[i].angle;

                        if (iIndex == 0)
                        {
                            float Long_Radius_Thorax = Long_Radius;
                            float Short_Radius_Thorax = Short_Radius;
                            float Radius_Circle_Thorax = radius[i];
                            float RelaxedLongRadiusThorax;
                            float RelaxedShortRadiusThorax;
                            double dRelaxedEllipse = 0;
                       
                            if (dRelaxedEllipse == 1)
                            {
                                if (Short_Radius_Thorax <= Long_Radius_Thorax)
                                {
                                    RelaxedLongRadiusThorax = Long_Radius_Thorax;
                                    RelaxedShortRadiusThorax = ((abs((Radius_Circle_Thorax * 2) - Short_Radius_Thorax) / 4) + Short_Radius_Thorax);

                                }
                                else
                                {
                                    RelaxedLongRadiusThorax = Short_Radius_Thorax;
                                    RelaxedShortRadiusThorax = ((abs((Radius_Circle_Thorax * 2) - Long_Radius_Thorax) / 4) + Long_Radius_Thorax);
                                }
                            }
                            else
                            {
                                //float new_short_radius = (((int)radius[i]) * 2 + Short_Radius_Heart) / 2;


                                if (Short_Radius_Thorax <= Long_Radius_Thorax)
                                {
                                    RelaxedLongRadiusThorax = Long_Radius_Thorax;
                                    RelaxedShortRadiusThorax = ((Radius_Circle_Thorax * 2) + Short_Radius_Thorax) / 2;

                                }
                                else
                                {
                                    RelaxedLongRadiusThorax = Short_Radius_Thorax;
                                    RelaxedShortRadiusThorax = ((Radius_Circle_Thorax * 2) + Long_Radius_Thorax) / 2;
                                }
                            }
                            float perimeter_beforesquarerroot1 = pow(RelaxedLongRadiusThorax, 2) + pow(RelaxedShortRadiusThorax, 2);
                            float Perimeter1 = sqrt(perimeter_beforesquarerroot1 / 2);
                            std::cout << " ///////////////////////////////////////////////////////////////////////////////" << std::endl;
                            std::cout << " Long_Radius_THORAX " << RelaxedLongRadiusThorax << std::endl;
                            std::cout << " short_radius_THORAX " << RelaxedShortRadiusThorax << std::endl;
                            std::cout << "THORAX Perimeter " << Perimeter1 << std::endl;
                            Tperimeter = Perimeter1;
                          /*  RotatedRect ThoraxrRect = RotatedRect(Point2f(centers_for_ellipse[i]), Size2f(RelaxedShortRadiusThorax, RelaxedLongRadiusThorax), angle[i]);
                            Point2f ThoraxPoints[4];
                            ThoraxrRect.points(ThoraxPoints);
                            ThoraxMajorAxisStartPoint.x = (ThoraxPoints[0].x + ThoraxPoints[3].x) / 2;
                            ThoraxMajorAxisStartPoint.y = (ThoraxPoints[0].y + ThoraxPoints[3].y) / 2;
                            ThoraxMajorAxisEndPoint.x = (ThoraxPoints[1].x + ThoraxPoints[2].x) / 2;
                            ThoraxMajorAxisEndPoint.y = (ThoraxPoints[1].y + ThoraxPoints[2].y) / 2;
                            ThoraxAxisRatio = RelaxedLongRadiusThorax / RelaxedShortRadiusThorax;*/

                            ellipse(inputOriginalImg, centers_for_ellipse[i], Size(RelaxedShortRadiusThorax / 2, RelaxedLongRadiusThorax / 2)
                                , angle[i], 0,
                                360, Scalar(255, 0, 255),
                                1, LINE_AA);

                        }


                        if (iIndex == 1)//Heart
                        {
                            float Long_Radius_Heart;
                            float Short_Radius_Heart;
                            Long_Radius_Heart = Long_Radius;
                            Short_Radius_Heart = Short_Radius;

                            float perimeter_beforesquarerroot = pow(Long_Radius_Heart, 2) + pow(Short_Radius_Heart, 2);
                            float Perimeter = sqrt(perimeter_beforesquarerroot / 2);
                            std::cout << " ///////////////////////////////////////////////////////////////////////////////" << std::endl;
                            std::cout << " Long_Radius_Heart " << Long_Radius_Heart << std::endl;
                            std::cout << " Short_Radius_Heart " << Short_Radius_Heart << std::endl;
                            std::cout << "HEART  Perimeter " << Perimeter << std::endl;
                            Hperimeter = Perimeter;
                            ellipse(inputOriginalImg, centers_for_ellipse[i], Size(Long_Radius_Heart / 2, Short_Radius_Heart / 2)
                                , angle[i], 0,
                                360, Scalar(255, 0, 255),
                                2, LINE_AA);

                         /*   RotatedRect HeartrRect = RotatedRect(Point2f(centers_for_ellipse[i]), Size2f(Long_Radius_Heart, Short_Radius_Heart), angle[i]);
                            Point2f HeartPoints[4];
                            HeartrRect.points(HeartPoints);
                            HeartMajorAxisStartPoint.x = (HeartPoints[0].x + HeartPoints[3].x) / 2;
                            HeartMajorAxisStartPoint.y = (HeartPoints[0].y + HeartPoints[3].y) / 2;
                            HeartMajorAxisEndPoint.x = (HeartPoints[1].x + HeartPoints[2].x) / 2;
                            HeartMajorAxisEndPoint.y = (HeartPoints[1].y + HeartPoints[2].y) / 2;*/
                            HeartAxisRatio = Long_Radius_Heart / Short_Radius_Heart;
                            if (HeartAxisRatio < 1) //here minor axis may have greater value compared to major axis , but for voyager Grpahics we need to pass the ratio greater than one (major axis /minor axis ) .
                                HeartAxisRatio = 1 / HeartAxisRatio;
                        }


                    }
                }

            }
            float CTR = Hperimeter / Tperimeter;
            //Temporary Display Support: Save image with CTR Result. This image will be used in AppUI for user display
            Point  pCADisplay = Point(30, img_height_original - 40);
            std::ostringstream oss;
            oss  << CTR;


            std::string CTRR = oss.str();          
            std::cout << " ==>>> CTR : " << CTRR << std::endl;
            cv::putText(inputOriginalImg, "CTR ::"+ CTRR, pCADisplay, FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 255), 2, 8, false);
            cv::imwrite("./Extra/CTR/" + r + ".png", inputOriginalImg);
            delete[] pTempFrameWithMaxPixelValueIndex;
        }
        std::cout << "APR :4CH CT RATIO MEASUREMENTS LABEL ENDS ..................\n";
        /* ****** End: Cardiac Thoracic Ratio Measurement ******** */
        /* ****** End: Compute Cardiac Axis Measurement ******** */
        delete[]pFrameWithMaxPixelValueIndexOfOutterThoraxMakingallInnerLabelsWhite;
    }

    /* ****************** End: 4CH view specific : Cardiac Thoracic Ratio(CTR) & Cardia Axis(CA) Angle measurements computation ******** */
    return true;
}


int main() 
    {
        Core ie;

        std::chrono::time_point<std::chrono::high_resolution_clock> ModelLoadstart, ModelLoadend, InferenceTimestart, InferenceTimeend, TotalTimeforAllFramesStart, TotalTimeforAllFramesEnd, TimeforOneFrameStart, TimeforOneFrameEnd, PreProcessstart, PreProcessend;
        fstream ModelFile;


        ModelFile.open("./Extra/Dependency/Models/ModelCurrentlyUsed.txt", ios::in); //open a file to perform read operation using file object
        string modelname;
        if (ModelFile.is_open()) { //checking whether the file is open

            while (getline(ModelFile, modelname)) { //read data from file object and put it into string.
                cout <<"Loaded Model ( from .txt file ) Name is ::  " <<modelname << "\n"; //print the data of the string
                break;
            }
            ModelFile.close(); //close the file object.
        }



        InferenceEngine::CNNNetwork network;
        string ModelPrefix = "./Extra/Dependency/Models/";
        string ModelPostfixbin = ".bin";
        string ModelPostfixXml = ".xml";
        string ModelPostfixOnnx = ".onnx";
        std::string model_path = ModelPrefix + modelname + ModelPostfixXml;
        std::string weights_path = ModelPrefix + modelname + ModelPostfixbin;
        std::string onnx_path = ModelPrefix + modelname + ModelPostfixOnnx;
        const char* filexml = model_path.data();
        const char* filebin = weights_path.data();
        const char* fileonnx = onnx_path.data();
       
       
        struct stat sb;     
        //if ((stat(filexml, &sb) == 0 && !(sb.st_mode & S_IFDIR))&& (stat(filebin, &sb) == 0 && !(sb.st_mode & S_IFDIR)))
        //{
        //    cout << "The path is valid!" << std::endl;
        //    network = ie.ReadNetwork(model_path, weights_path);
        //    //network = ie.ReadNetwork(onnx_path);  
        //}
        if ((stat(fileonnx, &sb) == 0 && !(sb.st_mode & S_IFDIR)) && (stat(fileonnx, &sb) == 0 && !(sb.st_mode & S_IFDIR)))
        {
            cout << "The path is valid!" << std::endl;
          //  network = ie.ReadNetwork(model_path, weights_path);
            network = ie.ReadNetwork(onnx_path);  
        }
        else
        {
            cout << "The Path is invalid!" << std::endl;
            return 0;
        }    
      

        ModelLoadstart = std::chrono::high_resolution_clock::now();
        // Load the network into the Inference Engine
        ExecutableNetwork executable_network = ie.LoadNetwork(network, "CPU");  //Most time taking step 
        ModelLoadend = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = ModelLoadend - ModelLoadstart;

        std::cout << "Model Loading Time  " << elapsed_seconds.count() << std::endl;
        std::cout << " This Happens Only once , So relax and take a seat " << std::endl;
        TotalTimeforAllFramesStart = std::chrono::high_resolution_clock::now();
        std::vector<InferRequest> inferRequests;
        inferRequests.push_back(executable_network.CreateInferRequest());
        vector<String> fn;
        string Image_Name;
        int counter = 0;
        EFetalHeartViewClasss inFetalHeartViewClasss;
        int a = -1;
        while (a < 0 || a >4)
        {
            cout << "---------------------Enter the correct values -------------------------" << std::endl;
            cout << "0 --> FOUR_CHAMBER " << std::endl;
            cout << "1 --> LVOT " << std::endl;
            cout << "2 --> RVOT" << std::endl;
            cout << "3 --> THREE_VT" << std::endl;
            cout << "4 --> THREE_VV" << std::endl;
            cout << "ENTER THE VIEW CLASS NUMBER TO CONTINUE  " << std::endl;           
            cin >> a;
        }
        inFetalHeartViewClasss = EFetalHeartViewClasss(a);
        glob("./Extra/input/*.*", fn);
        for (auto f : fn)
        {
            inferRequests.push_back(executable_network.CreateInferRequest());
            counter++;
            std::cout << "-------------------------------------------NEW FRAME PROCESSING-------------------------------------------" << std::endl;
            TimeforOneFrameStart = std::chrono::high_resolution_clock::now();
            // std::cout << f << std::endl;



            string str1 = "./Extra/input";



            // Find first occurrence of "geeks"
            size_t found = f.find(str1);
            /* std::cout << str1.size() << std::endl;*/
            string r = f.substr(str1.size() + 1, f.size());
            r.erase(r.length() - 4);
            // prints the result
            cout << "String is: " << r << std::endl;
            ImageNames.push_back(r);
            /* cout << "-------------------------------------" << std::endl;*/



            const std::string imageFile = f;



            // Set up the input and output blobs
            InputsDataMap input_info(network.getInputsInfo());
            const auto& input = input_info.begin()->second;
            input->setPrecision(Precision::FP32);
            input->setLayout(Layout::NCHW);



            OutputsDataMap output_info(network.getOutputsInfo());
            const auto& output = output_info.begin()->second;
            output->setPrecision(Precision::FP32);




            // Prepare the input data
            const size_t input_channels = input->getTensorDesc().getDims()[1];
            const size_t input_height = input->getTensorDesc().getDims()[2];
            const size_t input_width = input->getTensorDesc().getDims()[3];

            //Incase you want to check the input Dimension of the model
            std::cout << " input_channels :: " << input_channels << std::endl;
            std::cout << " input_height :: " << input_height << std::endl;
            std::cout << " input_width :: " << input_width << std::endl; 
           

            PreProcessstart = std::chrono::high_resolution_clock::now();
            cv::Mat Originalimage = cv::imread(imageFile, cv::IMREAD_GRAYSCALE);
            if (Originalimage.empty()) {
                std::cout << "No image found.";
            }
            int OriginalInputImageWidth = Originalimage.size().width;
            int OriginalInputImageHight = Originalimage.size().height;
            Mat PaddedImage = loadImageandPreProcess(imageFile);
            InferenceEngine::InferRequest infer_request = executable_network.CreateInferRequest();
            InferenceEngine::Blob::Ptr input_blob = inferRequests[counter].GetBlob(network.getInputsInfo().begin()->first);
            InferenceEngine::MemoryBlob::Ptr minput_blob = InferenceEngine::as<InferenceEngine::MemoryBlob>(input_blob);
            InferenceEngine::LockedMemory<void> minput_buffer = minput_blob->wmap();
            float* input_data = minput_buffer.as<float*>();
            const int inputH = IM_SEGMENTATION_HEIGHT;
            const int inputW = IM_SEGMENTATION_WIDTH;
            std::vector<float> vecR;

            vecR.resize(inputH * inputW);
            int iCount = 0;
            for (int i = 0; i < inputH; i++)
            {
                for (int j = 0; j < inputW; j++)
                {
                    float pixelValue = PaddedImage.at<float>(i, j);
                    input_data[iCount] = pixelValue;
                    iCount++;
                }
            }
            PreProcessend = std::chrono::high_resolution_clock::now();
            auto duration1 = duration_cast<milliseconds>(PreProcessend - PreProcessstart);
            std::cout << " PRE PROCESSING Time Taken (in milliseconds) :: " << duration1.count() << std::endl;

            // Infer
            InferenceTimestart = std::chrono::high_resolution_clock::now();
            infer_request.SetBlob(network.getInputsInfo().begin()->first, input_blob);
            infer_request.Infer();
            InferenceTimeend = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = InferenceTimeend - InferenceTimestart;
            std::cout << " Inference Time Taken :: " << elapsed_seconds.count() << std::endl;

            // Get output tensor
            InferenceEngine::Blob::Ptr output_blob = infer_request.GetBlob(network.getOutputsInfo().begin()->first);
            InferenceEngine::MemoryBlob::Ptr moutput_blob = InferenceEngine::as<InferenceEngine::MemoryBlob>(output_blob);
            InferenceEngine::LockedMemory<const void> moutput_buffer = moutput_blob->rmap();
            const float* output_data = moutput_buffer.as<float*>();

            const size_t output_channels = output->getTensorDesc().getDims()[1];
            const size_t output_height = output->getTensorDesc().getDims()[2];
            const size_t output_width = output->getTensorDesc().getDims()[3];

            //Incase you want to check the Output Dimensions
           /* std::cout << " output_channels :: " << output_channels << std::endl;
            std::cout << " output_height :: " << output_height << std::endl;
            std::cout << " output_width :: " << output_width << std::endl;*/

            const size_t output_size = output_channels * output_height * output_width;
            std::vector<float> output_vec(output_data, output_data + output_size);


            /////////////////////////////////////////////////////////////////

            int imgSize = IM_SEGMENTATION_WIDTH * IM_SEGMENTATION_HEIGHT;
            unsigned char frameWithMaxPixelValueIndex[IM_SEGMENTATION_WIDTH * IM_SEGMENTATION_HEIGHT];
            memset(frameWithMaxPixelValueIndex, 0, imgSize * sizeof(unsigned char));
            //#pragma omp parallel for
            for (int iPixelIndex = 0; iPixelIndex < imgSize; iPixelIndex++)
            {
                float pixelValue = 0;
                float pixelMaxValue = -INFINITY; //Initialzie max pixel value holder to negative INFINITY
                int   channelIndexWithMaxPixelValue = 0;
                for (int iChannelIndex = 0; iChannelIndex < output_channels; iChannelIndex++)
                {
                    pixelValue = *(output_data + (iChannelIndex * imgSize + iPixelIndex));
                    if (pixelMaxValue < pixelValue)
                    {
                        pixelMaxValue = pixelValue;
                        channelIndexWithMaxPixelValue = iChannelIndex;
                    }
                }



                frameWithMaxPixelValueIndex[iPixelIndex] = channelIndexWithMaxPixelValue;
            }


            cv::Mat cvframeWithMaxPixelValueIndex = cv::Mat(cv::Size(IM_SEGMENTATION_WIDTH, IM_SEGMENTATION_HEIGHT), CV_8UC1, frameWithMaxPixelValueIndex, cv::Mat::AUTO_STEP);
            cv::imwrite("./Extra/Output_Mask/" + r + ".png", cvframeWithMaxPixelValueIndex);


            //Remove applied padding  back to preprocessed cropped dimension
            cv::Mat preprocess_cvframeWithMaxPixelValueIndex;
            preprocess_cvframeWithMaxPixelValueIndex = cvframeWithMaxPixelValueIndex(Range(PaddingTop, IM_SEGMENTATION_HEIGHT - PaddingBottom), Range(PaddingLeft, IM_SEGMENTATION_WIDTH - PaddingRight));
            // cv::imwrite("frameWithMaxPixelValueIndex_after_preprocess_cropped_resize.png", preprocess_cvframeWithMaxPixelValueIndex);




             //Resize back to Cropped Size 
            cv::Mat FinalResized;
            cv::resize(preprocess_cvframeWithMaxPixelValueIndex, FinalResized, cv::Size(outCroppedWidth, outCroppedHeight), 0, 0, cv::INTER_NEAREST);
            cv::Rect preprocess_cropp_rect;
            preprocess_cropp_rect.x = outCroppedOriginX;
            preprocess_cropp_rect.y = outCroppedOriginY;
            preprocess_cropp_rect.width = outCroppedWidth;
            preprocess_cropp_rect.height = outCroppedHeight;



            //Resize Back to original Input Dimensions 
            cv::Mat maskapplied_orginputimgsize_cvframeWithMaxPixelValueIndex = cv::Mat::zeros(cv::Size(Original_Input_Width, Original_Input_Height), CV_8UC1);
            FinalResized.copyTo(maskapplied_orginputimgsize_cvframeWithMaxPixelValueIndex(preprocess_cropp_rect));
            //  cv::imwrite("frameWithMaxPixelValueIndex_after_maskapplied_orginputimgsize_resize.png", maskapplied_orginputimgsize_cvframeWithMaxPixelValueIndex);
            cv::imwrite("./Extra/Final_Resized_mask/" + r + ".png", maskapplied_orginputimgsize_cvframeWithMaxPixelValueIndex);

            //AUTOLABELS CODE FUNCTION CALL 
            Autolabels(maskapplied_orginputimgsize_cvframeWithMaxPixelValueIndex, Originalimage, inFetalHeartViewClasss,r);
            postprocessSegmentationResult_CA(maskapplied_orginputimgsize_cvframeWithMaxPixelValueIndex, Originalimage, inFetalHeartViewClasss, r);

            cvframeWithMaxPixelValueIndex.release();
            TimeforOneFrameEnd = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> outerelapsed_seconds = TimeforOneFrameEnd - TimeforOneFrameStart;
            std::cout << "********************* TIME TAKEN for 1 FRAME (seconds) :: " << outerelapsed_seconds.count() << "  ***************************" << std::endl;

    }
    // system("pause"); // <----------------------------------
    return 0;
}