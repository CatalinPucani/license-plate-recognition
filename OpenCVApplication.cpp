// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

Mat convolutie(Mat src, Mat kernel, float m)
{
	float S = 0;
	float k = 0;

	Mat dst = Mat(src.size(), CV_8UC1);

	k = kernel.cols / 2;

	if (k < 2) k = kernel.rows / 2;


	dst.convertTo(dst, CV_32FC1);

	for (int i = k; i < src.rows - k; i++) {
		for (int j = k; j < src.cols - k; j++) {
			S = 0.0;
			for (int ki = 0; ki < kernel.rows; ki++)
				for (int kj = 0; kj < kernel.cols; kj++) {
					S += (1 / m) * src.at<byte>(i + ki - k, j + kj - k) * kernel.at<float>(ki, kj);
				}
			dst.at<float>(i, j) = S;
		}
	}

	dst.convertTo(dst, CV_8UC1);

	return dst;
}

Mat Gaussian2D(Mat src, int n) {
	double t = (double)getTickCount();

	Mat dst;
	char fname[MAX_PATH];
	float data[100];


	float sigma = (float)n / 6;
	printf("%f\n", sigma);

	Mat kernel = Mat(n, n, CV_32F);

	float sum = 0;

		for (int ki = 0; ki < n; ki++) {

			for (int kj = 0; kj < n; kj++) {

				float intermed = -((ki - n / 2) * (ki - n / 2) + (kj - n / 2) * (kj - n / 2)) / (2 * sigma * sigma);

				kernel.at<float>(ki, kj) = (1 / (2 * PI * sigma * sigma)) * exp(intermed);

				sum += kernel.at<float>(ki, kj);
			}
		}

		dst = convolutie(src, kernel, sum);

		t = ((double)getTickCount() - t) / getTickFrequency();

		printf("Time = %.3f [s]\n", t);

		imshow("src", src);
		imshow("dst", dst);

	return dst;
}

bool isInside(Mat img, int i, int j) {
	return (i >= 0 && i < img.rows&& j >= 0 && j < img.cols);
}

int* makeHistogram(Mat src)
{
	int* histogram = (int*)calloc(256, sizeof(int));

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			histogram[src.at<uchar>(i, j)]++;

	return histogram;
}

Mat auto_threshold(Mat img) {
	float mg1 = 0, mg2 = 0;
	int N1 = 0, N2 = 0;
	float error=0.15;
	Mat dst;
	dst = Mat(img.rows, img.cols, CV_8UC1);
	int* histogram = makeHistogram(img);
	int min = histogram[0];
		int max = histogram[0];
		bool found = false;
		for (int i = 0; i < 256; ++i) {
			if (histogram[i] != 0) {
				if (found == false) {
					min = i;
					found = true;
				}
				else {
					max = i;
				}

			}
		}
		float Tk = (min + max) / 2;
		float Tk_1 = 0;
		do {
			for (int g = min; g < Tk; g++) {
				mg1 += g * histogram[g];
				N1 += histogram[g];
			}

			for (int g = Tk + 1; g < max; g++) {
				mg2 += g * histogram[g];
				N2 += histogram[g];
			}

			mg1 = mg1 / N1;
			mg2 = mg2 / N2;

			Tk_1 = Tk;
			Tk = (mg1 + mg2) / 2;
		} while (abs(Tk - Tk_1) < error);
		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++) {
				if (img.at<uchar>(i, j) < Tk)
					dst.at<uchar>(i, j) = 0;
				else
					dst.at<uchar>(i, j) = 255;
			}
		//imshow("Source", img);
		//showHistogram("Histogram", histogram, 256, 200);
		//showHistogram("Histogram_after", makeHistogram(dst), 256, 200);
		//printf("pragul este: %.2f", Tk);
		//imshow("Destination", dst);
		//waitKey(0);
	
	return dst;
}

void centering_transform(Mat img) {
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);
		}
	}
}

Mat sablon_filtru_gaussian_high_pass(Mat src) {

		Mat srcf(src.rows, src.cols, CV_32FC1); //stocam imaginea sursa ca float
		src.convertTo(srcf, CV_32FC1);

		centering_transform(srcf); //centram imaginea, pentru a o putea procesa in domeniul frecvential

		Mat fourier; //cream o matrice de numere complexe in care sa stocam transformata fourier directa
		dft(srcf, fourier, DFT_COMPLEX_OUTPUT); //aplicam transformata fourier directa pe imaginea sursa si rezultatul va fi stocat in matricea "fourier" de numere complexe

		//dorim sa desfacem numerele complexe din matricea "fourier" in doua matrici, una cu partile reale si una cu partile imaginare ale numerelor 
		Mat channels[] = { Mat::zeros(src.size(),CV_32F),Mat::zeros(src.size(),CV_32F) };
		split(fourier, channels);//separam partea reala de partea imaginara
		//in matricea channels[0] avem matricea care contine partile reale ale numerelor din "fourier"
		//in matricea channels[1] avem partea imaginara a numerelor

				//calculul magnitudinii


		//aici aplicam filtre (R=20 sau A=20)
		//parcurgem imaginea (0->H, 0->W) si aplicam modificarile pe channels[0] si channels[1]
		//pentru gauss putem folosi functia exp(exponent)

			//exemplu pentru filtru gaussian low pass
		int H = src.rows;
		int W = src.cols;
		int A = 20;
		for (int i = 0; i < H; i++) {
			for (int j = 0; j < W; j++) {
				channels[0].at<float>(i, j) *= (1 - exp(-((H / 2 - i) * (H / 2 - i) + (W / 2 - j) * (W / 2 - j)) / (A * A)));
				channels[1].at<float>(i, j) *= (1 - exp(-((H / 2 - i) * (H / 2 - i) + (W / 2 - j) * (W / 2 - j)) / (A * A)));
			}
		}

		Mat dst, dstf;
		merge(channels, 2, fourier); //reunim cele doua canale (real si imaginar), pentru a pregati matricea de revenirea in domeniul spatial
		dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE); //aplicam transformata fourier inversa pentru a reveni in domeniul spatial
		//in matricea dstf avem rezultatul transformatei fourier inverse

		centering_transform(dstf); //recentram imaginea, pentru a o vizualiza usor in domeniul spatial
		normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1); //normalizam matricea, ca sa nu avem valori in afara intervalului 0-255; punem rezultatul in dst

		//imshow("initial", src);
		//imshow("final", dst);
		//waitKey();

		return dst;

}

Mat sablon_filtru_ideal_high_pass(Mat src) {
	
		Mat srcf(src.rows, src.cols, CV_32FC1); //stocam imaginea sursa ca float
		src.convertTo(srcf, CV_32FC1);

		centering_transform(srcf); //centram imaginea, pentru a o putea procesa in domeniul frecvential

		Mat fourier; //cream o matrice de numere complexe in care sa stocam transformata fourier directa
		dft(srcf, fourier, DFT_COMPLEX_OUTPUT); //aplicam transformata fourier directa pe imaginea sursa si rezultatul va fi stocat in matricea "fourier" de numere complexe

		//dorim sa desfacem numerele complexe din matricea "fourier" in doua matrici, una cu partile reale si una cu partile imaginare ale numerelor 
		Mat channels[] = { Mat::zeros(src.size(),CV_32F),Mat::zeros(src.size(),CV_32F) };
		split(fourier, channels);//separam partea reala de partea imaginara
		//in matricea channels[0] avem matricea care contine partile reale ale numerelor din "fourier"
		//in matricea channels[1] avem partea imaginara a numerelor


		//aici aplicam filtre (R=20 sau A=20)
		//parcurgem imaginea (0->H, 0->W) si aplicam modificarile pe channels[0] si channels[1]
		//pentru gauss putem folosi functia exp(exponent)

			//exemplu pentru filtru gaussian low pass
		int H = src.rows;
		int W = src.cols;
		int R = 20;
		for (int i = 0; i < H; i++) {
			for (int j = 0; j < W; j++) {
				if (((H / 2 - i) * (H / 2 - i) + (W / 2 - j) * (W / 2 - j)) <= R * R) {
					channels[0].at<float>(i, j) = 0;
					channels[1].at<float>(i, j) = 0;
				}
			}
		}

		Mat dst, dstf;
		merge(channels, 2, fourier); //reunim cele doua canale (real si imaginar), pentru a pregati matricea de revenirea in domeniul spatial
		dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE); //aplicam transformata fourier inversa pentru a reveni in domeniul spatial
		//in matricea dstf avem rezultatul transformatei fourier inverse

		centering_transform(dstf); //recentram imaginea, pentru a o vizualiza usor in domeniul spatial
		normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1); //normalizam matricea, ca sa nu avem valori in afara intervalului 0-255; punem rezultatul in dst

	

		return dst;
	
}

Mat getS()
{
	Mat S(7, 7, CV_8UC1);
	for (int i = 0; i < 7; i++) {
		for (int j = 0; j < 7; j++) {
			S.at<uchar>(i, j) = 1;
		}
	}

	return S;
}

Mat getS2()
{
	Mat S(3, 3, CV_8UC1);
	S.at<uchar>(0, 0) = 1;
	S.at<uchar>(0, 1) = 1;
	S.at<uchar>(0, 2) = 1;
	S.at<uchar>(1, 0) = 1;
	S.at<uchar>(1, 1) = 1;
	S.at<uchar>(1, 2) = 1;
	S.at<uchar>(2, 0) = 1;
	S.at<uchar>(2, 1) = 1;
	S.at<uchar>(2, 2) = 1;

	return S;
}

Mat dilatare(Mat src, int N) {

	Mat dst = Mat(src.size(), CV_8UC1);

	char fname[MAX_PATH];


	dst = src.clone();
	int w = 3;

	Mat S = getS();

	for (int k = 0; k < N; k++) {
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (src.at<uchar>(i, j) == 0) {
					for (int x = 0; x < S.rows; x++) {
						for (int y = 0; y < S.cols; y++) {
							if (S.at<uchar>(x, y) == 1) {
								int auxx = i + x - (S.rows / 2);
								int auxy = j + y - (S.cols / 2);
								if (isInside(src, auxx, auxy)) {
									dst.at<uchar>(auxx, auxy) = 0;
								}
							}

						}
					}
				}
			}
		}
		src = dst.clone();
	}

	//imshow("dilatare", dst);

	return dst;
}

Mat dilatareMica(Mat src, int N) {

	Mat dst = Mat(src.size(), CV_8UC1);

	char fname[MAX_PATH];


	dst = src.clone();
	int w = 3;

	Mat S = getS2();

	for (int k = 0; k < N; k++) {
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (src.at<uchar>(i, j) == 0) {
					for (int x = 0; x < S.rows; x++) {
						for (int y = 0; y < S.cols; y++) {
							if (S.at<uchar>(x, y) == 1) {
								int auxx = i + x - (S.rows / 2);
								int auxy = j + y - (S.cols / 2);
								if (isInside(src, auxx, auxy)) {
									dst.at<uchar>(auxx, auxy) = 0;
								}
							}

						}
					}
				}
			}
		}
		src = dst.clone();
	}

	//imshow("dilatare", dst);

	return dst;
}



Mat eroziune(Mat src, int N) {
	Mat dst = Mat(src.size(), CV_8UC1);
	char fname[MAX_PATH];


	dst = src.clone();
	int w = 3;

	Mat S = getS();

	bool whiteFlag = false;

	for (int n = 0; n < N; n++) {
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (src.at<uchar>(i, j) == 0) {
					for (int x = 0; x < S.rows; x++) {
						for (int y = 0; y < S.cols; y++) {
							int auxx = i + x - (S.rows / 2);
							int auxy = j + y - (S.cols / 2);
							if (isInside(src, auxx, auxy) && S.at<uchar>(x, y) == 1 && src.at<uchar>(auxx, auxy) == 255) {
								dst.at<uchar>(i, j) = 255;
							}

						}
					}

				}
			}
		}
		src = dst.clone();
	}

	//imshow("eroziune", src);
	//imshow("eroziune", dst);

	return dst;
}

Mat eroziuneMica(Mat src, int N) {
	Mat dst = Mat(src.size(), CV_8UC1);
	char fname[MAX_PATH];


	dst = src.clone();
	int w = 3;

	Mat S = getS2();

	bool whiteFlag = false;

	for (int n = 0; n < N; n++) {
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				if (src.at<uchar>(i, j) == 0) {
					for (int x = 0; x < S.rows; x++) {
						for (int y = 0; y < S.cols; y++) {
							int auxx = i + x - (S.rows / 2);
							int auxy = j + y - (S.cols / 2);
							if (isInside(src, auxx, auxy) && S.at<uchar>(x, y) == 1 && src.at<uchar>(auxx, auxy) == 255) {
								dst.at<uchar>(i, j) = 255;
							}

						}
					}

				}
			}
		}
		src = dst.clone();
	}

	//imshow("eroziune", src);
	//imshow("eroziune", dst);

	return dst;
}

Mat inchidere(Mat src, int N) {

	//Mat dst = src.clone();
	//dst = convertToBlackAndWhiteSrc(src);

	Mat dst;
	for (int k = 0; k < N; k++) {
		dst = dilatareMica(src, 1);
		dst = eroziuneMica(dst, 1);
		
		
	}

	//imshow("inchidere", dst);

	return dst;
}


Mat maximumFilter(Mat img)
{
		Mat dst = Mat(img.rows, img.cols, CV_8UC1);

		dst = img.clone();

		int matrix[3][3];
		matrix[0][0] = -1;
		matrix[0][1] = -1;
		matrix[0][2] = -1;
		matrix[1][0] = -1;
		matrix[1][1] = 8;
		matrix[1][2] = -1;
		matrix[2][0] = -1;
		matrix[2][1] = -1;
		matrix[2][2] = -1;

		int array[9] = { 0 };


		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				int sum = 0;
				int k = 0;
				for (int ii = 0; ii < 3; ii++) {
					for (int jj = 0; jj < 3; jj++) {
						if (isInside(img, i + ii, j + jj)) {
							array[k++] = img.at<uchar>(i + ii, j + jj);
						}
					}
					//printf("%d\n", sum);
				}
				//sum /= 9;
				if (isInside(img, i + 1, j + 1)) {
					std::sort(array, array + 9);
					dst.at<uchar>(i + 1, j + 1) = array[8];

				}
			}
		}

		return dst;
}

Mat diff(Mat A, Mat B)
{
	Mat dst = Mat(A.size(), CV_8UC1);

	for (int i = 0; i < A.rows; i++)
	{
		for (int j = 0; j < A.cols; j++)
		{
			if ((A.at<uchar>(i, j) == 0) && (B.at<uchar>(i, j) != 0))
			{
				dst.at<uchar>(i, j) = 0;
			}
			else
			{
				dst.at<uchar>(i, j) = 255;
			}
		}
	}
	return dst;
}

Mat getContours(Mat src)
{
	Mat S = getS();

	Mat  intermediate, dst;


		//double t = (double)getTickCount(); // Get the current time [s]
		//src = imread(fname, IMREAD_GRAYSCALE);

		intermediate = eroziune(src, 1);

		dst = diff(src, intermediate);

		// Get the current time again and compute the time difference [s]
		//t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		//printf("Time = %.3f [ms]\n", t * 1000);

		//imshow("src", src);
		//imshow("dst", dst);

		return dst;
}

Mat negative_image(Mat img) 
{		
	Mat dst = Mat(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			int gin = img.at<uchar>(i, j);
			int gout = 255 - gin;
			if (gout < 0)
				gout = 0;
			if (gout > 255)
				gout = 255;

			dst.at<uchar>(i, j) = gout;
		}

	int* histogram_before = makeHistogram(img);
	int* histogram_after = makeHistogram(dst);

	//showHistogram("Before", histogram_before, 256, 200);
	//showHistogram("After", histogram_after, 256, 200);

	return dst;
}

void getLicensePlate()
{

	Mat src, dst;
	
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);

		Mat intermediate1 = Gaussian2D(src, 2);
		Mat intermediate2 = auto_threshold(intermediate1);

		Mat intermediate3 = getContours(intermediate2);

		//Mat intermediate2 = sablon_filtru_ideal_high_pass(intermediate1);

		//imshow("int3", intermediate3);

		//Mat intermediate3 = auto_threshold(intermediate2);

		Mat intermediate4 = negative_image(intermediate3);

		//imshow("int4", intermediate4);

		Mat intermediate5 = inchidere(intermediate4, 1);

		//dst = maximumFilter(intermediate4);
		dst = intermediate5;

		// Get the current time again and compute the time difference [s]
		//t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		//printf("Time = %.3f [ms]\n", t * 1000);

		imshow("src", src);
		imshow("dst", dst);

		waitKey();
	}
}

int main()
{
	int op;
	char fname[MAX_PATH];
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Gaussian Blur\n");
		printf(" 11 - Binary\n");
		printf(" 12 - Filtru-Gauss High Pass\n");
		printf(" 100 - Get License Plate\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				
				while (openFileDlg(fname))
				{
					Mat src = imread(fname, IMREAD_GRAYSCALE);
					imshow("dst", Gaussian2D(src, 2));

					waitKey(0);
				}
				break;
			case 11:
				while (openFileDlg(fname))
				{
					Mat src = imread(fname, IMREAD_GRAYSCALE);
					imshow("dst", auto_threshold(src));

					waitKey(0);
				}
				break;
			case 12:
				while (openFileDlg(fname))
				{
					Mat src = imread(fname, IMREAD_GRAYSCALE);
					imshow("dst", sablon_filtru_gaussian_high_pass(src));

					waitKey(0);
				}
			case 100: 
				getLicensePlate();
				break;
		}
	}
	while (op!=0);
	return 0;
}