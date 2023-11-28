#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void detectAndDraw(Mat& img, CascadeClassifier& cascade, CascadeClassifier& nestedCascade, double scale);
Scalar faceColor = Scalar(255, 0, 0); // Color for circle around the face
Scalar eyeColor = Scalar(0, 255, 0); // Color for circle around the eyes

int main(int argc, const char** argv)
{
	VideoCapture capture;
	Mat frame;
	double scale = 2;
	CascadeClassifier cascade, nestedCascade;
	int selectedInput, fileNumber;

	// Haar cascade classifiers
	// default eye cascade
	String eyeCascade = "src/haarcascades/haarcascade_eye.xml";

	// handles glasses better
	//String eyeCascade = "src/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

	// different versions of face cascades
	String faceCascade = "src/haarcascades/haarcascade_frontalface_default.xml";
	//String faceCascade = "src/haarcascades/haarcascade_frontalface_alt.xml";
	//String faceCascade = "src/haarcascades/haarcascade_frontalface_alt2.xml";
	//String faceCascade = "src/haarcascades/haarcascade_frontalface_alt_tree.xml";
	
	// Loading classifiers
	nestedCascade.load(eyeCascade);
	cascade.load(faceCascade);

	// Selecting input source
	// 1 - camera
	// 2 - file
	// files are located at src/samples/
	cout << "1 - Camera | 2 - File" << endl;
	cin >> selectedInput;
	switch (selectedInput)
	{
	case 1:
		capture.open(0);
		break;
	case 2:
		cout << "Enter the number of file: ";
		cin >> fileNumber;
		capture.open("src/samples/" + to_string(fileNumber) + ".MP4");
		break;
	}

	if (capture.isOpened())
	{
		// Capture frames from video and detect face
		while (1)
		{
			capture >> frame;
			char exit = (char)waitKey(10);

			if (frame.empty()) {
				break;
			}
			// "Q" to stop
			else if (exit == 'q' || exit == 'Q') {
				break;
			}
			detectAndDraw(frame, cascade, nestedCascade, scale);
		}
	}
	else
		cout << "Could not Open Camera";
	return 0;
}

static void detectAndDraw(Mat& img, CascadeClassifier& cascade, CascadeClassifier& nestedCascade, double scale) {
	vector<Rect> faces;
	Mat gray, smallImg;
	double fx = 1 / scale;

	// Convert to grayscale
	cvtColor(img, gray, COLOR_BGR2GRAY);

	// Resize the grayscale image
	resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR);
	equalizeHist(smallImg, smallImg);

	// Detecting face and eyes then drawing circles around them
	cascade.detectMultiScale(smallImg, faces, 1.2, 6);

	for (size_t i = 0; i < faces.size(); i++)
	{
		Rect r = faces[i];
		Point center;

		// Faces
		center.x = cvRound((r.x + r.width * 0.5) * scale);
		center.y = cvRound((r.y + r.height * 0.5) * scale);
		int radius = cvRound((r.width + r.height) * 0.25 * scale);
		circle(img, center, radius, faceColor, 2);

		/* // Eyes
		Mat smallImgROI = smallImg(r);
		vector<Rect> nestedObjects;
		
		nestedCascade.detectMultiScale(smallImgROI, nestedObjects, 1.3, 2);
		
		for (size_t j = 0; j < nestedObjects.size(); j++)
		{
			Rect nr = nestedObjects[j];
			center.x = cvRound((r.x + nr.x + nr.width * 0.5) * scale);
			center.y = cvRound((r.y + nr.y + nr.height * 0.5) * scale);
			radius = cvRound((nr.width + nr.height) * 0.225 * scale);
			circle(img, center, radius, eyeColor, 1.75);
		}
		*/
	}
	//resize(img, img, Size(1366, 768));
	imshow("Face Detection", img);
}
