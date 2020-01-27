package org.example.ahmad.android_test;
/**
 * @author ahmad.mirza001@stud.fh-dortmund.de (Ahmad H. Mirza)
 */
import android.content.Context;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.ImageView;

import org.example.ahmad.android_test.MvUtils.DetectLandmarks;
import org.example.ahmad.android_test.MvUtils.FaceDetection;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.ros.android.RosActivity;
import org.ros.node.NodeConfiguration;
import org.ros.node.NodeMainExecutor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

import static org.opencv.core.CvType.CV_8UC4;


/**
 * AvatarCameraActivity handles camera operations as well as face and landmarks detection
 */
public class AvatarCameraActivity extends RosActivity implements CvCameraViewListener2 {
    //private static final String TAG = "AvatarCameraActivity";
    //private static final std_msgs.String TAG = "AvatarCameraActivity";
    private static CascadeClassifier classifier;
    private CameraBridgeViewBase mOpenCvCameraView;
    private boolean              mIsJavaCamera = true;
    private MenuItem             mItemSwitchCamera = null;
    Mat mRGBA;
    Mat mGray;
    FaceDetection faceDetector;
    private int absoluteFaceSize;
    private MatOfRect faceDetections =new MatOfRect();
    private Mat grayscaleImage = new Mat();
    private Mat inputFrameProcessed = new Mat();
    private int frameCount= 0;
    private int frameSkipCount=20; //How many frames to skip for facedetection
    private ImageView imageCanvas;
    private boolean process = false;
    private DetectLandmarks landmarkDetector;

    private Rect faceRoi;
    //Bitmaps declarations
    private Bitmap processedImage;
    private Mat processedImageMat;
    private Bitmap croppedFaceImage;
    private int MAX_WIDTH = 800;
    private int MAX_HEIGHT = 800;
    private Sender imagePublisher;
    private boolean publishData;
    private int predictionFlag;

    private ArrayList<Integer> landmarksData = new ArrayList<>(); //Arraylist to hold landmarks coordinates


    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i("AvatarCameraActivity", "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };


    public AvatarCameraActivity() {
        // The RosActivity constructor configures the notification title
        super("AVATAR-ROS", "AVATAR ver4.0");
        Log.i("AvatarCameraActivity", "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. Several imporant variables are initialized here */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i("AvatarCameraActivity", "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_avatar_camera);
        mOpenCvCameraView = findViewById(R.id.tutorial1_activity_java_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        imageCanvas = findViewById(R.id.image_canvas);
        landmarkDetector = new DetectLandmarks(this);
        mOpenCvCameraView.setMaxFrameSize(MAX_WIDTH,MAX_HEIGHT);
        publishData=false;
        load_cascade();
        predictionFlag=1; //Inverse logic
    }

    //*****OpenCV Initialization ********//
    static {
        if (OpenCVLoader.initDebug()) {
            Log.i("AvatarCameraActivity", "OpenCV initialized successfully");
        } else {
            Log.i("AvatarCameraActivity", "OpenCV failed to initialize");
        }
    }
    //***************************************************************************************

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d("AvatarCameraActivity", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d("AvatarCameraActivity", "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mRGBA = new Mat(height, width, CV_8UC4);
        mGray = new Mat(height, width, CV_8UC4);
        faceDetector = new FaceDetection(this, mRGBA);

        //Bitmap must be initialized like this before using it in the code, otherwise android throws
        //a bmp==null exception
        processedImage = Bitmap.createBitmap(mRGBA.width(),mRGBA.height(),Bitmap.Config.ARGB_8888);
        // The faces will be a 20% of the height of the screen
        absoluteFaceSize = (int) (height * 0.2);

    }

    public void onCameraViewStopped() {
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        //return faceDetection(inputFrame.rgba());
        mRGBA =  inputFrame.rgba();
        mGray = inputFrame.gray();
        //faceDetection(mRGBA);
        predictionFlag=1;

        if(process){
            predictionFlag=0;
            faceDetection(mRGBA);
            detectLandmarks2(mRGBA);

        }
        imagePublisher.publishFlag(predictionFlag);
        return inputFrame.gray();
    }

    /**
     * Uses the FaceDetection Class to perform face detection
     * This method also sets the global variable FaceRoi after performing the
     * face detection steps for further processing e.g. cropping the full
     * image to the face ROI region
     * @param inputFrame input frame: CvCameraViewFrame matrix
     * @return inputFrame :MAT - Processed inputFrame, cropped to extracted faceROI
     */
    public Mat faceDetection(Mat inputFrame) {

        if(frameCount == 0){
            frameCount++;

            Imgproc.cvtColor(inputFrame, grayscaleImage, Imgproc.COLOR_RGBA2RGB);

            if (classifier.empty()) {
                Log.v("AvatarCameraActivity","Classifier not found");
                return inputFrame;

            }

            else {
                //classifier.detectMultiScale(grayscaleImage, faceDetections);

                classifier.detectMultiScale(grayscaleImage, faceDetections, 1.1, 2, 2,
                        new Size(absoluteFaceSize, absoluteFaceSize), new Size());
                // to draw rectangles around the detected faces
                for (Rect rect : faceDetections.toArray()) {
                    Imgproc.rectangle(
                            inputFrame,                                               // where to draw the box
                            new Point(rect.x, rect.y),                            // bottom left
                            new Point(rect.x + rect.width, rect.y + rect.height), // top right
                            new Scalar(0, 0, 255),                 // RGB colour and thickness of the box
                            1
                    );
                    setFaceRoi(rect);
                    //croppedFaceROI = new Mat(inputFrame,new Rect(rect.x, rect.y,rect.x +
                    //rect.width, rect.y + rect.height));
                }
                return inputFrame;
            }
        }

        else if(frameCount >0 && frameCount <= frameSkipCount){
            frameCount++;

            for (Rect rect : faceDetections.toArray()) {
/*                Imgproc.rectangle(
                        inputFrame,                                               // where to draw the box
                        new Point(rect.x, rect.y),                            // bottom left
                        new Point(rect.x + rect.width, rect.y + rect.height), // top right
                        new Scalar(0, 0, 255),                 // RGB colour and thickness of the box
                        1
                );*/
                setFaceRoi(rect);
            }
            return inputFrame;
        }
        else if(frameCount == frameSkipCount+1 ){
            frameCount = 0;
            for (Rect rect : faceDetections.toArray()) {
/*                Imgproc.rectangle(
                        inputFrame,                                               // where to draw the box
                        new Point(rect.x, rect.y),                            // bottom left
                        new Point(rect.x + rect.width, rect.y + rect.height), // top right
                        new Scalar(0, 0, 255),                 // RGB colour and thickness of the box
                        1
                );*/
                setFaceRoi(rect);
            }
            return inputFrame;
        }
        return inputFrame;
    }



    /**
     * This function loads the cascade classifier for face detection to be used by OpenCV facedetection
     * module. the cascade classifier is placed in /res/raw/lbpcascade_frontalface_improved.xml
     *
     * The classifier is loaded in the global variable of type CascadeClassifier classifier
     * @return -
     */
    public void load_cascade(){
        try {
            InputStream is = this.getResources().openRawResource(R.raw.lbpcascade_frontalface_improved);
            File cascadeDir = this.getDir("cascade", Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            classifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            if(classifier.empty())
            {
                Log.v("MyActivity","----(!)Error loading \n");
                return;
            }
            else
            {
                Log.v("MyActivity",
                        "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());
            }
        } catch (IOException e) {
            e.printStackTrace();
            Log.v("MyActivity", "Failed to load cascade. Exception thrown: " + e);
        }
    }

    //****************************Methods for landmark detection process****************************

    /**
     * This function sets the global variable process and is called from button press action from
     * the user, This variable is used to device whether to perform image processing tasks on the
     * inputFrame.
     * @return -
     */
    public void startProcess(View v){
        if(!process){
            process = true;
        }
        else{
            process = false;
        }

    }

    /**
     * This function takes in a Mat object and performs landmarks detection on the input image matrix
     * once the detection has been performed it also publishes the data over the ROS network, and draws
     * the resulting image on the canvas
     * @param inputFrame -Mat - CvCameraViewFrame matrix
     */
    public void detectLandmarks2(Mat inputFrame){
        try {

            Rect faceROI= getFaceRoi();

            //adding margins to the face ROI before cropping the image
            faceROI.x = faceROI.x - 50;
            faceROI.y = faceROI.y - 50;
            faceROI.width  = faceROI.width + 100;
            faceROI.height = faceROI.height + 100;

            if(faceROI.x <0 ){
                faceROI.x = 0;
            }
            if(faceROI.y <0 ){
                faceROI.y = 0;
            }
            if(faceROI.width > MAX_WIDTH ){
                faceROI.width = MAX_WIDTH;
            }
            if(faceROI.height <0 ){
                faceROI.height = MAX_HEIGHT;
            }

            //Matrix to hold the information of the cropped image
            Mat cropped = new Mat(inputFrame, faceROI);
            croppedFaceImage = Bitmap.createBitmap(cropped.width(),cropped.height(),Bitmap.Config.ARGB_8888);
            //processedImage = Bitmap.createBitmap(inputFrame.width(),inputFrame.height(),Bitmap.Config.ARGB_8888);
            processedImage = Bitmap.createBitmap(cropped.width(),cropped.height(),Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(cropped,croppedFaceImage);
            //get the results from dlib by passing in the cropped image and saving them in a new image
            processedImage = landmarkDetector.detFacesFromBitmap(croppedFaceImage);

            //landmarksData = landmarkDetector.getLandmarkPoints();

            // extract the landmarks related to the LipsROI
            landmarksData=landmarkDetector.getLandmarkPoints();

            //Pass the landmark array to the sender class,
            //The sender class should iterate through the array list and publish each point one by one

            imagePublisher.publishImageAndLandmarks(processedImage);

            //imagePublisher.publishImage(processedImage);
            //imagePublisher.publishImageAndLandmarks2(processedImage,landmarksData);

            //the following is in case a button is implemented for publishing data
            //if(publishData){
            //    imagePublisher.publishImage(processedImage);
            //}

            //Log.v("Size of inputframe: ", inputFrame.size().toString());
            //Log.v("Size of ROI: ", faceROI.toString());
        }
        catch (Exception E){
            Log.v("AvatarCameraActivity", E.toString());
        }

        //The portion of the background task that updates the UI has to run on the main thread.
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                //all the code that updates the UI goes in this block
                imageCanvas.setImageBitmap(processedImage);

                //imageCanvasLips.setImageBitmap(croppedLipsImage);
            }
        });

    }



    /**
     * @return faceROI rect object
     */
    public Rect getFaceRoi() {
        return faceRoi;
    }
    /**
     * Sets global variable faceROI
     * @param faceRoi - Rect object
     */
    public void setFaceRoi(Rect faceRoi) {
        this.faceRoi = faceRoi;
    }


    /**
     * Placeholder function for a button functionality for publishing data
     */
    public void publishToROS(View v){
        if(!publishData){
            publishData=true;
        }
        else{
            publishData=false;
        }


    }

    /**
     * Function to initialize Ros Nodes and corresponding classes
     * @param nodeMainExecutor
     */
    @Override
    protected void init(NodeMainExecutor nodeMainExecutor) {
        imagePublisher=new Sender();
        // At this point, the user has already been prompted to either enter the URI
        // of a master to use or to start a master locally.
        // The user can easily use the selected ROS Hostname in the master chooser
        // activity.
        NodeConfiguration nodeConfiguration = NodeConfiguration.newPublic(getRosHostname());
        nodeConfiguration.setMasterUri(getMasterUri());
        nodeMainExecutor.execute(imagePublisher, nodeConfiguration);

    }
}