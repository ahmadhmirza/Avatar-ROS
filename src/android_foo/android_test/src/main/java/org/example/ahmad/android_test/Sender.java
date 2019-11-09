package org.example.ahmad.android_test;
/**
 * @author ahmad.mirza001@stud.fh-dortmund.de (Ahmad H. Mirza)
*/

import android.graphics.Bitmap;
import android.util.Log;

import org.jboss.netty.buffer.ChannelBufferOutputStream;
import org.ros.internal.message.MessageBuffers;
import org.ros.message.Time;
import org.ros.namespace.GraphName;
import org.ros.node.AbstractNodeMain;
import org.ros.node.ConnectedNode;
import org.ros.node.topic.Publisher;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;

import geometry_msgs.Pose2D;
import sensor_msgs.CompressedImage;

public class Sender extends AbstractNodeMain {
    private String topic_name;
    private String topic_name_avatar;
    private String topic_name_avatarString;
    private String topic_name_avatarInt;
    private String goalMessage;
    private Publisher<std_msgs.String> stringPublisher;
    private Publisher<geometry_msgs.Pose2D> posePublisher;
    private Publisher<sensor_msgs.CompressedImage> imagePublisher;
    private Publisher<std_msgs.Int32> landMarksPublisher;
    private CompressedImage image;
    private Time currentTime;
    ConnectedNode connectedNode;
    String frameId = "AVATAR-ROS";

    public Sender() {
        this.topic_name = "destination_android";
        this.topic_name_avatar="image_transport";
        this.topic_name_avatarString="landmarks_transport";
        this.topic_name_avatarInt="landmarks_points";
    }

    /**
     * Function to set a custom topic name for the publishers
     * @param topic
     */
    public Sender(String topic) {
        this.topic_name = topic;
        this.topic_name_avatar = topic;
    }

    public GraphName getDefaultNodeName() {
        return GraphName.of("android_test/Publisher_Android");
    }

    /**
     * Initializes the publisher objects with Topic names, and message types
     * @param connectedNode Name of the connected node, passed internally
     */
    public void onStart(ConnectedNode connectedNode) {
        this.connectedNode = connectedNode;
        //publisher = connectedNode.newPublisher(this.topic_name, "std_msgs/String");
        this.posePublisher = connectedNode.newPublisher(this.topic_name, Pose2D._TYPE);
        this.imagePublisher = connectedNode.newPublisher(topic_name_avatar,sensor_msgs.CompressedImage._TYPE);
        this.stringPublisher = connectedNode.newPublisher(topic_name_avatarString, "std_msgs/String");
        this.landMarksPublisher = connectedNode.newPublisher(topic_name_avatarInt, "std_msgs/Int32");

        image = connectedNode.getTopicMessageFactory().newFromType(sensor_msgs.CompressedImage._TYPE);
    }

    /**
     * Creates and publishes the pose message
     * @param x
     * @param y
     * @param theta
     */
    public void publishMessage(double x,double y, double theta){
        geometry_msgs.Pose2D pose = posePublisher.newMessage();

        pose.setX(x);
        pose.setY(y);
        pose.setTheta(theta);

        posePublisher.publish(pose);
    }

    /**
     * Converts the incomming bitmap image to a compressed image format - jpeg
     * populates the image CompressedImage object with the received data
     * publishes image on the specified topic
     *
     * @param b Bitmap image to be published
     */
    public void publishImage(Bitmap b){

        ChannelBufferOutputStream stream = new ChannelBufferOutputStream(MessageBuffers.dynamicBuffer());

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        b.compress(Bitmap.CompressFormat.JPEG, 100, out);
        byte[] byteArray = out.toByteArray();

        try {
            stream.write(byteArray);
        } catch (IOException e) {
            e.printStackTrace();
        }
        currentTime = connectedNode.getCurrentTime();
        image.getHeader().setStamp(currentTime);
        image.getHeader().setFrameId(frameId);
        image.setFormat("bgr8; jpeg compressed bgr8");
        image.setData(stream.buffer().copy());
        stream.buffer().clear();
        imagePublisher.publish(image);
    }

    public void publishImageAndLandmarks(Bitmap b, String lms){

        ChannelBufferOutputStream stream = new ChannelBufferOutputStream(MessageBuffers.dynamicBuffer());

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        b.compress(Bitmap.CompressFormat.JPEG, 100, out);
        byte[] byteArray = out.toByteArray();

        try {
            stream.write(byteArray);
        } catch (IOException e) {
            e.printStackTrace();
        }
        currentTime = connectedNode.getCurrentTime();
        image.getHeader().setStamp(currentTime);
        image.getHeader().setFrameId(frameId);
        image.setFormat("bgr8; jpeg compressed bgr8");
        image.setData(stream.buffer().copy());
        stream.buffer().clear();


        std_msgs.String str =  stringPublisher.newMessage();
        str.setData(lms);

        //publish the messages over the ROS network
        stringPublisher.publish(str);
        imagePublisher.publish(image);
    }

    /**
     *
     * @param b Bitmap Image
     * @param lMarks Array list containg landMark point of datatype int
     */
    public void publishImageAndLandmarks2(Bitmap b, ArrayList<Integer> lMarks){

        ChannelBufferOutputStream stream = new ChannelBufferOutputStream(MessageBuffers.dynamicBuffer());

        ByteArrayOutputStream out = new ByteArrayOutputStream();
        b.compress(Bitmap.CompressFormat.JPEG, 100, out);
        byte[] byteArray = out.toByteArray();

        try {
            stream.write(byteArray);
        } catch (IOException e) {
            e.printStackTrace();
        }
        currentTime = connectedNode.getCurrentTime();
        image.getHeader().setStamp(currentTime);
        image.getHeader().setFrameId(frameId);
        image.setFormat("bgr8; jpeg compressed bgr8");
        image.setData(stream.buffer().copy());
        stream.buffer().clear();

        std_msgs.Int32 intMsg = landMarksPublisher.newMessage();

        //publish the messages over the ROS network
        imagePublisher.publish(image);

        for(int i=0; i<lMarks.size();i++){
            intMsg.setData(lMarks.get(i));
            Log.d("landmarkPointPublish: ",Integer.toString(intMsg.getData()));
            landMarksPublisher.publish(intMsg);
        }
        intMsg.setData(lMarks.size());
        landMarksPublisher.publish(intMsg);
    }


    private std_msgs.String extractLipsLandmarks(ArrayList dataArray){
        std_msgs.String landmarkString = stringPublisher.newMessage();
        String x="";
        //landmarkString.setData(dataArray.get(0).toString());
        x = dataArray.get(0).toString();
        for (int i=0; i<=dataArray.size(); i++){
            x = x +";" + dataArray.get(i).toString();

        }
        landmarkString.setData(x);
        return landmarkString;

    }
    public void createGoalString(String x, String y){
        goalMessage = ("x: " + x + ", y: " + y );
    }

}
