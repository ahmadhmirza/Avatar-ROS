package org.example.ahmad.android_test;
/**
 * @author ahmad.mirza001@stud.fh-dortmund.de (Ahmad H. Mirza)
 * Class to publish data over the ROS network.
*/

import android.graphics.Bitmap;

import org.jboss.netty.buffer.ChannelBufferOutputStream;
import org.ros.internal.message.MessageBuffers;
import org.ros.message.Time;
import org.ros.namespace.GraphName;
import org.ros.node.AbstractNodeMain;
import org.ros.node.ConnectedNode;
import org.ros.node.topic.Publisher;

import java.io.ByteArrayOutputStream;
import java.io.IOException;

import sensor_msgs.CompressedImage;

public class Sender extends AbstractNodeMain {
    private String topic_name;
    private String topic_name_avatar;
    private String topic_name_avatarFlag;
    private Publisher<std_msgs.Int32> flagPublisher;
    private Publisher<sensor_msgs.CompressedImage> imagePublisher;
    private CompressedImage image;
    private Time currentTime;
    ConnectedNode connectedNode;
    String frameId = "AVATAR-ROS";

    /**
     * Constructor for the class, initializes the topic names on which data will be published
     */
    public Sender() {
        this.topic_name = "destination_android";
        this.topic_name_avatar="image_transport";
        this.topic_name_avatarFlag="prediction_flag";
    }

    /**
     * Function to set a custom topic name for the publishers
     * @param topic - String type
     */
    public Sender(String topic) {
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
        this.imagePublisher = connectedNode.newPublisher(topic_name_avatar,sensor_msgs.CompressedImage._TYPE);
        this.flagPublisher = connectedNode.newPublisher(topic_name_avatarFlag, "std_msgs/Int32");

        image = connectedNode.getTopicMessageFactory().newFromType(sensor_msgs.CompressedImage._TYPE);
    }
    public void publishFlag(int flag){
        std_msgs.Int32 predictionFlag =  flagPublisher.newMessage();
        predictionFlag.setData(flag);

        //publish the messages over the ROS network
        flagPublisher.publish(predictionFlag);
    }

    /**
     * Publishes the landmarks data on the defined ROS topic
     * @param b object of type Bitmap, to be published
     */
    public void publishImageAndLandmarks(Bitmap b){

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

/*    *//**
     *
     * @param b Bitmap Image
     * @param lMarks Array list containg landMark point of datatype int
     *//*
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
    }*/


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

}
