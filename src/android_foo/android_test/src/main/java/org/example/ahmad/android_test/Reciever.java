package org.example.ahmad.android_test;

import org.apache.commons.logging.Log;
import org.ros.message.MessageListener;
import org.ros.namespace.GraphName;
import org.ros.node.AbstractNodeMain;
import org.ros.node.ConnectedNode;
import org.ros.node.topic.Subscriber;

/**
 * @author ahmad.mirza001@stud.fh-dortmund.de (Ahmad H. Mirza)
 */

/**
 * Class for recieving data over the ROS network.
 * Topic name can be set in a global varibale defined in the class body.
 * Currently only designed to receive String objects.
 */
public class Reciever extends AbstractNodeMain{
    private String dataReceived;
    private String topic_name;
    public Reciever() {
        this.topic_name = "recieive_android_data";
    }

        @Override
        public GraphName getDefaultNodeName() {
            return GraphName.of("android_test/receiver");
        }

        @Override
        public void onStart(ConnectedNode connectedNode) {
            final Log log = connectedNode.getLog();

            final Subscriber<std_msgs.String> subscriber = connectedNode.newSubscriber("chatter",
                    std_msgs.String._TYPE);

            subscriber.addMessageListener(new MessageListener<std_msgs.String>() {
                @Override
                public void onNewMessage(std_msgs.String message) {
                    log.info("I heard: \"" + message.getData() + "\"");
                    dataReceived = message.getData();
                }
            });
        }

    public String getDataReceived() {
        return dataReceived;
    }

}

