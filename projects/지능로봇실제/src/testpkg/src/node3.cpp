#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Int64.h"

int a = 0;
int b = 0;

// CALLBACK function
void node1Callback(const std_msgs::Int64::ConstPtr& msg)
{
    a = msg->data;
}

void node2Callback(const std_msgs::Int64::ConstPtr& msg)
{
    b = msg->data;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "a_plus_b");
    ros::NodeHandle n;

    ros::Publisher a_plus_b = n.advertise<std_msgs::Int64>("total_count", 1000);
    ros::Subscriber a_sub = n.subscribe("a", 1000, node1Callback);
    ros::Subscriber b_sub = n.subscribe("b", 1000, node2Callback);

    ros::Rate loop_rate(10);//hz

    while(ros::ok())
    {     
        int c = a + b;

        std_msgs::Int64 msg;
        msg.data = c;
        ROS_INFO("node3 : [%d] + [%d] = [%d]", a, b, c);

        a_plus_b.publish(msg);

        ros::spinOnce();

        loop_rate.sleep();
    }
    return 0;
}