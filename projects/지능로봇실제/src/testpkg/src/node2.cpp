//Int64 msg header Include
#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sstream>
#include "std_msgs/Int64.h"
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "random10_20");
    ros::NodeHandle n;
    ros::Publisher random10_20 = n.advertise<std_msgs::Int64>("b", 1000);

    ros::Rate loop_rate(10);//hz

    srand(time(NULL));

    while(ros::ok())
    {
        std_msgs::Int64 msg;
        int randNum = rand() % 10 + 11;

        msg.data = randNum;

        ROS_INFO("node2 : [%d]", msg.data);

        random10_20.publish(msg);

        ros::spinOnce();

        loop_rate.sleep();
    }
    return 0;
}