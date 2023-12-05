//Int64 msg header Include
#include "ros/ros.h"
#include "std_msgs/String.h"
#include <sstream>
#include "std_msgs/Int64.h"
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "random1_10");
    ros::NodeHandle n;
    ros::Publisher random1_10 = n.advertise<std_msgs::Int64>("a", 1000);

    ros::Rate loop_rate(10);//hz

    srand(time(NULL));

    while(ros::ok())
    {
        std_msgs::Int64 msg;
        int randNum = rand() % 10 + 1;

        msg.data = randNum;

        ROS_INFO("node1 : [%d]", msg.data);

        random1_10.publish(msg);

        ros::spinOnce();

        loop_rate.sleep();
    }
    return 0;
}