#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Int64.h"
#include "std_msgs/Float64.h"
#include "geometry_msgs/Point.h"

int totCnt = 0;

void node3Callback(const std_msgs::Int64::ConstPtr& msg)
{
    int cnt = msg->data;
    totCnt += cnt;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "cumulative_mean");
    ros::NodeHandle n;
    ros::Subscriber num_sub = n.subscribe("total_count", 1000, node3Callback);
    ros::Publisher result = n.advertise<std_msgs::Float64>("cumulative_mean", 1000);

    ros::Rate loop_rate(10);//hz
    int count = 0;

    while(ros::ok() && count <= 500)
    {
        count++;
        float mean = (totCnt * 1.0) / (count * 1.0);

        ROS_INFO("%d / %d = [%f]", totCnt, count, mean);
        ROS_INFO("Cumulative mean : [%f]", mean);

        std_msgs::Float64 msg;
        msg.data = mean;

        result.publish(msg);
        
        ros::spinOnce();
        loop_rate.sleep();
    }
    return 0;
}