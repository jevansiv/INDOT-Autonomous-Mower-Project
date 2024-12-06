// Fill out your copyright notice in the Description page of Project Settings.


#include "RRROS2IMUPublisher.h"

#include "Msgs/ROS2Imu.h"


URRROS2IMUPublisher::URRROS2IMUPublisher() 
{
   TopicName = TEXT("IMU");
   MsgClass = UROS2ImuMsg::StaticClass();

}