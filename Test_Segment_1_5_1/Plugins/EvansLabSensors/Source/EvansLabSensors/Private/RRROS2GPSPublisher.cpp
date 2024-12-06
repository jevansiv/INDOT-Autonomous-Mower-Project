// Fill out your copyright notice in the Description page of Project Settings.


#include "RRROS2GPSPublisher.h"

#include "Msgs/ROS2NavSatFix.h"


URRROS2GPSPublisher::URRROS2GPSPublisher() 
{
   TopicName = TEXT("GPS");
   MsgClass = UROS2NavSatFixMsg::StaticClass();

}