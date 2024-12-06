// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Tools/RRROS2BaseSensorPublisher.h"
#include "RRROS2GPSPublisher.generated.h"

/**
 * 
 * @brief GPS Publisher Class
 */
UCLASS(ClassGroup = (Custom), Blueprintable, meta = (BlueprintSpawnableComponent))
class EVANSLABSENSORS_API URRROS2GPSPublisher : public URRROS2BaseSensorPublisher
{
	GENERATED_BODY()
public:
    URRROS2GPSPublisher();
	
};
