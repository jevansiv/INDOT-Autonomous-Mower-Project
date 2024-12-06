// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Sensors/RRROS2BaseSensorComponent.h"

#include "Tools/RRROS2BaseSensorPublisher.h"
#include "Msgs/ROS2NavSatFix.h"
#include "GeoReferencingSystem.h"

#include "RRGPSComponent.generated.h"

/**
 *
 */
UCLASS(ClassGroup = (Custom), Blueprintable, meta = (BlueprintSpawnableComponent))
class EVANSLABSENSORS_API URRGPSComponent : public URRROS2BaseSensorComponent
{
	GENERATED_BODY()
public:
	/**
	 * @brief Construct a new URRROS2CameraComponent object
	 * Initialize #SceneCaptureComponent and #CameraComponent.
	 */
	URRGPSComponent();

	/**
	 * @brief Initialize #Data and #RenderTarget, set #SceneCaptureComponent parameters.
	 *
	 * @param InROS2Node ROS2Node which this publisher belongs to
	 * @Noiseparam InTopicName
	 */
	virtual void PreInitializePublisher(UROS2NodeComponent *InROS2Node, const FString &InTopicName) override;

    /**
     * @brief Set result of #GetROS2Data to InMessage.
     *
     * @param InMessage
     */
    virtual void SetROS2Msg(UROS2GenericMsg* InMessage) override;

    virtual void SensorUpdate() override;
protected:
	FVector ApplyGPSError(const FVector& TruePosition);
	AGeoReferencingSystem* sys;
	FROSNavSatFix fix;

private:
	UPROPERTY(EditDefaultsOnly, Category = "GPS Error")
    float SigmaWhiteNoise;

    UPROPERTY(EditDefaultsOnly, Category = "GPS Error")
    float SigmaDrift;

    UPROPERTY(EditDefaultsOnly, Category = "GPS Error")
    float PJump;

    UPROPERTY(EditDefaultsOnly, Category = "GPS Error")
    float MuJump;

    UPROPERTY(EditDefaultsOnly, Category = "GPS Error")
    float SigmaJump;

	UPROPERTY(EditDefaultsOnly, Category = "GPS Error")
	float Alpha;

	FVector Drift;
	FVector JumpEffect;

	std::default_random_engine RandomGenerator;
	std::normal_distribution<float> NormalDistribution;

	FVector GenerateGaussianNoise(float Mean, float StdDev);
};