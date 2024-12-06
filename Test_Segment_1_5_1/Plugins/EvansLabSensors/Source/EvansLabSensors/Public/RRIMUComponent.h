// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Sensors/RRROS2BaseSensorComponent.h"

#include "Tools/RRROS2BaseSensorPublisher.h"
#include "Msgs/ROS2Imu.h"
#include "PhysicsEngine/PhysicsConstraintComponent.h"

#include "RRIMUComponent.generated.h"

/**
 *
 */
UCLASS(ClassGroup = (Custom), Blueprintable, meta = (BlueprintSpawnableComponent))
class EVANSLABSENSORS_API URRIMUComponent : public URRROS2BaseSensorComponent
{
    GENERATED_BODY()

public:
    /**
     * @brief Construct a new URRIMUComponent object
     */
    URRIMUComponent();

    /**
     * @brief Initialize publisher
     *
     * @param InROS2Node ROS2Node which this publisher belongs to
     * @param InTopicName
     */
    virtual void PreInitializePublisher(UROS2NodeComponent *InROS2Node, const FString &InTopicName) override;

    /**
     * @brief Set result of IMU data to InMessage.
     *
     * @param InMessage
     */
    virtual void SetROS2Msg(UROS2GenericMsg* InMessage) override;

    virtual void SensorUpdate() override;
    FVector CalculateAcceleration();
    FVector CalculateAngularVelocity();

    virtual void Run() override;
    virtual void Stop() override;
    void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction);

protected:
    AGeoReferencingSystem* sys;
    FROSImu imuData;
    FVector LastLinearVelocity;
    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Inertial Mass", meta = (AllowPrivateAccess = "true"))
    UStaticMeshComponent* InertialMass;

    UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Spring Constraint", meta = (AllowPrivateAccess = "true"))
    UPhysicsConstraintComponent* SpringConstraint;

    // Tunable Parameters
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IMU Noise")
    float noise_accel_stddev_x = 0.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IMU Noise")
    float noise_accel_stddev_y = 0.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IMU Noise")
    float noise_accel_stddev_z = 0.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IMU Noise")
    float noise_gyro_bias_x = 0.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IMU Noise")
    float noise_gyro_bias_y = 0.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IMU Noise")
    float noise_gyro_bias_z = 0.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IMU Noise")
    float noise_gyro_stddev_x = 0.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IMU Noise")
    float noise_gyro_stddev_y = 0.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IMU Noise")
    float noise_gyro_stddev_z = 0.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IMU Noise")
    int noise_seed = 0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "IMU Noise")
    float sensor_tick = 0.0f;
};
