/**
 * @file RRROS2DepthCameraActor.h
 * @brief Standalone ROS2 Camera actor formed by #URRROS2CameraComponent
 * @copyright Copyright 2020-2022 Rapyuta Robotics Co., Ltd.
 */

#pragma once

#include "CoreMinimal.h"
#include "RRROS2DepthCameraComponent.h"
#include "RRROS2RgbCameraComponent.h"

// rclUE
#include "ROS2NodeComponent.h"

#include "RRROS2DepthCameraActor.generated.h"

/**
 * @brief Standalone ROS2 camera actor which can be placed in the level with #URRROS2CameraComponent.
 */
UCLASS()
class EVANSLABSENSORS_API ARRROS2DepthCameraActor : public AActor
{
    GENERATED_BODY()

public:
    /**
     * @brief Construct a new ARRROS2DepthCameraActor object
     *
     */
    ARRROS2DepthCameraActor();

    UPROPERTY(Transient)
    UROS2NodeComponent* Node;

    UPROPERTY(BlueprintReadWrite)
    FString NodeName;

    UPROPERTY(BlueprintReadWrite)
    FString NodeNamespace;

protected:
    /**
     * @brief Initialize ROS2 Node and #CameraComponent
     * If NodeName is empty, node name become This actor name + _RRROS2CameraNode
     */
    virtual void BeginPlay() override;

    //! ROS2 camera component for this camera
    UPROPERTY(Category = CameraActor, VisibleAnywhere, BlueprintReadOnly, meta = (AllowPrivateAccess = "true"))
    URRROS2DepthCameraComponent* DepthCameraComponent;

    UPROPERTY(Category = CameraActor, VisibleAnywhere, BlueprintReadOnly, meta = (AllowPrivateAccess = "true"))
    URRROS2RgbCameraComponent* RgbCameraComponent;

};
