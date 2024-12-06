#pragma once

#include "Camera/CameraComponent.h"
#include "Components/SceneCaptureComponent2D.h"
#include "CoreMinimal.h"
#include "Engine/TextureRenderTarget2D.h"

// rclUE
#include <Msgs/ROS2Img.h>

// RapyutaSimulationPlugins
#include "Sensors/RRROS2BaseSensorComponent.h"
#include "Tools/RRROS2ImagePublisher.h"
#include "Sensors/RRROS2CameraComponent.h"

#include "RRROS2RgbCameraComponent.generated.h"

USTRUCT()
struct FFloatRenderRequest {
    GENERATED_BODY()

    TArray<FFloat16Color> Image;
    FRenderCommandFence RenderFence;

    FFloatRenderRequest() {}
};

union Float16ToBytes {
    FFloat16 Value;
    uint8 Bytes[2];

    Float16ToBytes(FFloat16 val) {
        Value = val;
    }

    FFloat16 GetFloat() {
        return Value;
    }
};


/**
 * @brief ROS2 Camera component. Uses USceneCaptureComponent2D.
 *
 * @sa [USceneCaptureComponent2D](https://docs.unrealengine.com/5.1/en-US/API/Runtime/Engine/Components/USceneCaptureComponent2D/)
 * @sa [UE4 ShaderInPlugin](https://docs.unrealengine.com/5.1/en-US/ProgrammingAndScripting/Rendering/ShaderInPlugin/Overview/)
 * @sa implementation reference: https://github.com/TimmHess/UnrealImageCapture
 * @todo Support non RGB data support.
 */
UCLASS(ClassGroup = (Custom), Blueprintable, meta = (BlueprintSpawnableComponent))
class EVANSLABSENSORS_API URRROS2RgbCameraComponent : public URRROS2BaseSensorComponent
{
    GENERATED_BODY()

public:
    /**
     * @brief Construct a new URRROS2RgbCameraComponent object
     * Initialize #SceneCaptureComponent and #CameraComponent.
     */
    URRROS2RgbCameraComponent();

    /**
     * @brief Initialize #Data and #RenderTarget, set #SceneCaptureComponent parameters.
     *
     * @param InROS2Node ROS2Node which this publisher belongs to
     * @param InTopicName
     */
    virtual void PreInitializePublisher(UROS2NodeComponent* InROS2Node, const FString& InTopicName) override;

    /**
     * @brief Update sensor data by CaptureScene and #CaptureNonBlocking
     * @sa [CaptureScene](https://docs.unrealengine.com/5.1/en-US/API/Runtime/Engine/Components/USceneCaptureComponent2D/CaptureScene/)
     * @todo Should #CaptureNonBlocking called in TickComponents?
     */
    virtual void SensorUpdate() override;

protected:
    /**
     * @brief Capture data by notifing task in #RenderRequestQueue
     * @sa reference https://github.com/TimmHess/UnrealImageCapture
     */
    // Rgb
    UFUNCTION()
    void CaptureRgbNonBlocking();

    TQueue<FRenderRequest*> RgbRenderRequestQueue;
    int32 RgbQueueCount = 0;

    //!
    FROSImg Data;


public:
    //! Camera. Not necessary to capture but useful to see image in UE4 windows.
    UPROPERTY(VisibleAnywhere, BlueprintReadWrite)
    UCameraComponent* CameraComponent = nullptr;

    UPROPERTY(VisibleAnywhere, BlueprintReadWrite)
    USceneCaptureComponent2D* RgbSceneCaptureComponent = nullptr;

    UPROPERTY(VisibleAnywhere, BlueprintReadWrite)
    UTextureRenderTarget2D* RgbRenderTarget = nullptr;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    int32 Width = 1280;//640 standard setting to OAKD 1280

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    int32 Height = 720;//480 standard setting to OAKD 720

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    int32 QueueSize = 2;

    // ROS
    /**
     * @brief Update ROS2 Msg structure from #RenderRequestQueue
     *
     * @return FROSImg
     */
    UFUNCTION(BlueprintCallable)
    virtual FROSImg GetROS2Data();

    /**
     * @brief Set result of #GetROS2Data to InMessage.
     *
     * @param InMessage
     */
    virtual void SetROS2Msg(UROS2GenericMsg* InMessage) override;

    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString Encoding = TEXT("rgb16");
};
