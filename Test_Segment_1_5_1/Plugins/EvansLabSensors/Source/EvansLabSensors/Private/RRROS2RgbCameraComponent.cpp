// Copyright 2020-2021 Rapyuta Robotics Co., Ltd.

#include "RRROS2RgbCameraComponent.h"

URRROS2RgbCameraComponent::URRROS2RgbCameraComponent()
{
    // component initialization
    RgbSceneCaptureComponent = CreateDefaultSubobject<USceneCaptureComponent2D>(TEXT("RgbSceneCaptureComponent"));
    RgbSceneCaptureComponent->SetupAttachment(this);

    CameraComponent = CreateDefaultSubobject<UCameraComponent>(TEXT("CameraComponent"));
    CameraComponent->SetupAttachment(this);

    SensorPublisherClass = URRROS2ImagePublisher::StaticClass();
}

void URRROS2RgbCameraComponent::PreInitializePublisher(UROS2NodeComponent* InROS2Node, const FString& InTopicName)
{
    // Rgb
    RgbSceneCaptureComponent->FOVAngle = CameraComponent->FieldOfView;
    RgbSceneCaptureComponent->OrthoWidth = CameraComponent->OrthoWidth;

    RgbRenderTarget = NewObject<UTextureRenderTarget2D>(this, UTextureRenderTarget2D::StaticClass());
    RgbRenderTarget->InitCustomFormat(Width, Height, EPixelFormat::PF_B8G8R8A8, true);
    RgbSceneCaptureComponent->TextureTarget = RgbRenderTarget;

    // Initialize image data
    Data.Header.FrameId = FrameId;
    Data.Width = Width;
    Data.Height = Height;
    Data.Encoding = Encoding;
    Data.Step = Width * 3;    // todo should be variable based on encoding
    Data.Data.AddUninitialized(Width * Height * 3);

    QueueSize = QueueSize < 1 ? 1 : QueueSize;    // QueueSize should be more than 1

    Super::PreInitializePublisher(InROS2Node, InTopicName);
}

void URRROS2RgbCameraComponent::SensorUpdate()
{
    // Rgb
    RgbSceneCaptureComponent->CaptureScene();
    CaptureRgbNonBlocking();
}

// reference https://github.com/TimmHess/UnrealImageCapture
void URRROS2RgbCameraComponent::CaptureRgbNonBlocking()
{
    RgbSceneCaptureComponent->TextureTarget->TargetGamma = GEngine->GetDisplayGamma();
    FTextureRenderTargetResource* rgbRenderTargetResource = RgbSceneCaptureComponent->TextureTarget->GameThread_GetRenderTargetResource();

    struct FReadRgbSurfaceContext
    {
        FRenderTarget* SrcRenderTarget;
        TArray<FColor>* OutData;
        FIntRect Rect;
        FReadSurfaceDataFlags Flags;
    };

    FRenderRequest* rgbRenderRequest = new FRenderRequest();

    FReadRgbSurfaceContext readRgbSurfaceContext = {
        rgbRenderTargetResource,
        &(rgbRenderRequest->Image),
        FIntRect(0, 0, rgbRenderTargetResource->GetSizeXY().X, rgbRenderTargetResource->GetSizeXY().Y),
        FReadSurfaceDataFlags(RCM_UNorm, CubeFace_MAX)
    };

    ENQUEUE_RENDER_COMMAND(SceneDrawCompletion)
    (
        [readRgbSurfaceContext, this](FRHICommandListImmediate& RHICmdList)
        {
            RHICmdList.ReadSurfaceData(readRgbSurfaceContext.SrcRenderTarget->GetRenderTargetTexture(),
                                       readRgbSurfaceContext.Rect,
                                       *readRgbSurfaceContext.OutData,
                                       readRgbSurfaceContext.Flags);
        });

    RgbRenderRequestQueue.Enqueue(static_cast<FRenderRequest *>(rgbRenderRequest));
    if (RgbQueueCount > QueueSize)
    {
        RgbRenderRequestQueue.Pop();
    }
    else
    {
        RgbQueueCount++;
    }

    rgbRenderRequest->RenderFence.BeginFence();
}


FROSImg URRROS2RgbCameraComponent::GetROS2Data()
{
    if (!RgbRenderRequestQueue.IsEmpty())
    {
        // Peek the next RenderRequest from queue

        //Rgb
        FRenderRequest* nextRgbRenderRequest = nullptr;
        RgbRenderRequestQueue.Peek(nextRgbRenderRequest);

        if (nextRgbRenderRequest)
        {    // nullptr check
            if (nextRgbRenderRequest->RenderFence.IsFenceComplete())
            {    // Check if rendering is done, indicated by RenderFence
                auto world = GetWorld();
                ensure(world);
                Data.Header.Stamp = URRConversionUtils::FloatToROSStamp(UGameplayStatics::GetTimeSeconds(world));
                int sec = URRConversionUtils::FloatToROSStamp(UGameplayStatics::GetTimeSeconds(world)).Sec;

                for (int I = 0; I < nextRgbRenderRequest->Image.Num(); I++)
                {
                    Data.Data[I * 3 + 0] = nextRgbRenderRequest->Image[I].R;
                    Data.Data[I * 3 + 1] = nextRgbRenderRequest->Image[I].G;
                    Data.Data[I * 3 + 2] = nextRgbRenderRequest->Image[I].B;
                }
                // Rgb
                RgbRenderRequestQueue.Pop();
                RgbQueueCount--;
                delete nextRgbRenderRequest;
            }
        }
    }

    return Data;
}

void URRROS2RgbCameraComponent::SetROS2Msg(UROS2GenericMsg* InMessage)
{
    CastChecked<UROS2ImgMsg>(InMessage)->SetMsg(GetROS2Data());
}
