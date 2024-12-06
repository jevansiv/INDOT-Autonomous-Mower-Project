// Copyright 2020-2021 Rapyuta Robotics Co., Ltd.

#include "RRROS2DepthCameraComponent.h"

URRROS2DepthCameraComponent::URRROS2DepthCameraComponent()
{
    // component initialization
    DepthSceneCaptureComponent = CreateDefaultSubobject<USceneCaptureComponent2D>(TEXT("DepthSceneCaptureComponent"));
    DepthSceneCaptureComponent->SetupAttachment(this);

    CameraComponent = CreateDefaultSubobject<UCameraComponent>(TEXT("CameraComponent"));
    CameraComponent->SetupAttachment(this);

    SensorPublisherClass = URRROS2ImagePublisher::StaticClass();
}

void URRROS2DepthCameraComponent::PreInitializePublisher(UROS2NodeComponent* InROS2Node, const FString& InTopicName)
{
    // Depth
    DepthSceneCaptureComponent->FOVAngle = CameraComponent->FieldOfView;
    DepthSceneCaptureComponent->OrthoWidth = CameraComponent->OrthoWidth;

    DepthRenderTarget = NewObject<UTextureRenderTarget2D>(this, UTextureRenderTarget2D::StaticClass());
    DepthRenderTarget->InitCustomFormat(Width, Height, EPixelFormat::PF_FloatRGBA, true);
    DepthSceneCaptureComponent->TextureTarget = DepthRenderTarget;

    // Initialize image data
    Data.Header.FrameId = FrameId;
    Data.Width = Width;
    Data.Height = Height;
    Data.Encoding = Encoding;
    Data.Step = Width * 4;    // todo should be variable based on encoding
    Data.Data.AddUninitialized(Width * Height * 4);

    QueueSize = QueueSize < 1 ? 1 : QueueSize;    // QueueSize should be more than 1

    Super::PreInitializePublisher(InROS2Node, InTopicName);
}

void URRROS2DepthCameraComponent::SensorUpdate()
{
    // Depth
    DepthSceneCaptureComponent->CaptureScene();
    CaptureDepthNonBlocking();
}

// reference https://github.com/TimmHess/UnrealImageCapture
void URRROS2DepthCameraComponent::CaptureDepthNonBlocking()
{
    DepthSceneCaptureComponent->TextureTarget->TargetGamma = GEngine->GetDisplayGamma();
    FTextureRenderTargetResource* depthRenderTargetResource = DepthSceneCaptureComponent->TextureTarget->GameThread_GetRenderTargetResource();

    struct FReadDepthSurfaceContext
    {
        FRenderTarget* SrcRenderTarget;
        TArray<FFloat16Color>* OutData;
        FIntRect Rect;
        ECubeFace CubeFace;
    };

    FFloatRenderRequest* depthRenderRequest = new FFloatRenderRequest();

    FReadDepthSurfaceContext readDepthSurfaceContext = {
        depthRenderTargetResource,
        &(depthRenderRequest->Image),
        FIntRect(0, 0, depthRenderTargetResource->GetSizeXY().X, depthRenderTargetResource->GetSizeXY().Y),
        ECubeFace::CubeFace_MAX
    };

    ENQUEUE_RENDER_COMMAND(DepthSceneDrawCompletion)
    (
        [readDepthSurfaceContext, this](FRHICommandListImmediate& RHICmdList)
        {
            RHICmdList.ReadSurfaceFloatData(readDepthSurfaceContext.SrcRenderTarget->GetRenderTargetTexture(),
                                       readDepthSurfaceContext.Rect,
                                       *readDepthSurfaceContext.OutData,
                                       readDepthSurfaceContext.CubeFace,
                                       0,
                                       0);
        });

    DepthRenderRequestQueue.Enqueue(depthRenderRequest);
    if (DepthQueueCount > QueueSize)
    {
        DepthRenderRequestQueue.Pop();
    }
    else
    {
        DepthQueueCount++;
    }

    depthRenderRequest->RenderFence.BeginFence();
}

FROSImg URRROS2DepthCameraComponent::GetROS2Data()
{
    if (!DepthRenderRequestQueue.IsEmpty())
    {
        // Peek the next RenderRequest from queue

        //Depth
        FFloatRenderRequest* nextDepthRenderRequest = nullptr;
        DepthRenderRequestQueue.Peek(nextDepthRenderRequest);

        if (nextDepthRenderRequest)
        {    // nullptr check
            if (nextDepthRenderRequest->RenderFence.IsFenceComplete())
            {    // Check if rendering is done, indicated by RenderFence
                Data.Header.Stamp = URRConversionUtils::FloatToROSStamp(UGameplayStatics::GetTimeSeconds(GetWorld()));
                for (int I = 0; I < nextDepthRenderRequest->Image.Num(); I++)
                {

                    FFloat16 depth (nextDepthRenderRequest->Image[I].A);
                    Float16To4Bytes d;
                    if(depth <= MaxDepth) {
                        d.SetValue(depth.GetFloat());
                    }

                    Data.Data[I * 4 + 0] = d.Bytes[0];
                    Data.Data[I * 4 + 1] = d.Bytes[1];
                    Data.Data[I * 4 + 2] = d.Bytes[2];
                    Data.Data[I * 4 + 3] = d.Bytes[3];

                }

                // Depth
                DepthRenderRequestQueue.Pop();
                DepthQueueCount--;
                delete nextDepthRenderRequest;
            }
        }
    }

    return Data;
}

void URRROS2DepthCameraComponent::SetROS2Msg(UROS2GenericMsg* InMessage)
{
    CastChecked<UROS2ImgMsg>(InMessage)->SetMsg(GetROS2Data());
}
