// Copyright 2020-2021 Rapyuta Robotics Co., Ltd.

#include "RRROS2DepthCameraActor.h"

ARRROS2DepthCameraActor::ARRROS2DepthCameraActor()
{
	
	// Setup camera defaults
	DepthCameraComponent = CreateDefaultSubobject<URRROS2DepthCameraComponent>(TEXT("RgbCameraComponent"));
	DepthCameraComponent->CameraComponent->FieldOfView = 90.0f;
	DepthCameraComponent->CameraComponent->bConstrainAspectRatio = true;
	DepthCameraComponent->CameraComponent->AspectRatio = 1.777778f;
	DepthCameraComponent->CameraComponent->PostProcessBlendWeight = 1.0f;

	RgbCameraComponent = CreateDefaultSubobject<URRROS2RgbCameraComponent>(TEXT("DepthCameraComponent"));
	RgbCameraComponent->CameraComponent->FieldOfView = 90.0f;
	RgbCameraComponent->CameraComponent->bConstrainAspectRatio = true;
	RgbCameraComponent->CameraComponent->AspectRatio = 1.777778f;
	RgbCameraComponent->CameraComponent->PostProcessBlendWeight = 1.0f;

	
	RootComponent = DepthCameraComponent;
}

void ARRROS2DepthCameraActor::BeginPlay()
{
    Super::BeginPlay();
	
	// Node initialize
    Node = NewObject<UROS2NodeComponent>(this);
	Node->Name =
        NodeName.IsEmpty() ? FString::Printf(TEXT("%s_RRROS2CameraNode"), *(GetName())) : NodeName;
	Node->Namespace = NodeNamespace;
    Node->Init();
    
	RgbCameraComponent->InitalizeWithROS2(Node);
	DepthCameraComponent->InitalizeWithROS2(Node);
}
 