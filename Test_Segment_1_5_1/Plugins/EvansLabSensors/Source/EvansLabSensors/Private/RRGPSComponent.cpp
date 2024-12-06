// Fill out your copyright notice in the Description page of Project Settings.


#include "RRGPSComponent.h"
#include "RRROS2GPSPublisher.h"
#include "Math/UnrealMathUtility.h"
#include "Core/RRConversionUtils.h"

URRGPSComponent::URRGPSComponent()
{
    SensorPublisherClass = URRROS2GPSPublisher::StaticClass();

    //Noise Parameters
    SigmaWhiteNoise = 1.0f;

	//Drift Parameters
    SigmaDrift = 0.1f;

	//Jump Parameters
    PJump = 0.01f;
    MuJump = 0.0f;
    SigmaJump = 5.0f;
    Alpha = 0.9f;

    Drift = FVector::ZeroVector;
    JumpEffect = FVector::ZeroVector;
}

void URRGPSComponent::PreInitializePublisher(UROS2NodeComponent *InROS2Node, const FString &InTopicName)
{
	sys = AGeoReferencingSystem::GetGeoReferencingSystem(GetWorld());
	RandomGenerator = std::default_random_engine(static_cast<unsigned int>(FDateTime::Now().GetMillisecond()));
    Super::PreInitializePublisher(InROS2Node, InTopicName);
}

void URRGPSComponent::SetROS2Msg(UROS2GenericMsg* InMessage) {
	CastChecked<UROS2NavSatFixMsg>(InMessage)->SetMsg(fix);
}

void URRGPSComponent::SensorUpdate() {
    // HEY, DUMBASS, YOU'VE HAD THIS ISSUE BEFORE.  RE-ADD THE GPS COMPONENT
    // AND GO ABOUT YOUR DAY.  NOTHING WRONG HERE.
	FGeographicCoordinates coords;
	FVector Loc = this->GetComponentLocation();
    //UE_LOG(LogTemp, Warning, TEXT("Component Location: x=%f, y=%f, z=%f"), Loc.X, Loc.Y, Loc.Z);
    //FVector ParentLoc = GetOwner()->GetActorLocation();
    //UE_LOG(LogTemp, Warning, TEXT("Component Location: x=%f, y=%f, z=%f"), ParentLoc.X, ParentLoc.Y, ParentLoc.Z);

    Loc = ApplyGPSError(Loc);

	sys->EngineToGeographic(Loc, coords);

	//UE_LOG(LogTemp, Warning, TEXT("%s"), *coords.ToFullText().ToString());

    fix.Header.Stamp = URRConversionUtils::FloatToROSStamp(UGameplayStatics::GetTimeSeconds(GetWorld()));
	fix.Header.FrameId = FrameId;

	fix.Latitude = coords.Latitude;
	fix.Longitude = coords.Longitude;
}

FVector URRGPSComponent::ApplyGPSError(const FVector &TruePosition)
{
    FVector RandError = GenerateGaussianNoise(0, SigmaWhiteNoise);

    Drift += GenerateGaussianNoise(0, SigmaDrift);

    float UniformRandom = FMath::RandRange(0.0f, 1.0f);

    if (UniformRandom < PJump)
    {
        FVector Jump = GenerateGaussianNoise(MuJump, SigmaJump);

        // Generate a random float between -1.0f and 1.0f
        float RandomFloat = FMath::RandRange(-1.0f, 1.0f);

        // Calculate the sign
        float Sign = FMath::Sign(RandomFloat);

        // Multiply the Jump vector by the sign
        Jump *= Sign;

        JumpEffect += Jump;
    }

    JumpEffect *= Alpha;

    FVector MeasuredPosition = TruePosition + RandError + Drift + JumpEffect;
    return MeasuredPosition;
}

FVector URRGPSComponent::GenerateGaussianNoise(float Mean, float StdDev)
{
    float U1 = FMath::FRand();
    float U2 = FMath::FRand();

    // Box-Muller transform
    float Z0 = FMath::Sqrt(-2.0f * FMath::Loge(U1)) * FMath::Cos(2.0f * PI * U2);
    float Z1 = FMath::Sqrt(-2.0f * FMath::Loge(U1)) * FMath::Sin(2.0f * PI * U2);

    FVector Noise(Mean + Z0 * StdDev, Mean + Z1 * StdDev, 0.0f);
    return Noise;
}
