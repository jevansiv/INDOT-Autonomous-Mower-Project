// Fill out your copyright notice in the Description page of Project Settings.

#include "RRIMUComponent.h"
#include "GeoReferencingSystem.h"
#include "Msgs/ROS2Imu.h"
#include "Core/RRConversionUtils.h"

URRIMUComponent::URRIMUComponent()
{
    SensorPublisherClass = URRROS2IMUPublisher::StaticClass();

    PrimaryComponentTick.bCanEverTick = true;
    PrimaryComponentTick.TickGroup = TG_PrePhysics;
    PrimaryComponentTick.bStartWithTickEnabled = true;
}

void URRIMUComponent::PreInitializePublisher(UROS2NodeComponent *InROS2Node, const FString &InTopicName)
{
    sys = AGeoReferencingSystem::GetGeoReferencingSystem(GetWorld());
    Super::PreInitializePublisher(InROS2Node, InTopicName);
}

void URRIMUComponent::SetROS2Msg(UROS2GenericMsg *InMessage)
{
    CastChecked<UROS2ImuMsg>(InMessage)->SetMsg(imuData);
}

void URRIMUComponent::SensorUpdate()
{
    FTransform ComponentTransform = GetComponentTransform();
    FVector Acceleration = CalculateAcceleration();
    FVector AngularVelocity = CalculateAngularVelocity();
    FQuat Orientation = ComponentTransform.GetRotation().Inverse();

    FVector East, North, Up;
    sys->GetENUVectorsAtEngineLocation(ComponentTransform.GetLocation(), East, North, Up);
    FMatrix UnrealToENU = FMatrix(East, North, Up, FVector::ZeroVector);

    FQuat ENUOrientation;

    try {
        ENUOrientation = UnrealToENU.ToQuat() * Orientation;
    } catch (const std::exception& e) {
        FString MatrixString;
        for (int32 Row = 0; Row < 4; Row++) {
            FString RowString;
            for (int32 Col = 0; Col < 4; Col++) {
                RowString += FString::Printf(TEXT("%f\t"), UnrealToENU.M[Row][Col]);
            }
            MatrixString += RowString + "\n";
        }

        UE_LOG(LogTemp, Warning, TEXT("UnrealToENU matrix is not orthonormal! Matrix:\n%s"), *MatrixString);
        return;
    }

    imuData.Header.Stamp = URRConversionUtils::FloatToROSStamp(UGameplayStatics::GetTimeSeconds(GetWorld()));
    imuData.Header.FrameId = FrameId;

    imuData.Orientation.X = ENUOrientation.X;
    imuData.Orientation.Y = ENUOrientation.Y;
    imuData.Orientation.Z = ENUOrientation.Z;
    imuData.Orientation.W = ENUOrientation.W;

    // Linear acceleration in m/s^2
    imuData.LinearAcceleration.X = Acceleration.X / 100.0f;
    imuData.LinearAcceleration.Y = -Acceleration.Y / 100.0f;
    imuData.LinearAcceleration.Z = Acceleration.Z / 100.0f;

    // Angular velocity in rad/s
    imuData.AngularVelocity.X = FMath::DegreesToRadians(AngularVelocity.X);
    imuData.AngularVelocity.Y = -FMath::DegreesToRadians(AngularVelocity.Y);
    imuData.AngularVelocity.Z = -FMath::DegreesToRadians(AngularVelocity.Z);

    // Angular velocity covariance (wx, wy, wz)
    float angular_velocity_covariance = 0.001; // Example value, adjust according to your sensor
    for (int i = 0; i < 9; i += 4) {
        imuData.AngularVelocityCovariance[i] = angular_velocity_covariance;
    }

    // Linear acceleration covariance (ax, ay, az)
    float linear_acceleration_covariance = 0.1; // Example value, adjust according to your sensor
    for (int i = 0; i < 9; i += 4) {
        imuData.LinearAccelerationCovariance[i] = linear_acceleration_covariance;
    }

    float orientation_covariance = 0.01; // Example value, adjust according to your sensor
    for (int i = 0; i < 9; i += 4) {
        imuData.OrientationCovariance[i] = orientation_covariance;
    }
}

FVector URRIMUComponent::CalculateAcceleration()
{
    // Get the owner's primitive component
    UPrimitiveComponent* OwnerPrimitiveComponent = Cast<UPrimitiveComponent>(GetOwner()->GetRootComponent());
    if (OwnerPrimitiveComponent)
    {
        FVector LinearVelocity = OwnerPrimitiveComponent->GetPhysicsLinearVelocity();
        float DeltaTime = GetWorld()->GetDeltaSeconds();

        FVector Acceleration = (LinearVelocity - LastLinearVelocity) / DeltaTime;

        FVector LocalAcceleration = GetComponentTransform().InverseTransformVector(Acceleration);

        LastLinearVelocity = LinearVelocity;

        // Add noise to acceleration
        LocalAcceleration.X += FMath::FRandRange(-noise_accel_stddev_x, noise_accel_stddev_x);
        LocalAcceleration.Y += FMath::FRandRange(-noise_accel_stddev_y, noise_accel_stddev_y);
        LocalAcceleration.Z += FMath::FRandRange(-noise_accel_stddev_z, noise_accel_stddev_z);

        return LocalAcceleration;
    }
    return FVector::ZeroVector;
}

FVector URRIMUComponent::CalculateAngularVelocity()
{
    // Get the owner's primitive component
    UPrimitiveComponent* OwnerPrimitiveComponent = Cast<UPrimitiveComponent>(GetOwner()->GetRootComponent());

    if (OwnerPrimitiveComponent)
    {
        FVector AngularVelocity = OwnerPrimitiveComponent->GetPhysicsAngularVelocityInDegrees();

        // Add noise to angular velocity
        AngularVelocity.X += FMath::FRandRange(-noise_gyro_stddev_x, noise_gyro_stddev_x) + noise_gyro_bias_x;
        AngularVelocity.Y += FMath::FRandRange(-noise_gyro_stddev_y, noise_gyro_stddev_y) + noise_gyro_bias_y;
        AngularVelocity.Z += FMath::FRandRange(-noise_gyro_stddev_z, noise_gyro_stddev_z) + noise_gyro_bias_z;

        return AngularVelocity;
    }
    return FVector::ZeroVector;
}

void URRIMUComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
    SensorUpdate();
}

void URRIMUComponent::Run() {}
void URRIMUComponent::Stop() {}
