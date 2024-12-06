// Copyright Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Generated code exported from UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "UObject/GeneratedCppIncludes.h"
#include "EvansLabSensors/Public/RRIMUComponent.h"
PRAGMA_DISABLE_DEPRECATION_WARNINGS
void EmptyLinkFunctionForGeneratedCodeRRIMUComponent() {}
// Cross Module References
	ENGINE_API UClass* Z_Construct_UClass_UPhysicsConstraintComponent_NoRegister();
	ENGINE_API UClass* Z_Construct_UClass_UStaticMeshComponent_NoRegister();
	EVANSLABSENSORS_API UClass* Z_Construct_UClass_URRIMUComponent();
	EVANSLABSENSORS_API UClass* Z_Construct_UClass_URRIMUComponent_NoRegister();
	RAPYUTASIMULATIONPLUGINS_API UClass* Z_Construct_UClass_URRROS2BaseSensorComponent();
	UPackage* Z_Construct_UPackage__Script_EvansLabSensors();
// End Cross Module References
	void URRIMUComponent::StaticRegisterNativesURRIMUComponent()
	{
	}
	IMPLEMENT_CLASS_NO_AUTO_REGISTRATION(URRIMUComponent);
	UClass* Z_Construct_UClass_URRIMUComponent_NoRegister()
	{
		return URRIMUComponent::StaticClass();
	}
	struct Z_Construct_UClass_URRIMUComponent_Statics
	{
		static UObject* (*const DependentSingletons[])();
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Class_MetaDataParams[];
#endif
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_InertialMass_MetaData[];
#endif
		static const UECodeGen_Private::FObjectPropertyParams NewProp_InertialMass;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_SpringConstraint_MetaData[];
#endif
		static const UECodeGen_Private::FObjectPropertyParams NewProp_SpringConstraint;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_noise_accel_stddev_x_MetaData[];
#endif
		static const UECodeGen_Private::FFloatPropertyParams NewProp_noise_accel_stddev_x;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_noise_accel_stddev_y_MetaData[];
#endif
		static const UECodeGen_Private::FFloatPropertyParams NewProp_noise_accel_stddev_y;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_noise_accel_stddev_z_MetaData[];
#endif
		static const UECodeGen_Private::FFloatPropertyParams NewProp_noise_accel_stddev_z;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_noise_gyro_bias_x_MetaData[];
#endif
		static const UECodeGen_Private::FFloatPropertyParams NewProp_noise_gyro_bias_x;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_noise_gyro_bias_y_MetaData[];
#endif
		static const UECodeGen_Private::FFloatPropertyParams NewProp_noise_gyro_bias_y;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_noise_gyro_bias_z_MetaData[];
#endif
		static const UECodeGen_Private::FFloatPropertyParams NewProp_noise_gyro_bias_z;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_noise_gyro_stddev_x_MetaData[];
#endif
		static const UECodeGen_Private::FFloatPropertyParams NewProp_noise_gyro_stddev_x;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_noise_gyro_stddev_y_MetaData[];
#endif
		static const UECodeGen_Private::FFloatPropertyParams NewProp_noise_gyro_stddev_y;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_noise_gyro_stddev_z_MetaData[];
#endif
		static const UECodeGen_Private::FFloatPropertyParams NewProp_noise_gyro_stddev_z;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_noise_seed_MetaData[];
#endif
		static const UECodeGen_Private::FUnsizedIntPropertyParams NewProp_noise_seed;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_sensor_tick_MetaData[];
#endif
		static const UECodeGen_Private::FFloatPropertyParams NewProp_sensor_tick;
		static const UECodeGen_Private::FPropertyParamsBase* const PropPointers[];
		static const FCppClassTypeInfoStatic StaticCppClassTypeInfo;
		static const UECodeGen_Private::FClassParams ClassParams;
	};
	UObject* (*const Z_Construct_UClass_URRIMUComponent_Statics::DependentSingletons[])() = {
		(UObject* (*)())Z_Construct_UClass_URRROS2BaseSensorComponent,
		(UObject* (*)())Z_Construct_UPackage__Script_EvansLabSensors,
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRIMUComponent_Statics::Class_MetaDataParams[] = {
		{ "BlueprintSpawnableComponent", "" },
		{ "BlueprintType", "true" },
		{ "ClassGroupNames", "Custom" },
		{ "Comment", "/**\n *\n */" },
		{ "HideCategories", "Trigger PhysicsVolume" },
		{ "IncludePath", "RRIMUComponent.h" },
		{ "IsBlueprintBase", "true" },
		{ "ModuleRelativePath", "Public/RRIMUComponent.h" },
	};
#endif
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRIMUComponent_Statics::NewProp_InertialMass_MetaData[] = {
		{ "AllowPrivateAccess", "true" },
		{ "Category", "Inertial Mass" },
		{ "EditInline", "true" },
		{ "ModuleRelativePath", "Public/RRIMUComponent.h" },
	};
#endif
	const UECodeGen_Private::FObjectPropertyParams Z_Construct_UClass_URRIMUComponent_Statics::NewProp_InertialMass = { "InertialMass", nullptr, (EPropertyFlags)0x002008000008001d, UECodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRIMUComponent, InertialMass), Z_Construct_UClass_UStaticMeshComponent_NoRegister, METADATA_PARAMS(Z_Construct_UClass_URRIMUComponent_Statics::NewProp_InertialMass_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRIMUComponent_Statics::NewProp_InertialMass_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRIMUComponent_Statics::NewProp_SpringConstraint_MetaData[] = {
		{ "AllowPrivateAccess", "true" },
		{ "Category", "Spring Constraint" },
		{ "EditInline", "true" },
		{ "ModuleRelativePath", "Public/RRIMUComponent.h" },
	};
#endif
	const UECodeGen_Private::FObjectPropertyParams Z_Construct_UClass_URRIMUComponent_Statics::NewProp_SpringConstraint = { "SpringConstraint", nullptr, (EPropertyFlags)0x002008000008001d, UECodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRIMUComponent, SpringConstraint), Z_Construct_UClass_UPhysicsConstraintComponent_NoRegister, METADATA_PARAMS(Z_Construct_UClass_URRIMUComponent_Statics::NewProp_SpringConstraint_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRIMUComponent_Statics::NewProp_SpringConstraint_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_accel_stddev_x_MetaData[] = {
		{ "Category", "IMU Noise" },
		{ "Comment", "// Tunable Parameters\n" },
		{ "ModuleRelativePath", "Public/RRIMUComponent.h" },
		{ "ToolTip", "Tunable Parameters" },
	};
#endif
	const UECodeGen_Private::FFloatPropertyParams Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_accel_stddev_x = { "noise_accel_stddev_x", nullptr, (EPropertyFlags)0x0020080000000005, UECodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRIMUComponent, noise_accel_stddev_x), METADATA_PARAMS(Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_accel_stddev_x_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_accel_stddev_x_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_accel_stddev_y_MetaData[] = {
		{ "Category", "IMU Noise" },
		{ "ModuleRelativePath", "Public/RRIMUComponent.h" },
	};
#endif
	const UECodeGen_Private::FFloatPropertyParams Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_accel_stddev_y = { "noise_accel_stddev_y", nullptr, (EPropertyFlags)0x0020080000000005, UECodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRIMUComponent, noise_accel_stddev_y), METADATA_PARAMS(Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_accel_stddev_y_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_accel_stddev_y_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_accel_stddev_z_MetaData[] = {
		{ "Category", "IMU Noise" },
		{ "ModuleRelativePath", "Public/RRIMUComponent.h" },
	};
#endif
	const UECodeGen_Private::FFloatPropertyParams Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_accel_stddev_z = { "noise_accel_stddev_z", nullptr, (EPropertyFlags)0x0020080000000005, UECodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRIMUComponent, noise_accel_stddev_z), METADATA_PARAMS(Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_accel_stddev_z_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_accel_stddev_z_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_bias_x_MetaData[] = {
		{ "Category", "IMU Noise" },
		{ "ModuleRelativePath", "Public/RRIMUComponent.h" },
	};
#endif
	const UECodeGen_Private::FFloatPropertyParams Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_bias_x = { "noise_gyro_bias_x", nullptr, (EPropertyFlags)0x0020080000000005, UECodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRIMUComponent, noise_gyro_bias_x), METADATA_PARAMS(Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_bias_x_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_bias_x_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_bias_y_MetaData[] = {
		{ "Category", "IMU Noise" },
		{ "ModuleRelativePath", "Public/RRIMUComponent.h" },
	};
#endif
	const UECodeGen_Private::FFloatPropertyParams Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_bias_y = { "noise_gyro_bias_y", nullptr, (EPropertyFlags)0x0020080000000005, UECodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRIMUComponent, noise_gyro_bias_y), METADATA_PARAMS(Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_bias_y_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_bias_y_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_bias_z_MetaData[] = {
		{ "Category", "IMU Noise" },
		{ "ModuleRelativePath", "Public/RRIMUComponent.h" },
	};
#endif
	const UECodeGen_Private::FFloatPropertyParams Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_bias_z = { "noise_gyro_bias_z", nullptr, (EPropertyFlags)0x0020080000000005, UECodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRIMUComponent, noise_gyro_bias_z), METADATA_PARAMS(Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_bias_z_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_bias_z_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_stddev_x_MetaData[] = {
		{ "Category", "IMU Noise" },
		{ "ModuleRelativePath", "Public/RRIMUComponent.h" },
	};
#endif
	const UECodeGen_Private::FFloatPropertyParams Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_stddev_x = { "noise_gyro_stddev_x", nullptr, (EPropertyFlags)0x0020080000000005, UECodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRIMUComponent, noise_gyro_stddev_x), METADATA_PARAMS(Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_stddev_x_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_stddev_x_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_stddev_y_MetaData[] = {
		{ "Category", "IMU Noise" },
		{ "ModuleRelativePath", "Public/RRIMUComponent.h" },
	};
#endif
	const UECodeGen_Private::FFloatPropertyParams Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_stddev_y = { "noise_gyro_stddev_y", nullptr, (EPropertyFlags)0x0020080000000005, UECodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRIMUComponent, noise_gyro_stddev_y), METADATA_PARAMS(Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_stddev_y_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_stddev_y_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_stddev_z_MetaData[] = {
		{ "Category", "IMU Noise" },
		{ "ModuleRelativePath", "Public/RRIMUComponent.h" },
	};
#endif
	const UECodeGen_Private::FFloatPropertyParams Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_stddev_z = { "noise_gyro_stddev_z", nullptr, (EPropertyFlags)0x0020080000000005, UECodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRIMUComponent, noise_gyro_stddev_z), METADATA_PARAMS(Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_stddev_z_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_stddev_z_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_seed_MetaData[] = {
		{ "Category", "IMU Noise" },
		{ "ModuleRelativePath", "Public/RRIMUComponent.h" },
	};
#endif
	const UECodeGen_Private::FUnsizedIntPropertyParams Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_seed = { "noise_seed", nullptr, (EPropertyFlags)0x0020080000000005, UECodeGen_Private::EPropertyGenFlags::Int, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRIMUComponent, noise_seed), METADATA_PARAMS(Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_seed_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_seed_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRIMUComponent_Statics::NewProp_sensor_tick_MetaData[] = {
		{ "Category", "IMU Noise" },
		{ "ModuleRelativePath", "Public/RRIMUComponent.h" },
	};
#endif
	const UECodeGen_Private::FFloatPropertyParams Z_Construct_UClass_URRIMUComponent_Statics::NewProp_sensor_tick = { "sensor_tick", nullptr, (EPropertyFlags)0x0020080000000005, UECodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRIMUComponent, sensor_tick), METADATA_PARAMS(Z_Construct_UClass_URRIMUComponent_Statics::NewProp_sensor_tick_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRIMUComponent_Statics::NewProp_sensor_tick_MetaData)) };
	const UECodeGen_Private::FPropertyParamsBase* const Z_Construct_UClass_URRIMUComponent_Statics::PropPointers[] = {
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRIMUComponent_Statics::NewProp_InertialMass,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRIMUComponent_Statics::NewProp_SpringConstraint,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_accel_stddev_x,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_accel_stddev_y,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_accel_stddev_z,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_bias_x,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_bias_y,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_bias_z,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_stddev_x,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_stddev_y,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_gyro_stddev_z,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRIMUComponent_Statics::NewProp_noise_seed,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRIMUComponent_Statics::NewProp_sensor_tick,
	};
	const FCppClassTypeInfoStatic Z_Construct_UClass_URRIMUComponent_Statics::StaticCppClassTypeInfo = {
		TCppClassTypeTraits<URRIMUComponent>::IsAbstract,
	};
	const UECodeGen_Private::FClassParams Z_Construct_UClass_URRIMUComponent_Statics::ClassParams = {
		&URRIMUComponent::StaticClass,
		"Engine",
		&StaticCppClassTypeInfo,
		DependentSingletons,
		nullptr,
		Z_Construct_UClass_URRIMUComponent_Statics::PropPointers,
		nullptr,
		UE_ARRAY_COUNT(DependentSingletons),
		0,
		UE_ARRAY_COUNT(Z_Construct_UClass_URRIMUComponent_Statics::PropPointers),
		0,
		0x00B000A4u,
		METADATA_PARAMS(Z_Construct_UClass_URRIMUComponent_Statics::Class_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UClass_URRIMUComponent_Statics::Class_MetaDataParams))
	};
	UClass* Z_Construct_UClass_URRIMUComponent()
	{
		if (!Z_Registration_Info_UClass_URRIMUComponent.OuterSingleton)
		{
			UECodeGen_Private::ConstructUClass(Z_Registration_Info_UClass_URRIMUComponent.OuterSingleton, Z_Construct_UClass_URRIMUComponent_Statics::ClassParams);
		}
		return Z_Registration_Info_UClass_URRIMUComponent.OuterSingleton;
	}
	template<> EVANSLABSENSORS_API UClass* StaticClass<URRIMUComponent>()
	{
		return URRIMUComponent::StaticClass();
	}
	DEFINE_VTABLE_PTR_HELPER_CTOR(URRIMUComponent);
	URRIMUComponent::~URRIMUComponent() {}
	struct Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRIMUComponent_h_Statics
	{
		static const FClassRegisterCompiledInInfo ClassInfo[];
	};
	const FClassRegisterCompiledInInfo Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRIMUComponent_h_Statics::ClassInfo[] = {
		{ Z_Construct_UClass_URRIMUComponent, URRIMUComponent::StaticClass, TEXT("URRIMUComponent"), &Z_Registration_Info_UClass_URRIMUComponent, CONSTRUCT_RELOAD_VERSION_INFO(FClassReloadVersionInfo, sizeof(URRIMUComponent), 1315087167U) },
	};
	static FRegisterCompiledInInfo Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRIMUComponent_h_1979620389(TEXT("/Script/EvansLabSensors"),
		Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRIMUComponent_h_Statics::ClassInfo, UE_ARRAY_COUNT(Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRIMUComponent_h_Statics::ClassInfo),
		nullptr, 0,
		nullptr, 0);
PRAGMA_ENABLE_DEPRECATION_WARNINGS
