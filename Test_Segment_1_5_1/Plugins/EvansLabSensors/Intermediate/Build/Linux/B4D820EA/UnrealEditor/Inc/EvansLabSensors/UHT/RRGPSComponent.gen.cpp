// Copyright Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Generated code exported from UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "UObject/GeneratedCppIncludes.h"
#include "EvansLabSensors/Public/RRGPSComponent.h"
PRAGMA_DISABLE_DEPRECATION_WARNINGS
void EmptyLinkFunctionForGeneratedCodeRRGPSComponent() {}
// Cross Module References
	EVANSLABSENSORS_API UClass* Z_Construct_UClass_URRGPSComponent();
	EVANSLABSENSORS_API UClass* Z_Construct_UClass_URRGPSComponent_NoRegister();
	RAPYUTASIMULATIONPLUGINS_API UClass* Z_Construct_UClass_URRROS2BaseSensorComponent();
	UPackage* Z_Construct_UPackage__Script_EvansLabSensors();
// End Cross Module References
	void URRGPSComponent::StaticRegisterNativesURRGPSComponent()
	{
	}
	IMPLEMENT_CLASS_NO_AUTO_REGISTRATION(URRGPSComponent);
	UClass* Z_Construct_UClass_URRGPSComponent_NoRegister()
	{
		return URRGPSComponent::StaticClass();
	}
	struct Z_Construct_UClass_URRGPSComponent_Statics
	{
		static UObject* (*const DependentSingletons[])();
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Class_MetaDataParams[];
#endif
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_SigmaWhiteNoise_MetaData[];
#endif
		static const UECodeGen_Private::FFloatPropertyParams NewProp_SigmaWhiteNoise;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_SigmaDrift_MetaData[];
#endif
		static const UECodeGen_Private::FFloatPropertyParams NewProp_SigmaDrift;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_PJump_MetaData[];
#endif
		static const UECodeGen_Private::FFloatPropertyParams NewProp_PJump;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_MuJump_MetaData[];
#endif
		static const UECodeGen_Private::FFloatPropertyParams NewProp_MuJump;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_SigmaJump_MetaData[];
#endif
		static const UECodeGen_Private::FFloatPropertyParams NewProp_SigmaJump;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_Alpha_MetaData[];
#endif
		static const UECodeGen_Private::FFloatPropertyParams NewProp_Alpha;
		static const UECodeGen_Private::FPropertyParamsBase* const PropPointers[];
		static const FCppClassTypeInfoStatic StaticCppClassTypeInfo;
		static const UECodeGen_Private::FClassParams ClassParams;
	};
	UObject* (*const Z_Construct_UClass_URRGPSComponent_Statics::DependentSingletons[])() = {
		(UObject* (*)())Z_Construct_UClass_URRROS2BaseSensorComponent,
		(UObject* (*)())Z_Construct_UPackage__Script_EvansLabSensors,
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRGPSComponent_Statics::Class_MetaDataParams[] = {
		{ "BlueprintSpawnableComponent", "" },
		{ "BlueprintType", "true" },
		{ "ClassGroupNames", "Custom" },
		{ "Comment", "/**\n *\n */" },
		{ "HideCategories", "Trigger PhysicsVolume" },
		{ "IncludePath", "RRGPSComponent.h" },
		{ "IsBlueprintBase", "true" },
		{ "ModuleRelativePath", "Public/RRGPSComponent.h" },
	};
#endif
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRGPSComponent_Statics::NewProp_SigmaWhiteNoise_MetaData[] = {
		{ "Category", "GPS Error" },
		{ "ModuleRelativePath", "Public/RRGPSComponent.h" },
	};
#endif
	const UECodeGen_Private::FFloatPropertyParams Z_Construct_UClass_URRGPSComponent_Statics::NewProp_SigmaWhiteNoise = { "SigmaWhiteNoise", nullptr, (EPropertyFlags)0x0040000000010001, UECodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRGPSComponent, SigmaWhiteNoise), METADATA_PARAMS(Z_Construct_UClass_URRGPSComponent_Statics::NewProp_SigmaWhiteNoise_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRGPSComponent_Statics::NewProp_SigmaWhiteNoise_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRGPSComponent_Statics::NewProp_SigmaDrift_MetaData[] = {
		{ "Category", "GPS Error" },
		{ "ModuleRelativePath", "Public/RRGPSComponent.h" },
	};
#endif
	const UECodeGen_Private::FFloatPropertyParams Z_Construct_UClass_URRGPSComponent_Statics::NewProp_SigmaDrift = { "SigmaDrift", nullptr, (EPropertyFlags)0x0040000000010001, UECodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRGPSComponent, SigmaDrift), METADATA_PARAMS(Z_Construct_UClass_URRGPSComponent_Statics::NewProp_SigmaDrift_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRGPSComponent_Statics::NewProp_SigmaDrift_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRGPSComponent_Statics::NewProp_PJump_MetaData[] = {
		{ "Category", "GPS Error" },
		{ "ModuleRelativePath", "Public/RRGPSComponent.h" },
	};
#endif
	const UECodeGen_Private::FFloatPropertyParams Z_Construct_UClass_URRGPSComponent_Statics::NewProp_PJump = { "PJump", nullptr, (EPropertyFlags)0x0040000000010001, UECodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRGPSComponent, PJump), METADATA_PARAMS(Z_Construct_UClass_URRGPSComponent_Statics::NewProp_PJump_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRGPSComponent_Statics::NewProp_PJump_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRGPSComponent_Statics::NewProp_MuJump_MetaData[] = {
		{ "Category", "GPS Error" },
		{ "ModuleRelativePath", "Public/RRGPSComponent.h" },
	};
#endif
	const UECodeGen_Private::FFloatPropertyParams Z_Construct_UClass_URRGPSComponent_Statics::NewProp_MuJump = { "MuJump", nullptr, (EPropertyFlags)0x0040000000010001, UECodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRGPSComponent, MuJump), METADATA_PARAMS(Z_Construct_UClass_URRGPSComponent_Statics::NewProp_MuJump_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRGPSComponent_Statics::NewProp_MuJump_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRGPSComponent_Statics::NewProp_SigmaJump_MetaData[] = {
		{ "Category", "GPS Error" },
		{ "ModuleRelativePath", "Public/RRGPSComponent.h" },
	};
#endif
	const UECodeGen_Private::FFloatPropertyParams Z_Construct_UClass_URRGPSComponent_Statics::NewProp_SigmaJump = { "SigmaJump", nullptr, (EPropertyFlags)0x0040000000010001, UECodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRGPSComponent, SigmaJump), METADATA_PARAMS(Z_Construct_UClass_URRGPSComponent_Statics::NewProp_SigmaJump_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRGPSComponent_Statics::NewProp_SigmaJump_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRGPSComponent_Statics::NewProp_Alpha_MetaData[] = {
		{ "Category", "GPS Error" },
		{ "ModuleRelativePath", "Public/RRGPSComponent.h" },
	};
#endif
	const UECodeGen_Private::FFloatPropertyParams Z_Construct_UClass_URRGPSComponent_Statics::NewProp_Alpha = { "Alpha", nullptr, (EPropertyFlags)0x0040000000010001, UECodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRGPSComponent, Alpha), METADATA_PARAMS(Z_Construct_UClass_URRGPSComponent_Statics::NewProp_Alpha_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRGPSComponent_Statics::NewProp_Alpha_MetaData)) };
	const UECodeGen_Private::FPropertyParamsBase* const Z_Construct_UClass_URRGPSComponent_Statics::PropPointers[] = {
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRGPSComponent_Statics::NewProp_SigmaWhiteNoise,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRGPSComponent_Statics::NewProp_SigmaDrift,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRGPSComponent_Statics::NewProp_PJump,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRGPSComponent_Statics::NewProp_MuJump,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRGPSComponent_Statics::NewProp_SigmaJump,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRGPSComponent_Statics::NewProp_Alpha,
	};
	const FCppClassTypeInfoStatic Z_Construct_UClass_URRGPSComponent_Statics::StaticCppClassTypeInfo = {
		TCppClassTypeTraits<URRGPSComponent>::IsAbstract,
	};
	const UECodeGen_Private::FClassParams Z_Construct_UClass_URRGPSComponent_Statics::ClassParams = {
		&URRGPSComponent::StaticClass,
		"Engine",
		&StaticCppClassTypeInfo,
		DependentSingletons,
		nullptr,
		Z_Construct_UClass_URRGPSComponent_Statics::PropPointers,
		nullptr,
		UE_ARRAY_COUNT(DependentSingletons),
		0,
		UE_ARRAY_COUNT(Z_Construct_UClass_URRGPSComponent_Statics::PropPointers),
		0,
		0x00B000A4u,
		METADATA_PARAMS(Z_Construct_UClass_URRGPSComponent_Statics::Class_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UClass_URRGPSComponent_Statics::Class_MetaDataParams))
	};
	UClass* Z_Construct_UClass_URRGPSComponent()
	{
		if (!Z_Registration_Info_UClass_URRGPSComponent.OuterSingleton)
		{
			UECodeGen_Private::ConstructUClass(Z_Registration_Info_UClass_URRGPSComponent.OuterSingleton, Z_Construct_UClass_URRGPSComponent_Statics::ClassParams);
		}
		return Z_Registration_Info_UClass_URRGPSComponent.OuterSingleton;
	}
	template<> EVANSLABSENSORS_API UClass* StaticClass<URRGPSComponent>()
	{
		return URRGPSComponent::StaticClass();
	}
	DEFINE_VTABLE_PTR_HELPER_CTOR(URRGPSComponent);
	URRGPSComponent::~URRGPSComponent() {}
	struct Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRGPSComponent_h_Statics
	{
		static const FClassRegisterCompiledInInfo ClassInfo[];
	};
	const FClassRegisterCompiledInInfo Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRGPSComponent_h_Statics::ClassInfo[] = {
		{ Z_Construct_UClass_URRGPSComponent, URRGPSComponent::StaticClass, TEXT("URRGPSComponent"), &Z_Registration_Info_UClass_URRGPSComponent, CONSTRUCT_RELOAD_VERSION_INFO(FClassReloadVersionInfo, sizeof(URRGPSComponent), 1616399427U) },
	};
	static FRegisterCompiledInInfo Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRGPSComponent_h_2806076069(TEXT("/Script/EvansLabSensors"),
		Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRGPSComponent_h_Statics::ClassInfo, UE_ARRAY_COUNT(Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRGPSComponent_h_Statics::ClassInfo),
		nullptr, 0,
		nullptr, 0);
PRAGMA_ENABLE_DEPRECATION_WARNINGS
