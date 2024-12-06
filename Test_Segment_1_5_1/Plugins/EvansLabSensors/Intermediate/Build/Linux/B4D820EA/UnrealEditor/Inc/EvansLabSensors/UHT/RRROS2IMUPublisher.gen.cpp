// Copyright Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Generated code exported from UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "UObject/GeneratedCppIncludes.h"
#include "EvansLabSensors/Public/RRROS2IMUPublisher.h"
PRAGMA_DISABLE_DEPRECATION_WARNINGS
void EmptyLinkFunctionForGeneratedCodeRRROS2IMUPublisher() {}
// Cross Module References
	EVANSLABSENSORS_API UClass* Z_Construct_UClass_URRROS2IMUPublisher();
	EVANSLABSENSORS_API UClass* Z_Construct_UClass_URRROS2IMUPublisher_NoRegister();
	RAPYUTASIMULATIONPLUGINS_API UClass* Z_Construct_UClass_URRROS2BaseSensorPublisher();
	UPackage* Z_Construct_UPackage__Script_EvansLabSensors();
// End Cross Module References
	void URRROS2IMUPublisher::StaticRegisterNativesURRROS2IMUPublisher()
	{
	}
	IMPLEMENT_CLASS_NO_AUTO_REGISTRATION(URRROS2IMUPublisher);
	UClass* Z_Construct_UClass_URRROS2IMUPublisher_NoRegister()
	{
		return URRROS2IMUPublisher::StaticClass();
	}
	struct Z_Construct_UClass_URRROS2IMUPublisher_Statics
	{
		static UObject* (*const DependentSingletons[])();
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Class_MetaDataParams[];
#endif
		static const FCppClassTypeInfoStatic StaticCppClassTypeInfo;
		static const UECodeGen_Private::FClassParams ClassParams;
	};
	UObject* (*const Z_Construct_UClass_URRROS2IMUPublisher_Statics::DependentSingletons[])() = {
		(UObject* (*)())Z_Construct_UClass_URRROS2BaseSensorPublisher,
		(UObject* (*)())Z_Construct_UPackage__Script_EvansLabSensors,
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRROS2IMUPublisher_Statics::Class_MetaDataParams[] = {
		{ "BlueprintSpawnableComponent", "" },
		{ "BlueprintType", "true" },
		{ "ClassGroupNames", "Custom" },
		{ "Comment", "/**\n * \n * @brief GPS Publisher Class\n */" },
		{ "IncludePath", "RRROS2IMUPublisher.h" },
		{ "IsBlueprintBase", "true" },
		{ "ModuleRelativePath", "Public/RRROS2IMUPublisher.h" },
		{ "ToolTip", "@brief GPS Publisher Class" },
	};
#endif
	const FCppClassTypeInfoStatic Z_Construct_UClass_URRROS2IMUPublisher_Statics::StaticCppClassTypeInfo = {
		TCppClassTypeTraits<URRROS2IMUPublisher>::IsAbstract,
	};
	const UECodeGen_Private::FClassParams Z_Construct_UClass_URRROS2IMUPublisher_Statics::ClassParams = {
		&URRROS2IMUPublisher::StaticClass,
		nullptr,
		&StaticCppClassTypeInfo,
		DependentSingletons,
		nullptr,
		nullptr,
		nullptr,
		UE_ARRAY_COUNT(DependentSingletons),
		0,
		0,
		0,
		0x009000A0u,
		METADATA_PARAMS(Z_Construct_UClass_URRROS2IMUPublisher_Statics::Class_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UClass_URRROS2IMUPublisher_Statics::Class_MetaDataParams))
	};
	UClass* Z_Construct_UClass_URRROS2IMUPublisher()
	{
		if (!Z_Registration_Info_UClass_URRROS2IMUPublisher.OuterSingleton)
		{
			UECodeGen_Private::ConstructUClass(Z_Registration_Info_UClass_URRROS2IMUPublisher.OuterSingleton, Z_Construct_UClass_URRROS2IMUPublisher_Statics::ClassParams);
		}
		return Z_Registration_Info_UClass_URRROS2IMUPublisher.OuterSingleton;
	}
	template<> EVANSLABSENSORS_API UClass* StaticClass<URRROS2IMUPublisher>()
	{
		return URRROS2IMUPublisher::StaticClass();
	}
	DEFINE_VTABLE_PTR_HELPER_CTOR(URRROS2IMUPublisher);
	URRROS2IMUPublisher::~URRROS2IMUPublisher() {}
	struct Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRROS2IMUPublisher_h_Statics
	{
		static const FClassRegisterCompiledInInfo ClassInfo[];
	};
	const FClassRegisterCompiledInInfo Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRROS2IMUPublisher_h_Statics::ClassInfo[] = {
		{ Z_Construct_UClass_URRROS2IMUPublisher, URRROS2IMUPublisher::StaticClass, TEXT("URRROS2IMUPublisher"), &Z_Registration_Info_UClass_URRROS2IMUPublisher, CONSTRUCT_RELOAD_VERSION_INFO(FClassReloadVersionInfo, sizeof(URRROS2IMUPublisher), 208716020U) },
	};
	static FRegisterCompiledInInfo Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRROS2IMUPublisher_h_3742826586(TEXT("/Script/EvansLabSensors"),
		Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRROS2IMUPublisher_h_Statics::ClassInfo, UE_ARRAY_COUNT(Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRROS2IMUPublisher_h_Statics::ClassInfo),
		nullptr, 0,
		nullptr, 0);
PRAGMA_ENABLE_DEPRECATION_WARNINGS
