// Copyright Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Generated code exported from UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "UObject/GeneratedCppIncludes.h"
#include "EvansLabSensors/Public/RRROS2GPSPublisher.h"
PRAGMA_DISABLE_DEPRECATION_WARNINGS
void EmptyLinkFunctionForGeneratedCodeRRROS2GPSPublisher() {}
// Cross Module References
	EVANSLABSENSORS_API UClass* Z_Construct_UClass_URRROS2GPSPublisher();
	EVANSLABSENSORS_API UClass* Z_Construct_UClass_URRROS2GPSPublisher_NoRegister();
	RAPYUTASIMULATIONPLUGINS_API UClass* Z_Construct_UClass_URRROS2BaseSensorPublisher();
	UPackage* Z_Construct_UPackage__Script_EvansLabSensors();
// End Cross Module References
	void URRROS2GPSPublisher::StaticRegisterNativesURRROS2GPSPublisher()
	{
	}
	IMPLEMENT_CLASS_NO_AUTO_REGISTRATION(URRROS2GPSPublisher);
	UClass* Z_Construct_UClass_URRROS2GPSPublisher_NoRegister()
	{
		return URRROS2GPSPublisher::StaticClass();
	}
	struct Z_Construct_UClass_URRROS2GPSPublisher_Statics
	{
		static UObject* (*const DependentSingletons[])();
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Class_MetaDataParams[];
#endif
		static const FCppClassTypeInfoStatic StaticCppClassTypeInfo;
		static const UECodeGen_Private::FClassParams ClassParams;
	};
	UObject* (*const Z_Construct_UClass_URRROS2GPSPublisher_Statics::DependentSingletons[])() = {
		(UObject* (*)())Z_Construct_UClass_URRROS2BaseSensorPublisher,
		(UObject* (*)())Z_Construct_UPackage__Script_EvansLabSensors,
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRROS2GPSPublisher_Statics::Class_MetaDataParams[] = {
		{ "BlueprintSpawnableComponent", "" },
		{ "BlueprintType", "true" },
		{ "ClassGroupNames", "Custom" },
		{ "Comment", "/**\n * \n * @brief GPS Publisher Class\n */" },
		{ "IncludePath", "RRROS2GPSPublisher.h" },
		{ "IsBlueprintBase", "true" },
		{ "ModuleRelativePath", "Public/RRROS2GPSPublisher.h" },
		{ "ToolTip", "@brief GPS Publisher Class" },
	};
#endif
	const FCppClassTypeInfoStatic Z_Construct_UClass_URRROS2GPSPublisher_Statics::StaticCppClassTypeInfo = {
		TCppClassTypeTraits<URRROS2GPSPublisher>::IsAbstract,
	};
	const UECodeGen_Private::FClassParams Z_Construct_UClass_URRROS2GPSPublisher_Statics::ClassParams = {
		&URRROS2GPSPublisher::StaticClass,
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
		METADATA_PARAMS(Z_Construct_UClass_URRROS2GPSPublisher_Statics::Class_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UClass_URRROS2GPSPublisher_Statics::Class_MetaDataParams))
	};
	UClass* Z_Construct_UClass_URRROS2GPSPublisher()
	{
		if (!Z_Registration_Info_UClass_URRROS2GPSPublisher.OuterSingleton)
		{
			UECodeGen_Private::ConstructUClass(Z_Registration_Info_UClass_URRROS2GPSPublisher.OuterSingleton, Z_Construct_UClass_URRROS2GPSPublisher_Statics::ClassParams);
		}
		return Z_Registration_Info_UClass_URRROS2GPSPublisher.OuterSingleton;
	}
	template<> EVANSLABSENSORS_API UClass* StaticClass<URRROS2GPSPublisher>()
	{
		return URRROS2GPSPublisher::StaticClass();
	}
	DEFINE_VTABLE_PTR_HELPER_CTOR(URRROS2GPSPublisher);
	URRROS2GPSPublisher::~URRROS2GPSPublisher() {}
	struct Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRROS2GPSPublisher_h_Statics
	{
		static const FClassRegisterCompiledInInfo ClassInfo[];
	};
	const FClassRegisterCompiledInInfo Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRROS2GPSPublisher_h_Statics::ClassInfo[] = {
		{ Z_Construct_UClass_URRROS2GPSPublisher, URRROS2GPSPublisher::StaticClass, TEXT("URRROS2GPSPublisher"), &Z_Registration_Info_UClass_URRROS2GPSPublisher, CONSTRUCT_RELOAD_VERSION_INFO(FClassReloadVersionInfo, sizeof(URRROS2GPSPublisher), 2659393818U) },
	};
	static FRegisterCompiledInInfo Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRROS2GPSPublisher_h_256891078(TEXT("/Script/EvansLabSensors"),
		Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRROS2GPSPublisher_h_Statics::ClassInfo, UE_ARRAY_COUNT(Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRROS2GPSPublisher_h_Statics::ClassInfo),
		nullptr, 0,
		nullptr, 0);
PRAGMA_ENABLE_DEPRECATION_WARNINGS
