// Copyright Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Generated code exported from UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "UObject/GeneratedCppIncludes.h"
#include "EvansLabSensors/Public/RRROS2DepthCameraActor.h"
PRAGMA_DISABLE_DEPRECATION_WARNINGS
void EmptyLinkFunctionForGeneratedCodeRRROS2DepthCameraActor() {}
// Cross Module References
	ENGINE_API UClass* Z_Construct_UClass_AActor();
	EVANSLABSENSORS_API UClass* Z_Construct_UClass_ARRROS2DepthCameraActor();
	EVANSLABSENSORS_API UClass* Z_Construct_UClass_ARRROS2DepthCameraActor_NoRegister();
	EVANSLABSENSORS_API UClass* Z_Construct_UClass_URRROS2DepthCameraComponent_NoRegister();
	EVANSLABSENSORS_API UClass* Z_Construct_UClass_URRROS2RgbCameraComponent_NoRegister();
	RCLUE_API UClass* Z_Construct_UClass_UROS2NodeComponent_NoRegister();
	UPackage* Z_Construct_UPackage__Script_EvansLabSensors();
// End Cross Module References
	void ARRROS2DepthCameraActor::StaticRegisterNativesARRROS2DepthCameraActor()
	{
	}
	IMPLEMENT_CLASS_NO_AUTO_REGISTRATION(ARRROS2DepthCameraActor);
	UClass* Z_Construct_UClass_ARRROS2DepthCameraActor_NoRegister()
	{
		return ARRROS2DepthCameraActor::StaticClass();
	}
	struct Z_Construct_UClass_ARRROS2DepthCameraActor_Statics
	{
		static UObject* (*const DependentSingletons[])();
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Class_MetaDataParams[];
#endif
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_Node_MetaData[];
#endif
		static const UECodeGen_Private::FObjectPropertyParams NewProp_Node;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_NodeName_MetaData[];
#endif
		static const UECodeGen_Private::FStrPropertyParams NewProp_NodeName;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_NodeNamespace_MetaData[];
#endif
		static const UECodeGen_Private::FStrPropertyParams NewProp_NodeNamespace;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_DepthCameraComponent_MetaData[];
#endif
		static const UECodeGen_Private::FObjectPropertyParams NewProp_DepthCameraComponent;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_RgbCameraComponent_MetaData[];
#endif
		static const UECodeGen_Private::FObjectPropertyParams NewProp_RgbCameraComponent;
		static const UECodeGen_Private::FPropertyParamsBase* const PropPointers[];
		static const FCppClassTypeInfoStatic StaticCppClassTypeInfo;
		static const UECodeGen_Private::FClassParams ClassParams;
	};
	UObject* (*const Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::DependentSingletons[])() = {
		(UObject* (*)())Z_Construct_UClass_AActor,
		(UObject* (*)())Z_Construct_UPackage__Script_EvansLabSensors,
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::Class_MetaDataParams[] = {
		{ "Comment", "/**\n * @brief Standalone ROS2 camera actor which can be placed in the level with #URRROS2CameraComponent.\n */" },
		{ "IncludePath", "RRROS2DepthCameraActor.h" },
		{ "ModuleRelativePath", "Public/RRROS2DepthCameraActor.h" },
		{ "ToolTip", "@brief Standalone ROS2 camera actor which can be placed in the level with #URRROS2CameraComponent." },
	};
#endif
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::NewProp_Node_MetaData[] = {
		{ "EditInline", "true" },
		{ "ModuleRelativePath", "Public/RRROS2DepthCameraActor.h" },
	};
#endif
	const UECodeGen_Private::FObjectPropertyParams Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::NewProp_Node = { "Node", nullptr, (EPropertyFlags)0x0010000000082008, UECodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(ARRROS2DepthCameraActor, Node), Z_Construct_UClass_UROS2NodeComponent_NoRegister, METADATA_PARAMS(Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::NewProp_Node_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::NewProp_Node_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::NewProp_NodeName_MetaData[] = {
		{ "Category", "RRROS2DepthCameraActor" },
		{ "ModuleRelativePath", "Public/RRROS2DepthCameraActor.h" },
	};
#endif
	const UECodeGen_Private::FStrPropertyParams Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::NewProp_NodeName = { "NodeName", nullptr, (EPropertyFlags)0x0010000000000004, UECodeGen_Private::EPropertyGenFlags::Str, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(ARRROS2DepthCameraActor, NodeName), METADATA_PARAMS(Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::NewProp_NodeName_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::NewProp_NodeName_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::NewProp_NodeNamespace_MetaData[] = {
		{ "Category", "RRROS2DepthCameraActor" },
		{ "ModuleRelativePath", "Public/RRROS2DepthCameraActor.h" },
	};
#endif
	const UECodeGen_Private::FStrPropertyParams Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::NewProp_NodeNamespace = { "NodeNamespace", nullptr, (EPropertyFlags)0x0010000000000004, UECodeGen_Private::EPropertyGenFlags::Str, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(ARRROS2DepthCameraActor, NodeNamespace), METADATA_PARAMS(Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::NewProp_NodeNamespace_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::NewProp_NodeNamespace_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::NewProp_DepthCameraComponent_MetaData[] = {
		{ "AllowPrivateAccess", "true" },
		{ "Category", "CameraActor" },
		{ "Comment", "//! ROS2 camera component for this camera\n" },
		{ "EditInline", "true" },
		{ "ModuleRelativePath", "Public/RRROS2DepthCameraActor.h" },
		{ "ToolTip", "! ROS2 camera component for this camera" },
	};
#endif
	const UECodeGen_Private::FObjectPropertyParams Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::NewProp_DepthCameraComponent = { "DepthCameraComponent", nullptr, (EPropertyFlags)0x00200800000a001d, UECodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(ARRROS2DepthCameraActor, DepthCameraComponent), Z_Construct_UClass_URRROS2DepthCameraComponent_NoRegister, METADATA_PARAMS(Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::NewProp_DepthCameraComponent_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::NewProp_DepthCameraComponent_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::NewProp_RgbCameraComponent_MetaData[] = {
		{ "AllowPrivateAccess", "true" },
		{ "Category", "CameraActor" },
		{ "EditInline", "true" },
		{ "ModuleRelativePath", "Public/RRROS2DepthCameraActor.h" },
	};
#endif
	const UECodeGen_Private::FObjectPropertyParams Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::NewProp_RgbCameraComponent = { "RgbCameraComponent", nullptr, (EPropertyFlags)0x00200800000a001d, UECodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(ARRROS2DepthCameraActor, RgbCameraComponent), Z_Construct_UClass_URRROS2RgbCameraComponent_NoRegister, METADATA_PARAMS(Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::NewProp_RgbCameraComponent_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::NewProp_RgbCameraComponent_MetaData)) };
	const UECodeGen_Private::FPropertyParamsBase* const Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::PropPointers[] = {
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::NewProp_Node,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::NewProp_NodeName,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::NewProp_NodeNamespace,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::NewProp_DepthCameraComponent,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::NewProp_RgbCameraComponent,
	};
	const FCppClassTypeInfoStatic Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::StaticCppClassTypeInfo = {
		TCppClassTypeTraits<ARRROS2DepthCameraActor>::IsAbstract,
	};
	const UECodeGen_Private::FClassParams Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::ClassParams = {
		&ARRROS2DepthCameraActor::StaticClass,
		"Engine",
		&StaticCppClassTypeInfo,
		DependentSingletons,
		nullptr,
		Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::PropPointers,
		nullptr,
		UE_ARRAY_COUNT(DependentSingletons),
		0,
		UE_ARRAY_COUNT(Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::PropPointers),
		0,
		0x009000A4u,
		METADATA_PARAMS(Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::Class_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::Class_MetaDataParams))
	};
	UClass* Z_Construct_UClass_ARRROS2DepthCameraActor()
	{
		if (!Z_Registration_Info_UClass_ARRROS2DepthCameraActor.OuterSingleton)
		{
			UECodeGen_Private::ConstructUClass(Z_Registration_Info_UClass_ARRROS2DepthCameraActor.OuterSingleton, Z_Construct_UClass_ARRROS2DepthCameraActor_Statics::ClassParams);
		}
		return Z_Registration_Info_UClass_ARRROS2DepthCameraActor.OuterSingleton;
	}
	template<> EVANSLABSENSORS_API UClass* StaticClass<ARRROS2DepthCameraActor>()
	{
		return ARRROS2DepthCameraActor::StaticClass();
	}
	DEFINE_VTABLE_PTR_HELPER_CTOR(ARRROS2DepthCameraActor);
	ARRROS2DepthCameraActor::~ARRROS2DepthCameraActor() {}
	struct Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRROS2DepthCameraActor_h_Statics
	{
		static const FClassRegisterCompiledInInfo ClassInfo[];
	};
	const FClassRegisterCompiledInInfo Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRROS2DepthCameraActor_h_Statics::ClassInfo[] = {
		{ Z_Construct_UClass_ARRROS2DepthCameraActor, ARRROS2DepthCameraActor::StaticClass, TEXT("ARRROS2DepthCameraActor"), &Z_Registration_Info_UClass_ARRROS2DepthCameraActor, CONSTRUCT_RELOAD_VERSION_INFO(FClassReloadVersionInfo, sizeof(ARRROS2DepthCameraActor), 2917095499U) },
	};
	static FRegisterCompiledInInfo Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRROS2DepthCameraActor_h_249792812(TEXT("/Script/EvansLabSensors"),
		Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRROS2DepthCameraActor_h_Statics::ClassInfo, UE_ARRAY_COUNT(Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRROS2DepthCameraActor_h_Statics::ClassInfo),
		nullptr, 0,
		nullptr, 0);
PRAGMA_ENABLE_DEPRECATION_WARNINGS
