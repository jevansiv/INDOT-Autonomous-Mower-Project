// Copyright Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Generated code exported from UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "UObject/GeneratedCppIncludes.h"
#include "EvansLabSensors/Public/RRROS2DepthCameraComponent.h"
#include "rclUE/Public/Msgs/ROS2Img.h"
PRAGMA_DISABLE_DEPRECATION_WARNINGS
void EmptyLinkFunctionForGeneratedCodeRRROS2DepthCameraComponent() {}
// Cross Module References
	ENGINE_API UClass* Z_Construct_UClass_UCameraComponent_NoRegister();
	ENGINE_API UClass* Z_Construct_UClass_USceneCaptureComponent2D_NoRegister();
	ENGINE_API UClass* Z_Construct_UClass_UTextureRenderTarget2D_NoRegister();
	EVANSLABSENSORS_API UClass* Z_Construct_UClass_URRROS2DepthCameraComponent();
	EVANSLABSENSORS_API UClass* Z_Construct_UClass_URRROS2DepthCameraComponent_NoRegister();
	RAPYUTASIMULATIONPLUGINS_API UClass* Z_Construct_UClass_URRROS2BaseSensorComponent();
	RCLUE_API UScriptStruct* Z_Construct_UScriptStruct_FROSImg();
	UPackage* Z_Construct_UPackage__Script_EvansLabSensors();
// End Cross Module References
	DEFINE_FUNCTION(URRROS2DepthCameraComponent::execGetROS2Data)
	{
		P_FINISH;
		P_NATIVE_BEGIN;
		*(FROSImg*)Z_Param__Result=P_THIS->GetROS2Data();
		P_NATIVE_END;
	}
	DEFINE_FUNCTION(URRROS2DepthCameraComponent::execCaptureDepthNonBlocking)
	{
		P_FINISH;
		P_NATIVE_BEGIN;
		P_THIS->CaptureDepthNonBlocking();
		P_NATIVE_END;
	}
	void URRROS2DepthCameraComponent::StaticRegisterNativesURRROS2DepthCameraComponent()
	{
		UClass* Class = URRROS2DepthCameraComponent::StaticClass();
		static const FNameNativePtrPair Funcs[] = {
			{ "CaptureDepthNonBlocking", &URRROS2DepthCameraComponent::execCaptureDepthNonBlocking },
			{ "GetROS2Data", &URRROS2DepthCameraComponent::execGetROS2Data },
		};
		FNativeFunctionRegistrar::RegisterFunctions(Class, Funcs, UE_ARRAY_COUNT(Funcs));
	}
	struct Z_Construct_UFunction_URRROS2DepthCameraComponent_CaptureDepthNonBlocking_Statics
	{
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UECodeGen_Private::FFunctionParams FuncParams;
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_URRROS2DepthCameraComponent_CaptureDepthNonBlocking_Statics::Function_MetaDataParams[] = {
		{ "Comment", "/**\n     * @brief Capture data by notifing task in #RenderRequestQueue\n     * @sa reference https://github.com/TimmHess/UnrealImageCapture\n     */// Depth\n" },
		{ "ModuleRelativePath", "Public/RRROS2DepthCameraComponent.h" },
		{ "ToolTip", "@brief Capture data by notifing task in #RenderRequestQueue\n@sa reference https://github.com/TimmHess/UnrealImageCapture\n     // Depth" },
	};
#endif
	const UECodeGen_Private::FFunctionParams Z_Construct_UFunction_URRROS2DepthCameraComponent_CaptureDepthNonBlocking_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_URRROS2DepthCameraComponent, nullptr, "CaptureDepthNonBlocking", nullptr, nullptr, 0, nullptr, 0, RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x00080401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_URRROS2DepthCameraComponent_CaptureDepthNonBlocking_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_URRROS2DepthCameraComponent_CaptureDepthNonBlocking_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_URRROS2DepthCameraComponent_CaptureDepthNonBlocking()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UECodeGen_Private::ConstructUFunction(&ReturnFunction, Z_Construct_UFunction_URRROS2DepthCameraComponent_CaptureDepthNonBlocking_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_URRROS2DepthCameraComponent_GetROS2Data_Statics
	{
		struct RRROS2DepthCameraComponent_eventGetROS2Data_Parms
		{
			FROSImg ReturnValue;
		};
		static const UECodeGen_Private::FStructPropertyParams NewProp_ReturnValue;
		static const UECodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UECodeGen_Private::FFunctionParams FuncParams;
	};
	const UECodeGen_Private::FStructPropertyParams Z_Construct_UFunction_URRROS2DepthCameraComponent_GetROS2Data_Statics::NewProp_ReturnValue = { "ReturnValue", nullptr, (EPropertyFlags)0x0010000000000580, UECodeGen_Private::EPropertyGenFlags::Struct, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(RRROS2DepthCameraComponent_eventGetROS2Data_Parms, ReturnValue), Z_Construct_UScriptStruct_FROSImg, METADATA_PARAMS(nullptr, 0) }; // 1436882506
	const UECodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_URRROS2DepthCameraComponent_GetROS2Data_Statics::PropPointers[] = {
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_URRROS2DepthCameraComponent_GetROS2Data_Statics::NewProp_ReturnValue,
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_URRROS2DepthCameraComponent_GetROS2Data_Statics::Function_MetaDataParams[] = {
		{ "Comment", "/**\n     * @brief Update ROS2 Msg structure from #RenderRequestQueue\n     *\n     * @return FROSImg\n     */" },
		{ "ModuleRelativePath", "Public/RRROS2DepthCameraComponent.h" },
		{ "ToolTip", "@brief Update ROS2 Msg structure from #RenderRequestQueue\n\n@return FROSImg" },
	};
#endif
	const UECodeGen_Private::FFunctionParams Z_Construct_UFunction_URRROS2DepthCameraComponent_GetROS2Data_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_URRROS2DepthCameraComponent, nullptr, "GetROS2Data", nullptr, nullptr, sizeof(Z_Construct_UFunction_URRROS2DepthCameraComponent_GetROS2Data_Statics::RRROS2DepthCameraComponent_eventGetROS2Data_Parms), Z_Construct_UFunction_URRROS2DepthCameraComponent_GetROS2Data_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_URRROS2DepthCameraComponent_GetROS2Data_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x04020400, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_URRROS2DepthCameraComponent_GetROS2Data_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_URRROS2DepthCameraComponent_GetROS2Data_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_URRROS2DepthCameraComponent_GetROS2Data()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UECodeGen_Private::ConstructUFunction(&ReturnFunction, Z_Construct_UFunction_URRROS2DepthCameraComponent_GetROS2Data_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	IMPLEMENT_CLASS_NO_AUTO_REGISTRATION(URRROS2DepthCameraComponent);
	UClass* Z_Construct_UClass_URRROS2DepthCameraComponent_NoRegister()
	{
		return URRROS2DepthCameraComponent::StaticClass();
	}
	struct Z_Construct_UClass_URRROS2DepthCameraComponent_Statics
	{
		static UObject* (*const DependentSingletons[])();
		static const FClassFunctionLinkInfo FuncInfo[];
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Class_MetaDataParams[];
#endif
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_CameraComponent_MetaData[];
#endif
		static const UECodeGen_Private::FObjectPropertyParams NewProp_CameraComponent;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_DepthSceneCaptureComponent_MetaData[];
#endif
		static const UECodeGen_Private::FObjectPropertyParams NewProp_DepthSceneCaptureComponent;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_DepthRenderTarget_MetaData[];
#endif
		static const UECodeGen_Private::FObjectPropertyParams NewProp_DepthRenderTarget;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_Width_MetaData[];
#endif
		static const UECodeGen_Private::FIntPropertyParams NewProp_Width;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_Height_MetaData[];
#endif
		static const UECodeGen_Private::FIntPropertyParams NewProp_Height;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_QueueSize_MetaData[];
#endif
		static const UECodeGen_Private::FIntPropertyParams NewProp_QueueSize;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_MaxDepth_MetaData[];
#endif
		static const UECodeGen_Private::FFloatPropertyParams NewProp_MaxDepth;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_Encoding_MetaData[];
#endif
		static const UECodeGen_Private::FStrPropertyParams NewProp_Encoding;
		static const UECodeGen_Private::FPropertyParamsBase* const PropPointers[];
		static const FCppClassTypeInfoStatic StaticCppClassTypeInfo;
		static const UECodeGen_Private::FClassParams ClassParams;
	};
	UObject* (*const Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::DependentSingletons[])() = {
		(UObject* (*)())Z_Construct_UClass_URRROS2BaseSensorComponent,
		(UObject* (*)())Z_Construct_UPackage__Script_EvansLabSensors,
	};
	const FClassFunctionLinkInfo Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::FuncInfo[] = {
		{ &Z_Construct_UFunction_URRROS2DepthCameraComponent_CaptureDepthNonBlocking, "CaptureDepthNonBlocking" }, // 4233605758
		{ &Z_Construct_UFunction_URRROS2DepthCameraComponent_GetROS2Data, "GetROS2Data" }, // 4099207165
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::Class_MetaDataParams[] = {
		{ "BlueprintSpawnableComponent", "" },
		{ "BlueprintType", "true" },
		{ "ClassGroupNames", "Custom" },
		{ "Comment", "/**\n * @brief ROS2 Camera component. Uses USceneCaptureComponent2D.\n *\n * @sa [USceneCaptureComponent2D](https://docs.unrealengine.com/5.1/en-US/API/Runtime/Engine/Components/USceneCaptureComponent2D/)\n * @sa [UE4 ShaderInPlugin](https://docs.unrealengine.com/5.1/en-US/ProgrammingAndScripting/Rendering/ShaderInPlugin/Overview/)\n * @sa implementation reference: https://github.com/TimmHess/UnrealImageCapture\n * @todo Support non RGB data support.\n */" },
		{ "HideCategories", "Trigger PhysicsVolume" },
		{ "IncludePath", "RRROS2DepthCameraComponent.h" },
		{ "IsBlueprintBase", "true" },
		{ "ModuleRelativePath", "Public/RRROS2DepthCameraComponent.h" },
		{ "ToolTip", "@brief ROS2 Camera component. Uses USceneCaptureComponent2D.\n\n@sa [USceneCaptureComponent2D](https://docs.unrealengine.com/5.1/en-US/API/Runtime/Engine/Components/USceneCaptureComponent2D/)\n@sa [UE4 ShaderInPlugin](https://docs.unrealengine.com/5.1/en-US/ProgrammingAndScripting/Rendering/ShaderInPlugin/Overview/)\n@sa implementation reference: https://github.com/TimmHess/UnrealImageCapture\n@todo Support non RGB data support." },
	};
#endif
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_CameraComponent_MetaData[] = {
		{ "Category", "RRROS2DepthCameraComponent" },
		{ "Comment", "//! Camera. Not necessary to capture but useful to see image in UE4 windows.\n" },
		{ "EditInline", "true" },
		{ "ModuleRelativePath", "Public/RRROS2DepthCameraComponent.h" },
		{ "ToolTip", "! Camera. Not necessary to capture but useful to see image in UE4 windows." },
	};
#endif
	const UECodeGen_Private::FObjectPropertyParams Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_CameraComponent = { "CameraComponent", nullptr, (EPropertyFlags)0x00100000000a000d, UECodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRROS2DepthCameraComponent, CameraComponent), Z_Construct_UClass_UCameraComponent_NoRegister, METADATA_PARAMS(Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_CameraComponent_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_CameraComponent_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_DepthSceneCaptureComponent_MetaData[] = {
		{ "Category", "RRROS2DepthCameraComponent" },
		{ "EditInline", "true" },
		{ "ModuleRelativePath", "Public/RRROS2DepthCameraComponent.h" },
	};
#endif
	const UECodeGen_Private::FObjectPropertyParams Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_DepthSceneCaptureComponent = { "DepthSceneCaptureComponent", nullptr, (EPropertyFlags)0x00100000000a000d, UECodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRROS2DepthCameraComponent, DepthSceneCaptureComponent), Z_Construct_UClass_USceneCaptureComponent2D_NoRegister, METADATA_PARAMS(Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_DepthSceneCaptureComponent_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_DepthSceneCaptureComponent_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_DepthRenderTarget_MetaData[] = {
		{ "Category", "RRROS2DepthCameraComponent" },
		{ "ModuleRelativePath", "Public/RRROS2DepthCameraComponent.h" },
	};
#endif
	const UECodeGen_Private::FObjectPropertyParams Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_DepthRenderTarget = { "DepthRenderTarget", nullptr, (EPropertyFlags)0x0010000000020005, UECodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRROS2DepthCameraComponent, DepthRenderTarget), Z_Construct_UClass_UTextureRenderTarget2D_NoRegister, METADATA_PARAMS(Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_DepthRenderTarget_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_DepthRenderTarget_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_Width_MetaData[] = {
		{ "Category", "RRROS2DepthCameraComponent" },
		{ "ModuleRelativePath", "Public/RRROS2DepthCameraComponent.h" },
	};
#endif
	const UECodeGen_Private::FIntPropertyParams Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_Width = { "Width", nullptr, (EPropertyFlags)0x0010000000000005, UECodeGen_Private::EPropertyGenFlags::Int, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRROS2DepthCameraComponent, Width), METADATA_PARAMS(Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_Width_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_Width_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_Height_MetaData[] = {
		{ "Category", "RRROS2DepthCameraComponent" },
		{ "ModuleRelativePath", "Public/RRROS2DepthCameraComponent.h" },
	};
#endif
	const UECodeGen_Private::FIntPropertyParams Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_Height = { "Height", nullptr, (EPropertyFlags)0x0010000000000005, UECodeGen_Private::EPropertyGenFlags::Int, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRROS2DepthCameraComponent, Height), METADATA_PARAMS(Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_Height_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_Height_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_QueueSize_MetaData[] = {
		{ "Category", "RRROS2DepthCameraComponent" },
		{ "ModuleRelativePath", "Public/RRROS2DepthCameraComponent.h" },
	};
#endif
	const UECodeGen_Private::FIntPropertyParams Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_QueueSize = { "QueueSize", nullptr, (EPropertyFlags)0x0010000000000005, UECodeGen_Private::EPropertyGenFlags::Int, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRROS2DepthCameraComponent, QueueSize), METADATA_PARAMS(Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_QueueSize_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_QueueSize_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_MaxDepth_MetaData[] = {
		{ "Category", "RRROS2DepthCameraComponent" },
		{ "Comment", "/** max depth in cm */" },
		{ "ModuleRelativePath", "Public/RRROS2DepthCameraComponent.h" },
		{ "ToolTip", "max depth in cm" },
	};
#endif
	const UECodeGen_Private::FFloatPropertyParams Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_MaxDepth = { "MaxDepth", nullptr, (EPropertyFlags)0x0010000000000005, UECodeGen_Private::EPropertyGenFlags::Float, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRROS2DepthCameraComponent, MaxDepth), METADATA_PARAMS(Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_MaxDepth_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_MaxDepth_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_Encoding_MetaData[] = {
		{ "Category", "RRROS2DepthCameraComponent" },
		{ "ModuleRelativePath", "Public/RRROS2DepthCameraComponent.h" },
	};
#endif
	const UECodeGen_Private::FStrPropertyParams Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_Encoding = { "Encoding", nullptr, (EPropertyFlags)0x0010000000000005, UECodeGen_Private::EPropertyGenFlags::Str, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRROS2DepthCameraComponent, Encoding), METADATA_PARAMS(Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_Encoding_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_Encoding_MetaData)) };
	const UECodeGen_Private::FPropertyParamsBase* const Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::PropPointers[] = {
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_CameraComponent,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_DepthSceneCaptureComponent,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_DepthRenderTarget,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_Width,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_Height,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_QueueSize,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_MaxDepth,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::NewProp_Encoding,
	};
	const FCppClassTypeInfoStatic Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::StaticCppClassTypeInfo = {
		TCppClassTypeTraits<URRROS2DepthCameraComponent>::IsAbstract,
	};
	const UECodeGen_Private::FClassParams Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::ClassParams = {
		&URRROS2DepthCameraComponent::StaticClass,
		"Engine",
		&StaticCppClassTypeInfo,
		DependentSingletons,
		FuncInfo,
		Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::PropPointers,
		nullptr,
		UE_ARRAY_COUNT(DependentSingletons),
		UE_ARRAY_COUNT(FuncInfo),
		UE_ARRAY_COUNT(Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::PropPointers),
		0,
		0x00B000A4u,
		METADATA_PARAMS(Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::Class_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::Class_MetaDataParams))
	};
	UClass* Z_Construct_UClass_URRROS2DepthCameraComponent()
	{
		if (!Z_Registration_Info_UClass_URRROS2DepthCameraComponent.OuterSingleton)
		{
			UECodeGen_Private::ConstructUClass(Z_Registration_Info_UClass_URRROS2DepthCameraComponent.OuterSingleton, Z_Construct_UClass_URRROS2DepthCameraComponent_Statics::ClassParams);
		}
		return Z_Registration_Info_UClass_URRROS2DepthCameraComponent.OuterSingleton;
	}
	template<> EVANSLABSENSORS_API UClass* StaticClass<URRROS2DepthCameraComponent>()
	{
		return URRROS2DepthCameraComponent::StaticClass();
	}
	DEFINE_VTABLE_PTR_HELPER_CTOR(URRROS2DepthCameraComponent);
	URRROS2DepthCameraComponent::~URRROS2DepthCameraComponent() {}
	struct Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRROS2DepthCameraComponent_h_Statics
	{
		static const FClassRegisterCompiledInInfo ClassInfo[];
	};
	const FClassRegisterCompiledInInfo Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRROS2DepthCameraComponent_h_Statics::ClassInfo[] = {
		{ Z_Construct_UClass_URRROS2DepthCameraComponent, URRROS2DepthCameraComponent::StaticClass, TEXT("URRROS2DepthCameraComponent"), &Z_Registration_Info_UClass_URRROS2DepthCameraComponent, CONSTRUCT_RELOAD_VERSION_INFO(FClassReloadVersionInfo, sizeof(URRROS2DepthCameraComponent), 1604971144U) },
	};
	static FRegisterCompiledInInfo Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRROS2DepthCameraComponent_h_3139124108(TEXT("/Script/EvansLabSensors"),
		Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRROS2DepthCameraComponent_h_Statics::ClassInfo, UE_ARRAY_COUNT(Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRROS2DepthCameraComponent_h_Statics::ClassInfo),
		nullptr, 0,
		nullptr, 0);
PRAGMA_ENABLE_DEPRECATION_WARNINGS
