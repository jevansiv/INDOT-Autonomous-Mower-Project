// Copyright Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Generated code exported from UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "UObject/GeneratedCppIncludes.h"
#include "EvansLabSensors/Public/RRROS2RgbCameraComponent.h"
#include "rclUE/Public/Msgs/ROS2Img.h"
PRAGMA_DISABLE_DEPRECATION_WARNINGS
void EmptyLinkFunctionForGeneratedCodeRRROS2RgbCameraComponent() {}
// Cross Module References
	ENGINE_API UClass* Z_Construct_UClass_UCameraComponent_NoRegister();
	ENGINE_API UClass* Z_Construct_UClass_USceneCaptureComponent2D_NoRegister();
	ENGINE_API UClass* Z_Construct_UClass_UTextureRenderTarget2D_NoRegister();
	EVANSLABSENSORS_API UClass* Z_Construct_UClass_URRROS2RgbCameraComponent();
	EVANSLABSENSORS_API UClass* Z_Construct_UClass_URRROS2RgbCameraComponent_NoRegister();
	EVANSLABSENSORS_API UScriptStruct* Z_Construct_UScriptStruct_FFloatRenderRequest();
	RAPYUTASIMULATIONPLUGINS_API UClass* Z_Construct_UClass_URRROS2BaseSensorComponent();
	RCLUE_API UScriptStruct* Z_Construct_UScriptStruct_FROSImg();
	UPackage* Z_Construct_UPackage__Script_EvansLabSensors();
// End Cross Module References
	static FStructRegistrationInfo Z_Registration_Info_UScriptStruct_FloatRenderRequest;
class UScriptStruct* FFloatRenderRequest::StaticStruct()
{
	if (!Z_Registration_Info_UScriptStruct_FloatRenderRequest.OuterSingleton)
	{
		Z_Registration_Info_UScriptStruct_FloatRenderRequest.OuterSingleton = GetStaticStruct(Z_Construct_UScriptStruct_FFloatRenderRequest, Z_Construct_UPackage__Script_EvansLabSensors(), TEXT("FloatRenderRequest"));
	}
	return Z_Registration_Info_UScriptStruct_FloatRenderRequest.OuterSingleton;
}
template<> EVANSLABSENSORS_API UScriptStruct* StaticStruct<FFloatRenderRequest>()
{
	return FFloatRenderRequest::StaticStruct();
}
	struct Z_Construct_UScriptStruct_FFloatRenderRequest_Statics
	{
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Struct_MetaDataParams[];
#endif
		static void* NewStructOps();
		static const UECodeGen_Private::FStructParams ReturnStructParams;
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UScriptStruct_FFloatRenderRequest_Statics::Struct_MetaDataParams[] = {
		{ "ModuleRelativePath", "Public/RRROS2RgbCameraComponent.h" },
	};
#endif
	void* Z_Construct_UScriptStruct_FFloatRenderRequest_Statics::NewStructOps()
	{
		return (UScriptStruct::ICppStructOps*)new UScriptStruct::TCppStructOps<FFloatRenderRequest>();
	}
	const UECodeGen_Private::FStructParams Z_Construct_UScriptStruct_FFloatRenderRequest_Statics::ReturnStructParams = {
		(UObject* (*)())Z_Construct_UPackage__Script_EvansLabSensors,
		nullptr,
		&NewStructOps,
		"FloatRenderRequest",
		sizeof(FFloatRenderRequest),
		alignof(FFloatRenderRequest),
		nullptr,
		0,
		RF_Public|RF_Transient|RF_MarkAsNative,
		EStructFlags(0x00000001),
		METADATA_PARAMS(Z_Construct_UScriptStruct_FFloatRenderRequest_Statics::Struct_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UScriptStruct_FFloatRenderRequest_Statics::Struct_MetaDataParams))
	};
	UScriptStruct* Z_Construct_UScriptStruct_FFloatRenderRequest()
	{
		if (!Z_Registration_Info_UScriptStruct_FloatRenderRequest.InnerSingleton)
		{
			UECodeGen_Private::ConstructUScriptStruct(Z_Registration_Info_UScriptStruct_FloatRenderRequest.InnerSingleton, Z_Construct_UScriptStruct_FFloatRenderRequest_Statics::ReturnStructParams);
		}
		return Z_Registration_Info_UScriptStruct_FloatRenderRequest.InnerSingleton;
	}
	DEFINE_FUNCTION(URRROS2RgbCameraComponent::execGetROS2Data)
	{
		P_FINISH;
		P_NATIVE_BEGIN;
		*(FROSImg*)Z_Param__Result=P_THIS->GetROS2Data();
		P_NATIVE_END;
	}
	DEFINE_FUNCTION(URRROS2RgbCameraComponent::execCaptureRgbNonBlocking)
	{
		P_FINISH;
		P_NATIVE_BEGIN;
		P_THIS->CaptureRgbNonBlocking();
		P_NATIVE_END;
	}
	void URRROS2RgbCameraComponent::StaticRegisterNativesURRROS2RgbCameraComponent()
	{
		UClass* Class = URRROS2RgbCameraComponent::StaticClass();
		static const FNameNativePtrPair Funcs[] = {
			{ "CaptureRgbNonBlocking", &URRROS2RgbCameraComponent::execCaptureRgbNonBlocking },
			{ "GetROS2Data", &URRROS2RgbCameraComponent::execGetROS2Data },
		};
		FNativeFunctionRegistrar::RegisterFunctions(Class, Funcs, UE_ARRAY_COUNT(Funcs));
	}
	struct Z_Construct_UFunction_URRROS2RgbCameraComponent_CaptureRgbNonBlocking_Statics
	{
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UECodeGen_Private::FFunctionParams FuncParams;
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_URRROS2RgbCameraComponent_CaptureRgbNonBlocking_Statics::Function_MetaDataParams[] = {
		{ "Comment", "/**\n     * @brief Capture data by notifing task in #RenderRequestQueue\n     * @sa reference https://github.com/TimmHess/UnrealImageCapture\n     */// Rgb\n" },
		{ "ModuleRelativePath", "Public/RRROS2RgbCameraComponent.h" },
		{ "ToolTip", "@brief Capture data by notifing task in #RenderRequestQueue\n@sa reference https://github.com/TimmHess/UnrealImageCapture\n     // Rgb" },
	};
#endif
	const UECodeGen_Private::FFunctionParams Z_Construct_UFunction_URRROS2RgbCameraComponent_CaptureRgbNonBlocking_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_URRROS2RgbCameraComponent, nullptr, "CaptureRgbNonBlocking", nullptr, nullptr, 0, nullptr, 0, RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x00080401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_URRROS2RgbCameraComponent_CaptureRgbNonBlocking_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_URRROS2RgbCameraComponent_CaptureRgbNonBlocking_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_URRROS2RgbCameraComponent_CaptureRgbNonBlocking()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UECodeGen_Private::ConstructUFunction(&ReturnFunction, Z_Construct_UFunction_URRROS2RgbCameraComponent_CaptureRgbNonBlocking_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_URRROS2RgbCameraComponent_GetROS2Data_Statics
	{
		struct RRROS2RgbCameraComponent_eventGetROS2Data_Parms
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
	const UECodeGen_Private::FStructPropertyParams Z_Construct_UFunction_URRROS2RgbCameraComponent_GetROS2Data_Statics::NewProp_ReturnValue = { "ReturnValue", nullptr, (EPropertyFlags)0x0010000000000580, UECodeGen_Private::EPropertyGenFlags::Struct, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(RRROS2RgbCameraComponent_eventGetROS2Data_Parms, ReturnValue), Z_Construct_UScriptStruct_FROSImg, METADATA_PARAMS(nullptr, 0) }; // 1436882506
	const UECodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_URRROS2RgbCameraComponent_GetROS2Data_Statics::PropPointers[] = {
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_URRROS2RgbCameraComponent_GetROS2Data_Statics::NewProp_ReturnValue,
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_URRROS2RgbCameraComponent_GetROS2Data_Statics::Function_MetaDataParams[] = {
		{ "Comment", "/**\n     * @brief Update ROS2 Msg structure from #RenderRequestQueue\n     *\n     * @return FROSImg\n     */" },
		{ "ModuleRelativePath", "Public/RRROS2RgbCameraComponent.h" },
		{ "ToolTip", "@brief Update ROS2 Msg structure from #RenderRequestQueue\n\n@return FROSImg" },
	};
#endif
	const UECodeGen_Private::FFunctionParams Z_Construct_UFunction_URRROS2RgbCameraComponent_GetROS2Data_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_URRROS2RgbCameraComponent, nullptr, "GetROS2Data", nullptr, nullptr, sizeof(Z_Construct_UFunction_URRROS2RgbCameraComponent_GetROS2Data_Statics::RRROS2RgbCameraComponent_eventGetROS2Data_Parms), Z_Construct_UFunction_URRROS2RgbCameraComponent_GetROS2Data_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_URRROS2RgbCameraComponent_GetROS2Data_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x04020400, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_URRROS2RgbCameraComponent_GetROS2Data_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_URRROS2RgbCameraComponent_GetROS2Data_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_URRROS2RgbCameraComponent_GetROS2Data()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UECodeGen_Private::ConstructUFunction(&ReturnFunction, Z_Construct_UFunction_URRROS2RgbCameraComponent_GetROS2Data_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	IMPLEMENT_CLASS_NO_AUTO_REGISTRATION(URRROS2RgbCameraComponent);
	UClass* Z_Construct_UClass_URRROS2RgbCameraComponent_NoRegister()
	{
		return URRROS2RgbCameraComponent::StaticClass();
	}
	struct Z_Construct_UClass_URRROS2RgbCameraComponent_Statics
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
		static const UECodeGen_Private::FMetaDataPairParam NewProp_RgbSceneCaptureComponent_MetaData[];
#endif
		static const UECodeGen_Private::FObjectPropertyParams NewProp_RgbSceneCaptureComponent;
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam NewProp_RgbRenderTarget_MetaData[];
#endif
		static const UECodeGen_Private::FObjectPropertyParams NewProp_RgbRenderTarget;
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
		static const UECodeGen_Private::FMetaDataPairParam NewProp_Encoding_MetaData[];
#endif
		static const UECodeGen_Private::FStrPropertyParams NewProp_Encoding;
		static const UECodeGen_Private::FPropertyParamsBase* const PropPointers[];
		static const FCppClassTypeInfoStatic StaticCppClassTypeInfo;
		static const UECodeGen_Private::FClassParams ClassParams;
	};
	UObject* (*const Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::DependentSingletons[])() = {
		(UObject* (*)())Z_Construct_UClass_URRROS2BaseSensorComponent,
		(UObject* (*)())Z_Construct_UPackage__Script_EvansLabSensors,
	};
	const FClassFunctionLinkInfo Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::FuncInfo[] = {
		{ &Z_Construct_UFunction_URRROS2RgbCameraComponent_CaptureRgbNonBlocking, "CaptureRgbNonBlocking" }, // 958736519
		{ &Z_Construct_UFunction_URRROS2RgbCameraComponent_GetROS2Data, "GetROS2Data" }, // 3355136828
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::Class_MetaDataParams[] = {
		{ "BlueprintSpawnableComponent", "" },
		{ "BlueprintType", "true" },
		{ "ClassGroupNames", "Custom" },
		{ "Comment", "/**\n * @brief ROS2 Camera component. Uses USceneCaptureComponent2D.\n *\n * @sa [USceneCaptureComponent2D](https://docs.unrealengine.com/5.1/en-US/API/Runtime/Engine/Components/USceneCaptureComponent2D/)\n * @sa [UE4 ShaderInPlugin](https://docs.unrealengine.com/5.1/en-US/ProgrammingAndScripting/Rendering/ShaderInPlugin/Overview/)\n * @sa implementation reference: https://github.com/TimmHess/UnrealImageCapture\n * @todo Support non RGB data support.\n */" },
		{ "HideCategories", "Trigger PhysicsVolume" },
		{ "IncludePath", "RRROS2RgbCameraComponent.h" },
		{ "IsBlueprintBase", "true" },
		{ "ModuleRelativePath", "Public/RRROS2RgbCameraComponent.h" },
		{ "ToolTip", "@brief ROS2 Camera component. Uses USceneCaptureComponent2D.\n\n@sa [USceneCaptureComponent2D](https://docs.unrealengine.com/5.1/en-US/API/Runtime/Engine/Components/USceneCaptureComponent2D/)\n@sa [UE4 ShaderInPlugin](https://docs.unrealengine.com/5.1/en-US/ProgrammingAndScripting/Rendering/ShaderInPlugin/Overview/)\n@sa implementation reference: https://github.com/TimmHess/UnrealImageCapture\n@todo Support non RGB data support." },
	};
#endif
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_CameraComponent_MetaData[] = {
		{ "Category", "RRROS2RgbCameraComponent" },
		{ "Comment", "//! Camera. Not necessary to capture but useful to see image in UE4 windows.\n" },
		{ "EditInline", "true" },
		{ "ModuleRelativePath", "Public/RRROS2RgbCameraComponent.h" },
		{ "ToolTip", "! Camera. Not necessary to capture but useful to see image in UE4 windows." },
	};
#endif
	const UECodeGen_Private::FObjectPropertyParams Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_CameraComponent = { "CameraComponent", nullptr, (EPropertyFlags)0x00100000000a000d, UECodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRROS2RgbCameraComponent, CameraComponent), Z_Construct_UClass_UCameraComponent_NoRegister, METADATA_PARAMS(Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_CameraComponent_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_CameraComponent_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_RgbSceneCaptureComponent_MetaData[] = {
		{ "Category", "RRROS2RgbCameraComponent" },
		{ "EditInline", "true" },
		{ "ModuleRelativePath", "Public/RRROS2RgbCameraComponent.h" },
	};
#endif
	const UECodeGen_Private::FObjectPropertyParams Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_RgbSceneCaptureComponent = { "RgbSceneCaptureComponent", nullptr, (EPropertyFlags)0x00100000000a000d, UECodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRROS2RgbCameraComponent, RgbSceneCaptureComponent), Z_Construct_UClass_USceneCaptureComponent2D_NoRegister, METADATA_PARAMS(Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_RgbSceneCaptureComponent_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_RgbSceneCaptureComponent_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_RgbRenderTarget_MetaData[] = {
		{ "Category", "RRROS2RgbCameraComponent" },
		{ "ModuleRelativePath", "Public/RRROS2RgbCameraComponent.h" },
	};
#endif
	const UECodeGen_Private::FObjectPropertyParams Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_RgbRenderTarget = { "RgbRenderTarget", nullptr, (EPropertyFlags)0x0010000000020005, UECodeGen_Private::EPropertyGenFlags::Object, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRROS2RgbCameraComponent, RgbRenderTarget), Z_Construct_UClass_UTextureRenderTarget2D_NoRegister, METADATA_PARAMS(Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_RgbRenderTarget_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_RgbRenderTarget_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_Width_MetaData[] = {
		{ "Category", "RRROS2RgbCameraComponent" },
		{ "ModuleRelativePath", "Public/RRROS2RgbCameraComponent.h" },
	};
#endif
	const UECodeGen_Private::FIntPropertyParams Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_Width = { "Width", nullptr, (EPropertyFlags)0x0010000000000005, UECodeGen_Private::EPropertyGenFlags::Int, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRROS2RgbCameraComponent, Width), METADATA_PARAMS(Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_Width_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_Width_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_Height_MetaData[] = {
		{ "Category", "RRROS2RgbCameraComponent" },
		{ "Comment", "//640 standard setting to OAKD 1280\n" },
		{ "ModuleRelativePath", "Public/RRROS2RgbCameraComponent.h" },
		{ "ToolTip", "640 standard setting to OAKD 1280" },
	};
#endif
	const UECodeGen_Private::FIntPropertyParams Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_Height = { "Height", nullptr, (EPropertyFlags)0x0010000000000005, UECodeGen_Private::EPropertyGenFlags::Int, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRROS2RgbCameraComponent, Height), METADATA_PARAMS(Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_Height_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_Height_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_QueueSize_MetaData[] = {
		{ "Category", "RRROS2RgbCameraComponent" },
		{ "Comment", "//480 standard setting to OAKD 720\n" },
		{ "ModuleRelativePath", "Public/RRROS2RgbCameraComponent.h" },
		{ "ToolTip", "480 standard setting to OAKD 720" },
	};
#endif
	const UECodeGen_Private::FIntPropertyParams Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_QueueSize = { "QueueSize", nullptr, (EPropertyFlags)0x0010000000000005, UECodeGen_Private::EPropertyGenFlags::Int, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRROS2RgbCameraComponent, QueueSize), METADATA_PARAMS(Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_QueueSize_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_QueueSize_MetaData)) };
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_Encoding_MetaData[] = {
		{ "Category", "RRROS2RgbCameraComponent" },
		{ "ModuleRelativePath", "Public/RRROS2RgbCameraComponent.h" },
	};
#endif
	const UECodeGen_Private::FStrPropertyParams Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_Encoding = { "Encoding", nullptr, (EPropertyFlags)0x0010000000000005, UECodeGen_Private::EPropertyGenFlags::Str, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(URRROS2RgbCameraComponent, Encoding), METADATA_PARAMS(Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_Encoding_MetaData, UE_ARRAY_COUNT(Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_Encoding_MetaData)) };
	const UECodeGen_Private::FPropertyParamsBase* const Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::PropPointers[] = {
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_CameraComponent,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_RgbSceneCaptureComponent,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_RgbRenderTarget,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_Width,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_Height,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_QueueSize,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::NewProp_Encoding,
	};
	const FCppClassTypeInfoStatic Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::StaticCppClassTypeInfo = {
		TCppClassTypeTraits<URRROS2RgbCameraComponent>::IsAbstract,
	};
	const UECodeGen_Private::FClassParams Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::ClassParams = {
		&URRROS2RgbCameraComponent::StaticClass,
		"Engine",
		&StaticCppClassTypeInfo,
		DependentSingletons,
		FuncInfo,
		Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::PropPointers,
		nullptr,
		UE_ARRAY_COUNT(DependentSingletons),
		UE_ARRAY_COUNT(FuncInfo),
		UE_ARRAY_COUNT(Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::PropPointers),
		0,
		0x00B000A4u,
		METADATA_PARAMS(Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::Class_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::Class_MetaDataParams))
	};
	UClass* Z_Construct_UClass_URRROS2RgbCameraComponent()
	{
		if (!Z_Registration_Info_UClass_URRROS2RgbCameraComponent.OuterSingleton)
		{
			UECodeGen_Private::ConstructUClass(Z_Registration_Info_UClass_URRROS2RgbCameraComponent.OuterSingleton, Z_Construct_UClass_URRROS2RgbCameraComponent_Statics::ClassParams);
		}
		return Z_Registration_Info_UClass_URRROS2RgbCameraComponent.OuterSingleton;
	}
	template<> EVANSLABSENSORS_API UClass* StaticClass<URRROS2RgbCameraComponent>()
	{
		return URRROS2RgbCameraComponent::StaticClass();
	}
	DEFINE_VTABLE_PTR_HELPER_CTOR(URRROS2RgbCameraComponent);
	URRROS2RgbCameraComponent::~URRROS2RgbCameraComponent() {}
	struct Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRROS2RgbCameraComponent_h_Statics
	{
		static const FStructRegisterCompiledInInfo ScriptStructInfo[];
		static const FClassRegisterCompiledInInfo ClassInfo[];
	};
	const FStructRegisterCompiledInInfo Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRROS2RgbCameraComponent_h_Statics::ScriptStructInfo[] = {
		{ FFloatRenderRequest::StaticStruct, Z_Construct_UScriptStruct_FFloatRenderRequest_Statics::NewStructOps, TEXT("FloatRenderRequest"), &Z_Registration_Info_UScriptStruct_FloatRenderRequest, CONSTRUCT_RELOAD_VERSION_INFO(FStructReloadVersionInfo, sizeof(FFloatRenderRequest), 1252738353U) },
	};
	const FClassRegisterCompiledInInfo Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRROS2RgbCameraComponent_h_Statics::ClassInfo[] = {
		{ Z_Construct_UClass_URRROS2RgbCameraComponent, URRROS2RgbCameraComponent::StaticClass, TEXT("URRROS2RgbCameraComponent"), &Z_Registration_Info_UClass_URRROS2RgbCameraComponent, CONSTRUCT_RELOAD_VERSION_INFO(FClassReloadVersionInfo, sizeof(URRROS2RgbCameraComponent), 2993860112U) },
	};
	static FRegisterCompiledInInfo Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRROS2RgbCameraComponent_h_3049958459(TEXT("/Script/EvansLabSensors"),
		Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRROS2RgbCameraComponent_h_Statics::ClassInfo, UE_ARRAY_COUNT(Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRROS2RgbCameraComponent_h_Statics::ClassInfo),
		Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRROS2RgbCameraComponent_h_Statics::ScriptStructInfo, UE_ARRAY_COUNT(Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_EvansLabSensors_Source_EvansLabSensors_Public_RRROS2RgbCameraComponent_h_Statics::ScriptStructInfo),
		nullptr, 0);
PRAGMA_ENABLE_DEPRECATION_WARNINGS
