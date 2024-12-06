// Copyright Epic Games, Inc. All Rights Reserved.
/*===========================================================================
	Generated code exported from UnrealHeaderTool.
	DO NOT modify this manually! Edit the corresponding .h files instead!
===========================================================================*/

#include "UObject/GeneratedCppIncludes.h"
#include "CSV_Creator/Public/CSVCreatorComponent.h"
PRAGMA_DISABLE_DEPRECATION_WARNINGS
void EmptyLinkFunctionForGeneratedCodeCSVCreatorComponent() {}
// Cross Module References
	CSV_CREATOR_API UClass* Z_Construct_UClass_UCSVCreatorComponent();
	CSV_CREATOR_API UClass* Z_Construct_UClass_UCSVCreatorComponent_NoRegister();
	ENGINE_API UClass* Z_Construct_UClass_UActorComponent();
	UPackage* Z_Construct_UPackage__Script_CSV_Creator();
// End Cross Module References
	DEFINE_FUNCTION(UCSVCreatorComponent::execLoadANSI_TextFile)
	{
		P_GET_PROPERTY(FStrProperty,Z_Param_SaveDirectory);
		P_GET_PROPERTY(FStrProperty,Z_Param_FileName);
		P_FINISH;
		P_NATIVE_BEGIN;
		*(TArray<FString>*)Z_Param__Result=UCSVCreatorComponent::LoadANSI_TextFile(Z_Param_SaveDirectory,Z_Param_FileName);
		P_NATIVE_END;
	}
	DEFINE_FUNCTION(UCSVCreatorComponent::execLoadText_File)
	{
		P_GET_PROPERTY(FStrProperty,Z_Param_Directory);
		P_GET_PROPERTY(FStrProperty,Z_Param_FileName);
		P_FINISH;
		P_NATIVE_BEGIN;
		*(TArray<FString>*)Z_Param__Result=UCSVCreatorComponent::LoadText_File(Z_Param_Directory,Z_Param_FileName);
		P_NATIVE_END;
	}
	DEFINE_FUNCTION(UCSVCreatorComponent::execSaveText_File)
	{
		P_GET_PROPERTY(FStrProperty,Z_Param_SaveDirectory);
		P_GET_PROPERTY(FStrProperty,Z_Param_FileName);
		P_GET_TARRAY(FString,Z_Param_SaveText);
		P_GET_UBOOL(Z_Param_AllowOverWriting);
		P_FINISH;
		P_NATIVE_BEGIN;
		*(bool*)Z_Param__Result=UCSVCreatorComponent::SaveText_File(Z_Param_SaveDirectory,Z_Param_FileName,Z_Param_SaveText,Z_Param_AllowOverWriting);
		P_NATIVE_END;
	}
	void UCSVCreatorComponent::StaticRegisterNativesUCSVCreatorComponent()
	{
		UClass* Class = UCSVCreatorComponent::StaticClass();
		static const FNameNativePtrPair Funcs[] = {
			{ "LoadANSI_TextFile", &UCSVCreatorComponent::execLoadANSI_TextFile },
			{ "LoadText_File", &UCSVCreatorComponent::execLoadText_File },
			{ "SaveText_File", &UCSVCreatorComponent::execSaveText_File },
		};
		FNativeFunctionRegistrar::RegisterFunctions(Class, Funcs, UE_ARRAY_COUNT(Funcs));
	}
	struct Z_Construct_UFunction_UCSVCreatorComponent_LoadANSI_TextFile_Statics
	{
		struct CSVCreatorComponent_eventLoadANSI_TextFile_Parms
		{
			FString SaveDirectory;
			FString FileName;
			TArray<FString> ReturnValue;
		};
		static const UECodeGen_Private::FStrPropertyParams NewProp_SaveDirectory;
		static const UECodeGen_Private::FStrPropertyParams NewProp_FileName;
		static const UECodeGen_Private::FStrPropertyParams NewProp_ReturnValue_Inner;
		static const UECodeGen_Private::FArrayPropertyParams NewProp_ReturnValue;
		static const UECodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UECodeGen_Private::FFunctionParams FuncParams;
	};
	const UECodeGen_Private::FStrPropertyParams Z_Construct_UFunction_UCSVCreatorComponent_LoadANSI_TextFile_Statics::NewProp_SaveDirectory = { "SaveDirectory", nullptr, (EPropertyFlags)0x0010000000000080, UECodeGen_Private::EPropertyGenFlags::Str, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(CSVCreatorComponent_eventLoadANSI_TextFile_Parms, SaveDirectory), METADATA_PARAMS(nullptr, 0) };
	const UECodeGen_Private::FStrPropertyParams Z_Construct_UFunction_UCSVCreatorComponent_LoadANSI_TextFile_Statics::NewProp_FileName = { "FileName", nullptr, (EPropertyFlags)0x0010000000000080, UECodeGen_Private::EPropertyGenFlags::Str, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(CSVCreatorComponent_eventLoadANSI_TextFile_Parms, FileName), METADATA_PARAMS(nullptr, 0) };
	const UECodeGen_Private::FStrPropertyParams Z_Construct_UFunction_UCSVCreatorComponent_LoadANSI_TextFile_Statics::NewProp_ReturnValue_Inner = { "ReturnValue", nullptr, (EPropertyFlags)0x0000000000000000, UECodeGen_Private::EPropertyGenFlags::Str, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, 0, METADATA_PARAMS(nullptr, 0) };
	const UECodeGen_Private::FArrayPropertyParams Z_Construct_UFunction_UCSVCreatorComponent_LoadANSI_TextFile_Statics::NewProp_ReturnValue = { "ReturnValue", nullptr, (EPropertyFlags)0x0010000000000580, UECodeGen_Private::EPropertyGenFlags::Array, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(CSVCreatorComponent_eventLoadANSI_TextFile_Parms, ReturnValue), EArrayPropertyFlags::None, METADATA_PARAMS(nullptr, 0) };
	const UECodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_UCSVCreatorComponent_LoadANSI_TextFile_Statics::PropPointers[] = {
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UCSVCreatorComponent_LoadANSI_TextFile_Statics::NewProp_SaveDirectory,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UCSVCreatorComponent_LoadANSI_TextFile_Statics::NewProp_FileName,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UCSVCreatorComponent_LoadANSI_TextFile_Statics::NewProp_ReturnValue_Inner,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UCSVCreatorComponent_LoadANSI_TextFile_Statics::NewProp_ReturnValue,
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_UCSVCreatorComponent_LoadANSI_TextFile_Statics::Function_MetaDataParams[] = {
		{ "Category", "SaveLoadTextFilePlugin" },
		{ "Keywords", "Load" },
		{ "ModuleRelativePath", "Public/CSVCreatorComponent.h" },
	};
#endif
	const UECodeGen_Private::FFunctionParams Z_Construct_UFunction_UCSVCreatorComponent_LoadANSI_TextFile_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_UCSVCreatorComponent, nullptr, "LoadANSI_TextFile", nullptr, nullptr, sizeof(Z_Construct_UFunction_UCSVCreatorComponent_LoadANSI_TextFile_Statics::CSVCreatorComponent_eventLoadANSI_TextFile_Parms), Z_Construct_UFunction_UCSVCreatorComponent_LoadANSI_TextFile_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_UCSVCreatorComponent_LoadANSI_TextFile_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x04082401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_UCSVCreatorComponent_LoadANSI_TextFile_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_UCSVCreatorComponent_LoadANSI_TextFile_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_UCSVCreatorComponent_LoadANSI_TextFile()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UECodeGen_Private::ConstructUFunction(&ReturnFunction, Z_Construct_UFunction_UCSVCreatorComponent_LoadANSI_TextFile_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_UCSVCreatorComponent_LoadText_File_Statics
	{
		struct CSVCreatorComponent_eventLoadText_File_Parms
		{
			FString Directory;
			FString FileName;
			TArray<FString> ReturnValue;
		};
		static const UECodeGen_Private::FStrPropertyParams NewProp_Directory;
		static const UECodeGen_Private::FStrPropertyParams NewProp_FileName;
		static const UECodeGen_Private::FStrPropertyParams NewProp_ReturnValue_Inner;
		static const UECodeGen_Private::FArrayPropertyParams NewProp_ReturnValue;
		static const UECodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UECodeGen_Private::FFunctionParams FuncParams;
	};
	const UECodeGen_Private::FStrPropertyParams Z_Construct_UFunction_UCSVCreatorComponent_LoadText_File_Statics::NewProp_Directory = { "Directory", nullptr, (EPropertyFlags)0x0010000000000080, UECodeGen_Private::EPropertyGenFlags::Str, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(CSVCreatorComponent_eventLoadText_File_Parms, Directory), METADATA_PARAMS(nullptr, 0) };
	const UECodeGen_Private::FStrPropertyParams Z_Construct_UFunction_UCSVCreatorComponent_LoadText_File_Statics::NewProp_FileName = { "FileName", nullptr, (EPropertyFlags)0x0010000000000080, UECodeGen_Private::EPropertyGenFlags::Str, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(CSVCreatorComponent_eventLoadText_File_Parms, FileName), METADATA_PARAMS(nullptr, 0) };
	const UECodeGen_Private::FStrPropertyParams Z_Construct_UFunction_UCSVCreatorComponent_LoadText_File_Statics::NewProp_ReturnValue_Inner = { "ReturnValue", nullptr, (EPropertyFlags)0x0000000000000000, UECodeGen_Private::EPropertyGenFlags::Str, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, 0, METADATA_PARAMS(nullptr, 0) };
	const UECodeGen_Private::FArrayPropertyParams Z_Construct_UFunction_UCSVCreatorComponent_LoadText_File_Statics::NewProp_ReturnValue = { "ReturnValue", nullptr, (EPropertyFlags)0x0010000000000580, UECodeGen_Private::EPropertyGenFlags::Array, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(CSVCreatorComponent_eventLoadText_File_Parms, ReturnValue), EArrayPropertyFlags::None, METADATA_PARAMS(nullptr, 0) };
	const UECodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_UCSVCreatorComponent_LoadText_File_Statics::PropPointers[] = {
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UCSVCreatorComponent_LoadText_File_Statics::NewProp_Directory,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UCSVCreatorComponent_LoadText_File_Statics::NewProp_FileName,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UCSVCreatorComponent_LoadText_File_Statics::NewProp_ReturnValue_Inner,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UCSVCreatorComponent_LoadText_File_Statics::NewProp_ReturnValue,
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_UCSVCreatorComponent_LoadText_File_Statics::Function_MetaDataParams[] = {
		{ "Category", "SaveLoadTextFilePlugin" },
		{ "Keywords", "Load" },
		{ "ModuleRelativePath", "Public/CSVCreatorComponent.h" },
	};
#endif
	const UECodeGen_Private::FFunctionParams Z_Construct_UFunction_UCSVCreatorComponent_LoadText_File_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_UCSVCreatorComponent, nullptr, "LoadText_File", nullptr, nullptr, sizeof(Z_Construct_UFunction_UCSVCreatorComponent_LoadText_File_Statics::CSVCreatorComponent_eventLoadText_File_Parms), Z_Construct_UFunction_UCSVCreatorComponent_LoadText_File_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_UCSVCreatorComponent_LoadText_File_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x04082401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_UCSVCreatorComponent_LoadText_File_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_UCSVCreatorComponent_LoadText_File_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_UCSVCreatorComponent_LoadText_File()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UECodeGen_Private::ConstructUFunction(&ReturnFunction, Z_Construct_UFunction_UCSVCreatorComponent_LoadText_File_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	struct Z_Construct_UFunction_UCSVCreatorComponent_SaveText_File_Statics
	{
		struct CSVCreatorComponent_eventSaveText_File_Parms
		{
			FString SaveDirectory;
			FString FileName;
			TArray<FString> SaveText;
			bool AllowOverWriting;
			bool ReturnValue;
		};
		static const UECodeGen_Private::FStrPropertyParams NewProp_SaveDirectory;
		static const UECodeGen_Private::FStrPropertyParams NewProp_FileName;
		static const UECodeGen_Private::FStrPropertyParams NewProp_SaveText_Inner;
		static const UECodeGen_Private::FArrayPropertyParams NewProp_SaveText;
		static void NewProp_AllowOverWriting_SetBit(void* Obj);
		static const UECodeGen_Private::FBoolPropertyParams NewProp_AllowOverWriting;
		static void NewProp_ReturnValue_SetBit(void* Obj);
		static const UECodeGen_Private::FBoolPropertyParams NewProp_ReturnValue;
		static const UECodeGen_Private::FPropertyParamsBase* const PropPointers[];
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Function_MetaDataParams[];
#endif
		static const UECodeGen_Private::FFunctionParams FuncParams;
	};
	const UECodeGen_Private::FStrPropertyParams Z_Construct_UFunction_UCSVCreatorComponent_SaveText_File_Statics::NewProp_SaveDirectory = { "SaveDirectory", nullptr, (EPropertyFlags)0x0010000000000080, UECodeGen_Private::EPropertyGenFlags::Str, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(CSVCreatorComponent_eventSaveText_File_Parms, SaveDirectory), METADATA_PARAMS(nullptr, 0) };
	const UECodeGen_Private::FStrPropertyParams Z_Construct_UFunction_UCSVCreatorComponent_SaveText_File_Statics::NewProp_FileName = { "FileName", nullptr, (EPropertyFlags)0x0010000000000080, UECodeGen_Private::EPropertyGenFlags::Str, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(CSVCreatorComponent_eventSaveText_File_Parms, FileName), METADATA_PARAMS(nullptr, 0) };
	const UECodeGen_Private::FStrPropertyParams Z_Construct_UFunction_UCSVCreatorComponent_SaveText_File_Statics::NewProp_SaveText_Inner = { "SaveText", nullptr, (EPropertyFlags)0x0000000000000000, UECodeGen_Private::EPropertyGenFlags::Str, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, 0, METADATA_PARAMS(nullptr, 0) };
	const UECodeGen_Private::FArrayPropertyParams Z_Construct_UFunction_UCSVCreatorComponent_SaveText_File_Statics::NewProp_SaveText = { "SaveText", nullptr, (EPropertyFlags)0x0010000000000080, UECodeGen_Private::EPropertyGenFlags::Array, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, STRUCT_OFFSET(CSVCreatorComponent_eventSaveText_File_Parms, SaveText), EArrayPropertyFlags::None, METADATA_PARAMS(nullptr, 0) };
	void Z_Construct_UFunction_UCSVCreatorComponent_SaveText_File_Statics::NewProp_AllowOverWriting_SetBit(void* Obj)
	{
		((CSVCreatorComponent_eventSaveText_File_Parms*)Obj)->AllowOverWriting = 1;
	}
	const UECodeGen_Private::FBoolPropertyParams Z_Construct_UFunction_UCSVCreatorComponent_SaveText_File_Statics::NewProp_AllowOverWriting = { "AllowOverWriting", nullptr, (EPropertyFlags)0x0010000000000080, UECodeGen_Private::EPropertyGenFlags::Bool | UECodeGen_Private::EPropertyGenFlags::NativeBool, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, sizeof(bool), sizeof(CSVCreatorComponent_eventSaveText_File_Parms), &Z_Construct_UFunction_UCSVCreatorComponent_SaveText_File_Statics::NewProp_AllowOverWriting_SetBit, METADATA_PARAMS(nullptr, 0) };
	void Z_Construct_UFunction_UCSVCreatorComponent_SaveText_File_Statics::NewProp_ReturnValue_SetBit(void* Obj)
	{
		((CSVCreatorComponent_eventSaveText_File_Parms*)Obj)->ReturnValue = 1;
	}
	const UECodeGen_Private::FBoolPropertyParams Z_Construct_UFunction_UCSVCreatorComponent_SaveText_File_Statics::NewProp_ReturnValue = { "ReturnValue", nullptr, (EPropertyFlags)0x0010000000000580, UECodeGen_Private::EPropertyGenFlags::Bool | UECodeGen_Private::EPropertyGenFlags::NativeBool, RF_Public|RF_Transient|RF_MarkAsNative, 1, nullptr, nullptr, sizeof(bool), sizeof(CSVCreatorComponent_eventSaveText_File_Parms), &Z_Construct_UFunction_UCSVCreatorComponent_SaveText_File_Statics::NewProp_ReturnValue_SetBit, METADATA_PARAMS(nullptr, 0) };
	const UECodeGen_Private::FPropertyParamsBase* const Z_Construct_UFunction_UCSVCreatorComponent_SaveText_File_Statics::PropPointers[] = {
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UCSVCreatorComponent_SaveText_File_Statics::NewProp_SaveDirectory,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UCSVCreatorComponent_SaveText_File_Statics::NewProp_FileName,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UCSVCreatorComponent_SaveText_File_Statics::NewProp_SaveText_Inner,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UCSVCreatorComponent_SaveText_File_Statics::NewProp_SaveText,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UCSVCreatorComponent_SaveText_File_Statics::NewProp_AllowOverWriting,
		(const UECodeGen_Private::FPropertyParamsBase*)&Z_Construct_UFunction_UCSVCreatorComponent_SaveText_File_Statics::NewProp_ReturnValue,
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UFunction_UCSVCreatorComponent_SaveText_File_Statics::Function_MetaDataParams[] = {
		{ "Category", "SaveLoadTextFilePlugin" },
		{ "Keywords", "Save" },
		{ "ModuleRelativePath", "Public/CSVCreatorComponent.h" },
	};
#endif
	const UECodeGen_Private::FFunctionParams Z_Construct_UFunction_UCSVCreatorComponent_SaveText_File_Statics::FuncParams = { (UObject*(*)())Z_Construct_UClass_UCSVCreatorComponent, nullptr, "SaveText_File", nullptr, nullptr, sizeof(Z_Construct_UFunction_UCSVCreatorComponent_SaveText_File_Statics::CSVCreatorComponent_eventSaveText_File_Parms), Z_Construct_UFunction_UCSVCreatorComponent_SaveText_File_Statics::PropPointers, UE_ARRAY_COUNT(Z_Construct_UFunction_UCSVCreatorComponent_SaveText_File_Statics::PropPointers), RF_Public|RF_Transient|RF_MarkAsNative, (EFunctionFlags)0x04082401, 0, 0, METADATA_PARAMS(Z_Construct_UFunction_UCSVCreatorComponent_SaveText_File_Statics::Function_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UFunction_UCSVCreatorComponent_SaveText_File_Statics::Function_MetaDataParams)) };
	UFunction* Z_Construct_UFunction_UCSVCreatorComponent_SaveText_File()
	{
		static UFunction* ReturnFunction = nullptr;
		if (!ReturnFunction)
		{
			UECodeGen_Private::ConstructUFunction(&ReturnFunction, Z_Construct_UFunction_UCSVCreatorComponent_SaveText_File_Statics::FuncParams);
		}
		return ReturnFunction;
	}
	IMPLEMENT_CLASS_NO_AUTO_REGISTRATION(UCSVCreatorComponent);
	UClass* Z_Construct_UClass_UCSVCreatorComponent_NoRegister()
	{
		return UCSVCreatorComponent::StaticClass();
	}
	struct Z_Construct_UClass_UCSVCreatorComponent_Statics
	{
		static UObject* (*const DependentSingletons[])();
		static const FClassFunctionLinkInfo FuncInfo[];
#if WITH_METADATA
		static const UECodeGen_Private::FMetaDataPairParam Class_MetaDataParams[];
#endif
		static const FCppClassTypeInfoStatic StaticCppClassTypeInfo;
		static const UECodeGen_Private::FClassParams ClassParams;
	};
	UObject* (*const Z_Construct_UClass_UCSVCreatorComponent_Statics::DependentSingletons[])() = {
		(UObject* (*)())Z_Construct_UClass_UActorComponent,
		(UObject* (*)())Z_Construct_UPackage__Script_CSV_Creator,
	};
	const FClassFunctionLinkInfo Z_Construct_UClass_UCSVCreatorComponent_Statics::FuncInfo[] = {
		{ &Z_Construct_UFunction_UCSVCreatorComponent_LoadANSI_TextFile, "LoadANSI_TextFile" }, // 4062374800
		{ &Z_Construct_UFunction_UCSVCreatorComponent_LoadText_File, "LoadText_File" }, // 2734488494
		{ &Z_Construct_UFunction_UCSVCreatorComponent_SaveText_File, "SaveText_File" }, // 1172513866
	};
#if WITH_METADATA
	const UECodeGen_Private::FMetaDataPairParam Z_Construct_UClass_UCSVCreatorComponent_Statics::Class_MetaDataParams[] = {
		{ "BlueprintSpawnableComponent", "" },
		{ "ClassGroupNames", "Custom" },
		{ "IncludePath", "CSVCreatorComponent.h" },
		{ "ModuleRelativePath", "Public/CSVCreatorComponent.h" },
	};
#endif
	const FCppClassTypeInfoStatic Z_Construct_UClass_UCSVCreatorComponent_Statics::StaticCppClassTypeInfo = {
		TCppClassTypeTraits<UCSVCreatorComponent>::IsAbstract,
	};
	const UECodeGen_Private::FClassParams Z_Construct_UClass_UCSVCreatorComponent_Statics::ClassParams = {
		&UCSVCreatorComponent::StaticClass,
		"Engine",
		&StaticCppClassTypeInfo,
		DependentSingletons,
		FuncInfo,
		nullptr,
		nullptr,
		UE_ARRAY_COUNT(DependentSingletons),
		UE_ARRAY_COUNT(FuncInfo),
		0,
		0,
		0x00B000A4u,
		METADATA_PARAMS(Z_Construct_UClass_UCSVCreatorComponent_Statics::Class_MetaDataParams, UE_ARRAY_COUNT(Z_Construct_UClass_UCSVCreatorComponent_Statics::Class_MetaDataParams))
	};
	UClass* Z_Construct_UClass_UCSVCreatorComponent()
	{
		if (!Z_Registration_Info_UClass_UCSVCreatorComponent.OuterSingleton)
		{
			UECodeGen_Private::ConstructUClass(Z_Registration_Info_UClass_UCSVCreatorComponent.OuterSingleton, Z_Construct_UClass_UCSVCreatorComponent_Statics::ClassParams);
		}
		return Z_Registration_Info_UClass_UCSVCreatorComponent.OuterSingleton;
	}
	template<> CSV_CREATOR_API UClass* StaticClass<UCSVCreatorComponent>()
	{
		return UCSVCreatorComponent::StaticClass();
	}
	DEFINE_VTABLE_PTR_HELPER_CTOR(UCSVCreatorComponent);
	UCSVCreatorComponent::~UCSVCreatorComponent() {}
	struct Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_CSV_Creator_Source_CSV_Creator_Public_CSVCreatorComponent_h_Statics
	{
		static const FClassRegisterCompiledInInfo ClassInfo[];
	};
	const FClassRegisterCompiledInInfo Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_CSV_Creator_Source_CSV_Creator_Public_CSVCreatorComponent_h_Statics::ClassInfo[] = {
		{ Z_Construct_UClass_UCSVCreatorComponent, UCSVCreatorComponent::StaticClass, TEXT("UCSVCreatorComponent"), &Z_Registration_Info_UClass_UCSVCreatorComponent, CONSTRUCT_RELOAD_VERSION_INFO(FClassReloadVersionInfo, sizeof(UCSVCreatorComponent), 2184133530U) },
	};
	static FRegisterCompiledInInfo Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_CSV_Creator_Source_CSV_Creator_Public_CSVCreatorComponent_h_661354853(TEXT("/Script/CSV_Creator"),
		Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_CSV_Creator_Source_CSV_Creator_Public_CSVCreatorComponent_h_Statics::ClassInfo, UE_ARRAY_COUNT(Z_CompiledInDeferFile_FID_Documents_Unreal_Projects_Test_Segment_1_5_1_Plugins_CSV_Creator_Source_CSV_Creator_Public_CSVCreatorComponent_h_Statics::ClassInfo),
		nullptr, 0,
		nullptr, 0);
PRAGMA_ENABLE_DEPRECATION_WARNINGS
