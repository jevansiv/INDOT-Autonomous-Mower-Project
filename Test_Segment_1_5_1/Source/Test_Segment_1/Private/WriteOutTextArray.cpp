// Fill out your copyright notice in the Description page of Project Settings.


#include "WriteOutTextArray.h"
#include "Misc/FileHelper.h"
#include "HAL/PlatformFileManager.h"
#include <cstdlib>
#include <iostream>
#include <string>

bool UWriteOutTextArray::SaveArrayText(FString SaveDirectory, FString FileName, TArray<FString> SaveText, bool AllowOverWriting)
{
	// Set complete file path
	SaveDirectory += "\\";
	SaveDirectory += FileName;

	if (!AllowOverWriting)
	{
		if (FPlatformFileManager::Get().GetPlatformFile().FileExists(*SaveDirectory))
		{
			/*
			int random = rand();
			std::string s = std::to_string(random);
			FileName = s + "_" + FileName;
			*/
			return false;
		}
	}

	FString FinalString = "";
	for (FString& Each : SaveText)
	{
		FinalString += Each;
		FinalString += LINE_TERMINATOR;
	}

	return FFileHelper::SaveStringToFile(FinalString, *SaveDirectory);
}


