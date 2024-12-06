#!/bin/bash
enginedir="/home/evanslab/UnrealEngine-5.1/Engine"
projectfile=$(ls . | grep ".uproject")
projectname="${projectfile%.*}"
projectfile=$(realpath $projectfile)
builddir="dist"
mkdir dist -p


map="FirstPersonMap"

TEMP=$(getopt -o cbrg --long cook,build,run,gen -- "$@")
eval set -- "$TEMP"
#default values
COOK=0
BUILD=0
RUN=0
GEN=0
while true; do
    case "$1" in
        -c | --cook ) COOK=1; shift ;;
        -b | --build ) BUILD=1; shift ;;
        -r | --run ) RUN=1; shift ;;
        -g | --gen ) GEN=1; shift ;;
        -- ) if [ -n "$2" ]
            then
                ARGUMENT1=$2
                if [ -n "$3" ]
                then
                    ARGUMENT2=$3
                    if [ -n "$4" ]
                    then
                        shift 3
                        echo "Unexpected options: \"$@\" . exiting."
                        exit 1;
                    fi
                fi
            fi
            shift 2; break;;
        * ) break ;;
    esac
done

# $projectname = (Get-Item $projectfilepath ).Basename


# gen solution
if [ $GEN -ne 0 ]
then
$enginedir/Binaries/DotNET/UnrealBuildTool/UnrealBuildTool -projectfiles -vscode -project=$projectfile -game -engine
fi

# run editor project
if [ $RUN -ne 0 ]
then
$enginedir/Binaries/Linux/UnrealEditor $projectfile
fi

# cook project
if [ $COOK -ne 0 ]
then
$enginedir/Binaries/Linux/UnrealEditor-Cmd $projectfilepath -run=cook -targetplatform=Linux -iterate -map=$map
fi


# build project
if [ $BUILD -ne 0 ]
then
$enginedir/Binaries/DotNET/AutomationTool/AutomationTool -ScriptsForProject=$projectfilepath BuildCookRun -project=$projectfilepath -noP4 -clientconfig=Shipping -serverconfig=Shipping -nocompile -nocompileeditor -installed -ue4exe=$enginedir/Binaries/Linux/UnrealEditor-Cmd -utf8output -platform=Linux -build -skipcook -compressed -stage -deploy -stagingdirectory=$builddir
fi

