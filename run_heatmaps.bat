@echo off
setlocal enabledelayedexpansion

echo =====================================================
echo Running heatmaps for ALL slides in coords directory
echo =====================================================

REM Set paths
set COORDS_DIR=D:\IPD\CLAM-master\datasets\coords
set CLAM_OUT=D:\IPD\CLAM-master\heatmaps
set DTFD_OUT=D:\IPD\DTFD-MIL\IPD_Brain-main\IPD-Brain-main\attention_maps
set OUT_DIR=combined

mkdir %OUT_DIR% 2>nul

REM Loop through all CSV files
for %%F in ("%COORDS_DIR%\*.csv") do (
    
    REM Extract slide ID (remove folder + .csv)
    set FILE=%%~nF
    set SLIDE_ID=!FILE!

    echo.
    echo -----------------------------------------
    echo Processing slide: !SLIDE_ID!
    echo -----------------------------------------

    REM ============= RUN CLAM =============
    echo Running CLAM heatmap...
    python D:\IPD\CLAM-master\generate_attention_maps.py ^
        --slide_id "!SLIDE_ID!" ^
        --coords_dir "D:\IPD\CLAM-master\datasets\coords" ^
        --feats_dir  "D:\IPD\CLAM-master\datasets\features\pt_files" ^
        --wsi_dir    "D:\IPD\CLAM-master\datasets\labelled" ^
        --out_dir    "%CLAM_OUT%"

    REM ============= RUN DTFD =============
    echo Running DTFD heatmap...
    python D:\IPD\DTFD-MIL\IPD_Brain-main\IPD-Brain-main\generate_attention_maps.py ^
        --slide_id "!SLIDE_ID!" ^
        --coords_dir "D:\IPD\CLAM-master\datasets\coords" ^
        --feats_dir  "D:\IPD\CLAM-master\datasets\features\pt_files" ^
        --wsi_dir    "D:\IPD\CLAM-master\datasets\labelled" ^
        --save_dir   "%DTFD_OUT%"

    REM ============= COMBINE HEATMAPS =============
    echo Combining heatmaps...

    set CLAM_IMG=%CLAM_OUT%\!SLIDE_ID!_CLAM_heatmap.png
    set DTFD_IMG=%DTFD_OUT%\!SLIDE_ID!_DTFD_heatmap.png
    set FINAL=%OUT_DIR%\!SLIDE_ID!_combined.png

    python concat.py "!CLAM_IMG!" "!DTFD_IMG!" "!FINAL!"

    echo Saved combined heatmap → !FINAL!
)

echo.
echo =====================================================
echo ALL SLIDES PROCESSED SUCCESSFULLY
echo =====================================================

exit /b 0
