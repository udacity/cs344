SET OPENCV_DIR=I:\Projekte\OpenCV\opencv_bin\bin\Release
SET OPENCV_DEBUG_DIR=I:\Projekte\OpenCV\opencv_bin\bin\Debug
SET HW_DIR=I:\Projekte\CUDA\Udacity GIT\cs344_bin\

PATH=%OPENCV_DIR%;%OPENCV_DEBUG_DIR%;%PATH%

call "C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\vcvarsall.bat x64"

cd /D %HW_DIR%
call cmake .
call cs344.sln
cmd
