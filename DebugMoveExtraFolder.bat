@echo off
set sourcePath=.\Extra
set targetPath=.\x64\Debug\Extra

xcopy "%sourcePath%" "%targetPath%" /E /I /Y
