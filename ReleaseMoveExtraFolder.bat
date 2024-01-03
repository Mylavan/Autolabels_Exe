@echo off
set sourcePath=.\Extra
set targetPath=.\x64\Release\Extra

xcopy "%sourcePath%" "%targetPath%" /E /I /Y
