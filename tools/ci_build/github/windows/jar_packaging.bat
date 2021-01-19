@echo on
pushd onnxruntime-java-linux-x64
jar uf  %BUILD_BINARIESDIRECTORY%\java-artifact\onnxruntime-java-win-x64\testing.jar libcustom_op_library.so
del /F /Q libcustom_op_library.so
jar uf  %BUILD_BINARIESDIRECTORY%\java-artifact\onnxruntime-java-win-x64\onnxruntime-%ONNXRUNTIMEVERSION%.jar .
popd
pushd onnxruntime-java-osx-x64
jar uf  %BUILD_BINARIESDIRECTORY%\java-artifact\onnxruntime-java-win-x64\testing.jar libcustom_op_library.dylib
del /F /Q libcustom_op_library.dylib
jar uf  %BUILD_BINARIESDIRECTORY%\java-artifact\onnxruntime-java-win-x64\onnxruntime-%ONNXRUNTIMEVERSION%.jar .
popd