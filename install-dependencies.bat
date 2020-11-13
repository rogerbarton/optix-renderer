call .\ext\vcpkg\bootstrap-vcpkg.bat

:: call .\ext\vcpkg\vcpkg install zlib:x64-windows
call .\ext\vcpkg\vcpkg install blosc:x64-windows
:: call .\ext\vcpkg\vcpkg install openexr:x64-windows
:: call .\ext\vcpkg\vcpkg install tbb:x64-windows
call .\ext\vcpkg\vcpkg install boost-iostreams:x64-windows
call .\ext\vcpkg\vcpkg install boost-system:x64-windows
call .\ext\vcpkg\vcpkg install boost-any:x64-windows
call .\ext\vcpkg\vcpkg install boost-algorithm:x64-windows
call .\ext\vcpkg\vcpkg install boost-uuid:x64-windows
call .\ext\vcpkg\vcpkg install boost-interprocess:x64-windows