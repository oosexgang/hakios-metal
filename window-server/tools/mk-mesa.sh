#!/bin/bash
# Build Mesa for the iOS userland: the OpenGL ES library behind userland/src/eagl.c (see there and
# userland/src/gl_osmesa.c). Mesa is fetched (a pinned release), patched with tools/mesa/*.patch and
# built with the cross compiler of the kernel (xnu-deps/host/bin/xnu-clang, see xnu-deps/setup.sh)
# against the iOS SDK that userland/build.sh uses; nothing of it is in the repository.
#
#   DRIVERS="softpipe"   the gallium drivers to build (softpipe: a rasterizer in plain C; it needs
#                        no LLVM and nothing of the machine but its processor)
#
# What comes out, in xnu-deps/build/mesa/out:
#   libOSMesa.8.dylib    everything in one library (install name /redwood/lib/libOSMesa.8.dylib): the
#                        OpenGL (ES) entry points gl*, the compiler of the shading language, the driver,
#                        and the off-screen interface that hands out contexts
#   exported-gl.txt      the gl* functions it exports, and
#   gl.xml               their prototypes (the Khronos registry as Mesa has it): for userland/mkeagl.py
# Needs on this machine: meson, ninja, bison, flex, python3 with venv (Mako is installed into a
# private environment), llvm-ar.
set -e
VERSION=25.0.7
SHA256=592272df3cf01e85e7db300c449df5061092574d099da275d19e97ef0510f8a6
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
source "$ROOT/xnu-env.sh"
B=$ROOT/xnu-deps/build/mesa
SDK=$THEOS/sdks/iPhoneOS16.5.sdk
CC=$ROOT/xnu-deps/host/bin/xnu-clang
mkdir -p "$B/out"
cd "$B"

if [ ! -d mesa-$VERSION ]; then
	[ -f mesa-$VERSION.tar.xz ] || curl -sL --fail -o mesa-$VERSION.tar.xz https://archive.mesa3d.org/mesa-$VERSION.tar.xz
	echo "$SHA256  mesa-$VERSION.tar.xz" | sha256sum -c --quiet
	tar xJf mesa-$VERSION.tar.xz
	for p in "$ROOT"/tools/mesa/*.patch; do patch -s -d mesa-$VERSION -p1 < "$p"; done
fi
if [ ! -x venv/bin/python3 ]; then
	python3 -m venv venv
	venv/bin/pip -q install mako pyyaml packaging
fi
export PATH=$B/venv/bin:$PATH

# the compiler is the kernel's wrapper: it names the target (arm64-apple-ios) and the linker (ld64)
cat > cross.ini <<INI
[binaries]
c = ['$CC', '-arch', 'arm64', '-isysroot', '$SDK']
cpp = ['$CC++', '-arch', 'arm64', '-isysroot', '$SDK', '-fuse-cxa-atexit']
objc = ['$CC', '-arch', 'arm64', '-isysroot', '$SDK']
ar = 'llvm-ar'
ranlib = 'llvm-ranlib'
strip = 'llvm-strip'
pkg-config = 'false'

[host_machine]
system = 'darwin'
subsystem = 'ios'
cpu_family = 'aarch64'
cpu = 'arm64'
endian = 'little'

[properties]
needs_exe_wrapper = true
INI
[ -f build/build.ninja ] || meson setup build mesa-$VERSION --cross-file cross.ini --prefix=/redwood --libdir=lib \
	--buildtype=release -Db_ndebug=true \
	-Dplatforms= -Dgallium-drivers="${DRIVERS:-softpipe}" -Dvulkan-drivers= -Dglx=disabled -Degl=disabled -Dgbm=disabled \
	-Dosmesa=true -Dllvm=disabled -Dshared-glapi=enabled -Dgles1=disabled -Dgles2=enabled -Dopengl=true \
	-Dzstd=disabled -Dxmlconfig=disabled -Dexpat=disabled -Dshader-cache=disabled -Dvalgrind=disabled \
	-Dlibunwind=disabled -Dglvnd=disabled -Dvideo-codecs= -Dtools= -Dgallium-vdpau=disabled -Dgallium-va=disabled \
	-Dgallium-xa=disabled -Dlmsensors=disabled -Dbuild-tests=false > meson-setup.log
ninja -C build > ninja.log

llvm-strip -x -o out/libOSMesa.8.dylib build/src/gallium/targets/osmesa/libOSMesa.8.dylib
install_name_tool -id /redwood/lib/libOSMesa.8.dylib out/libOSMesa.8.dylib
nm -gU out/libOSMesa.8.dylib | awk '$3 ~ /^_gl/ { print substr($3, 2) }' > out/exported-gl.txt
cp mesa-$VERSION/src/mapi/glapi/registry/gl.xml out/gl.xml
echo "mk-mesa.sh: Mesa $VERSION (${DRIVERS:-softpipe}): $B/out/libOSMesa.8.dylib, $(stat -c %s out/libOSMesa.8.dylib) bytes," \
	"$(wc -l < out/exported-gl.txt) gl* functions"
