#!/usr/bin/env bash

set -e
set -x

# Parse inputs
TSAN=false
COMPRESS=false

while (( $# )); do
  case "$1" in
    --compress)
      COMPRESS=true
      ;;
    --tsan)
      TSAN=true
      ;;
    *)
      break
      ;;
  esac
  shift
done

HTTP_DEPS="https://dependencies.mapd.com/thirdparty"

SUFFIX=${SUFFIX:=$(date +%Y%m%d)}
PREFIX=/usr/local/mapd-deps

SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $SCRIPTS_DIR/common-functions.sh

# Establish distro
source /etc/os-release
if [ "$ID" == "ubuntu" ] ; then
  PACKAGER="apt -y"
  if [ "$VERSION_ID" != "20.04" ] && [ "$VERSION_ID" != "19.10" ] && [ "$VERSION_ID" != "19.04" ] && [ "$VERSION_ID" != "18.04" ]; then
    echo "Ubuntu 20.04, 19.10, 19.04, and 18.04 are the only debian-based releases supported by this script"
    exit 1
  fi
else
  echo "Only Ubuntu is supported by this script"
  exit 1
fi

sudo mkdir -p $PREFIX
sudo chown -R $(id -u) $PREFIX

DEBIAN_FRONTEND=noninteractive sudo apt update
DEBIAN_FRONTEND=noninteractive sudo apt install -y \
    software-properties-common \
    build-essential \
    ccache \
    git \
    wget \
    curl \
    gcc-8 \
    g++-8 \
    libboost-all-dev \
    libgoogle-glog-dev \
    libssl-dev \
    libevent-dev \
    default-jre \
    default-jre-headless \
    default-jdk \
    default-jdk-headless \
    libncurses5-dev \
    libldap2-dev \
    binutils-dev \
    google-perftools \
    libdouble-conversion-dev \
    libevent-dev \
    libgflags-dev \
    libgoogle-perftools-dev \
    libiberty-dev \
    libjemalloc-dev \
    libglu1-mesa-dev \
    liblz4-dev \
    liblzma-dev \
    libbz2-dev \
    libarchive-dev \
    libcurl4-openssl-dev \
    libedit-dev \
    uuid-dev \
    libsnappy-dev \
    zlib1g-dev \
    autoconf \
    autoconf-archive \
    automake \
    bison \
    flex \
    libpng-dev \
    rsync \
    unzip \
    jq \
    python-dev \
    python-yaml \
    swig \
    pkg-config \
    libxerces-c-dev \
    libxmlsec1-dev

# Set up gcc-8 as default gcc
sudo update-alternatives \
  --install /usr/bin/gcc gcc /usr/bin/gcc-8 800 \
  --slave /usr/bin/g++ g++ /usr/bin/g++-8

# Needed to find sqlite3, xmltooling, and xml_security_c
export PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig:$PREFIX/lib64/pkgconfig:$PKG_CONFIG_PATH
export PATH=$PREFIX/bin:$PATH

install_ninja

install_maven

install_cmake

# llvm
# (see common-functions.sh)
LLVM_BUILD_DYLIB=false
install_llvm

# Geo Support
install_gdal
install_geos

# install AWS core and s3 sdk
install_awscpp -j $(nproc)

VERS=0.13.0
wget --continue http://apache.claz.org/thrift/$VERS/thrift-$VERS.tar.gz
tar xvf thrift-$VERS.tar.gz
pushd thrift-$VERS
CFLAGS="-fPIC" CXXFLAGS="-fPIC" JAVA_PREFIX=$PREFIX/lib ./configure \
    --with-lua=no \
    --with-python=no \
    --with-php=no \
    --with-ruby=no \
    --with-qt4=no \
    --with-qt5=no \
    --with-java=no \
    --prefix=$PREFIX
make -j $(nproc)
make install
popd

VERS=3.52.15
CFLAGS="-fPIC" CXXFLAGS="-fPIC" download_make_install ${HTTP_DEPS}/libiodbc-${VERS}.tar.gz

#c-blosc
VERS=1.14.4
wget --continue https://github.com/Blosc/c-blosc/archive/v$VERS.tar.gz
tar xvf v$VERS.tar.gz
BDIR="c-blosc-$VERS/build"
rm -rf "$BDIR"
mkdir -p "$BDIR"
pushd "$BDIR"
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DBUILD_BENCHMARKS=off \
    -DBUILD_TESTS=off \
    -DPREFER_EXTERNAL_SNAPPY=off \
    -DPREFER_EXTERNAL_ZLIB=off \
    -DPREFER_EXTERNAL_ZSTD=off \
    ..
make -j $(nproc)
make install
popd

install_folly

download_make_install ${HTTP_DEPS}/bisonpp-1.21-45.tar.gz bison++-1.21

# TBB
install_tbb

# Apache Arrow (see common-functions.sh)
ARROW_BOOST_USE_SHARED="ON"
install_arrow

# Go
install_go

VERS=3.1.0
wget --continue https://github.com/cginternals/glbinding/archive/v$VERS.tar.gz
tar xvf v$VERS.tar.gz
mkdir -p glbinding-$VERS/build
pushd glbinding-$VERS/build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DOPTION_BUILD_DOCS=OFF \
    -DOPTION_BUILD_EXAMPLES=OFF \
    -DOPTION_BUILD_GPU_TESTS=OFF \
    -DOPTION_BUILD_TESTS=OFF \
    -DOPTION_BUILD_TOOLS=OFF \
    -DOPTION_BUILD_WITH_BOOST_THREAD=OFF \
    ..
make -j $(nproc)
make install
popd

# librdkafka
install_rdkafka

# libuv
install_libuv

# glslang (with spirv-tools)
VERS=8.13.3743 # stable 4/27/20
rm -rf glslang
mkdir -p glslang
pushd glslang
wget --continue https://github.com/KhronosGroup/glslang/archive/$VERS.tar.gz
tar xvf $VERS.tar.gz
pushd glslang-$VERS
./update_glslang_sources.py
mkdir build
pushd build
cmake \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    ..
make -j $(nproc)
make install
popd # build
popd # glslang-$VERS
popd # glslang

# spirv-cross
VERS=2020-06-29 # latest from 6/29/20
rm -rf spirv-cross
mkdir -p spirv-cross
pushd spirv-cross
wget --continue https://github.com/KhronosGroup/SPIRV-Cross/archive/$VERS.tar.gz
tar xvf $VERS.tar.gz
pushd SPIRV-Cross-$VERS
mkdir build
pushd build
cmake \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_INSTALL_PREFIX=$PREFIX \
    -DCMAKE_POSITION_INDEPENDENT_CODE=on \
    -DSPIRV_CROSS_ENABLE_TESTS=off \
    ..
make -j $(nproc)
make install
popd # build
popd # SPIRV-Cross-$VERS
popd # spirv-cross

# Vulkan
# Custom tarball which excludes the spir-v toolchain
VERS=1.2.162.0 # stable 12/11/20
rm -rf vulkan
mkdir -p vulkan
pushd vulkan
wget --continue ${HTTP_DEPS}/vulkansdk-linux-x86_64-no-spirv-$VERS.tar.gz -O vulkansdk-linux-x86_64-no-spirv-$VERS.tar.gz
tar xvf vulkansdk-linux-x86_64-no-spirv-$VERS.tar.gz
rsync -av $VERS/x86_64/* $PREFIX
popd # vulkan

# OpenSAML
download_make_install ${HTTP_DEPS}/xml-security-c-2.0.2.tar.gz "" "--without-xalan"
download_make_install ${HTTP_DEPS}/xmltooling-3.0.4-nolog4shib.tar.gz
download_make_install ${HTTP_DEPS}/opensaml-3.0.1-nolog4shib.tar.gz

cat > $PREFIX/mapd-deps.sh <<EOF
PREFIX=$PREFIX

LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
LD_LIBRARY_PATH=\$PREFIX/lib:\$LD_LIBRARY_PATH
LD_LIBRARY_PATH=\$PREFIX/lib64:\$LD_LIBRARY_PATH

PATH=/usr/local/cuda/bin:\$PATH
PATH=\$PREFIX/go/bin:\$PATH
PATH=\$PREFIX/maven/bin:\$PATH
PATH=\$PREFIX/bin:\$PATH

VULKAN_SDK=\$PREFIX
VK_LAYER_PATH=\$PREFIX/etc/vulkan/explicit_layer.d

CMAKE_PREFIX_PATH=\$PREFIX:\$CMAKE_PREFIX_PATH

GOROOT=\$PREFIX/go

export LD_LIBRARY_PATH PATH VULKAN_SDK VK_LAYER_PATH CMAKE_PREFIX_PATH GOROOT
EOF

echo
echo "Done. Be sure to source the 'mapd-deps.sh' file to pick up the required environment variables:"
echo "    source $PREFIX/mapd-deps.sh"

if [ "$COMPRESS" = "true" ] ; then
    if [ "$TSAN" = "false" ]; then
      TARBALL_TSAN=""
    elif [ "$TSAN" = "true" ]; then
      TARBALL_TSAN="tsan-"
    fi
    tar acvf mapd-deps-ubuntu-${VERSION_ID}-${TARBALL_TSAN}${SUFFIX}.tar.xz -C ${PREFIX} .
fi
