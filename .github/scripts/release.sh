#!/bin/bash
script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
root_dir=$script_dir/../..

# folder where the artifacts where downloaded
release_dir=$1

cd $release_dir
ls -R
# extract servers
mkdir -p servers
for arch in win-x64_avx2 osx-universal;do
    if [ -f "$arch.zip/$arch.zip" ]; then
        unzip -o $arch.zip/$arch.zip -d servers llamalib*server* 2>/dev/null || true
    fi
done
chmod a+x servers/* 2>/dev/null || true

# extract runtimes
for d in *.zip;do
    platform=`echo $d|cut -d'.' -f1|cut -d'_' -f1`
    mkdir -p runtimes/$platform/native
    unzip -o $d/$d -d runtimes/$platform/native -x llamalib*server*
done

rm -r *.zip

# copy includes
mkdir include
cp $root_dir/include/*.h* include/
cp $root_dir/third_party/llama.cpp/vendor/nlohmann/json.hpp include/
cp $root_dir/third_party/llama.cpp/vendor/cpp-httplib/httplib.h include/

# licenses
mkdir -p third_party_licenses
cp $root_dir/third_party/llama.cpp/LICENSE third_party_licenses/llama.cpp.LICENSE.txt

# copy files from repo
cp $root_dir/LICENSE ./
cp $root_dir/VERSION ./
cp $root_dir/cmake/*.cmake ./
