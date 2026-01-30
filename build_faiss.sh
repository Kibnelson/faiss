#!/bin/bash
cmake -B build \
  -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON -DBUILD_TESTING=ON \
  -DBUILD_SHARED_LIBS=ON -DFAISS_ENABLE_C_API=ON -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_OPT_LEVEL=avx512 \
  -DCMAKE_CXX_FLAGS="-mavx512vpopcntdq" \
  .
  
make -C build -j faiss_avx512
make -C build -j swigfaiss_avx512

(cd build/faiss/python && python setup.py install)
