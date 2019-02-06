debug=-DUNSAMPLED -DNUMA_MAP -O3 -lmemkind -lnuma -DSCATTER -DPRINT_ESTIMATION #-I/usr/src/kernels/3.10.0-327.36.1.el7.x86_64/include/linux/ #-DBUFFER_SIZE=32 #-DMCDRAM
macro=-DPREFETCHING #-DTIMELOG #-qno-opt-prefetch #-DTIMELOG_DETAIL
include=#-I/home/s/shuhao-z/install/include
lib=#-L/home/s/shuhao-z/install/lib

write: write.cpp
	icpc $(debug) write.cpp -lpthread -std=c++0x -lrt -o write $(macro) $(include) $(lib)


hj_hma: hj_hma.cpp
	icpc -m64 -xmic-avx512 hj_hma.cpp hash_table.cpp chained_hash_table.cpp -lpthread -std=c++0x -lrt -o hj_hma $(debug) $(macro) $(include) $(lib)

