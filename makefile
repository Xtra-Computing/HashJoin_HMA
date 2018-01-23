debug=-DUNSAMPLED -DNUMA_MAP -O3 -lmemkind -lnuma -DSCATTER -DPRINT_ESTIMATION #-I/usr/src/kernels/3.10.0-327.36.1.el7.x86_64/include/linux/ #-DBUFFER_SIZE=32 #-DMCDRAM
macro=-DPREFETCHING #-DTIMELOG #-qno-opt-prefetch #-DTIMELOG_DETAIL
include=-I/home/s/shuhao-z/install/include
lib=-L/home/s/shuhao-z/install/lib

all: cpra phj npj tpj

cpra: cpra2.cpp
	icpc -m64 cpra2.cpp -lpthread -std=c++0x -lrt -o cpra $(debug) $(lib) $(include) $(macro)

phj: phj.cpp
	icpc $(debug) phj.cpp -lpthread -std=c++0x -lrt -o phj $(macro) $(include) $(lib)

write: write.cpp
	icpc $(debug) write.cpp -lpthread -std=c++0x -lrt -o write $(macro) $(include) $(lib)
	
phj_block: phj_block.cpp
	icpc $(debug) phj_block.cpp -lpthread -std=c++0x -lrt -o phj_block $(macro) $(include) $(lib)

phj_mem: phj_mem.cpp
	icpc $(debug) phj_mem.cpp -lpthread -std=c++0x -lrt -o phj_mem $(macro) $(include) $(lib)

npj: npj.cpp
	icpc -m64 -xmic-avx512 npj.cpp -lpthread -std=c++0x -lrt -o npj $(debug) $(macro) $(include) $(lib)

npj_new: npj_new.cpp
	icpc -m64 -xmic-avx512 npj_new.cpp hash_table.cpp chained_hash_table.cpp -lpthread -std=c++0x -lrt -o numa_npj $(debug) $(macro) $(include) $(lib)

tpj: tpj.cpp partition_ddr.cpp p_m.cpp tpj.h
	icpc tpj.cpp partition_ddr.cpp p_m.cpp -lpthread -std=c++0x -lrt -o tpj $(debug) $(include) $(lib) $(macro)

clean:
	rm cpra phj npj tpj
