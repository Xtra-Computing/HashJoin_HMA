#ifndef HEADER__
#define HEADER__


#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <immintrin.h>
#include <assert.h>
#include <vector>
#include <algorithm>    

#define BUCKET_SIZE 1024
#define HW_NUM_NUMA_NODES 6  
#define SW_NUM_NUMA_NODES 4
//float HBM_SPEEDUP;

typedef uint32_t KEY_TYPE;
typedef uint32_t PAYLOAD_TYPE;
typedef struct 
{
	KEY_TYPE* keys;
	PAYLOAD_TYPE* vals;
	long size;
} RELATION_TYPE;

typedef struct BUCKET
{
	KEY_TYPE key;
	PAYLOAD_TYPE val;
	BUCKET *next;
} BUCKET;

typedef struct 
{
	BUCKET *next;
} BUCKET_HEAD;

typedef struct bucket_rec
{
	uint32_t id;
	uint32_t workload;
}bucket_rec;

typedef struct thread_NUMA_info
{
	uint32_t id;
	float freq_percentage[4];
	size_t rank[4];
}thread_NUMA_info;

void *align(const void *p);
__m512i simd_hash(__m512i k, __m512i Nbins);

class Hash_Table 
{
protected:
	RELATION_TYPE relation;
	size_t buckets;
	size_t sampling_buckets;
	double ratio;
	int empty;
	uint32_t factor;
	uint32_t factor2;
	//volatile bool *placement_bitmap;
	volatile uint64_t *NUMA_tables[HW_NUM_NUMA_NODES];
	volatile uint32_t *data_placement;
	volatile thread_NUMA_info *threads_info;
public:
	void Sample_outer_relation (RELATION_TYPE *outer_rel);
	void histogram(const uint32_t *keys, size_t size, uint32_t *counts,uint32_t factor, size_t partitions);
	//size_t interleave(uint32_t **counts, uint32_t *offsets, uint32_t *aggr_counts,size_t partitions, size_t thread, size_t threads);
	void subset_sum_solver (uint32_t *inner_counts, uint32_t *outer_counts, size_t sampling_n, size_t inner_N, size_t outer_N, double HBM_SPEEDUP);
	void subset_select (uint32_t *inner_counts, uint32_t *outer_counts, 
		size_t sampling_n, size_t inner_N, size_t outer_N, double HBM_SPEEDUP);	
	void printSubsetsRec(std::vector<bucket_rec> arr, int i, int sum, std::vector<bucket_rec>& p);
	Hash_Table () = default;
	virtual void Build () = 0;
	virtual size_t Probe (const uint32_t *keys, const uint32_t *vals, size_t size, uint32_t *keys_out, uint32_t *vals_out, uint32_t *tabs_out,size_t block_size, size_t block_limit, volatile size_t *counter) = 0;
	void NUMA_mapping_prepare(size_t thread, uint32_t *local_counts);
	void NUMA_mapping_solver(size_t *remapping, size_t threads, size_t NUMA_nodes);
};

class Linear_Hash_Table: public Hash_Table {
public:
	Linear_Hash_Table (RELATION_TYPE inner_relation, size_t input_buckets, int input_empty, uint32_t input_factor, uint32_t input_factor2, double input_ratio, 
					   volatile uint64_t *NUMA_tables_input[HW_NUM_NUMA_NODES], volatile uint32_t * input_data_placement, size_t sampling_buckets_input, volatile thread_NUMA_info *input_threads_info)
	{
		relation=inner_relation;
		buckets=input_buckets;
		empty=input_empty;
		ratio=input_ratio;
		factor=input_factor;
		factor2=input_factor2;
		//placement_bitmap=input_placement_bitmap;
		data_placement=input_data_placement;
		for (int i=0;i<HW_NUM_NUMA_NODES;++i)
			NUMA_tables[i]=NUMA_tables_input[i];
		sampling_buckets=sampling_buckets_input;
		threads_info=input_threads_info;
	}
	void Build();
	size_t Probe(const uint32_t *keys, const uint32_t *vals, size_t size, uint32_t *keys_out, uint32_t *vals_out, uint32_t *tabs_out,size_t block_size, size_t block_limit, volatile size_t *counter);
	void info()
	{
		printf("size %d %d %d\n", relation.size, buckets, factor);
	}
}; 

class Chained_Hash_Table: public Hash_Table{
protected: 
	BUCKET_HEAD *chained_hash_table;
public:
	Chained_Hash_Table (RELATION_TYPE inner_relation, size_t input_buckets, int input_empty, uint32_t input_factor, double input_ratio, volatile uint64_t *hash_table_ddr)
	{
		relation=inner_relation;
		buckets=input_buckets;
		empty=input_empty;
		ratio=input_ratio;
		factor=input_factor;
		//table=hash_table_ddr;
	}
	void Insert(KEY_TYPE key, PAYLOAD_TYPE val);
	void Build () {};
	void Build(uint32_t inner_beg, uint32_t inner_end);
	size_t Probe(const uint32_t *keys, const uint32_t *vals, size_t size, uint32_t *keys_out, uint32_t *vals_out, uint32_t *tabs_out,size_t block_size, size_t block_limit, volatile size_t *counter);
	void info ()
	{
		printf("size %d %d %d\n", relation.size, buckets, factor);
	}
};

#endif
