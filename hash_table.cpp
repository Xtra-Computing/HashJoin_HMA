#include "hash_table.h"
int precision = 100;


void *align(const void *p)
{
	size_t i = 63 & (size_t) p;
	return (void*) (i ? p + 64 - i : p);
}
__m512i _mm512_fmadd_epi32(__m512i a, __m512i b, __m512i c)
{
	__m512i temp=_mm512_mullo_epi32(a,b);
	temp=_mm512_add_epi32 (temp,c);
	return temp;
}
__m512i simd_hash(__m512i k, __m512i Nbins)
{
	__m512i permute_2 = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
	__m512i blend_0 = _mm512_set1_epi32(0);
	__mmask16 blend_interleave = _mm512_int2mask(21845);
	__m512i Nbins2 = _mm512_permutevar_epi32 (permute_2,Nbins);
	Nbins=_mm512_mask_blend_epi32(blend_interleave,blend_0,Nbins);
	Nbins2=_mm512_mask_blend_epi32(blend_interleave,blend_0,Nbins2);
	__m512i k2=_mm512_permutevar_epi32 (permute_2,k);
	k=_mm512_mask_blend_epi32(blend_interleave,blend_0,k);
	k2=_mm512_mask_blend_epi32(blend_interleave,blend_0,k2);
	k=_mm512_mul_epu32 (k,Nbins);
	k2=_mm512_mul_epu32 (k2,Nbins2);
	k=_mm512_permutevar_epi32 (permute_2,k);
	k=_mm512_mask_blend_epi32(blend_interleave,k2,k);
	return k;
}
__m512i simd_hash_new(__m512i k, __m512i factors, __m512i Nbins)
{
	k = _mm512_mullo_epi32(k, factors);
	__m512i permute_2 = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
	__m512i blend_0 = _mm512_set1_epi32(0);
	__mmask16 blend_interleave = _mm512_int2mask(21845);
	__m512i Nbins2 = _mm512_permutevar_epi32 (permute_2,Nbins);
	Nbins=_mm512_mask_blend_epi32(blend_interleave,blend_0,Nbins);
	Nbins2=_mm512_mask_blend_epi32(blend_interleave,blend_0,Nbins2);
	__m512i k2=_mm512_permutevar_epi32 (permute_2,k);
	k=_mm512_mask_blend_epi32(blend_interleave,blend_0,k);
	k2=_mm512_mask_blend_epi32(blend_interleave,blend_0,k2);
	k=_mm512_mul_epu32 (k,Nbins);
	k2=_mm512_mul_epu32 (k2,Nbins2);
	k=_mm512_permutevar_epi32 (permute_2,k);
	k=_mm512_mask_blend_epi32(blend_interleave,k2,k);
	return k;
}
__m512i simd_hash_ratio(__m512i k, __m512i factors_1, __m512i factors_2, size_t partitions, double ratio)
{
	//threads are equally divided to the DDR and the HBM
	//ratio is 0, 0.1, 0.2, ..., 0.9, 1
	int cut = (int)((1-ratio)*precision);
	//first level hasing
	__m512i mask_ratio_par = _mm512_set1_epi32(precision);
	__m512i hash1 = simd_hash_new (k, factors_1, mask_ratio_par);
	//second level hashing
	__m512i mask_cut = _mm512_set1_epi32(cut);
	__m512i mask_partitions = _mm512_set1_epi32(partitions/2);
	__mmask16 is_hbm = ~_mm512_cmp_epi32_mask(hash1, mask_cut, _MM_CMPINT_LT);
	__m512i hash2 = simd_hash_new (k, factors_2, mask_partitions);
	hash2 = _mm512_mask_add_epi32 (hash2, is_hbm, hash2, mask_partitions);
	return hash2;
}
void Linear_Hash_Table:: Build ()
{
	//volatile uint64_t *NUMA_tables[NUM_NUMA_NODES];
	//volatile uint32_t *data_placement;
	//FILE *fl;
	//fl=fopen("bucket_build_log.txt", "wr");
	uint32_t i;
	for (i = 0 ; i != relation.size ; ++i) {
		uint32_t key = relation.keys[i];
		uint64_t pair = relation.vals[i];
		pair = (pair << 32) | key;
		//caculate the big hash value, and find out the node
		uint64_t big_hash = (uint32_t) (key * factor2);
		big_hash = (big_hash * sampling_buckets) >> 32;
		uint32_t to_node = data_placement[big_hash];
		//fprintf (fl, "%d\n", to_node);
		volatile uint64_t *hash_table = NUMA_tables[to_node];
		//caculate the small hash value, and find out the bucket
		uint64_t h = (uint32_t) (key * factor);
		h = (h * buckets) >> 32;
		uint64_t tab = hash_table[h];
		while (empty != (uint32_t) tab ||
		       !__sync_bool_compare_and_swap(&hash_table[h], tab, pair)) {
			if (++h == buckets) h = 0;
			tab = hash_table[h];
		}
	}
	//fclose(fl);
}	

size_t Linear_Hash_Table:: Probe(const uint32_t *keys, const uint32_t *vals, size_t size, uint32_t *keys_out, uint32_t *vals_out, uint32_t *tabs_out, 
	size_t block_size, size_t block_limit, volatile size_t *counter)
{
	assert(keys_out == align(keys_out));
	assert(vals_out == align(vals_out));
	// generate masks
	__m512i mask_1 = _mm512_set1_epi32(1);
	__m512i mask_empty = _mm512_set1_epi32(empty);
	__m512i mask_factor = _mm512_set1_epi32(factor);
	__m512i mask_factor2 = _mm512_set1_epi32(factor2);
	__m512i mask_buckets = _mm512_set1_epi32(buckets);
	__m512i mask_buckets2 = _mm512_set1_epi32(sampling_buckets);
	__m512i mask_unpack = _mm512_set_epi32(15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0);
	//__m512i mask_ddr = _mm512_set1_epi32(size_ddr);
	__mmask16 blend_0000 = _mm512_int2mask(0x0000);
	__mmask16 blend_AAAA = _mm512_int2mask(0xAAAA);
	__mmask16 blend_5555 = _mm512_int2mask(0x5555);
	__m512i MASK_0 = _mm512_set1_epi32(0);
	__m512i MASK_1 = _mm512_set1_epi32(1);
	__m512i MASK_2 = _mm512_set1_epi32(2);
	__m512i MASK_3 = _mm512_set1_epi32(3);
	__m512i MASK_4 = _mm512_set1_epi32(4);
	__m512i MASK_5 = _mm512_set1_epi32(5);
	// space for buffers
	const size_t buffer_size = 256;
	uint32_t buffer_space[(buffer_size + 16) * 3 + 15];
	uint32_t *keys_buf = (uint32_t *)align(buffer_space);
	uint32_t *vals_buf = &keys_buf[buffer_size + 16];
	uint32_t *tabs_buf = &vals_buf[buffer_size + 16];
	// main loop
	const size_t size_vec = size - 16;
	size_t b, i = 0, j = 0;
	size_t o = __sync_fetch_and_add(counter, 1);
	assert(o <= block_limit);
	o *= block_size;
	__mmask16 k = _mm512_kxnor(k, k);
	__m512i key, val, off;
	if (size >= 16) do {
		// replace invalid keys & payloads
		key = _mm512_mask_expandloadu_epi32 (key, k, &keys[i]);
		val = _mm512_mask_expandloadu_epi32 (val, k, &vals[i]);
		off = _mm512_mask_xor_epi32(off, k, off, off);
		i += _mm_countbits_64(_mm512_kconcatlo_64(blend_0000, k));
		// hash keys to get the big hash value
		__m512i big_hash = _mm512_mullo_epi32(key, mask_factor2);
		big_hash = simd_hash(big_hash, mask_buckets2);
		__m512i big_hash_NUMA_node = _mm512_i32gather_epi32(big_hash, (const void *)data_placement, 1);
		__mmask16 node_0_mask=_mm512_cmpeq_epi32_mask(big_hash_NUMA_node, MASK_0);
		__mmask16 node_1_mask=_mm512_cmpeq_epi32_mask(big_hash_NUMA_node, MASK_1);
		__mmask16 node_2_mask=_mm512_cmpeq_epi32_mask(big_hash_NUMA_node, MASK_2);
		__mmask16 node_3_mask=_mm512_cmpeq_epi32_mask(big_hash_NUMA_node, MASK_3);
		__mmask16 node_4_mask=_mm512_cmpeq_epi32_mask(big_hash_NUMA_node, MASK_4);
		__mmask16 node_5_mask=_mm512_cmpeq_epi32_mask(big_hash_NUMA_node, MASK_5);
		// hash keys to get the small hash value and add offsets
		__m512i hash = _mm512_mullo_epi32(key, mask_factor);
		hash = simd_hash(hash, mask_buckets);
		hash = _mm512_add_epi32(hash, off);
		k = _mm512_cmpge_epu32_mask(hash, mask_buckets);
		hash = _mm512_mask_sub_epi32(hash, k, hash, mask_buckets);

		//__mmask16 in_ddr = _mm512_cmp_epi32_mask(hash, mask_ddr, _MM_CMPINT_LT);
		//__mmask16 in_hbm = ~ in_ddr;
		
		//lo
		__m512i lo = _mm512_mask_i32logather_epi64(lo, node_0_mask&255, hash, (void const *)(NUMA_tables[0]), 8);
		lo = _mm512_mask_i32logather_epi64(lo, node_1_mask&255, hash, (void const *)(NUMA_tables[1]), 8);
		lo = _mm512_mask_i32logather_epi64(lo, node_2_mask&255, hash, (void const *)(NUMA_tables[2]), 8);
		lo = _mm512_mask_i32logather_epi64(lo, node_3_mask&255, hash, (void const *)(NUMA_tables[3]), 8);
		lo = _mm512_mask_i32logather_epi64(lo, node_4_mask&255, hash, (void const *)(NUMA_tables[4]), 8);
		lo = _mm512_mask_i32logather_epi64(lo, node_5_mask&255, hash, (void const *)(NUMA_tables[5]), 8);
		//hi
		hash = _mm512_permute4f128_epi32(hash, _MM_PERM_BADC);
		__m512i hi = _mm512_mask_i32logather_epi64(hi, (node_0_mask>>8)&255, hash, (void const *)(NUMA_tables[0]), 8);
		hi = _mm512_mask_i32logather_epi64(hi, (node_1_mask>>8)&255, hash, (void const *)(NUMA_tables[1]), 8);
		hi = _mm512_mask_i32logather_epi64(hi, (node_2_mask>>8)&255, hash, (void const *)(NUMA_tables[2]), 8);
		hi = _mm512_mask_i32logather_epi64(hi, (node_3_mask>>8)&255, hash, (void const *)(NUMA_tables[3]), 8);
		hi = _mm512_mask_i32logather_epi64(hi, (node_4_mask>>8)&255, hash, (void const *)(NUMA_tables[4]), 8);
		hi = _mm512_mask_i32logather_epi64(hi, (node_5_mask>>8)&255, hash, (void const *)(NUMA_tables[5]), 8);

		// load keys from table and update offsets
		//__mmask8 temp_in_ddr = in_ddr & 255;
		//__m512i lo = _mm512_mask_i32logather_epi64(lo, temp_in_ddr, hash, (void const *)table, 8);
		//__mmask8 temp_in_hbm = in_hbm & 255;
		//lo = _mm512_mask_i32logather_epi64(lo, temp_in_hbm, hash, (void const *)table_hbm, 8);
		//__m512i lo = _mm512_i32logather_epi64(hash, table, 8);
		//hash = _mm512_permute4f128_epi32(hash, _MM_PERM_BADC);
		//temp_in_ddr = (in_ddr >> 8) & 255;
		//__m512i hi = _mm512_mask_i32logather_epi64(hi, temp_in_ddr, hash, (void const *)table, 8);
		//temp_in_hbm = (in_hbm >> 8) & 255;
		//hi = _mm512_mask_i32logather_epi64(hi, temp_in_hbm, hash, (void const *)table_hbm, 8);
		//__m512i hi = _mm512_i32logather_epi64(hash, table, 8);

		off = _mm512_add_epi32(off, mask_1);
		// split keys and payloads
		__m512i tab_key = _mm512_mask_blend_epi32(blend_AAAA, lo, _mm512_swizzle_epi32(hi, _MM_SWIZ_REG_CDAB));
		__m512i tab_val = _mm512_mask_blend_epi32(blend_5555, hi, _mm512_swizzle_epi32(lo, _MM_SWIZ_REG_CDAB));
		tab_key = _mm512_permutevar_epi32(mask_unpack, tab_key);
		tab_val = _mm512_permutevar_epi32(mask_unpack, tab_val);
		// compare
		__mmask16 m = _mm512_cmpeq_epi32_mask(tab_key, key);
		k = _mm512_cmpeq_epi32_mask(tab_key, mask_empty);
#ifdef _UNIQUE
		k = _mm512_kor(k, m);
#endif
		// pack store matches
		_mm512_mask_compressstoreu_epi32(&keys_buf[j +  0], m, key);
		_mm512_mask_compressstoreu_epi32(&vals_buf[j +  0], m, val);
		_mm512_mask_compressstoreu_epi32(&tabs_buf[j +  0], m, tab_val);
		j += _mm_countbits_64(_mm512_kconcatlo_64(blend_0000, m));
		if (j >= buffer_size) {
			j -= buffer_size;
			for (b = 0 ; b != buffer_size ; b += 16, o += 16) {
				__m512 x = _mm512_load_ps(&keys_buf[b]);
				__m512 y = _mm512_load_ps(&vals_buf[b]);
				__m512 z = _mm512_load_ps(&tabs_buf[b]);
				_mm512_stream_ps (&keys_out[o], x);
				_mm512_stream_ps (&vals_out[o], y);
				_mm512_stream_ps (&tabs_out[o], z);
			}
			__m512 x = _mm512_load_ps(&keys_buf[b]);
			__m512 y = _mm512_load_ps(&vals_buf[b]);
			__m512 z = _mm512_load_ps(&tabs_buf[b]);
			_mm512_store_ps(keys_buf, x);
			_mm512_store_ps(vals_buf, y);
			_mm512_store_ps(tabs_buf, z);
			if ((o & (block_size - 1)) == 0) {
				o = __sync_fetch_and_add(counter, 1);
				assert(o <= block_limit);
				o *= block_size;
			}
		}
	} while (i <= size_vec);
	// flush last items
	for (b = 0 ; b != j ; ++b, ++o) {
		keys_out[o] = keys_buf[b];
		vals_out[o] = vals_buf[b];
		tabs_out[o] = tabs_buf[b];
	}
	// save last items
	uint32_t keys_last[32];
	uint32_t vals_last[32];
	uint32_t offs_last[32];
	k = _mm512_knot(k);
	_mm512_mask_compressstoreu_epi32 (&keys_last[0],  k, key);
	_mm512_mask_compressstoreu_epi32 (&vals_last[0],  k, val);
	_mm512_mask_compressstoreu_epi32 (&offs_last[0],  k, off);
	j = _mm_countbits_64(_mm512_kconcatlo_64(blend_0000, k));
	for (; i != size ; ++i, ++j) {
		keys_last[j] = keys[i];
		vals_last[j] = vals[i];
		offs_last[j] = 0;
	}
	// process last items in scalar code
	for (i = 0 ; i != j ; ++i) {
		uint32_t k = keys_last[i];
		uint32_t r = vals_last[i];
		
		//caculate the big hash value, and find out the node
		uint64_t big_hash = (uint32_t) (k * factor2);
		big_hash = (big_hash * sampling_buckets) >> 32;
		uint32_t to_node = data_placement[big_hash];
		volatile uint64_t *hash_table = NUMA_tables[to_node];
		//caculate the small hash value, and find out the bucket
		uint64_t h = (uint32_t) (k * factor);
		h = (h * buckets) >> 32;
		h += (uint32_t) offs_last[i];

		uint64_t t = hash_table[h];
		while (empty != (uint32_t) t) {
			if (k == (uint32_t) t) {
				tabs_out[o] = t >> 32;
				vals_out[o] = r;
				keys_out[o++] = k;
				if ((o & (block_size - 1)) == 0) {
					o = __sync_fetch_and_add(counter, 1);
					assert(o <= block_limit);
					o *= block_size;
				}
			}
			if (++h == buckets) h = 0;
			t = hash_table[h];
		}
	}
	return o;	
}
void Hash_Table::histogram(const uint32_t *keys, size_t size, uint32_t *counts,
		uint32_t factor, size_t partitions)
{
	// partition vector space
	uint32_t parts_space[31];
	uint32_t *parts = (uint32_t *)align(parts_space);
	// create masks
	__mmask16 blend_0 = _mm512_int2mask(0);
	__m512i mask_0 = _mm512_set1_epi32(0);
	__m512i mask_1 = _mm512_set1_epi32(1);
	__m512i mask_16 = _mm512_set1_epi32(16);
	__m512i mask_255 = _mm512_set1_epi32(255);
	__m512i mask_factor = _mm512_set1_epi32(factor);
	__m512i mask_partitions = _mm512_set1_epi32(partitions);
	__m512i mask_lanes = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
	// reset counts
	size_t p, partitions_x16 = partitions << 4;
	uint32_t all_counts_space[partitions_x16 + 127];
	uint32_t *all_counts = (uint32_t *)align(all_counts_space);
	for (p = 0 ; p < partitions_x16 ; p += 16)
		_mm512_store_epi32(&all_counts[p], mask_0);
	for (p = 0 ; p != partitions ; ++p)
		counts[p] = 0;
	// before alignment
	const uint32_t *keys_end = &keys[size];
	const uint32_t *keys_aligned = (uint32_t *)align(keys);
	while (keys != keys_end && keys != keys_aligned) {
		uint32_t key = *keys++;
		p = (uint32_t) (key * factor);
		p = (p * partitions) >> 32;
		counts[p]++;
	}

	// aligned
	keys_aligned = &keys[(keys_end - keys) & -16];
	while (keys != keys_aligned) {
		//printf ("align\n");
		__m512i key = _mm512_load_epi32(keys);
		keys += 16;
		//__m512i part = _mm512_mullo_epi32(key, mask_factor);
		//part = _mm512_mulhi_epi32(part, mask_partitions);
		__m512i part = simd_hash_new(key, mask_factor, mask_partitions);
		__m512i part_lanes = _mm512_fmadd_epi32(part, mask_16, mask_lanes);
		__m512i count = _mm512_i32gather_epi32(part_lanes, all_counts, 4);
		__mmask16 k = _mm512_cmpeq_epi32_mask(count, mask_255);
		count = _mm512_add_epi32(count, mask_1);
		count = _mm512_and_epi32(count, mask_255);
		_mm512_i32scatter_epi32(all_counts, part_lanes, count, 4);
		if (!_mm512_kortestz(k, k)) {
			_mm512_store_epi32(parts, part);
			size_t mask = _mm512_kconcatlo_64(blend_0, k);
			size_t b = _mm_tzcnt_64(mask);
			do {
				p = parts[b];
				counts[p] += 256;
				//b = _mm_tzcnti_64(b, mask);
				mask=mask&(~(1<<b));
				b = _mm_tzcnt_64(mask);
			} while (b != 64);
		}
	}
	// after alignment
	while (keys != keys_end) {
		uint32_t key = *keys++;
		p = (uint32_t) (key * factor);
		p = (p * partitions) >> 32;
		counts[p]++;
	}
	// merge counts
	for (p = 0 ; p != partitions ; ++p) {
		__m512i sum = _mm512_load_epi32(&all_counts[p << 4]);
		counts[p] += _mm512_reduce_add_epi32(sum);
	}
#ifdef BG
	size_t i;
	for (i = p = 0 ; p != partitions ; ++p)
		i += counts[p];
	//assert(i == size);
#endif
}


bool myfunction (bucket_rec i,bucket_rec j) { return (i.workload<j.workload); }

void Hash_Table::subset_select(uint32_t *inner_counts, uint32_t *outer_counts, 
		size_t sampling_n, size_t inner_N, size_t outer_N, double HBM_SPEEDUP)
{
	
}
void Hash_Table::subset_sum_solver (uint32_t *inner_counts, uint32_t *outer_counts, 
										size_t sampling_n, size_t inner_N, size_t outer_N, double HBM_SPEEDUP)
{
	//n = sampling_buckets
	std::vector<bucket_rec> list;
	std::vector<float> HMA_arc (HW_NUM_NUMA_NODES,0.0);
	size_t sum;
	double c=0.1;
	int i,j,k;
	uint64_t total_sum=0;
	sum=0.6*outer_N;
	for (i=0;i<sampling_n;++i)
	{
		bucket_rec temp;
		temp.id=i;
		temp.workload=inner_counts[i]*outer_counts[i]*0.5;
		list.push_back(temp);
		total_sum+=temp.workload;
	}
 	std::sort(list.begin(), list.end(), myfunction);
	double min=~0;
	int min_id;
	for (i=0;i<sampling_n;++i)
		
	{
		double t[HW_NUM_NUMA_NODES];
		for (j=0;j<HW_NUM_NUMA_NODES;++j)
		//for (j=HW_NUM_NUMA_NODES-1;j>=0;--j)
		{
			t[j]=1.0*(HMA_arc[j]*total_sum+list[i].workload)/total_sum;
			if (1<j)
			{
				t[j]/=HBM_SPEEDUP;
			}
		}
		min=t[0];
		min_id=0;
		for (j=1;j<HW_NUM_NUMA_NODES;++j)
		//for (j=HW_NUM_NUMA_NODES-1;j>=0;--j)
		{
			if (t[j]<min)
			{
				min=t[j];
				min_id=j;
				assert(min_id<HW_NUM_NUMA_NODES);
			}
		}
		HMA_arc[min_id]=min_id>1?t[min_id]*HBM_SPEEDUP:t[min_id];
		data_placement[i]=min_id;
#ifdef PRINT_ESTIMATION
		//printf ("%d\t%d\t%d\t%d\n", i, inner_counts[i], outer_counts[i], min_id);
		//printf ("%d\t%d\n", i, min_id);
#endif 
	}
	//for (j=0;j<HW_NUM_NUMA_NODES;j++)
	//{
	//	printf ("%lf\n", HMA_arc[j]);
	//}
}
void Hash_Table::NUMA_mapping_prepare(size_t thread, uint32_t *local_counts)
{
	thread_NUMA_info *local_info = (thread_NUMA_info *)&threads_info[thread];
	size_t NUMA_nodes=SW_NUM_NUMA_NODES;
	int i,ptr_node;
	uint64_t all_count[5]={0,0,0,0,0};
	local_info->id=thread;
	local_info->freq_percentage[0]=0.0;
	local_info->freq_percentage[1]=0.0;
	local_info->freq_percentage[2]=0.0;
	local_info->freq_percentage[3]=0.0;
	for (i=0;i<sampling_buckets;++i)
	{
		all_count[4]+=local_counts[i];
		if (data_placement[i]>1)
			ptr_node=data_placement[i]-2;
		else
			ptr_node=data_placement[i];
		//printf ("%d\n", ptr_node);
		all_count[ptr_node]+=local_counts[i];
	}
	local_info->freq_percentage[0]=(float)all_count[0]/all_count[4];
	local_info->freq_percentage[1]=(float)all_count[1]/all_count[4];
	local_info->freq_percentage[2]=(float)all_count[2]/all_count[4];
	local_info->freq_percentage[3]=(float)all_count[3]/all_count[4];	
	//printf ("%d\t%.2lf\t%.2lf\t%.2lf\t%.2lf\n", thread, local_info->freq_percentage[0], local_info->freq_percentage[1], local_info->freq_percentage[2], local_info->freq_percentage[3]);
	local_info->rank[0]=0;
	local_info->rank[1]=1;
	local_info->rank[2]=2;
	local_info->rank[3]=3;
	for (int i=0;i<4;++i)
	{
		for (int j=i;j<4;++j)
		{
			if (local_info->freq_percentage[i]<local_info->freq_percentage[j])
			{
				local_info->rank[i]^=local_info->rank[j];
				local_info->rank[j]^=local_info->rank[i];
				local_info->rank[i]^=local_info->rank[j];
			}
		}
	}
	//printf ("%d\t%d\t%d\t%d\t%d\n", thread, local_info->rank[0], local_info->rank[1], local_info->rank[2], local_info->rank[3]);//local_info->freq_percentage[1], local_info->freq_percentage[2], local_info->freq_percentage[3]);
}
typedef struct NUMA_node_list_cell
{
	size_t id;
	float val;
}NUMA_node_list_cell;
bool NUMA_compare(NUMA_node_list_cell i, NUMA_node_list_cell j)
{
	return (i.val>j.val);
}
void Hash_Table::NUMA_mapping_solver(size_t *remapping, size_t threads, size_t NUMA_nodes)
{
	int i,j,q,k;
	NUMA_node_list_cell NUMA_list[4][256];
	size_t candidate_counts[4]={0,0,0,0};
	volatile thread_NUMA_info *local_info;
	bool thread_mark[256];
	for (j=0;j<4;++j)
	{
		//#NUMA_nodes rounds of GS algorithms
		//Within each round, each thread first proposes to the jth NUMA node
		for (i=0;i<threads;++i)
		{
			if (j==0)
				thread_mark[i]=false;
			if (!thread_mark[i])
			{
				local_info = &threads_info[i];
				size_t to_node=local_info->rank[j];
				NUMA_list[to_node][candidate_counts[to_node]].id=i;
				NUMA_list[to_node][candidate_counts[to_node]].val=local_info->freq_percentage[to_node];
				candidate_counts[to_node]++;
				thread_mark[i]=true;				
			}
		}
		//Then, each NUMA node rejects extra proposals
		for (q=0;q<4;++q)
		{
			if (candidate_counts[q]>64)
			{
				std::sort(&(NUMA_list[q][0]),&(NUMA_list[q][candidate_counts[q]]),NUMA_compare);
				for (k=64;k<candidate_counts[q];++k)
				{
					thread_mark[NUMA_list[q][k].id]=false;
				}
				candidate_counts[q]=64;
			}
		}
		//repeat
	}
	//update ids
	j=0;
	for (q=0;q<4;++q)
	{
		assert(candidate_counts[q]==64);
		for (i=0;i<candidate_counts[q];++i)
		{
	//		printf ("%d\t%d\n", q, NUMA_list[q][i].id);
			remapping[j++]=NUMA_list[q][i].id;
		}	
	}
	assert(j==256);
}













