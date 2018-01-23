#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifdef _NO_VECTOR
#ifndef _NO_VECTOR_HASHING
#define _NO_VECTOR_HASHING
#endif
#endif

#ifdef _NO_VECTOR_HASHING
#ifndef _NO_VECTOR
#define _NO_VECTOR
#endif
#endif

#include <unistd.h>

#ifndef _NO_VECTOR

#include <immintrin.h>
#endif
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <pthread.h>
#include <stdlib.h>
#include <stdint.h>	
#include <assert.h>
#include <stdio.h>
#include <sched.h>
#include <time.h>
#include <math.h>
#include "rand.h"
#include <hbwmalloc.h>
#include <numa.h>
#include "hash_table.h"
#include <stdatomic.h>

#define NUM_SAMPLE_BUCKETS 100

double sampling_ratio;
int prefetching_distance;
double HBM_SPEEDUP;
uint64_t thread_time(void)
{
	struct timespec t;
	assert(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t) == 0);
	return t.tv_sec * 1000 * 1000 * 1000 + t.tv_nsec;
}

uint64_t real_time(void)
{
	struct timespec t;
	assert(clock_gettime(CLOCK_REALTIME, &t) == 0);
	return t.tv_sec * 1000 * 1000 * 1000 + t.tv_nsec;
}

int hardware_threads(void)
{
	char name[64];
	struct stat st;
	int threads = -1;
	do {
		sprintf(name, "/sys/devices/system/cpu/cpu%d", ++threads);
	} while (stat(name, &st) == 0);
	return threads;
}
int get_cpu_id (int i, int t)
{
	//t is the total number of threads
#ifdef COMPACT
	//printf ("%d\n", (int)(i/4)+(i%4)*64);
	return (int)(i/4)+(i%4)*64;
#else

#ifdef SCATTER
	return i;
#else
	//BALANCED
	int threads_per_core=floor(t/64);
	int id;
	if (threads_per_core!=0)
	{
		if (i<threads_per_core*64)
			id=(int)(i/threads_per_core)+(i%threads_per_core)*64;
		else
		{
			id=i;
		}
	}
	else id=i;
	//printf ("%d\t%d\n", i, id);
	return id;
#endif

#endif
}
void bind_thread(int thread, int threads)
{
	size_t size = CPU_ALLOC_SIZE(threads);
	cpu_set_t *cpu_set = CPU_ALLOC(threads);
	assert(cpu_set != NULL);
	CPU_ZERO_S(size, cpu_set);
	CPU_SET_S(thread, size, cpu_set);
	assert(pthread_setaffinity_np(pthread_self(), size, cpu_set) == 0);
	CPU_FREE(cpu_set);
}

void *mamalloc(size_t size)
{
	void *ptr = NULL;
#ifdef MCDRAM
	return posix_memalign((void **)&ptr, 64, size)	? NULL : ptr;
#else
	return posix_memalign(&ptr, 64, size) ? NULL : ptr;
#endif
}

typedef struct rand_state_32 {
	uint32_t num[625];
	size_t index;
} rand32_t;

rand32_t *rand32_init(unsigned int seed)
{
	rand32_t *state = (rand32_t*)malloc(sizeof(rand32_t));
	uint32_t *n = state->num;
	size_t i;
	n[0] = seed;
	for (i = 0 ; i != 623 ; ++i)
		n[i + 1] = 0x6c078965 * (n[i] ^ (n[i] >> 30));
	state->index = 624;
	return state;
}
uint32_t rand32_next(rand32_t *state)
{
	uint32_t y, *n = state->num;
	if (state->index == 624) {
		size_t i = 0;
		do {
			y = n[i] & 0x80000000;
			y += n[i + 1] & 0x7fffffff;
			n[i] = n[i + 397] ^ (y >> 1);
			n[i] ^= 0x9908b0df & -(y & 1);
		} while (++i != 227);
		n[624] = n[0];
		do {
			y = n[i] & 0x80000000;
			y += n[i + 1] & 0x7fffffff;
			n[i] = n[i - 227] ^ (y >> 1);
			n[i] ^= 0x9908b0df & -(y & 1);
		} while (++i != 624);
		state->index = 0;
	}
	y = n[state->index++];
	y ^= (y >> 11);
	y ^= (y << 7) & 0x9d2c5680;
	y ^= (y << 15) & 0xefc60000;
	y ^= (y >> 18);
	return y;
}
int power_of_2(uint64_t x)
{
	return x > 0 && (x & (x - 1)) == 0;
}

int prime(uint64_t x)
{
	if ((x & 1) == 0) return x == 2;
	uint64_t d, sx = sqrt(x);
	for (d = 3 ; d <= sx ; d += 2)
		if (x % d == 0) return 0;
	return x > 2;
}

size_t interleave(uint32_t **counts, uint32_t *offsets, uint32_t *aggr_counts,
		size_t partitions, size_t thread, size_t threads)
{
	size_t i = 0, p = 0;
	assert(offsets == align(offsets));
	assert(aggr_counts == align(aggr_counts));
	while (p != partitions) {
		size_t s = partitions - p;
		if (s > 16) s = 16;
		__mmask16 k = _mm512_int2mask((1 << s) - 1);
		__m512i sum = _mm512_xor_epi32(sum, sum);
		size_t t = 0;
		if (thread) do {
			__m512i cur = _mm512_mask_load_epi32(cur, k, &counts[t][p]);
			sum = _mm512_add_epi32(sum, cur);
		} while (++t != thread);
		_mm512_mask_store_epi32(&offsets[p], k, sum);
		do {
			__m512i cur = _mm512_mask_load_epi32(cur, k, &counts[t][p]);
			sum = _mm512_add_epi32(sum, cur);
		} while (++t != threads);
		_mm512_mask_store_epi32(&aggr_counts[p], k, sum);
		do {
			offsets[p] += i;
			i += aggr_counts[p++];
		} while (--s);
	}
	return i;
}

void build(const uint32_t *keys, const uint32_t *vals, size_t size,
		volatile uint64_t *table, volatile uint64_t *table_hbm, size_t buckets, uint32_t factor, uint32_t empty, double ratio)
{
	//size_t count=0;
	size_t i;
	size_t size_ddr = buckets * (1-ratio), size_hbm = buckets - size_ddr;
	for (i = 0 ; i != size ; ++i) {
		uint32_t key = keys[i];
		uint64_t pair = vals[i];
		pair = (pair << 32) | key;
		uint64_t h = (uint32_t) (key * factor);
		h = (h * buckets) >> 32;
		volatile uint64_t *hash_table = h<size_ddr?table:table_hbm;
		//if (hash_table==table_hbm)count++;
		uint64_t tab = hash_table[h];
		while (empty != (uint32_t) tab ||
				!__sync_bool_compare_and_swap(&hash_table[h], tab, pair)) {
			if (++h == buckets) h = 0;
			tab = hash_table[h];
		}
	}
	//printf ("%d\n", count);
}

#ifndef _NO_VECTOR_HASHING

size_t probe(const uint32_t *keys, const uint32_t *vals, size_t size,
		const uint64_t *table, const uint64_t *table_hbm, size_t buckets, uint32_t factor, uint32_t empty,
		uint32_t *keys_out, uint32_t *vals_out, uint32_t *tabs_out,
		size_t block_size, size_t block_limit, volatile size_t *counter, double ratio)
{
	size_t size_ddr = buckets * (1-ratio) ; 
	size_t size_hbm = buckets - size_ddr;
	assert(keys_out == align(keys_out));
	assert(vals_out == align(vals_out));
	// generate masks
	__m512i mask_1 = _mm512_set1_epi32(1);
	__m512i mask_empty = _mm512_set1_epi32(empty);
	__m512i mask_factor = _mm512_set1_epi32(factor);
	__m512i mask_buckets = _mm512_set1_epi32(buckets);
	__m512i mask_unpack = _mm512_set_epi32(15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0);
	__m512i mask_ddr = _mm512_set1_epi32(size_ddr);
	__mmask16 blend_0000 = _mm512_int2mask(0x0000);
	__mmask16 blend_AAAA = _mm512_int2mask(0xAAAA);
	__mmask16 blend_5555 = _mm512_int2mask(0x5555);
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
		// hash keys and add offsets
		__m512i hash = _mm512_mullo_epi32(key, mask_factor);
		hash = simd_hash(hash, mask_buckets);
		hash = _mm512_add_epi32(hash, off);
		k = _mm512_cmpge_epu32_mask(hash, mask_buckets);

		hash = _mm512_mask_sub_epi32(hash, k, hash, mask_buckets);

		__mmask16 in_ddr = _mm512_cmp_epi32_mask(hash, mask_ddr, _MM_CMPINT_LT);
		__mmask16 in_hbm = ~ in_ddr ;

		// load keys from table and update offsets
		__mmask8 temp_in_ddr = in_ddr & 255;
		__m512i lo = _mm512_mask_i32logather_epi64(lo, temp_in_ddr, hash, table, 8);
		__mmask8 temp_in_hbm = in_hbm & 255;
		lo = _mm512_mask_i32logather_epi64(lo, temp_in_hbm, hash, table_hbm, 8);
		//__m512i lo = _mm512_i32logather_epi64(hash, table, 8);
		hash = _mm512_permute4f128_epi32(hash, _MM_PERM_BADC);
		temp_in_ddr = (in_ddr >> 8) & 255;
		__m512i hi = _mm512_mask_i32logather_epi64(hi, temp_in_ddr, hash, table, 8);
		temp_in_hbm = (in_hbm >> 8) & 255;
		hi = _mm512_mask_i32logather_epi64(hi, temp_in_hbm, hash, table_hbm, 8);
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
		size_t h = (uint32_t) (k * factor);
		h = (h * buckets) >> 32;
		h += (uint32_t) offs_last[i];
		const uint64_t *hash_table = h < size_ddr ? table : table_hbm;
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

void set(uint64_t *dst, size_t size, uint32_t value)
{
	uint64_t *dst_end = &dst[size];
	uint64_t *dst_aligned = (uint64_t *)align(dst);
	__m512i x = _mm512_set1_epi64(value);
	while (dst != dst_end && dst != dst_aligned)
		*dst++ = value;
	dst_aligned = &dst[(dst_end - dst) & ~7];
	while (dst != dst_aligned) {
		_mm512_store_epi64(dst, x);
		dst += 8;
	}
	while (dst != dst_end)
		*dst++ = value;
}

void copy(uint32_t *dst, const uint32_t *src, size_t size)
{
	uint32_t *dst_end = &dst[size];
	uint32_t *dst_aligned = (uint32_t *)align(dst);
	__mmask16 k = _mm512_kxnor(k, k);
	while (dst != dst_end && dst != dst_aligned)
		*dst++ = *src++;
	dst_aligned = &dst[(dst_end - dst) & -16];
	if (src == align(src))
		while (dst != dst_aligned) {
			__m512 x = _mm512_load_ps(src);
			_mm512_store_ps(dst, x);
			src += 16;
			dst += 16;
		}
	else
		while (dst != dst_aligned) {
			__m512 x;
			x = _mm512_mask_loadu_ps(x, k, src);
			src += 16;
			//x = _mm512_loadunpackhi_ps(x, src);
			_mm512_store_ps(dst, x);
			dst += 16;
		}
	while (dst != dst_end)
		*dst++ = *src++;
}

#else

size_t probe(const uint32_t *keys, const uint32_t *vals, size_t size,
		const uint64_t *table, size_t buckets, uint32_t factor, uint32_t empty,
		uint32_t *keys_out, uint32_t *vals_out, uint32_t *tabs_out,
		size_t block_size, size_t block_limit, volatile size_t *counter)
{
	size_t i, o = __sync_fetch_and_add(counter, 1);
	assert(o <= block_limit);
	o *= block_size;
	for (i = 0 ; i != size ; ++i) {
		uint32_t key = keys[i];
		uint32_t val = vals[i];
		uint64_t h = (uint32_t) (key * factor);
		h = (h * buckets) >> 32;
		uint64_t tab = table[h];
		while (empty != (uint32_t) tab) {
			if (key == (uint32_t) tab) {
				tabs_out[o] = tab >> 32;
				vals_out[o] = val;
				keys_out[o++] = key;
				if ((o & (block_size - 1)) == 0) {
					o = __sync_fetch_and_add(counter, 1);
					assert(o <= block_limit);
					o *= block_size;
				}
#ifdef _UNIQUE
				break;
#endif
			}
			if (++h == buckets) h = 0;
			tab = table[h];
		}
	}
	return o;
}

void set(uint64_t *dst, size_t size, uint32_t value)
{
	uint64_t *dst_end = &dst[size];
	while (dst != dst_end)
		*dst++ = value;
}

void copy(uint32_t *dst, const uint32_t *src, size_t size)
{
	uint32_t *dst_end = &dst[size];
	while (dst != dst_end)
		*dst++ = *src++;
}

#endif

typedef struct {
	size_t beg;
	size_t end;
} size_pair_t;

int pair_cmp(const void *x, const void *y)
{
	size_t a = ((size_pair_t*) x)->beg;
	size_t b = ((size_pair_t*) y)->beg;
	return a < b ? -1 : a > b ? 1 : 0;
}

size_t close_gaps(uint32_t *keys, uint32_t *vals, uint32_t *tabs,
		const size_t *offsets, size_t count,
		size_t block_size, volatile size_t *counter)
{
	assert(power_of_2(block_size));
	size_pair_t *holes = (size_pair_t *)malloc(count * sizeof(size_pair_t));
	size_t i, c = 0, l = 0, h = count - 1;
	for (i = 0 ; i != count ; ++i) {
		holes[i].beg = offsets[i];
		holes[i].end = (offsets[i] & ~(block_size - 1)) + block_size;
	}
	qsort(holes, count, sizeof(size_pair_t), pair_cmp);
	size_t src = holes[h].end;
	i = __sync_fetch_and_add(counter, 1);
	while (l <= h) {
		size_t fill = src - holes[h].end;
		if (fill == 0) {
			src = holes[h].beg;
			if (!h--) break;
			continue;
		}
		size_t hole = holes[l].end - holes[l].beg;
		if (hole == 0) {
			l++;
			continue;
		}
		size_t cnt = fill < hole ? fill : hole;
		size_t dst = holes[l].beg;
		holes[l].beg += cnt;
		src -= cnt;
		if (c++ == i) {
			copy(&keys[dst], &keys[src], cnt);
			copy(&vals[dst], &vals[src], cnt);
			copy(&tabs[dst], &tabs[src], cnt);
			i = __sync_fetch_and_add(counter, 1);
		}
	}
	free(holes);
	return src;
}

size_t thread_beg(size_t size, size_t alignment, size_t thread, size_t threads)
{
	assert(power_of_2(alignment));
	size_t part = (size / threads) & ~(alignment - 1);
	return part * thread;
}

size_t thread_end(size_t size, size_t alignment, size_t thread, size_t threads)
{
	assert(power_of_2(alignment));
	size_t part = (size / threads) & ~(alignment - 1);
	if (thread + 1 == threads) return size;
	return part * (thread + 1);
}

void swap(uint32_t **x, uint32_t **y)
{
	uint32_t *t = *x; *x = *y; *y = t;
}

size_t max(size_t x, size_t y)
{
	return x > y ? x : y;
}

size_t min(size_t x, size_t y)
{
	return x < y ? x : y;
}

size_t binary_search(const size_t *array, size_t size, size_t value)
{
	size_t lo = 0;
	size_t hi = size;
	do {
		size_t mid = (lo + hi) >> 1;
		if (value > array[mid])
			lo = mid + 1;
		else
			hi = mid;
	} while (lo < hi);
	return array[lo];
}

void shuffle(uint32_t *data, size_t size, rand32_t *gen)
{
	size_t i, j;
	for (i = 0 ; i != size ; ++i) {
		j = rand32_next(gen);
		j *= size - i;
		j >>= 32;
		j += i;
		uint32_t t = data[i];
		data[i] = data[j];
		data[j] = t;
	}
}

void unique(uint32_t *keys, size_t size, volatile uint32_t *table, size_t buckets,
		uint32_t factor, uint32_t empty, rand32_t *gen)
{
	assert(prime(buckets));
	size_t i = 0;
	while (i != size) {
		uint32_t key;
		do {
			key = rand32_next(gen);
		} while (key == empty);
		size_t h = (uint32_t) (key * factor);
		h = (h * buckets) >> 32;
		uint32_t tab = table[h];
		while (tab != key) {
			if (tab == empty) {
				tab = __sync_val_compare_and_swap(&table[h], empty, key);
				if (tab == empty) {
					keys[i++] = key;
					break;
				}
				if (tab == key) break;
			}
			if (++h == buckets) h = 0;
			tab = table[h];
		}
	}
}

typedef struct {
	pthread_t id;
	int seed;
	int thread;
	int threads;
	uint32_t thread_factor;
	uint32_t inner_factor;
	uint32_t outer_factor;
	size_t inner_tuples;
	size_t outer_tuples;
	size_t join_tuples;
	size_t inner_distinct;
	size_t outer_distinct;
	size_t join_distinct;
	uint32_t *inner_keys;
	uint32_t *inner_vals;
	uint32_t *outer_keys;
	uint32_t *outer_vals;
	uint32_t *join_keys;
	uint32_t *join_outer_vals;
	uint32_t *join_inner_vals;
	size_t *final_offsets;
	uint64_t real_time;
	uint64_t thread_time;
	volatile uint64_t *hash_table;
	volatile uint64_t *hash_table_hbm;
	volatile uint64_t *NUMA_tables_ori[HW_NUM_NUMA_NODES];
	double ratio;
	size_t hash_buckets;
	uint32_t hash_factor;
	uint32_t hash_factor2;
	uint32_t sample_factor;
	uint32_t **outer_counts;
	uint32_t **inner_counts;
	uint32_t *unique;
	uint32_t *unique_table;
	size_t unique_buckets;
	uint32_t unique_factor;
	size_t block_size;
	size_t block_limit;
	uint64_t times[6];
	size_t sampling_buckets;
	volatile size_t *counters;
	volatile uint32_t *input_data_placement;
	pthread_barrier_t *barrier;
	size_t *remapping;
	volatile thread_NUMA_info *threads_info;
	uint32_t *sample_counter;
	uint32_t *sentry;
	uint32_t *sampled_inner_keys;
	uint32_t *sampled_outer_keys;
} info_t;
void *gen_data(void *arg)
{
	info_t *d = (info_t*) arg;
	assert(pthread_equal(pthread_self(), d->id));
	pthread_barrier_t *barrier = d->barrier;
	bind_thread(d->thread, d->threads);
	uint32_t *inner_keys  = d->inner_keys;
	uint32_t *inner_vals  = d->inner_vals;
	uint32_t *outer_keys = d->outer_keys;
	uint32_t *outer_vals  = d->outer_vals;
	size_t i, o, j, u, t, f, h;
	size_t thread  = d->thread;
	size_t threads = d->threads;
	rand32_t *gen = rand32_init(d->seed);
	// generate unique items
	size_t join_distinct = d->join_distinct;
	size_t inner_distinct = d->inner_distinct;
	size_t outer_distinct = d->outer_distinct;
	size_t distinct = inner_distinct + outer_distinct - join_distinct;
	size_t distinct_beg = thread_beg(distinct, 1, thread, threads);
	size_t distinct_end = thread_end(distinct, 1, thread, threads);
	unique(&d->unique[distinct_beg], distinct_end - distinct_beg,
			d->unique_table, d->unique_buckets, d->unique_factor, 0, gen);
	// generate keys from unique items
	pthread_barrier_wait(barrier++);
	if (thread == 0) free((void*) d->unique_table);
	uint32_t *inner_unique = d->unique;
	size_t inner_beg = thread_beg(d->inner_tuples, 16, thread, threads);
	size_t inner_end = thread_end(d->inner_tuples, 16, thread, threads);
	size_t inner_distinct_beg = thread_beg(inner_distinct, 16, thread, threads);
	size_t inner_distinct_end = thread_end(inner_distinct, 16, thread, threads);
	uint64_t inner_checksum = 0;
	u = inner_distinct_beg;
	for (i = inner_beg ; i != inner_end ; ++i) {
		if (u != inner_distinct_end)
			inner_keys[i] = inner_unique[u++];
		else {
			uint64_t r = rand32_next(gen);
			r = (r * inner_distinct) >> 32;
			inner_keys[i] = inner_unique[r];
		}
		assert(inner_keys[i] != 0);
		inner_checksum += inner_keys[i];
	}
	assert(u == inner_distinct_end);
	uint32_t *outer_unique = &d->unique[inner_distinct - join_distinct];
	size_t outer_beg = thread_beg(d->outer_tuples, 16, thread, threads);
	size_t outer_end = thread_end(d->outer_tuples, 16, thread, threads);
	size_t outer_distinct_beg = thread_beg(outer_distinct, 16, thread, threads);
	size_t outer_distinct_end = thread_end(outer_distinct, 16, thread, threads);
	u = outer_distinct_beg;
	for (o = outer_beg ; o != outer_end ; ++o) {
		if (u != outer_distinct_end)
			outer_keys[o] = outer_unique[u++];
		else {
			uint64_t r = rand32_next(gen);
			r = (r * outer_distinct) >> 32;
			outer_keys[o] = outer_unique[r];
		}
		assert(outer_keys[o] != 0);
	}
	assert(u == outer_distinct_end);
	pthread_barrier_wait(barrier++);
	if (thread == 0) {
		free(d->unique);
		//fprintf(stderr, "Shuffling ... ");
		shuffle(inner_keys, d->inner_tuples, gen);
		shuffle(outer_keys, d->outer_tuples, gen);
		//fprintf(stderr, "done!\n");
	}
	// generate payloads and outputs
	pthread_barrier_wait(barrier++);
	uint32_t inner_factor = d->inner_factor;
	uint32_t temp[16];
	size_t output=0;
	for (i = inner_beg ; i != inner_end ; ++i) {
		//inner_vals_in[i] = inner_keys_in[i] * inner_factor;
		temp[i%16]=inner_keys[i] * inner_factor;
		if ((i&15)==15)
		{
			__m512 key=_mm512_set_ps(temp[15],temp[14],temp[13],temp[12],temp[11],temp[10],temp[9],temp[8],temp[7],temp[6],temp[5],
					temp[4],temp[3],temp[2],temp[1],temp[0]);
			_mm512_stream_ps(&inner_vals[output],key);
			//_mm512_stream_ps(&inner_keys_out[output],key);
			//_mm512_stream_ps(&inner_vals_out[output],key);
			output+=16;
		}
		//inner_keys_out[i] = 0xCAFEBABE;
		//inner_vals_out[i] = 0xDEADBEEF;
	}
	uint32_t outer_factor = d->outer_factor;
	output=0;
	for (o = outer_beg ; o != outer_end ; ++o) {
		//outer_vals_in[o] = outer_keys_in[o] * outer_factor;
		temp[o%16]=outer_keys[o] * outer_factor;
		if ((o&15)==15)
		{
			__m512 key=_mm512_set_ps(temp[15],temp[14],temp[13],temp[12],temp[11],temp[10],temp[9],temp[8],temp[7],temp[6],temp[5],
					temp[4],temp[3],temp[2],temp[1],temp[0]);
			_mm512_stream_ps(&outer_vals[output],key);
			//_mm512_stream_ps(&outer_keys_out[output],key);
			//_mm512_stream_ps(&outer_vals_out[output],key);
			output+=16;
		}
		//outer_keys_out[o] = 0xBABECAFE;
		//outer_vals_out[o] = 0xBEEFDEAD; 
	}
	size_t join_beg = thread_beg(d->block_limit * d->block_size, 1, thread, threads);
	size_t join_end = thread_end(d->block_limit * d->block_size, 1, thread, threads);
	uint32_t *join_keys = d->join_keys;
	uint32_t *join_vals = d->join_outer_vals;
	uint32_t *join_tabs = d->join_inner_vals;
	output=0;
	for (j = join_beg ; j != join_end ; ++j) {
		temp[j%16]=0xCAFEBABE;
		if ((j&15)==15)
		{
			__m512 key=_mm512_set_ps(temp[15],temp[14],temp[13],temp[12],temp[11],temp[10],temp[9],temp[8],temp[7],temp[6],temp[5],
					temp[4],temp[3],temp[2],temp[1],temp[0]);
			_mm512_stream_ps(&join_keys[output],key);
			_mm512_stream_ps(&join_vals[output],key);
			_mm512_stream_ps(&join_tabs[output],key);
			output+=16;
		}
	}
	free(gen);
	pthread_exit(NULL);
}
void two_level_offline_sampling (const uint32_t *keys, size_t size, 
		uint32_t *sampled_keys, size_t *sample_size, uint32_t factor, uint32_t sample_buckets, 
		uint32_t p, float q, uint32_t *counter, uint32_t *sentry)
{
	int i,o;
	for (i=0;i<size;i++)
	{
		uint32_t key=keys[i];
		size_t h = (uint32_t) key * factor;
		h = (h * sample_buckets) >> 32;
		if (h<p)
		{
			sentry[h]=key;
			counter[h]=1;
		}
		else
		{
            counter[h]+=1;
			uint64_t probability = (int)(rand())%counter[h];
			if (probability == 1)
			{
				sentry[h]=key;
			}
		}
		float prob = 1.0*rand()/RAND_MAX;
		if (prob < q)
		{
			o=__sync_fetch_and_add(sample_size, 1);
			sampled_keys[o]=key;
		}
	}
}
void *run(void *arg)
{
	info_t *d = (info_t*) arg;
	assert(pthread_equal(pthread_self(), d->id));
	pthread_barrier_t *barrier = d->barrier;
	bind_thread(d->thread, d->threads);
	uint32_t *inner_keys  = d->inner_keys;
	uint32_t *inner_vals  = d->inner_vals;
	uint32_t *outer_keys = d->outer_keys;
	uint32_t *outer_vals  = d->outer_vals;
	size_t i, o, j, u, t, f, h;
	size_t thread  = d->thread;
	size_t threads = d->threads;
	rand32_t *gen = rand32_init(d->seed);
	size_t inner_beg = thread_beg(d->inner_tuples, 16, thread, threads);
	size_t inner_end = thread_end(d->inner_tuples, 16, thread, threads);
	size_t outer_beg = thread_beg(d->outer_tuples, 16, thread, threads);
	size_t outer_end = thread_end(d->outer_tuples, 16, thread, threads);
	size_t join_beg = thread_beg(d->block_limit * d->block_size, 1, thread, threads);
	size_t join_end = thread_end(d->block_limit * d->block_size, 1, thread, threads);
	uint64_t ptt;
	uint64_t ntt;
	// start timinig

	size_t sampling_buckets = d->sampling_buckets;
	size_t partitions_aligned = (sampling_buckets + 15) & -16;
	uint32_t counts_space[partitions_aligned * 6 + 15];
	uint32_t *counts = (uint32_t *)align(counts_space);

	uint32_t *outer_counts  = &counts[0];
	uint32_t *outer_offsets = &counts[partitions_aligned * 1];
	d->outer_counts[thread] = outer_counts;

	uint32_t *inner_counts = &counts[partitions_aligned * 3];
	uint32_t *inner_offsets = &counts[partitions_aligned * 4];
	d->inner_counts[thread] = inner_counts;

	size_t sampled_inner_size=0, sampled_outer_size=0;
	uint32_t sample_buckets = NUM_SAMPLE_BUCKETS;
	float p = 0.5, q = 0.5;
	uint64_t rt;
#ifndef UNSAMPLED
	if (thread==0)
	{
		memset(d->sample_counter, 0, sample_buckets);
	}
    pthread_barrier_wait(barrier++);
	//Try to adopt the 2-level sampling here offline, not included in the runtime
	two_level_offline_sampling (&inner_keys[inner_beg], inner_end-inner_beg, 
			d->sampled_inner_keys, &sampled_inner_size, d->sample_factor, sample_buckets, p*sample_buckets , q, d->sample_counter, d->sentry);
	pthread_barrier_wait(barrier++);	
	if (thread==0)
	{
		memset(d->sample_counter, 0, sample_buckets);
	}
    pthread_barrier_wait(barrier++);
	two_level_offline_sampling (&outer_keys[inner_beg], outer_end-outer_beg, 
			d->sampled_outer_keys, &sampled_outer_size, d->sample_factor, sample_buckets, p*sample_buckets , q, d->sample_counter, d->sentry); 
	//printf ("sampled size: %d\t%d\n", sampled_inner_size, sampled_outer_size);
	pthread_barrier_wait(barrier++);
#endif
	// initialize table
	size_t table_beg = thread_beg(d->hash_buckets, 1, thread, threads);
	size_t table_end = thread_end(d->hash_buckets, 1, thread, threads);
	//set((uint64_t*) &d->hash_table[table_beg], table_end - table_beg, 0);
	//set((uint64_t*) &d->hash_table_hbm[table_beg], table_end - table_beg, 0);
	for (i=0;i<HW_NUM_NUMA_NODES;++i)
		set((uint64_t*) &(d->NUMA_tables_ori[i])[table_beg], table_end-table_beg, 0);
	pthread_barrier_wait(barrier++);
	// build global table
	RELATION_TYPE inner_relation;
	inner_relation.keys=&inner_keys[inner_beg];
	inner_relation.vals=&inner_vals[inner_beg];
	inner_relation.size=inner_end-inner_beg;
	uint64_t tt = thread_time();
	uint64_t time_stamp = tt;
	rt = real_time();	
	uint64_t exe_time = 0;//thread_time();

	//printf("start %d %d %d\n", inner_relation.size, d->hash_table, d->hash_table_hbm);
	Linear_Hash_Table local_hash_table (inner_relation, d->hash_buckets, 0, d->hash_factor, d->hash_factor2, 
			d->ratio, d->NUMA_tables_ori, 
			d->input_data_placement, sampling_buckets, d->threads_info);

	//##### The following is building a histogram, not sampling. Now replaced with building histograms on sampled sets. ####
#ifdef UNSAMPLED
	local_hash_table.histogram(&outer_keys[outer_beg], outer_end-outer_beg, outer_counts, d->hash_factor2, sampling_buckets);
	local_hash_table.histogram(&inner_keys[inner_beg], inner_end-inner_beg, inner_counts, d->hash_factor2, sampling_buckets);
#else
	//The following is building a histogram on sampled sets. 
	//caculate offsets for each thread
	size_t sampled_inner_beg=thread_beg(sampled_inner_size, 16, thread, threads);
	size_t sampled_inner_end=thread_end(sampled_inner_size, 16, thread, threads);
	size_t sampled_outer_beg=thread_beg(sampled_outer_size, 16, thread, threads);
	size_t sampled_outer_end=thread_end(sampled_outer_size, 16, thread, threads);
	//build histogram
	local_hash_table.histogram(&d->sampled_outer_keys[sampled_outer_beg], sampled_outer_end-sampled_outer_beg, outer_counts, d->hash_factor2, sampling_buckets);
	local_hash_table.histogram(&d->sampled_inner_keys[sampled_inner_beg], sampled_inner_end-sampled_inner_beg, inner_counts, d->hash_factor2, sampling_buckets);
#endif

	exe_time+=(thread_time()-tt);
	pthread_barrier_wait(barrier++);
	time_stamp=thread_time();
	outer_counts = &counts[partitions_aligned * 2];
	inner_counts = &counts[partitions_aligned * 5];
	o = interleave(d->outer_counts, outer_offsets, outer_counts, sampling_buckets, thread, threads);
	//assert(o == sampled_outer_size);  //size does not match for some unkown reason, leave it here for now. 
	i = interleave(d->inner_counts, inner_offsets, inner_counts, sampling_buckets, thread, threads);
	//assert(i == sampled_inner_size);
	//begin of the subset sum solution

	//for (size_t i = 0; i < sampling_buckets; i++)
	//{
	//	printf ("%d\t%d\n", inner_counts[i], outer_counts[i]);
	//}
	if (thread==0)
	{
		local_hash_table.subset_sum_solver (inner_counts, outer_counts, sampling_buckets, d->inner_tuples, d->outer_tuples, HBM_SPEEDUP);
	}


	exe_time+=(thread_time()-time_stamp);
	pthread_barrier_wait(barrier++);
	time_stamp=thread_time();
	ptt = tt;
	ntt = thread_time();
	d->times[0] = ntt - ptt;

	local_hash_table.Build();

	ptt = ntt;
	ntt = thread_time();
	d->times[1] = ntt - ptt;

#ifdef NUMA_MAP
	//Stable marriage problem solver
	local_hash_table.NUMA_mapping_prepare(thread, &counts[0]);
	exe_time+=(thread_time()-time_stamp);
	pthread_barrier_wait(barrier++);
	time_stamp=thread_time();
	if (thread==0)
		local_hash_table.NUMA_mapping_solver(d->remapping, threads, 4);
	exe_time+=(thread_time()-time_stamp);
	pthread_barrier_wait(barrier++);
	time_stamp=thread_time();
#endif
	size_t new_thread = d->remapping[thread];
	//if (thread!=new_thread)
	//	printf ("%d -> %d\n", thread, new_thread);
	outer_beg = thread_beg(d->outer_tuples, 16, new_thread, threads);
	outer_end = thread_end(d->outer_tuples, 16, new_thread, threads);

	ptt = ntt;
	ntt = thread_time();
	d->times[2] = ntt - ptt;
	exe_time+=(thread_time()-time_stamp);
	pthread_barrier_wait(barrier++);
	time_stamp=thread_time();

	// probe hash table
	o = local_hash_table.Probe(
			&outer_keys[outer_beg],
			&outer_vals[outer_beg],
			outer_end - outer_beg,	          
			d->join_keys,
			d->join_outer_vals,
			d->join_inner_vals,
			d->block_size,
			d->block_limit,
			&d->counters[0]);

	d->final_offsets[thread] = o;
	ptt = ntt;
	ntt = thread_time();
	d->times[3] = ntt - ptt;
	// sync
	exe_time+=(thread_time()-time_stamp);
	pthread_barrier_wait(barrier++);
	time_stamp=thread_time();
	// close gaps
	d->join_tuples = close_gaps(d->join_keys,
			d->join_outer_vals,
			d->join_inner_vals,
			d->final_offsets,
			threads, 
			d->block_size,
			&d->counters[1]); 
	ptt = ntt;
	ntt = thread_time();
	d->times[4] = ntt - ptt;
	// finish timinig
	rt = real_time() - rt;
	tt = thread_time() - tt;
	exe_time+=(thread_time()-time_stamp);
	d->times[5]=exe_time;
	pthread_barrier_wait(barrier++);
	d->real_time = rt;
	d->thread_time = tt;
	//	fprintf(stderr, "Join verified [%ld / %ld]\n", thread + 1, threads);
	free(gen);
	pthread_exit(NULL);
}

int main(int argc, char **argv)
{
	// arguments
	//uint64_t tt = thread_time();
	int t, threads = argc > 1 ? atoi(argv[1]) : hardware_threads();
	size_t outer_tuples = argc > 2 ? atoll(argv[2]) : 200 * 1000 * 1000;
	size_t inner_tuples = argc > 3 ? atoll(argv[3]) : 200 * 1000 * 1000;
	//double ratio= argc > 4 ? std::stod(argv[4]):1;
	int id = argc > 4 ? std::atoll(argv[4]):0;
	double zipf = argc > 5 ? std::stod(argv[5]):0;
	double hash_table_load = 0.9;//argc > 4 ? std::stod(argv[4]):0.90;;//0.90;
	sampling_ratio = 0.0000006;
	HBM_SPEEDUP = 3;//argc > 5 ? std::stod(argv[5]):5;
	prefetching_distance= 10;
	size_t outer_distinct = min(inner_tuples, outer_tuples);
	size_t inner_distinct = min(inner_tuples, outer_tuples);
	size_t join_distinct = min(inner_distinct, outer_distinct); 
	double outer_repeats = outer_tuples * 1.0 / outer_distinct;
	double inner_repeats = inner_tuples * 1.0 / inner_distinct;
	size_t join_tuples = outer_repeats * inner_repeats * join_distinct;
	// other parameters
	size_t block_size = 256 * 256;
	// compute hash buckets
	size_t hash_buckets = inner_tuples / hash_table_load;
	// print info
	assert(inner_distinct <= inner_tuples);
	assert(outer_distinct <= outer_tuples);
	assert(join_distinct <= min(inner_distinct, outer_distinct));
	assert(threads > 0 && threads <= hardware_threads());
	assert(power_of_2(block_size));
#ifdef _UNIQUE
	assert(inner_tuples == inner_distinct);
	//fprintf(stderr, "Enforcing unique keys\n");
#endif
#ifdef _NO_VECTOR_HASHING
	//fprintf(stderr, "Vectorized hashing (probing) disabled!\n");
#endif

	// compute space
	double space = inner_tuples * 8 + outer_tuples * 8 + hash_buckets * 8 + join_tuples * 12;
	space /= 1024 * 1024 * 1024;
	//fprintf(stderr, "Space: %.2f GB\n", space);
	// parameters
	srand(time(NULL));
	uint32_t factors[6];
	for (t = 0 ; t != 6 ; ++t)
		factors[t] = (rand() << 1) | 1;
	uint32_t sample_factor=(rand() << 1) | 1;
	//set_policy(POLICY_INTERLEAVE);

	// inner side & buffers
	uint32_t *inner_keys = (uint32_t *)mamalloc(inner_tuples * sizeof(uint32_t));
	uint32_t *inner_vals = (uint32_t *)mamalloc(inner_tuples * sizeof(uint32_t));
	// outer side & buffers
	uint32_t *outer_keys = (uint32_t *)mamalloc(outer_tuples * sizeof(uint32_t));
	uint32_t *outer_vals = (uint32_t *)mamalloc(outer_tuples * sizeof(uint32_t));
	// allocate hash table
	//uint64_t *hash_table = (uint64_t *)mamalloc(hash_buckets * sizeof(uint64_t));
	//uint64_t *hash_table_hbm = (uint64_t *)hbw_malloc(hash_buckets * sizeof(uint64_t));

	float SHORTERN_RATIO = 1;

	uint64_t *NUMA_tables_ori[HW_NUM_NUMA_NODES];
	for (int i=0;i<HW_NUM_NUMA_NODES;++i)
	{
		NUMA_tables_ori[i]=(uint64_t *)numa_alloc_onnode(hash_buckets * sizeof(uint64_t) * SHORTERN_RATIO, i>1?i+2:i);
	}

	size_t sampling_buckets = hash_buckets * sampling_ratio;
	volatile uint32_t *input_data_placement=(volatile uint32_t*)malloc(sizeof(uint32_t)*sampling_buckets);

	// unique table and unique items
	size_t distinct = outer_distinct + inner_distinct - join_distinct;
	size_t unique_buckets = distinct * 2 + 1;
	while (!prime(unique_buckets)) unique_buckets += 2;
	uint32_t *unique = (uint32_t *)mamalloc(distinct * sizeof(uint32_t));
	uint32_t *unique_table = (uint32_t *)calloc(unique_buckets, sizeof(uint32_t));
	// join result
	size_t block_limit = join_tuples * 1.05 / block_size + threads * 2;
	uint32_t *join_keys       = (uint32_t *)mamalloc(block_limit * block_size * sizeof(uint32_t));
	uint32_t *join_inner_vals = (uint32_t *)mamalloc(block_limit * block_size * sizeof(uint32_t));
	uint32_t *join_outer_vals = (uint32_t *)mamalloc(block_limit * block_size * sizeof(uint32_t));

	thread_NUMA_info *input_threads_info;
	input_threads_info=(thread_NUMA_info*)malloc(sizeof(thread_NUMA_info)*threads);

	// run threads
	int b, barriers = 64;
	pthread_barrier_t barrier[barriers];
	for (b = 0 ; b != barriers ; ++b)
		pthread_barrier_init(&barrier[b], NULL, threads);
	info_t info[threads];
	size_t final_offsets[threads];
	volatile size_t counters[2] = {0, 0};
	pthread_attr_t attr;
	cpu_set_t cpuset;
	pthread_attr_init(&attr);

	double selc=1.0;
	std::string name1,name2,name3,name4;
	name1.append("./ik");
	name2.append(".iv");
	name3.append("./ok");
	name4.append("./ov");
	name1.append("_");
	name2.append("_");
	name3.append("_");
	name4.append("_");
	name1.append(std::to_string(inner_tuples));	
	name2.append(std::to_string(inner_tuples));	
	name3.append(std::to_string(outer_tuples));	
	name4.append(std::to_string(outer_tuples));	
	name1.append("_");
	name2.append("_");
	name3.append("_");
	name4.append("_");
 	/*name1.append(std::to_string(zipf));	
	name2.append(std::to_string(zipf));	
	name3.append(std::to_string(zipf));	
	name4.append(std::to_string(zipf)); 
	name1.append("_");
	name2.append("_");
	name3.append("_");
	name4.append("_");*/
	name1.append(std::to_string(id));
	name2.append(std::to_string(id));
	name3.append(std::to_string(id));
	name4.append(std::to_string(id));	
	name1.append(".txt");
	name2.append(".txt");
	name3.append(".txt");
	name4.append(".txt");
	FILE *f_inner_keys=fopen(name1.c_str(), "rb");
	FILE *f_inner_vals=fopen(name2.c_str(), "rb");
	FILE *f_outer_keys=fopen(name3.c_str(), "rb");
	FILE *f_outer_vals=fopen(name4.c_str(), "rb");

	assert(f_inner_keys!=NULL);
	assert(f_inner_vals!=NULL);
	assert(f_outer_keys!=NULL);
	assert(f_outer_vals!=NULL);

	fread(inner_keys, inner_tuples, sizeof(int), f_inner_keys);
	fread(inner_vals, inner_tuples, sizeof(int), f_inner_vals);
	//fread(inner_keys, inner_tuples, sizeof(int), f_outer_keys);
	//fread(inner_vals, inner_tuples, sizeof(int), f_outer_vals);
	fread(outer_keys, outer_tuples, sizeof(int), f_outer_keys);
	fread(outer_vals, outer_tuples, sizeof(int), f_outer_vals);

	uint32_t *outer_counts[threads];
	uint32_t *inner_counts[threads];
	size_t *remapping=(size_t*)mamalloc(sizeof(size_t)*threads);

	uint32_t *sample_counter, *sentry, *sampled_inner_keys, *sampled_outer_keys;
	size_t sample_buckets = NUM_SAMPLE_BUCKETS;
	sample_counter = (uint32_t*)mamalloc(sample_buckets*sizeof(uint32_t));
	sentry = (uint32_t*)mamalloc(sample_buckets*sizeof(uint32_t));
	sampled_inner_keys = (uint32_t*)mamalloc(inner_tuples*sizeof(uint32_t));
	sampled_outer_keys = (uint32_t*)mamalloc(outer_tuples*sizeof(uint32_t));
	//initialize all the counters for samples
	memset(sample_counter, 0, sample_buckets);

	for (t = 0 ; t != threads ; ++t) {
		int cpu_idx=get_cpu_id(t, threads);
		CPU_ZERO((void*)&cpuset);
		CPU_SET(cpu_idx,&cpuset);
		pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpuset);
		info[t].sample_counter=sample_counter;
		info[t].sentry=sentry;
		info[t].sampled_inner_keys=sampled_inner_keys;
		info[t].sampled_outer_keys=sampled_outer_keys;
		info[t].input_data_placement=input_data_placement;
		info[t].thread = t;
		info[t].threads = threads;
		info[t].outer_counts=outer_counts;
		info[t].inner_counts=inner_counts;
		info[t].seed = rand();
		info[t].join_tuples = join_tuples;
		info[t].block_limit = block_limit;
		info[t].block_size = block_size;
		info[t].outer_tuples = outer_tuples;
		info[t].inner_tuples = inner_tuples;
		info[t].join_distinct = join_distinct;
		info[t].inner_distinct = inner_distinct;
		info[t].outer_distinct = outer_distinct;
		info[t].inner_keys = inner_keys;
		info[t].inner_vals = inner_vals;
		info[t].outer_keys = outer_keys;
		info[t].outer_vals = outer_vals;
		info[t].final_offsets = final_offsets;
		info[t].join_keys = join_keys;
		info[t].join_inner_vals = join_inner_vals;
		info[t].join_outer_vals = join_outer_vals;
		info[t].unique = unique;
		info[t].unique_table = unique_table;
		info[t].unique_buckets = unique_buckets;
		info[t].unique_factor = factors[0];
		info[t].inner_factor = factors[1];
		info[t].outer_factor = factors[2];
		info[t].hash_factor = factors[3];
		info[t].hash_factor2 = factors[4];
		info[t].threads_info=input_threads_info;
		//info[t].hash_table = hash_table;
		info[t].sampling_buckets=sampling_buckets;
		info[t].hash_buckets = hash_buckets;
		info[t].counters = counters;
		info[t].barrier = barrier;
		info[t].remapping=remapping;
		info[t].sample_factor=sample_factor;
		for (int i=0;i<HW_NUM_NUMA_NODES;++i)
			info[t].NUMA_tables_ori[i]=NUMA_tables_ori[i];
		pthread_create(&info[t].id, &attr, run, (void*) &info[t]);
	}
	for (t = 0 ; t != threads ; ++t)
		pthread_join(info[t].id, NULL);
	for (b = 0 ; b != barriers ; ++b)
		pthread_barrier_destroy(&barrier[b]);
	//tt = thread_time()-tt;
	//printf ("overall time %lf\n", 1.0 * tt / 1000000000);
	join_tuples = info[0].join_tuples;
	uint64_t min_tt = ~0, max_tt = 0, avg_tt = 0;
	uint64_t min_rt = ~0, max_rt = 0, avg_rt = 0;
	uint64_t min_exe_time =~0, max_exe_time = 0;
	for (t = 0 ; t != threads ; ++t) {
		uint64_t rt = info[t].real_time;
		uint64_t tt = info[t].thread_time;
		min_exe_time = min (min_exe_time, info[t].times[5]);
		max_exe_time = max (max_exe_time, info[t].times[5]);
		//printf ("%.4f\n", info[t].real_time * 1.0 / 1000000000);
		max_rt = max(rt, max_rt);
		max_tt = max(tt, max_tt);
		min_rt = min(rt, min_rt);
		min_tt = min(tt, min_tt);
		avg_rt += rt;
		avg_tt += tt;
		assert(info[t].join_tuples == join_tuples);
	}
	avg_tt /= threads;
	avg_rt /= threads;
	double billion_th = 1.0 / 1000000000;
	size_t p;
	for (p = 0 ; p != 5 ; ++p) {
		uint64_t partial_tt = 0;
		for (t = 0 ; t != threads ; ++t)
			partial_tt += info[t].times[p];
		if (partial_tt == 0) continue;
		partial_tt /= threads;
		double r = partial_tt * 1.0 / avg_tt;
		//printf("Phase %ld: %5.2f%% (%.4f)\n", p + 1, r * 100, r * max_rt * billion_th);
		//printf("%.4f\t", r * max_rt * billion_th);
		//printf("%.4f\t", r * max_rt * billion_th);
	}
	//printf("%.4f\t%.4f\t%.4f\n", max_rt * billion_th, min_exe_time * billion_th, max_exe_time * billion_th);
	printf("%.4f\n", max_rt * billion_th);
	free(join_keys);
	free(join_inner_vals);
	free(join_outer_vals);
	free(inner_keys);
	free(inner_vals);
	free(outer_keys);
	free(outer_vals);
	//free(hash_table);
	for (int i=0;i<HW_NUM_NUMA_NODES;++i)
	{
		numa_free(NUMA_tables_ori[i], hash_buckets * sizeof(uint64_t) * SHORTERN_RATIO);
	}
	return EXIT_SUCCESS;
}
