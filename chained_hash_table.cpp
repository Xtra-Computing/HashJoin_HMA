#include "hash_table.h"
#include <stdio.h>
#include <string.h>
void Chained_Hash_Table :: Build (uint32_t inner_beg, uint32_t inner_end)
{
	int i;
	KEY_TYPE *keys = relation.keys;
	PAYLOAD_TYPE *vals = relation.vals;
	chained_hash_table = new BUCKET_HEAD[buckets];
	memset(chained_hash_table, NULL, buckets * sizeof(BUCKET_HEAD));
	for (i = inner_beg ; i != inner_end ; ++i) 
	{
		Insert(keys[i],vals[i]);
	}
}
void Chained_Hash_Table :: Insert (KEY_TYPE key, PAYLOAD_TYPE val)
{
	BUCKET *ptr;

	uint64_t h = (uint32_t) (key * factor);
	h = (h * buckets) >> 32;
	ptr=chained_hash_table[h].next;
	
	BUCKET *new_bucket = new BUCKET;
	new_bucket->key=key;
	new_bucket->val=val;
	new_bucket->next=NULL;

	while (!__sync_bool_compare_and_swap(&ptr, NULL, new_bucket))
	{
		ptr=ptr->next;
	}
}
size_t Chained_Hash_Table :: Probe (const uint32_t *keys, const uint32_t *vals, size_t size, uint32_t *keys_out, uint32_t *vals_out, uint32_t *tabs_out, 
	size_t block_size, size_t block_limit, volatile size_t *counter)
{
	uint32_t i;
	BUCKET *ptr;
	size_t o = __sync_fetch_and_add(counter, 1);
	assert(o <= block_limit);
	o *= block_size;
	for (i=0;i<size;++i) {
		uint64_t h = (uint32_t) (keys[i] * factor);		
		h = (h * buckets) >> 32;	
		ptr=chained_hash_table[h].next;		
		while (ptr!=NULL)
		{
			if (ptr->key==keys[i])
			{
				tabs_out[o] = ptr->key;
				vals_out[o] = ptr->val;
				keys_out[o++] = ptr->key;
				if ((o & (block_size - 1)) == 0) 
				{
					o = __sync_fetch_and_add(counter, 1);
					assert(o <= block_limit);
					o *= block_size;
				}
			}
			ptr=ptr->next;
		}
	}
	return 0;
}