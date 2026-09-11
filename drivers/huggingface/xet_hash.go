package huggingface

// Merkle / aggregate hashing rules from the Xet "hashing" spec. Each entry is
// a (hash,size) pair; strings use the DataHash "string form" (4×u64
// little-endian), because the reference implementation emits those strings in
// the internal-node hash computation.

import (
	"encoding/hex"
	"strconv"
)

// xetHashSize is one merkle entry: a chunk (or inner node) hash and its size.
type xetHashSize struct {
	hash [32]byte
	size int64
}

// xetHashString formats h the way the Xet reference implementation prints
// hashes: the raw 32 bytes grouped into 4 little-endian u64 words (each word
// byte-reversed relative to memory order).
func xetHashString(h [32]byte) string {
	b := h
	for i := 0; i < 4; i++ {
		lo, hi := i*8, i*8+7
		for j := 0; j < 4; j++ {
			b[lo+j], b[hi-j] = b[hi-j], b[lo+j]
		}
	}
	return hex.EncodeToString(b[:])
}

// xetNodeHash computes the inner-node hash of an entry sequence:
// blake3(INTERNAL_KEY, "<string-hash> : <size>\n" for each entry).
func xetNodeHash(entries []xetHashSize) [32]byte {
	sum := make([]byte, 0, 96*len(entries))
	for _, e := range entries {
		sum = append(sum, xetHashString(e.hash)...)
		sum = append(sum, ' ', ':', ' ')
		sum = append(sum, strconv.FormatInt(e.size, 10)...)
		sum = append(sum, '\n')
	}
	var h [32]byte
	copy(h[:], b3SumKeyed(xetInternalKey[:], sum))
	return h
}

// xetAggregatedHash iteratively merges pairs into a single root hash.
// Merge order follows the reference implementation's next_merge_cut: take 2
// through min(9,n) entries, stop at the first whose last u64 (little-endian)
// is divisible by 4.
func xetAggregatedHash(pairs []xetHashSize) [32]byte {
	cur := pairs
	for len(cur) > 1 {
		var next []xetHashSize
		for len(cur) > 0 {
			cut := xetMergeCut(cur)
			group, rest := cur[:cut], cur[cut:]
			node := xetNodeHash(group)
			var size int64
			for _, e := range group {
				size += e.size
			}
			next = append(next, xetHashSize{hash: node, size: size})
			cur = rest
		}
		cur = next
	}
	return cur[0].hash
}

// xetMergeCut returns how many leading pairs form the next merge group
// (reference: next_merge_cut on aggregated_hashes). The scan starts at index
// 2: the first candidate group is 3 entries, matching the reference loop
// `for i in range(2, end)` / `for i in 2..end` checking hashes[i].
func xetMergeCut(pairs []xetHashSize) int {
	if len(pairs) <= 2 {
		return len(pairs)
	}
	end := len(pairs)
	if end > 9 {
		end = 9
	}
	for i := 2; i < end; i++ {
		if xetLastWord(pairs[i].hash)%4 == 0 {
			return i + 1
		}
	}
	return end
}

// xetLastWord interprets the last 8 bytes of h as a little-endian u64.
func xetLastWord(h [32]byte) uint64 {
	var v uint64
	for i := 0; i < 8; i++ {
		v |= uint64(h[24+i]) << (8 * i)
	}
	return v
}

// xetFileHash is the file hash registered with the bucket API: the
// aggregated hash salted with an all-zero key (reference: salted_file_hash).
func xetFileHash(pairs []xetHashSize) [32]byte {
	agg := xetAggregatedHash(pairs)
	var h [32]byte
	copy(h[:], b3SumKeyed(make([]byte, 32), agg[:]))
	return h
}

// xetChunkHash hashes one raw chunk (blake3 keyed with the DATA key).
func xetChunkHash(data []byte) [32]byte {
	var h [32]byte
	copy(h[:], b3SumKeyed(xetDataKey[:], data))
	return h
}

// xetRangeHash is the per-segment verification hash: blake3 keyed by
// xetVerifyKey over the concatenated raw chunk hashes of the segment.
func xetRangeHash(pairs []xetHashSize) [32]byte {
	var sum []byte
	for _, e := range pairs {
		sum = append(sum, e.hash[:]...)
	}
	var h [32]byte
	copy(h[:], b3SumKeyed(xetVerifyKey[:], sum))
	return h
}
