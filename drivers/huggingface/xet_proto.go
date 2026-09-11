package huggingface

// Xet protocol primitives as pure functions: gearhash chunking, xorb and
// shard serialization. Every layout was verified against the official Xet
// docs (https://huggingface.co/docs/xet, /tmp/xet-shard.md) and the official
// reference shard (xet-team/xet-spec-reference-files), and round-tripped
// live through cas-server.xethub.hf.co. See xet_hash.go for hashing and
// xet_blake3.go for the vendored BLAKE3.

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
)

const (
	xetChunkMin  = 8 * 1024
	xetChunkMax  = 128 * 1024
	xetChunkMask = 0xFFFF000000000000
)

// xetChunker cuts a stream into content-defined chunks (gearhash CDC).
type xetChunker struct {
	r     io.Reader
	h     uint64
	carry []byte
}

func newXetChunker(r io.Reader) *xetChunker {
	return &xetChunker{r: r}
}

// Next returns the next chunk (freshly allocated) or io.EOF.
func (c *xetChunker) Next() ([]byte, error) {
	for {
		if cut := c.scan(); cut > 0 {
			chunk := make([]byte, cut)
			copy(chunk, c.carry[:cut])
			c.carry = append(c.carry[:0], c.carry[cut:]...)
			return chunk, nil
		}
		buf := make([]byte, 1<<20)
		n, err := c.r.Read(buf)
		if n > 0 {
			c.carry = append(c.carry, buf[:n]...)
			continue
		}
		if err == io.EOF {
			if len(c.carry) > 0 {
				chunk := make([]byte, len(c.carry))
				copy(chunk, c.carry)
				c.carry = c.carry[:0]
				return chunk, nil
			}
			return nil, io.EOF
		}
		if err != nil {
			return nil, err
		}
	}
}

// scan advances the rolling hash over carry and returns the cut length, or 0.
func (c *xetChunker) scan() int {
	h := c.h
	for i := range c.carry {
		h = (h << 1) + xetGearTable[c.carry[i]]
		k := i + 1
		if k >= xetChunkMax || (k >= xetChunkMin && h&xetChunkMask == 0) {
			c.h = h
			return k
		}
	}
	c.h = h
	return 0
}

// ---------- xorb ----------

type xetChunkData struct {
	hash [32]byte
	data []byte
}

type xetXorbMeta struct {
	hash      [32]byte
	pairs     []xetHashSize // chunks in order
	diskBytes int64         // serialized xorb size
}

// serializeXorb renders the on-disk xorb layout: one 8-byte header per chunk
// (version 0, 3-byte LE compressed size, 0 = no compression, 3-byte LE raw
// size) followed by the chunk data. It returns the bytes and the xorb hash
// (aggregate of the chunk hashes).
func serializeXorb(chunks []xetChunkData) ([]byte, [32]byte, error) {
	var b bytes.Buffer
	b.Grow(len(chunks)*8 + len(chunks)<<17)
	var pairs []xetHashSize
	for _, c := range chunks {
		sz := len(c.data)
		if sz > 0xFFFFFF {
			return nil, [32]byte{}, fmt.Errorf("huggingface: chunk too large for xorb header: %d", sz)
		}
		hdr := [8]byte{0}
		hdr[1] = byte(sz)
		hdr[2] = byte(sz >> 8)
		hdr[3] = byte(sz >> 16)
		hdr[5] = byte(sz)
		hdr[6] = byte(sz >> 8)
		hdr[7] = byte(sz >> 16)
		b.Write(hdr[:])
		b.Write(c.data)
		pairs = append(pairs, xetHashSize{hash: c.hash, size: int64(sz)})
	}
	return b.Bytes(), xetAggregatedHash(pairs), nil
}

// ---------- shard ----------

const xetShardVersion = 2

// MDB_FILE_FLAG_WITH_VERIFICATION: file section includes one FileVerificationEntry
// per FileDataSequenceEntry.
const xetShardFlags = 0x80000000

// buildShard serializes the upload shard for a single file. Layout (verified
// against the official reference shard and xet.shard format spec):
//
//	header:  tag(32) + version u64 + footer_size u64         48 B
//	file:    FileDataSequenceHeader(48)                      48 B
//	         (hash32 + flags u32 + num_entries u32 + unused 8)
//	         FileDataSequenceEntry × N (48 each)
//	         FileVerificationEntry × N (48 each)
//	         bookend(48)
//	cas:     CASChunkSequenceHeader(48) per xorb (sorted by xorb hash)
//	         (hash32 + flags u32 + num_entries u32
//	          + num_bytes_in_cas u32 + num_bytes_on_disk u32)
//	         CASChunkSequenceEntry × num_entries (48 each)
//	         bookend(48)
//
// No footer is appended (footer MUST NOT be included on upload).
func buildShard(allPairs []xetHashSize, xorbs []xetXorbMeta) []byte {
	var b bytes.Buffer

	// ---- header ----
	b.Write(xetMDBTag[:])
	var hv [16]byte // version u64, footer_size u64
	binary.LittleEndian.PutUint64(hv[0:8], xetShardVersion)
	b.Write(hv[:])

	// ---- file info section ----
	fhVal := xetFileHash(allPairs)
	b.Write(fhVal[:])
	var fh [16]byte // flags u32, num_entries u32, unused 8
	binary.LittleEndian.PutUint32(fh[0:4], xetShardFlags)
	binary.LittleEndian.PutUint32(fh[4:8], uint32(len(xorbs)))
	b.Write(fh[:])

	// 1. All FileDataSequenceEntries first
	for _, x := range xorbs {
		var segBytes uint64
		for _, p := range x.pairs {
			segBytes += uint64(p.size)
		}
		var e [48]byte // hash32 + cas_flags u32 + unpacked u32 + start u32 + end u32
		copy(e[0:32], x.hash[:])
		binary.LittleEndian.PutUint32(e[36:40], uint32(segBytes))
		binary.LittleEndian.PutUint32(e[44:48], uint32(len(x.pairs))) // chunk_index_end
		b.Write(e[:])
	}

	// 2. All FileVerificationEntries second (one per FileDataSequenceEntry)
	for _, x := range xorbs {
		var ve [48]byte // range_hash(32) + zeros(16)
		rv := xetRangeHash(x.pairs)
		copy(ve[0:32], rv[:])
		b.Write(ve[:])
	}
	b.Write(xetBookmark())

	// ---- CAS info section ----
	for _, x := range xorbs {
		var casBytes uint64
		for _, p := range x.pairs {
			casBytes += uint64(p.size)
		}
		var h [48]byte
		copy(h[0:32], x.hash[:])
		binary.LittleEndian.PutUint32(h[36:40], uint32(len(x.pairs))) // num_entries
		binary.LittleEndian.PutUint32(h[40:44], uint32(casBytes))     // num_bytes_in_cas
		binary.LittleEndian.PutUint32(h[44:48], uint32(x.diskBytes))  // num_bytes_on_disk
		b.Write(h[:])
		var acc uint32
		for _, p := range x.pairs {
			var e [48]byte
			copy(e[0:32], p.hash[:])
			binary.LittleEndian.PutUint32(e[32:36], acc) // chunk_byte_range_start
			binary.LittleEndian.PutUint32(e[36:40], uint32(p.size))
			b.Write(e[:])
			acc += uint32(p.size)
		}
	}
	b.Write(xetBookmark())

	return b.Bytes()
}

// xetBookmark returns the bookmark: 32 bytes of 0xFF followed by 16 zeros.
func xetBookmark() []byte {
	book := make([]byte, 48)
	for i := 0; i < 32; i++ {
		book[i] = 0xFF
	}
	return book
}
