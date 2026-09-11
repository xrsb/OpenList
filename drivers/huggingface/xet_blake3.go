package huggingface

// Vendored BLAKE3 (keyed mode, 32-byte digest) so the Hugging Face driver
// ships with zero third-party dependencies. The implementation follows the
// BLAKE3 reference algorithm (blake3-team/blake3, CC0/CC-BY) in its scalar
// unrolled form as used by lukechampine/blake3 (BSD-3-Clause).

import (
	"encoding/binary"
	"math/bits"
)

const (
	b3FlagChunkStart = 1 << iota
	b3FlagChunkEnd
	b3FlagParent
	b3FlagRoot
	b3FlagKeyedHash
	b3FlagDeriveKeyContext
	b3FlagDeriveKeyMaterial

	b3BlockSize = 64
	b3ChunkSize = 1024
)

var b3IV = [8]uint32{
	0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
	0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
}

func b3Words(b []byte) [16]uint32 {
	var w [16]uint32
	for i := range w {
		w[i] = binary.LittleEndian.Uint32(b[i*4:])
	}
	return w
}

func b3Bytes(w [16]uint32) [64]byte {
	var b [64]byte
	for i, v := range w {
		binary.LittleEndian.PutUint32(b[i*4:], v)
	}
	return b
}

// b3Compress runs the 7-round BLAKE3 permutation on a node and returns the
// full 16-word output.
func b3Compress(n b3Node) (out [16]uint32) {
	g := func(a, b, c, d, mx, my uint32) (uint32, uint32, uint32, uint32) {
		a += b + mx
		d = bits.RotateLeft32(d^a, -16)
		c += d
		b = bits.RotateLeft32(b^c, -12)
		a += b + my
		d = bits.RotateLeft32(d^a, -8)
		c += d
		b = bits.RotateLeft32(b^c, -7)
		return a, b, c, d
	}
	m := n.block

	// round 1 (columns then diagonals)
	s0, s4, s8, s12 := g(n.cv[0], n.cv[4], b3IV[0], uint32(n.counter), m[0], m[1])
	s1, s5, s9, s13 := g(n.cv[1], n.cv[5], b3IV[1], uint32(n.counter>>32), m[2], m[3])
	s2, s6, s10, s14 := g(n.cv[2], n.cv[6], b3IV[2], n.blockLen, m[4], m[5])
	s3, s7, s11, s15 := g(n.cv[3], n.cv[7], b3IV[3], n.flags, m[6], m[7])
	s0, s5, s10, s15 = g(s0, s5, s10, s15, m[8], m[9])
	s1, s6, s11, s12 = g(s1, s6, s11, s12, m[10], m[11])
	s2, s7, s8, s13 = g(s2, s7, s8, s13, m[12], m[13])
	s3, s4, s9, s14 = g(s3, s4, s9, s14, m[14], m[15])

	// round 2
	s0, s4, s8, s12 = g(s0, s4, s8, s12, m[2], m[6])
	s1, s5, s9, s13 = g(s1, s5, s9, s13, m[3], m[10])
	s2, s6, s10, s14 = g(s2, s6, s10, s14, m[7], m[0])
	s3, s7, s11, s15 = g(s3, s7, s11, s15, m[4], m[13])
	s0, s5, s10, s15 = g(s0, s5, s10, s15, m[1], m[11])
	s1, s6, s11, s12 = g(s1, s6, s11, s12, m[12], m[5])
	s2, s7, s8, s13 = g(s2, s7, s8, s13, m[9], m[14])
	s3, s4, s9, s14 = g(s3, s4, s9, s14, m[15], m[8])

	// round 3
	s0, s4, s8, s12 = g(s0, s4, s8, s12, m[3], m[4])
	s1, s5, s9, s13 = g(s1, s5, s9, s13, m[10], m[12])
	s2, s6, s10, s14 = g(s2, s6, s10, s14, m[13], m[2])
	s3, s7, s11, s15 = g(s3, s7, s11, s15, m[7], m[14])
	s0, s5, s10, s15 = g(s0, s5, s10, s15, m[6], m[5])
	s1, s6, s11, s12 = g(s1, s6, s11, s12, m[9], m[0])
	s2, s7, s8, s13 = g(s2, s7, s8, s13, m[11], m[15])
	s3, s4, s9, s14 = g(s3, s4, s9, s14, m[8], m[1])

	// round 4
	s0, s4, s8, s12 = g(s0, s4, s8, s12, m[10], m[7])
	s1, s5, s9, s13 = g(s1, s5, s9, s13, m[12], m[9])
	s2, s6, s10, s14 = g(s2, s6, s10, s14, m[14], m[3])
	s3, s7, s11, s15 = g(s3, s7, s11, s15, m[13], m[15])
	s0, s5, s10, s15 = g(s0, s5, s10, s15, m[4], m[0])
	s1, s6, s11, s12 = g(s1, s6, s11, s12, m[11], m[2])
	s2, s7, s8, s13 = g(s2, s7, s8, s13, m[5], m[8])
	s3, s4, s9, s14 = g(s3, s4, s9, s14, m[1], m[6])

	// round 5
	s0, s4, s8, s12 = g(s0, s4, s8, s12, m[12], m[13])
	s1, s5, s9, s13 = g(s1, s5, s9, s13, m[9], m[11])
	s2, s6, s10, s14 = g(s2, s6, s10, s14, m[15], m[10])
	s3, s7, s11, s15 = g(s3, s7, s11, s15, m[14], m[8])
	s0, s5, s10, s15 = g(s0, s5, s10, s15, m[7], m[2])
	s1, s6, s11, s12 = g(s1, s6, s11, s12, m[5], m[3])
	s2, s7, s8, s13 = g(s2, s7, s8, s13, m[0], m[1])
	s3, s4, s9, s14 = g(s3, s4, s9, s14, m[6], m[4])

	// round 6
	s0, s4, s8, s12 = g(s0, s4, s8, s12, m[9], m[14])
	s1, s5, s9, s13 = g(s1, s5, s9, s13, m[11], m[5])
	s2, s6, s10, s14 = g(s2, s6, s10, s14, m[8], m[12])
	s3, s7, s11, s15 = g(s3, s7, s11, s15, m[15], m[1])
	s0, s5, s10, s15 = g(s0, s5, s10, s15, m[13], m[3])
	s1, s6, s11, s12 = g(s1, s6, s11, s12, m[0], m[10])
	s2, s7, s8, s13 = g(s2, s7, s8, s13, m[2], m[6])
	s3, s4, s9, s14 = g(s3, s4, s9, s14, m[4], m[7])

	// round 7
	s0, s4, s8, s12 = g(s0, s4, s8, s12, m[11], m[15])
	s1, s5, s9, s13 = g(s1, s5, s9, s13, m[5], m[0])
	s2, s6, s10, s14 = g(s2, s6, s10, s14, m[1], m[9])
	s3, s7, s11, s15 = g(s3, s7, s11, s15, m[8], m[6])
	s0, s5, s10, s15 = g(s0, s5, s10, s15, m[14], m[10])
	s1, s6, s11, s12 = g(s1, s6, s11, s12, m[2], m[12])
	s2, s7, s8, s13 = g(s2, s7, s8, s13, m[3], m[4])
	s3, s4, s9, s14 = g(s3, s4, s9, s14, m[7], m[13])

	return [16]uint32{
		s0 ^ s8, s1 ^ s9, s2 ^ s10, s3 ^ s11,
		s4 ^ s12, s5 ^ s13, s6 ^ s14, s7 ^ s15,
		s8 ^ n.cv[0], s9 ^ n.cv[1], s10 ^ n.cv[2], s11 ^ n.cv[3],
		s12 ^ n.cv[4], s13 ^ n.cv[5], s14 ^ n.cv[6], s15 ^ n.cv[7],
	}
}

type b3Node struct {
	cv       [8]uint32
	block    [16]uint32
	counter  uint64
	blockLen uint32
	flags    uint32
}

func b3Chain(n b3Node) [8]uint32 {
	out := b3Compress(n)
	var cv [8]uint32
	copy(cv[:], out[:])
	return cv
}

func b3Parent(left, right [8]uint32, key *[8]uint32, flags uint32) b3Node {
	n := b3Node{cv: *key, counter: 0, blockLen: b3BlockSize, flags: flags | b3FlagParent}
	copy(n.block[:8], left[:])
	copy(n.block[8:], right[:])
	return n
}

// b3Chunk compresses one 1024-byte chunk (message tree leaf). Full blocks are
// compressed left to right; the final (possibly partial) block gets
// CHUNK_END. The returned node's output is the chunk chaining value.
func b3Chunk(data []byte, key *[8]uint32, counter uint64, flags uint32) b3Node {
	cv := *key
	start := uint32(b3FlagChunkStart)
	for len(data) > b3BlockSize {
		cv = b3Chain(b3Node{cv: cv, block: b3Words(data[:b3BlockSize]), counter: counter, blockLen: b3BlockSize, flags: flags | start})
		start = 0
		data = data[b3BlockSize:]
	}
	var tail [b3BlockSize]byte
	copy(tail[:], data)
	return b3Node{cv: cv, block: b3Words(tail[:]), counter: counter, blockLen: uint32(len(data)), flags: flags | start | b3FlagChunkEnd}
}

// b3SumKeyed returns blake3-keyed(data) truncated to 32 bytes.
func b3SumKeyed(key, data []byte) []byte {
	var kw [8]uint32
	for i := range kw {
		kw[i] = binary.LittleEndian.Uint32(key[i*4:])
	}
	var out [32]byte
	var root b3Node
	switch {
	case len(data) <= b3BlockSize:
		var block [b3BlockSize]byte
		copy(block[:], data)
		full := b3Bytes(b3Compress(b3Node{
			cv: kw, block: b3Words(block[:]), blockLen: uint32(len(data)),
			flags: b3FlagChunkStart | b3FlagChunkEnd | b3FlagKeyedHash | b3FlagRoot,
		}))
		copy(out[:], full[:32])
		return out[:]
	case len(data) <= b3ChunkSize:
		root = b3Chunk(data, &kw, 0, b3FlagKeyedHash)
		root.flags |= b3FlagRoot
	default:
		var h b3Hasher
		h.init(&kw)
		h.write(data)
		root = h.root()
	}
	full := b3Bytes(b3Compress(root))
	copy(out[:], full[:32])
	return out[:]
}

// b3Hasher builds the Merkle tree for inputs larger than one chunk.
type b3Hasher struct {
	key     *[8]uint32
	flags   uint32
	counter uint64
	stack   [64][8]uint32
	buf     [b3ChunkSize]byte
	buflen  int
}

func (h *b3Hasher) init(key *[8]uint32) {
	h.key = key
	h.flags = b3FlagKeyedHash
}

func (h *b3Hasher) has(i int) bool { return h.counter&(1<<uint(i)) != 0 }

func (h *b3Hasher) push(cv [8]uint32, height int) {
	i := height
	for h.has(i) {
		cv = b3Chain(b3Parent(h.stack[i], cv, h.key, h.flags))
		i++
	}
	h.stack[i] = cv
	h.counter += 1 << uint(height)
}

func (h *b3Hasher) write(p []byte) {
	if h.buflen > 0 {
		n := copy(h.buf[h.buflen:], p)
		h.buflen += n
		p = p[n:]
	}
	if h.buflen == len(h.buf) && len(p) > 0 {
		h.push(b3Chain(b3Chunk(h.buf[:], h.key, h.counter, h.flags)), 0)
		h.buflen = 0
	}
	if len(p) > len(h.buf) {
		rem := len(p) % len(h.buf)
		if rem == 0 {
			rem = len(h.buf)
		}
		for len(p) > rem {
			h.push(b3Chain(b3Chunk(p[:b3ChunkSize], h.key, h.counter, h.flags)), 0)
			p = p[b3ChunkSize:]
		}
	}
	n := copy(h.buf[h.buflen:], p)
	h.buflen += n
}

func (h *b3Hasher) root() b3Node {
	n := b3Chunk(h.buf[:h.buflen], h.key, h.counter, h.flags)
	for i := bits.TrailingZeros64(h.counter); i < bits.Len64(h.counter); i++ {
		if h.has(i) {
			n = b3Parent(h.stack[i], b3Chain(n), h.key, h.flags)
		}
	}
	n.flags |= b3FlagRoot
	return n
}
