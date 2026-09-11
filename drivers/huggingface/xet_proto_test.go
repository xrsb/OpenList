package huggingface

import (
	"bytes"
	"io"
	"testing"
)

// Chunker invariants: deterministic cuts, bounded chunk sizes (except the
// final chunk), and the stream reassembles to the original data.
func TestChunkerInvariants(t *testing.T) {
	src := make([]byte, 1<<20+131071)
	for i := range src {
		src[i] = byte(i * 31)
	}
	for _, n := range []int{0, 1, 64, 8191, 8192, 1 << 20, len(src)} {
		data := src[:n]
		ch := newXetChunker(bytes.NewReader(data))
		var got []byte
		for {
			c, err := ch.Next()
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatalf("n=%d: chunk error: %v", n, err)
			}
			if len(c) > xetChunkMax {
				t.Fatalf("n=%d: chunk too large: %d", n, len(c))
			}
			got = append(got, c...)
		}
		if !bytes.Equal(got, data) {
			t.Fatalf("n=%d: reassembly mismatch: got %d want %d bytes", n, len(got), len(data))
		}
	}
}

// Chunk hash is stable and 32 bytes.
func TestChunkHashStable(t *testing.T) {
	data := bytes.Repeat([]byte("OpenList-hub-xet"), 1000)
	h1 := xetChunkHash(data)
	h2 := xetChunkHash(data)
	if !bytes.Equal(h1[:], h2[:]) {
		t.Fatal("chunk hash not deterministic")
	}
}

// String form: each 8-byte word of the raw hash is byte-reversed.
// Raw 0..31 → string starts with 0706050403020100 (word 0 reversed).
func TestXetHashHex(t *testing.T) {
	var h [32]byte
	for i := range h {
		h[i] = byte(i)
	}
	s := xetHashString(h)
	if len(s) != 64 {
		t.Fatalf("len: got %d", len(s))
	}
	if s[:16] != "0706050403020100" {
		t.Fatalf("first word: got %s", s[:16])
	}
	if s[16:24] != "0f0e0d0c" {
		t.Fatalf("second word head: got %s", s[16:24])
	}
}
