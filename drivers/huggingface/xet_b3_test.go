package huggingface

import (
	"bytes"
	"encoding/hex"
	"testing"
)

// Vectors generated with the reference python-blake3 build (the same oracle
// used to validate the Xet pipeline); key = 0x00..0x1f, data = i % 251.
func TestB3KeyedVectors(t *testing.T) {
	var key [32]byte
	for i := range key {
		key[i] = byte(i)
	}
	cases := []struct {
		n    int
		want string
	}{
		{0, "73492b19995d71cdb1e9d74decc09809eb732f1b00bc95c27cb15f9dd4d6478f"},
		{1, "d08b45c6b127ee94f3f8527a0b82a5f80be1695a0eaec6022e772c0eb95a7e8b"},
		{2, "3a771bec5c84aa7ad8c0e214a0598c1d7091113e60595bd2b6db9d4725955e6a"},
		{3, "e5326c9674055c012371eb5e26424317732fee320660bdd86d4f719edd7caa29"},
		{31, "ec50185fc5d513c0a3a6411c746d638f4762c3077a478b329809532f8eb5dd18"},
		{63, "e471df92f6f7dee100138af7da29695906b0dc34ccde2142a730dd4ebcbc09cc"},
		{64, "cfaf838ff320e0d87301dcba02b1a4bb397d65119f57403df2817a51d4025f9b"},
		{127, "85f533d496cd9c62af45735983026a3f343afd1b3d8ef440991180f9945e46d6"},
		{128, "fe43a847dfccdfa5f070664fb8b51d7b906341ff81ac4adafbf6a3ffac564def"},
		{1023, "da1f18069871512af22af9f13dc005800dfd52c55f42753b5ae718086fe2ee44"},
		{1024, "f45a9249a627fdf1fcf13c0e6376f6a9a9b2056d6e1b5693a4b119a3453665f9"},
		{1025, "82223147a9b804a0c3f9a921b8d8aee250d1a51bb76be72152e6d5e8f27349b3"},
		{2048, "636bfa717d4f9fc3e59da9b2e5cce6a2b78eb70469c0fce49da38b5419892423"},
		{3072, "66315151ac08f5cdf077f76e1b5f584a4da7b48a75036de5729be38dac835fb7"},
		{4096, "e8c6e859e0480c4b062457defd04d2f4303b6cc280a0fe080ec5c4346a171937"},
		{65536, "ca2a089711002f4987989e5fab9c11ca9940e94ee258ea062d2bcb402de11ca9"},
		{131072, "0eed93ee0e31b0d5ba7c0feaf30758ac652cf202ed65e63a380a11369e95a086"},
		{262144, "1a785033535f360e5df070101a89f2aab941e32cdf9db0cedd023022c65c3d5a"},
	}
	for _, c := range cases {
		data := make([]byte, c.n)
		for i := range data {
			data[i] = byte(i % 251)
		}
		got := b3SumKeyed(key[:], data)
		if gotHex := hex.EncodeToString(got); gotHex != c.want {
			t.Errorf("b3 keyed n=%d:\n got %s\nwant %s", c.n, gotHex, c.want)
		}
	}
}

// The empty-input keyed hash with an all-zero key (zero-size file hash).
func TestB3SumKeyedEmptyKey(t *testing.T) {
	got := b3SumKeyed(make([]byte, 32), nil)
	want, _ := hex.DecodeString("a7f91ced0533c12cd59706f2dc38c2a8c39c007ae89ab6492698778c8684c483")
	if !bytes.Equal(got, want) {
		t.Fatalf("b3SumKeyed(0, nil) = %x, want %x", got, want)
	}
}
