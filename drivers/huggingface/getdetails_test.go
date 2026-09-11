//go:build integration

package huggingface

import (
	"crypto/rand"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/OpenListTeam/OpenList/v4/internal/model"
	"github.com/OpenListTeam/OpenList/v4/internal/stream"
)

// randomString returns n random hex chars for unique temp repo names.
func randomString(n int) string {
	b := make([]byte, (n+1)/2)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)[:n]
}

// deleteRepo removes the remote repo (best-effort test cleanup).

// TestGetDetailsRealUpload verifies storage accounting against a real
// private repo: upload a file of known size, then GetDetails must report
// at least that many bytes used, and the total must be the 100 GB quota.
func TestGetDetailsRealUpload(t *testing.T) {
	ctx := t.Context()
	tok := strings.TrimSpace(os.Getenv("HF_TOKEN"))
	if tok == "" {
		t.Skip("HF_TOKEN not set")
	}
	d := &HuggingFace{}
	d.Addition = Addition{
		RepoType: "model",
		RepoID:   userTestRepo(t, tok),
		Revision: "main",
		Token:    tok,
	}
	if err := d.Init(ctx); err != nil {
		t.Fatalf("Init: %v", err)
	}

	const size = 2 << 20 // 2 MiB, small enough to be routed as a regular blob
	tmp := filepath.Join(t.TempDir(), "blob.bin")
	if err := os.WriteFile(tmp, make([]byte, size), 0o644); err != nil {
		t.Fatal(err)
	}
	root, err := d.GetRoot(ctx)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(tmp)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if _, err := d.Put(ctx, root, &stream.FileStream{
		Obj:    &model.Object{Name: "blob.bin", Size: size},
		Reader: f,
	}, nil); err != nil {
		t.Fatalf("Put: %v", err)
	}

	// The Hub's official accounting lags uploads by a few seconds; poll.
	deadline := time.Now().Add(60 * time.Second)
	var used int64
	var total int64
	for {
		details, err := d.GetDetails(ctx)
		if err != nil {
			t.Fatalf("GetDetails: %v", err)
		}
		used = details.UsedSpace
		total = details.TotalSpace
		if used >= size || time.Now().After(deadline) {
			break
		}
		time.Sleep(1 * time.Second)
	}
	if total != int64(100)*1000*1000*1000 {
		t.Fatalf("total = %d, want 100GB", total)
	}
	if used < size {
		t.Fatalf("used = %d, want >= %d after upload", used, size)
	}
	t.Logf("after %d-byte upload: used=%d (%.3f GiB)", size, used, float64(used)/(1<<30))
}
