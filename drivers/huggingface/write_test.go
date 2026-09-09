//go:build integration

package huggingface

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/OpenListTeam/OpenList/v4/internal/model"
	"github.com/OpenListTeam/OpenList/v4/internal/stream"
)

// These tests perform real write operations on a throwaway public repo on the
// Hugging Face Hub and clean up afterwards. They require:
//   HF_TOKEN=<user access token with write permission>
//   HF_USER=<your huggingface username>
//
// Run with: go test ./drivers/huggingface -tags integration -run TestWrite -v

func writeToken(t *testing.T) string {
	t.Helper()
	tok := strings.TrimSpace(os.Getenv("HF_TOKEN"))
	if tok == "" {
		t.Skip("HF_TOKEN not set; skipping write tests")
	}
	return tok
}

func createTestRepo(t *testing.T, token string) string {
	t.Helper()
	name := fmt.Sprintf("openlist-driver-test-%d", time.Now().UnixNano()%100000000)
	body := strings.NewReader(fmt.Sprintf(`{"name":%q,"type":"model","private":true}`, name))
	req, err := http.NewRequest("POST", Endpoint+"/api/repos/create", body)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	rb, _ := io.ReadAll(res.Body)
	if res.StatusCode >= 400 {
		t.Fatalf("create repo failed: %s: %s", res.Status, string(rb))
	}
	t.Cleanup(func() { deleteTestRepo(t, token, name) })
	return name
}

func deleteTestRepo(t *testing.T, token, name string) {
	t.Helper()
	body := strings.NewReader(fmt.Sprintf(`{"name":%q,"type":"model"}`, name))
	req, err := http.NewRequest(http.MethodDelete, Endpoint+"/api/repos/delete", body)
	if err != nil {
		t.Logf("cleanup: %v", err)
		return
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Logf("cleanup: %v", err)
		return
	}
	res.Body.Close()
}

func newWriteDriver(t *testing.T, token, repo string) *HuggingFace {
	t.Helper()
	d := &HuggingFace{}
	d.Addition.RepoType = "model"
	d.Addition.RepoID = repo
	d.Addition.Revision = "main"
	d.Addition.Token = token
	return d
}

func makeStream(t *testing.T, ctx context.Context, name string, content []byte) *stream.FileStream {
	t.Helper()
	return &stream.FileStream{
		Obj:    &model.Object{Name: name, Size: int64(len(content))},
		Reader: bytes.NewReader(content),
	}
}

// userTestRepo creates a private throwaway repo and returns "username/repo".
func userTestRepo(t *testing.T, token string) string {
	t.Helper()
	req, _ := http.NewRequest("GET", Endpoint+"/api/whoami-v2", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	rb, _ := io.ReadAll(res.Body)
	if res.StatusCode >= 400 {
		t.Fatalf("whoami failed: %s: %s", res.Status, string(rb))
	}
	var who struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal(rb, &who); err != nil {
		t.Fatalf("whoami decode: %v", err)
	}
	if who.Name == "" {
		t.Fatalf("whoami returned empty name: %s", string(rb))
	}
	return who.Name + "/" + createTestRepo(t, token)
}

func TestWriteLifecycle(t *testing.T) {
	ctx := context.Background()
	token := writeToken(t)
	repo := userTestRepo(t, token)

	d := &HuggingFace{}
	d.Addition.RepoType = "model"
	d.Addition.RepoID = repo
	d.Addition.Revision = "main"
	d.Addition.Token = token
	if err := d.Init(ctx); err != nil {
		t.Fatalf("Init: %v", err)
	}

	root := &model.Object{Name: "/", IsFolder: true, Path: "/"}

	// MakeDir
	if err := d.MakeDir(ctx, root, "docs"); err != nil {
		t.Fatalf("MakeDir: %v", err)
	}

	// upload small (regular) file
	small := []byte("hello huggingface driver!\n")
	if err := d.Put(ctx, root, makeStream(t, ctx, "hello.txt", small), nil); err != nil {
		t.Fatalf("Put small: %v", err)
	}

	// upload large (LFS) file (>5MB)
	big := make([]byte, 6<<20)
	if _, err := rand.Read(big); err != nil {
		t.Fatal(err)
	}
	if err := d.Put(ctx, root, makeStream(t, ctx, "big.bin", big), nil); err != nil {
		t.Fatalf("Put big: %v", err)
	}

	// verify listing
	objs, err := d.List(ctx, root, model.ListArgs{})
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	names := map[string]bool{}
	for _, o := range objs {
		names[o.GetName()] = true
	}
	for _, want := range []string{"hello.txt", "big.bin", "docs"} {
		if !names[want] {
			t.Fatalf("missing %q after upload, listing has %v", want, names)
		}
	}

	// verify big.bin is tracked via LFS
	info, err := d.pathsInfo(ctx, "big.bin")
	if err != nil {
		t.Fatal(err)
	}
	if info.LFS == nil || info.LFS.OID == "" {
		t.Fatalf("big.bin is not LFS: %+v", info)
	}

	// download and verify the small file content via Link
	link, err := d.Link(ctx, &model.Object{Name: "hello.txt", Path: "/hello.txt"}, model.LinkArgs{})
	if err != nil {
		t.Fatal(err)
	}
	req, _ := http.NewRequest("GET", link.URL, nil)
	for k, vv := range link.Header {
		for _, v := range vv {
			req.Header.Add(k, v)
		}
	}
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	got, _ := io.ReadAll(res.Body)
	if !bytes.Equal(got, small) {
		t.Fatalf("downloaded content mismatch: %q", got)
	}

	// Rename
	if err := d.Rename(ctx, &model.Object{Name: "hello.txt", Path: "/hello.txt"}, "renamed.txt"); err != nil {
		t.Fatalf("Rename: %v", err)
	}
	// Move into docs/
	if err := d.Move(ctx, &model.Object{Name: "renamed.txt", Path: "/renamed.txt"}, &model.Object{Name: "docs", IsFolder: true, Path: "/docs"}); err != nil {
		t.Fatalf("Move: %v", err)
	}
	// Copy back to root
	if err := d.Copy(ctx, &model.Object{Name: "renamed.txt", Path: "/docs/renamed.txt"}, root); err != nil {
		t.Fatalf("Copy: %v", err)
	}

	// verify after rename/move/copy
	rootObjs, err := d.List(ctx, root, model.ListArgs{})
	if err != nil {
		t.Fatal(err)
	}
	rootNames := map[string]bool{}
	for _, o := range rootObjs {
		rootNames[o.GetName()] = true
	}
	if !rootNames["renamed.txt"] {
		t.Fatalf("renamed.txt missing at root: %v", rootNames)
	}
	if rootNames["hello.txt"] {
		t.Fatalf("hello.txt should be gone after rename: %v", rootNames)
	}
	docsObjs, err := d.List(ctx, &model.Object{Name: "docs", IsFolder: true, Path: "/docs"}, model.ListArgs{})
	if err != nil {
		t.Fatal(err)
	}
	if len(docsObjs) != 1 || docsObjs[0].GetName() != "renamed.txt" {
		t.Fatalf("docs should contain only renamed.txt: %v", docsObjs)
	}

	// Remove everything
	if err := d.Remove(ctx, &model.Object{Name: "renamed.txt", Path: "/renamed.txt"}); err != nil {
		t.Fatalf("Remove file: %v", err)
	}
	if err := d.Remove(ctx, &model.Object{Name: "big.bin", Path: "/big.bin"}); err != nil {
		t.Fatalf("Remove big: %v", err)
	}
	if err := d.Remove(ctx, &model.Object{Name: "docs", IsFolder: true, Path: "/docs"}); err != nil {
		t.Fatalf("Remove folder: %v", err)
	}
	finalList, err := d.List(ctx, root, model.ListArgs{})
	if err != nil {
		t.Fatal(err)
	}
	if len(finalList) != 0 {
		left := []string{}
		for _, o := range finalList {
			left = append(left, o.GetName())
		}
		t.Fatalf("repo should be empty after cleanup, left: %v", left)
	}
}
