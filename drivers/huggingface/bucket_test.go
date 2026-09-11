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
	"strings"
	"testing"
	"time"

	"github.com/OpenListTeam/OpenList/v4/internal/model"
)

// Real Storage Bucket tests on the Hugging Face Hub. They create a throwaway
// private bucket, exercise the whole lifecycle over the native Xet upload
// pipeline and delete the bucket afterwards.
//
// Run with:
//
//	go test ./drivers/huggingface -tags integration -run TestBucket -v
//
// Requires HF_TOKEN with bucket (write) permission.

func createTestBucket(t *testing.T, token string) string {
	t.Helper()
	name := fmt.Sprintf("ol-bucket-test-%d", time.Now().UnixNano()%100000000)

	// bucket endpoint is /api/buckets/{owner}/{name}
	var who struct {
		Name string `json:"name"`
	}
	req, _ := http.NewRequest(http.MethodGet, Endpoint+"/api/whoami-v2", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	rb, _ := io.ReadAll(res.Body)
	res.Body.Close()
	if res.StatusCode >= 400 {
		t.Fatalf("whoami failed: %s", res.Status)
	}
	if err := json.Unmarshal(rb, &who); err != nil {
		t.Fatalf("whoami decode: %v", err)
	}

	body := strings.NewReader(`{"private":true}`)
	req, err = http.NewRequest(http.MethodPost, Endpoint+"/api/buckets/"+who.Name+"/"+name, body)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")
	res, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	rb, _ = io.ReadAll(res.Body)
	if res.StatusCode >= 400 {
		t.Fatalf("create bucket failed: %s: %s", res.Status, string(rb))
	}
	t.Cleanup(func() { deleteTestBucket(t, token, name) })
	return who.Name + "/" + name
}

func deleteTestBucket(t *testing.T, token, name string) {
	t.Helper()
	req, _ := http.NewRequest(http.MethodDelete, Endpoint+"/api/buckets/"+name, nil)
	req.Header.Set("Authorization", "Bearer "+token)
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Logf("cleanup: %v", err)
		return
	}
	res.Body.Close()
}

func bucketDriver(t *testing.T, token, name string) *HuggingFace {
	t.Helper()
	d := &HuggingFace{}
	d.Addition.RepoType = "bucket"
	d.Addition.RepoID = name
	d.Addition.Token = token
	return d
}

// downloadURL fetches the file and returns its bytes.
func downloadBucketURL(t *testing.T, d *HuggingFace, url string) []byte {
	t.Helper()
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer "+d.Addition.Token)
	res, err := d.client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	b, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	if res.StatusCode >= 400 {
		t.Fatalf("download %s: %s: %s", url, res.Status, b)
	}
	return b
}

// bucketDownload fetches a file through the driver's Link URL and returns the
// exact bytes.
func bucketDownload(t *testing.T, ctx context.Context, d *HuggingFace, name string) []byte {
	t.Helper()
	obj, err := d.Get(ctx, "/"+name)
	if err != nil {
		t.Fatalf("Get %s: %v", name, err)
	}
	link, err := d.Link(ctx, obj, model.LinkArgs{})
	if err != nil {
		t.Fatalf("Link %s: %v", name, err)
	}
	return downloadBucketURL(t, d, link.URL)
}

func TestBucketLifecycle(t *testing.T) {
	ctx := context.Background()
	token := writeToken(t)
	name := createTestBucket(t, token)
	d := bucketDriver(t, token, name)
	if err := d.Init(ctx); err != nil {
		t.Fatalf("Init: %v", err)
	}

	root := &model.Object{Name: "/", IsFolder: true, Path: "/"}

	// empty listing
	objs, err := d.List(ctx, root, model.ListArgs{})
	if err != nil {
		t.Fatalf("List empty: %v", err)
	}
	if len(objs) != 0 {
		t.Fatalf("new bucket not empty: %v", objs)
	}

	// small single-chunk file
	small := []byte("hello storage bucket!\nproject.OpenList xet\n")
	if _, err := d.Put(ctx, root, makeStream(t, ctx, "hello.txt", small), nil); err != nil {
		t.Fatalf("Put small: %v", err)
	}
	if got := bucketDownload(t, ctx, d, "hello.txt"); !bytes.Equal(got, small) {
		t.Fatalf("small download mismatch: %d bytes", len(got))
	}

	// multi-chunk file (>64KiB, several chunks)
	mid := make([]byte, 3<<20) // 3 MiB, ~48 chunks across ~2 xorbs
	if _, err := rand.Read(mid); err != nil {
		t.Fatal(err)
	}
	if _, err := d.Put(ctx, root, makeStream(t, ctx, "mid.bin", mid), nil); err != nil {
		t.Fatalf("Put mid: %v", err)
	}
	if got := bucketDownload(t, ctx, d, "mid.bin"); !bytes.Equal(got, mid) {
		t.Fatalf("mid download mismatch: %d bytes", len(got))
	}

	// nested paths
	if _, err := d.Put(ctx, root, makeStream(t, ctx, "a/b/c.txt", []byte("nested")), nil); err != nil {
		t.Fatalf("Put nested: %v", err)
	}
	sub, err := d.List(ctx, root, model.ListArgs{})
	if err != nil {
		t.Fatalf("List root: %v", err)
	}
	names := map[string]bool{}
	for _, o := range sub {
		names[o.GetName()] = true
	}
	for _, want := range []string{"hello.txt", "mid.bin", "a"} {
		if !names[want] {
			t.Errorf("root listing missing %q (got %v)", want, names)
		}
	}
	if objs, err := d.List(ctx, &model.Object{Name: "a", IsFolder: true, Path: "/a"}, model.ListArgs{}); err != nil {
		t.Fatalf("List a: %v", err)
	} else if len(objs) != 1 || objs[0].GetName() != "b" {
		t.Fatalf("List a = %v", objs)
	}

	// overwrite in place (mutable bucket semantic)
	if _, err := d.Put(ctx, root, makeStream(t, ctx, "hello.txt", []byte("overwritten")), nil); err != nil {
		t.Fatalf("Put overwrite: %v", err)
	}
	if got := bucketDownload(t, ctx, d, "hello.txt"); string(got) != "overwritten" {
		t.Fatalf("overwrite failed: %q", got)
	}

	// copy file, move file, rename file
	hi, err := d.Get(ctx, "/hello.txt")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := d.Copy(ctx, hi, root); err != nil {
		t.Fatalf("Copy hello.txt: %v", err)
	}
	if _, err := d.Move(ctx, hi, &model.Object{Name: "a", IsFolder: true, Path: "/a"}); err != nil {
		t.Fatalf("Move hello.txt→a/: %v", err)
	}
	ren, err := d.Get(ctx, "/a/hello.txt")
	if err != nil {
		t.Fatalf("Get moved: %v", err)
	}
	if _, err := d.Rename(ctx, ren, "renamed.txt"); err != nil {
		t.Fatalf("Rename: %v", err)
	}
	if got := bucketDownload(t, ctx, d, "a/renamed.txt"); string(got) != "overwritten" {
		t.Fatalf("copy/move/rename chain broke: %q", got)
	}

	// remove a file and a directory
	midObj, err := d.Get(ctx, "/mid.bin")
	if err != nil {
		t.Fatal(err)
	}
	if err := d.Remove(ctx, midObj); err != nil {
		t.Fatalf("Remove mid.bin: %v", err)
	}
	if _, err := d.Get(ctx, "/mid.bin"); err == nil {
		t.Fatalf("Get removed file succeeded")
	}
	aObj, err := d.Get(ctx, "/a")
	if err != nil {
		t.Fatal(err)
	}
	if err := d.Remove(ctx, aObj); err != nil {
		t.Fatalf("Remove dir a: %v", err)
	}
	if _, err := d.Get(ctx, "/a/b/c.txt"); err == nil {
		t.Fatalf("nested file survived dir removal")
	}

	// GetDetails (used space grows with the files we uploaded; overwritten +
	// removed files shrink again, so just check it holds the big file)
	det, err := d.GetDetails(ctx)
	if err != nil {
		t.Fatalf("GetDetails: %v", err)
	}
	if det.TotalSpace <= 0 || det.UsedSpace < int64(len(mid)) {
		t.Fatalf("GetDetails = %+v", det)
	}
}

func TestBucketPathsInfo(t *testing.T) {
	ctx := context.Background()
	token := writeToken(t)
	name := createTestBucket(t, token)
	d := bucketDriver(t, token, name)
	if err := d.Init(ctx); err != nil {
		t.Fatal(err)
	}
	root := &model.Object{Name: "/", IsFolder: true, Path: "/"}
	if _, err := d.Put(ctx, root, makeStream(t, ctx, "x/y.bin", []byte("xy")), nil); err != nil {
		t.Fatal(err)
	}
	got, err := d.Get(ctx, "/x/y.bin")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got.GetSize() != 2 {
		t.Fatalf("size=%d", got.GetSize())
	}
}

// TestBucketLinkSignedRedirect locks the download-link contract: for a
// private bucket, Link must exchange our token for the Hub's signed CDN
// redirect so browsers can fetch the file with NO Authorization header
// (OpenList web_proxy stays off; no server-relayed traffic).
func TestBucketLinkSignedRedirect(t *testing.T) {
	ctx := context.Background()
	token := writeToken(t)
	name := createTestBucket(t, token)
	d := bucketDriver(t, token, name)
	if err := d.Init(ctx); err != nil {
		t.Fatal(err)
	}
	root := &model.Object{Name: "/", IsFolder: true, Path: "/"}
	payload := []byte("signed-redirect-check: " + name)
	if _, err := d.Put(ctx, root, makeStream(t, ctx, "dl.bin", payload), nil); err != nil {
		t.Fatal(err)
	}
	obj, err := d.Get(ctx, "/dl.bin")
	if err != nil {
		t.Fatal(err)
	}
	link, err := d.Link(ctx, obj, model.LinkArgs{})
	if err != nil {
		t.Fatal(err)
	}
	if link.URL == "" || strings.Contains(link.URL, "/resolve/") {
		t.Fatalf("Link URL not a signed CDN redirect: %s", link.URL)
	}
	if strings.HasPrefix(link.URL, Endpoint) && !strings.Contains(link.URL, "cdn") {
		t.Fatalf("Link URL still points at the Hub (would need auth): %s", link.URL)
	}
	// fetch without any Authorization header, exactly like a browser would
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, link.URL, nil)
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := io.ReadAll(res.Body)
	res.Body.Close()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("signed CDN fetch: %s: %s", res.Status, b)
	}
	if !bytes.Equal(b, payload) {
		t.Fatalf("payload mismatch via signed link: %d bytes", len(b))
	}
}

// silence unused-import guard (rand used in this file's plans)
var _ = rand.Read
