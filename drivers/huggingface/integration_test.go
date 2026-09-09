//go:build integration

package huggingface

import (
	"context"
	"net/http"
	"os"
	"path"
	"strings"
	"testing"

	"github.com/OpenListTeam/OpenList/v4/drivers/base"
	"github.com/OpenListTeam/OpenList/v4/internal/conf"
	"github.com/OpenListTeam/OpenList/v4/internal/model"
)

func TestMain(m *testing.M) {
	conf.Conf = conf.DefaultConfig("")
	base.InitClient()
	os.Exit(m.Run())
}

// These tests hit the live Hugging Face Hub (read-only, public repo).
// Run with: go test ./drivers/huggingface -run TestPublic -v

func newTestDriver(t *testing.T) *HuggingFace {
	t.Helper()
	d := &HuggingFace{}
	d.Addition.RepoType = "model"
	d.Addition.RepoID = "openai-community/gpt2"
	d.Addition.Revision = "main"
	return d
}

func TestPublicInit(t *testing.T) {
	d := newTestDriver(t)
	if err := d.Init(context.Background()); err != nil {
		t.Fatalf("Init: %v", err)
	}
	if !strings.Contains(d.apiURL(), "/api/models/openai-community/gpt2") {
		t.Fatalf("unexpected apiURL: %s", d.apiURL())
	}
}

func TestPublicList(t *testing.T) {
	ctx := context.Background()
	d := newTestDriver(t)
	if err := d.Init(ctx); err != nil {
		t.Fatal(err)
	}
	objs, err := d.List(ctx, &model.Object{Name: "/", IsFolder: true, Path: "/"}, model.ListArgs{})
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	names := map[string]bool{}
	for _, o := range objs {
		names[o.GetName()] = true
	}
	for _, want := range []string{"README.md", "config.json", "onnx"} {
		if !names[want] {
			t.Errorf("missing entry %s (got %d entries)", want, len(objs))
		}
	}
}

func TestPublicListSubdirAndEmptyFolder(t *testing.T) {
	ctx := context.Background()
	d := newTestDriver(t)
	if err := d.Init(ctx); err != nil {
		t.Fatal(err)
	}
	// subfolder listing
	objs, err := d.List(ctx, &model.Object{Name: "onnx", IsFolder: true, Path: "/onnx"}, model.ListArgs{})
	if err != nil || len(objs) == 0 {
		t.Fatalf("subdir List: %v (%d items)", err, len(objs))
	}
	// an existing empty folder (has no files at all) should return empty, not error
	objs, err = d.List(ctx, &model.Object{Name: "nonexistent", IsFolder: true, Path: "/nonexistent"}, model.ListArgs{})
	if err != nil {
		t.Fatalf("empty folder List returned error: %v", err)
	}
	if len(objs) != 0 {
		t.Fatalf("empty folder List returned %d items", len(objs))
	}
}

func TestPublicLink(t *testing.T) {
	ctx := context.Background()
	d := newTestDriver(t)
	if err := d.Init(ctx); err != nil {
		t.Fatal(err)
	}
	link, err := d.Link(ctx, &model.Object{Name: "config.json", Path: "/config.json"}, model.LinkArgs{})
	if err != nil {
		t.Fatalf("Link: %v", err)
	}
	if !strings.Contains(link.URL, "/openai-community/gpt2/resolve/main/config.json") {
		t.Fatalf("unexpected link URL: %s", link.URL)
	}
	resp, err := http.Get(link.URL)
	if err != nil {
		t.Fatalf("resolve GET: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("resolve status: %s", resp.Status)
	}
}

func TestPublicGet(t *testing.T) {
	ctx := context.Background()
	d := newTestDriver(t)
	if err := d.Init(ctx); err != nil {
		t.Fatal(err)
	}
	obj, err := d.Get(ctx, "/config.json")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if obj.IsDir() || obj.GetName() != "config.json" || obj.GetSize() <= 0 {
		t.Fatalf("unexpected object: %+v", obj)
	}
	if obj.GetPath() != path.Clean("/config.json") {
		t.Fatalf("unexpected path: %s", obj.GetPath())
	}
}
