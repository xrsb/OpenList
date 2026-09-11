package huggingface

// Storage Bucket ("bucket") desktop implementation for the Hugging Face
// driver. Buckets are Xet-backed, mutable, non-versioned storage; they are
// manipulated through the bucket REST API (/api/buckets/...) and the Xet
// CAS protocol for data upload. There is no .gitattributes / LFS layer and no
// revision: paths map directly to object keys.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	stdpath "path"
	"sort"
	"strings"
	"time"

	"github.com/OpenListTeam/OpenList/v4/internal/driver"
	"github.com/OpenListTeam/OpenList/v4/internal/errs"
	"github.com/OpenListTeam/OpenList/v4/internal/model"
	"github.com/OpenListTeam/OpenList/v4/pkg/utils"
	"github.com/pkg/errors"
)

// isBucket reports whether the driver instance is configured for a bucket.
func (d *HuggingFace) isBucket() bool { return d.RepoType == "bucket" }

func (d *HuggingFace) bucketAPIURL() string { return "https://huggingface.co/api/buckets/" + d.RepoID }

func (d *HuggingFace) bucketResolveURL(path string) string {
	return "https://huggingface.co/buckets/" + d.RepoID + "/resolve/" + path
}

// bucketKey normalizes a model path ("/a/b") into an object key ("a/b").
func (d *HuggingFace) bucketKey(p string) string {
	p = utils.FixAndCleanPath(p)
	return strings.TrimPrefix(p, "/")
}

// bucketEntry mirrors one entry of the bucket tree API.
type bucketEntry struct {
	Path    string `json:"path"`
	Type    string `json:"type"`
	Size    int64  `json:"size"`
	XetHash string `json:"xetHash"`
}

// bucketTree lists a bucket directory (non-recursive, collapsed).
func (d *HuggingFace) bucketTree(ctx context.Context, path string) ([]bucketEntry, error) {
	key := d.bucketKey(path)
	q := url.Values{}
	q.Set("recursive", "false")
	q.Set("limit", "1000")
	u := d.bucketAPIURL() + "/tree/" + pathEscapeWildcard(key) + "?" + q.Encode()
	req, err := d.newRequest(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	var entries []bucketEntry
	if err := doRequestJSON(d.client, req, &entries); err != nil {
		return nil, err
	}
	return entries, nil
}

// pathEscapeWildcard escapes each path segment separately, leaving slashes
// in place (the bucket tree endpoint accepts multi-segment wildcard paths).
func pathEscapeWildcard(p string) string {
	parts := strings.Split(strings.Trim(p, "/"), "/")
	for i := range parts {
		parts[i] = url.PathEscape(parts[i])
	}
	return strings.Join(parts, "/")
}

// bucketInfo mirrors GET /api/buckets/{id}.
type bucketInfo struct {
	ID         string `json:"id"`
	Name       string `json:"name"`
	Private    bool   `json:"private"`
	Size       int64  `json:"size"`
	TotalFiles int64  `json:"totalFiles"`
}

func (d *HuggingFace) bucketInfoAPI(ctx context.Context) (*bucketInfo, error) {
	req, err := d.newRequest(ctx, http.MethodGet, d.bucketAPIURL(), nil)
	if err != nil {
		return nil, err
	}
	var info bucketInfo
	if err := doRequestJSON(d.client, req, &info); err != nil {
		return nil, err
	}
	return &info, nil
}

// bucketPathsInfo returns the file info for a single path.
func (d *HuggingFace) bucketPathsInfo(ctx context.Context, path string) (*bucketEntry, error) {
	entries, err := d.bucketPathsInfoMany(ctx, []string{d.bucketKey(path)})
	if err != nil {
		return nil, err
	}
	if len(entries) == 0 {
		return nil, errs.ObjectNotFound
	}
	return &entries[0], nil
}

func (d *HuggingFace) bucketPathsInfoMany(ctx context.Context, paths []string) ([]bucketEntry, error) {
	req, err := d.newRequest(ctx, http.MethodPost, d.bucketAPIURL()+"/paths-info", nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	type pathsInfoReq struct {
		Paths []string `json:"paths"`
	}
	b, err := json.Marshal(pathsInfoReq{Paths: paths})
	if err != nil {
		return nil, err
	}
	req.Body = io.NopCloser(bytes.NewReader(b))
	req.ContentLength = int64(len(b))
	var resp []bucketEntry
	if err := doRequestJSON(d.client, req, &resp); err != nil {
		return nil, err
	}
	return resp, nil
}

// bucketBatch sends NDJSON operations to the batch endpoint. All add/copy
// operations MUST precede delete operations (API contract).
func (d *HuggingFace) bucketBatch(ctx context.Context, ops []map[string]any) error {
	for len(ops) > 0 {
		n := len(ops)
		if n > 1000 {
			n = 1000
		}
		chunk := ops[:n]
		ops = ops[n:]
		var b strings.Builder
		for _, op := range chunk {
			line, err := json.Marshal(op)
			if err != nil {
				return err
			}
			b.Write(line)
			b.WriteByte('\n')
		}
		req, err := d.newRequest(ctx, http.MethodPost, d.bucketAPIURL()+"/batch", []byte(b.String()))
		if err != nil {
			return err
		}
		req.Header.Set("Content-Type", "application/x-ndjson")
		res, err := d.client.Do(req)
		if err != nil {
			return err
		}
		res.Body.Close()
		if res.StatusCode != http.StatusOK {
			return fmt.Errorf("huggingface: bucket batch failed (status %d)", res.StatusCode)
		}
	}
	return nil
}

// bucketAddPath registers a freshly uploaded file (xetHash) at path.
func (d *HuggingFace) bucketAddPath(ctx context.Context, path string, fileHash [32]byte) error {
	op := map[string]any{
		"type":    "addFile",
		"path":    d.bucketKey(path),
		"xetHash": xetHashString(fileHash),
		"mtime":   time.Now().UnixMilli(),
	}
	return d.bucketBatch(ctx, []map[string]any{op})
}

// bucketDownload fetches the file bytes (follows the xet bridge redirect).
func (d *HuggingFace) bucketDownload(ctx context.Context, path string) ([]byte, error) {
	req, err := d.newRequest(ctx, http.MethodGet, d.bucketResolveURL(d.bucketKey(path)), nil)
	if err != nil {
		return nil, err
	}
	res, err := d.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("huggingface: bucket download %s: %s", path, res.Status)
	}
	return io.ReadAll(res.Body)
}

// ---------- driver methods (bucket mode) ----------

func (d *HuggingFace) bucketList(ctx context.Context, dir model.Obj) ([]model.Obj, error) {
	p := utils.FixAndCleanPath(dir.GetPath())
	prefix := d.bucketKey(p)
	entries, err := d.bucketTree(ctx, p)
	if err != nil {
		if isNotFoundErr(err) {
			return []model.Obj{}, nil
		}
		return nil, err
	}
	out := make([]model.Obj, 0, len(entries))
	seen := map[string]bool{}
	for _, e := range entries {
		name := e.Path
		if prefix != "" {
			name = strings.TrimPrefix(e.Path, prefix+"/")
		}
		if name == "" || strings.Contains(name, "/") || seen[name] {
			continue
		}
		seen[name] = true
		out = append(out, &model.Object{
			Name:     name,
			Size:     e.Size,
			IsFolder: e.Type == "directory",
			Path:     utils.FixAndCleanPath(stdpath.Join(dir.GetPath(), name)),
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].GetName() < out[j].GetName() })
	return out, nil
}

func (d *HuggingFace) bucketGet(ctx context.Context, path string) (model.Obj, error) {
	info, err := d.bucketPathsInfo(ctx, path)
	if err == nil && info.Type == "directory" {
		return d.dirObj(path), nil
	}
	if err == nil {
		return &model.Object{
			Name:     stdpath.Base(path),
			Size:     info.Size,
			IsFolder: false,
			Path:     utils.FixAndCleanPath(path),
		}, nil
	}
	if !errors.Is(err, errs.ObjectNotFound) && !strings.Contains(err.Error(), "404") {
		return nil, err
	}
	// paths-info does not list "virtual" directories; fall back to a subtree
	// probe: if the prefix has any children it is a directory.
	if entries, lerr := d.bucketTree(ctx, path); lerr == nil && len(entries) > 0 {
		return d.dirObj(path), nil
	}
	return nil, errs.ObjectNotFound
}

func (d *HuggingFace) bucketPut(
	ctx context.Context,
	dstDir model.Obj,
	stream model.FileStreamer,
	up driver.UpdateProgress,
) (model.Obj, error) {
	path := stdpath.Join(dstDir.GetPath(), stream.GetName())
	key := d.bucketKey(path)

	tmp, _, size, err := d.spoolToTemp(stream, stream.GetSize(), up)
	if err != nil {
		return nil, err
	}
	defer func() { tmp.Close(); os.Remove(tmp.Name()) }()

	if size == 0 {
		// empty file: no CAS round-trip needed; register the path directly.
		if err := d.bucketBatch(ctx, []map[string]any{{
			"type":    "addFile",
			"path":    key,
			"xetHash": "",
			"mtime":   time.Now().UnixMilli(),
		}}); err != nil {
			return nil, err
		}
	} else {
		if err := d.xetUpload(ctx, tmp, size, key, up); err != nil {
			return nil, err
		}
	}

	return &model.Object{
		Name:     stream.GetName(),
		Size:     size,
		IsFolder: false,
		Path:     utils.FixAndCleanPath(path),
	}, nil
}

func (d *HuggingFace) bucketRemove(ctx context.Context, obj model.Obj) error {
	// Fetch all descendant paths (recursive tree) for folders, then send a
	// delete batch. Buckets free space immediately; single-file deletes are
	// one op.
	var ops []map[string]any
	if !obj.IsDir() {
		ops = append(ops, map[string]any{"type": "deleteFile", "path": d.bucketKey(obj.GetPath())})
		return d.bucketBatch(ctx, ops)
	}

	var collect func(ctx2 context.Context, o model.Obj) error
	collect = func(ctx2 context.Context, o model.Obj) error {
		list, err := d.bucketTree(ctx2, o.GetPath())
		if err != nil {
			return err
		}
		for _, e := range list {
			child := e.Path
			if e.Type == "directory" {
				// collapsed: need to descend
				if err := collect(ctx2, &model.Object{Path: "/" + child, IsFolder: true}); err != nil {
					return err
				}
			} else {
				ops = append(ops, map[string]any{"type": "deleteFile", "path": child})
			}
		}
		return nil
	}
	if err := collect(ctx, obj); err != nil {
		return err
	}
	return d.bucketBatch(ctx, ops)
}

// bucketCopyOne copies an existing file server-side (Xet dedupes the CAS
// content; no bytes cross the wire).
func (d *HuggingFace) bucketCopyPath(ctx context.Context, srcKey, dstKey, xetHash string, size int64) error {
	// addFile registers the same hash at the new path — the CAS already
	// holds the content, so this is a metadata-only op.
	return d.bucketBatch(ctx, []map[string]any{{
		"type":    "addFile",
		"path":    dstKey,
		"xetHash": xetHash,
		"mtime":   time.Now().UnixMilli(),
	}})
}

// bucketListRecursive returns every file under a path (recursive=true).
func (d *HuggingFace) bucketListAll(ctx context.Context, path string) ([]bucketEntry, error) {
	key := d.bucketKey(path)
	u := d.bucketAPIURL() + "/tree/" + pathEscapeWildcard(key) + "?recursive=true&limit=1000"
	var all []bucketEntry
	for {
		req, err := d.newRequest(ctx, http.MethodGet, u, nil)
		if err != nil {
			return nil, err
		}
		var page []bucketEntry
		if err := doRequestJSON(d.client, req, &page); err != nil {
			return nil, err
		}
		if len(page) == 0 {
			break
		}
		all = append(all, page...)
		// The bucket tree API returns a full listing per query in one page
		// (no cursor); larger buckets are bounded by per-request limits.
		break
	}
	return all, nil
}

// ---------- copy / move / rename ----------

// bucketCopy duplicates a file or folder server-side: same xetHash is
// registered at the destination (content is already in the CAS).
func (d *HuggingFace) bucketCopy(ctx context.Context, srcObj, dstDir model.Obj) (model.Obj, error) {
	srcKey := d.bucketKey(srcObj.GetPath())
	dstPrefix := d.bucketKey(dstDir.GetPath())
	if err := d.bucketClone(ctx, srcKey, dstPrefix, srcObj.GetName()); err != nil {
		return nil, err
	}
	return d.objByName(srcObj, stdpath.Join(dstDir.GetPath(), srcObj.GetName())), nil
}

// bucketRename renames inside the same directory.
func (d *HuggingFace) bucketRename(ctx context.Context, srcObj model.Obj, newName string) (model.Obj, error) {
	srcKey := d.bucketKey(srcObj.GetPath())
	dstPrefix := d.bucketKey(stdpath.Dir(srcObj.GetPath()))
	if err := d.bucketClone(ctx, srcKey, dstPrefix, newName); err != nil {
		return nil, err
	}
	if err := d.bucketRemoveByKey(ctx, srcKey); err != nil {
		return nil, err
	}
	return d.objByName(srcObj, stdpath.Join(stdpath.Dir(srcObj.GetPath()), newName)), nil
}

// bucketMove moves across directories (clone + delete).
func (d *HuggingFace) bucketMove(ctx context.Context, srcObj, dstDir model.Obj) (model.Obj, error) {
	srcKey := d.bucketKey(srcObj.GetPath())
	dstPrefix := d.bucketKey(dstDir.GetPath())
	if err := d.bucketClone(ctx, srcKey, dstPrefix, srcObj.GetName()); err != nil {
		return nil, err
	}
	if err := d.bucketRemoveByKey(ctx, srcKey); err != nil {
		return nil, err
	}
	return d.objByName(srcObj, stdpath.Join(dstDir.GetPath(), srcObj.GetName())), nil
}

// bucketClone copies every file under srcKey to dstPrefix/topName (metadata
// only: same xetHash, content is already in the CAS).
func (d *HuggingFace) bucketClone(ctx context.Context, srcKey, dstPrefix, topName string) error {
	if srcKey == "" {
		return errors.New("huggingface: bucket clone: empty source")
	}
	entries, err := d.bucketListAll(ctx, srcKey)
	if err != nil {
		return err
	}
	var ops []map[string]any
	for _, e := range entries {
		if e.Type != "file" || e.XetHash == "" {
			continue
		}
		rel := strings.TrimPrefix(strings.TrimPrefix(e.Path, srcKey), "/")
		if rel == e.Path { // exact file match (no subtree prefix)
			rel = ""
		}
		dst := topName
		if rel != "" {
			dst = topName + "/" + rel
		}
		if dstPrefix != "" {
			dst = dstPrefix + "/" + dst
		}
		ops = append(ops, map[string]any{
			"type":    "addFile",
			"path":    dst,
			"xetHash": e.XetHash,
			"mtime":   time.Now().UnixMilli(),
		})
	}
	if len(ops) == 0 {
		return errors.New("huggingface: bucket clone: no files found under " + srcKey)
	}
	return d.bucketBatch(ctx, ops)
}

// bucketRemoveByKey deletes a key or the whole subtree under it.
func (d *HuggingFace) bucketRemoveByKey(ctx context.Context, key string) error {
	key = strings.TrimSuffix(d.bucketKey(key), "/")
	entries, err := d.bucketListAll(ctx, "/"+key)
	if err != nil {
		return err
	}
	ops := []map[string]any{{"type": "deleteFile", "path": key}}
	for _, e := range entries {
		if e.Type == "file" {
			ops = append(ops, map[string]any{"type": "deleteFile", "path": e.Path})
		}
	}
	return d.bucketBatch(ctx, ops)
}
