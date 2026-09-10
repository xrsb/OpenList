package huggingface

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	stdpath "path"
	"strconv"
	"strings"

	"github.com/OpenListTeam/OpenList/v4/internal/driver"
	"github.com/OpenListTeam/OpenList/v4/pkg/utils"
)

// repoPath joins the mount-root with p and returns the repo-relative path
// (no leading slash), as the Hub API expects.
func (d *HuggingFace) repoPath(p string) string {
	return strings.TrimPrefix(utils.FixAndCleanPath(stdpath.Join(d.GetRootPath(), p)), "/")
}

// apiURL returns e.g. https://huggingface.co/api/datasets/{ns}/{name}
func (d *HuggingFace) apiURL() string {
	return fmt.Sprintf("%s/api/%ss/%s", Endpoint, d.RepoType, d.RepoID)
}

func (d *HuggingFace) resolveURL(path string) string {
	prefix := repoTypeURLPrefix[d.RepoType]
	return fmt.Sprintf("%s/%s%s/resolve/%s/%s", Endpoint, prefix, d.RepoID, d.Revision, escapePath(path))
}

// escapePath encodes every segment so the result is a valid URL path.
func escapePath(p string) string {
	segs := strings.Split(strings.Trim(p, "/"), "/")
	for i, s := range segs {
		segs[i] = escapeSegment(s)
	}
	return strings.Join(segs, "/")
}

func escapeSegment(s string) string {
	var b strings.Builder
	for _, c := range s {
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9',
			c == '-', c == '_', c == '.', c == '~', c == '!', c == '*', c == '\'', c == '(', c == ')':
			b.WriteRune(c)
		default:
			b.WriteString(fmt.Sprintf("%%%02X", c))
		}
	}
	return b.String()
}

// listTree lists a folder (or recursively the whole tree) of the repo.
func (d *HuggingFace) listTree(ctx context.Context, path string, recursive bool) ([]TreeEntry, error) {
	url := d.apiURL() + "/tree/" + d.Revision
	if path != "" {
		url += "/" + escapePath(path)
	}
	var out []TreeEntry
	cursor := ""
	for {
		url := d.apiURL() + "/tree/" + d.Revision
		if path != "" {
			url += "/" + escapePath(path)
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return nil, err
		}
		if t := strings.TrimSpace(d.Token); t != "" {
			req.Header.Set("Authorization", "Bearer "+t)
		}
		q := req.URL.Query()
		q.Set("recursive", fmt.Sprintf("%v", recursive))
		q.Set("expand", "false")
		q.Set("limit", "1000")
		if cursor != "" {
			q.Set("cursor", cursor)
		}
		req.URL.RawQuery = q.Encode()
		res, err := d.client.Do(req)
		if err != nil {
			return nil, err
		}
		rb, readErr := io.ReadAll(res.Body)
		res.Body.Close()
		if readErr != nil {
			return nil, readErr
		}
		if res.StatusCode != 200 {
			return nil, fmt.Errorf("huggingface list failed: %s: %s", res.Status, string(rb))
		}
		var page []TreeEntry
		if err := json.Unmarshal(rb, &page); err != nil {
			return nil, err
		}
		out = append(out, page...)
		if len(page) < 1000 {
			break
		}
		cursor = firstNonEmpty(
			res.Header.Get("X-Next-Cursor"),
			res.Header.Get("X-Link-Cursor"),
		)
		if cursor == "" {
			break
		}
	}
	return out, nil
}

func firstNonEmpty(s ...string) string {
	for _, v := range s {
		if v != "" {
			return v
		}
	}
	return ""
}

/* ---------------- commit (ndjson) ---------------- */

type commitOp struct {
	Key   string
	Value map[string]any
}

// renameOp builds a commit op for a move/rename of one file (oldPath
// semantics of the Hub commit API). LFS entries keep their blob oid.
func renameOp(info *TreeEntry, dst string) commitOp {
	if info.LFS != nil && info.LFS.OID != "" {
		return commitOp{Key: "lfsFile", Value: map[string]any{
			"path":    dst,
			"oldPath": info.Path,
			"algo":    "sha256",
			"oid":     info.LFS.OID,
			"size":    info.LFS.Size,
		}}
	}
	return commitOp{Key: "file", Value: map[string]any{
		"path":    dst,
		"oldPath": info.Path,
	}}
}

// commitPayload serializes ops as ndjson lines with a header line.
func commitPayload(ops []commitOp, summary string) ([]byte, error) {
	var b strings.Builder
	h, _ := json.Marshal(map[string]string{"summary": summary})
	b.WriteString(`{"key":"header","value":` + string(h) + "}\n")
	for _, op := range ops {
		kb, _ := json.Marshal(op.Key)
		vb, err := json.Marshal(op.Value)
		if err != nil {
			return nil, err
		}
		b.WriteString(`{"key":` + string(kb) + `,"value":` + string(vb) + "}\n")
	}
	return []byte(b.String()), nil
}

func (d *HuggingFace) commit(ctx context.Context, ops []commitOp, summary string) error {
	if len(ops) == 0 {
		return nil
	}
	body, err := commitPayload(ops, summary)
	if err != nil {
		return err
	}
	req, err := d.newRequest(ctx, "POST", fmt.Sprintf("%s/commit/%s", d.apiURL(), d.Revision), body)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/x-ndjson")
	res, err := d.client.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		rb, _ := io.ReadAll(res.Body)
		return fmt.Errorf("huggingface commit failed: %s: %s", res.Status, string(rb))
	}
	return nil
}

// newRequest builds an authenticated request against the Hub.
func (d *HuggingFace) newRequest(ctx context.Context, method, url string, body []byte) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	if t := strings.TrimSpace(d.Token); t != "" {
		req.Header.Set("Authorization", "Bearer "+t)
	}
	req.Header.Set("User-Agent", userAgent)
	return req, nil
}

// preupload asks the Hub how a file should be uploaded. Returns "regular" or "lfs".
func (d *HuggingFace) preupload(ctx context.Context, path string, size int64, sample []byte) (string, error) {
	payload := map[string]any{
		"files": []map[string]any{{
			"path":   path,
			"size":   size,
			"sample": base64.StdEncoding.EncodeToString(sample),
		}},
	}
	b, _ := json.Marshal(payload)
	req, err := d.newRequest(ctx, "POST", d.apiURL()+"/preupload/"+d.Revision, b)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := d.client.Do(req)
	if err != nil {
		return "", err
	}
	defer res.Body.Close()
	rb, _ := io.ReadAll(res.Body)
	if res.StatusCode != 200 {
		return "", fmt.Errorf("huggingface preupload failed: %s: %s", res.Status, string(rb))
	}
	var resp struct {
		Files []struct {
			Path       string `json:"path"`
			UploadMode string `json:"uploadMode"`
		} `json:"files"`
	}
	if err := json.Unmarshal(rb, &resp); err != nil {
		return "", err
	}
	for _, f := range resp.Files {
		if f.Path == path {
			return f.UploadMode, nil
		}
	}
	return "", fmt.Errorf("huggingface preupload: file %q not in response", path)
}

// lfsBatch requests git-lfs upload actions for a single object. We advertise
// both "basic" and "multipart" transfers: the Hub answers "basic" for files
// under its ~5GB limit and "multipart" (chunked presigned uploads) above it.
func (d *HuggingFace) lfsBatch(ctx context.Context, oid string, size int64) (map[string]LFSAction, error) {
	prefix := repoTypeURLPrefix[d.RepoType]
	url := fmt.Sprintf("%s/%s%s.git/info/lfs/objects/batch", Endpoint, prefix, d.RepoID)
	payload := map[string]any{
		"operation": "upload",
		"transfers": []string{"basic", "multipart"},
		"objects": []map[string]any{{
			"oid":  oid,
			"size": size,
		}},
	}
	body, _ := json.Marshal(payload)
	req, err := d.newRequest(ctx, "POST", url, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/vnd.git-lfs+json")
	req.Header.Set("Accept", "application/vnd.git-lfs+json")
	res, err := d.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	rb, _ := io.ReadAll(res.Body)
	if res.StatusCode >= 400 {
		return nil, fmt.Errorf("huggingface lfs batch failed: %s: %s", res.Status, string(rb))
	}
	var resp struct {
		Objects []struct {
			OID     string               `json:"oid"`
			Actions map[string]LFSAction `json:"actions"`
			Error   *LFSBatchError       `json:"error"`
		} `json:"objects"`
	}
	if err := json.Unmarshal(rb, &resp); err != nil {
		return nil, err
	}
	for _, o := range resp.Objects {
		if o.OID == oid {
			if o.Error != nil {
				return nil, fmt.Errorf("lfs batch error: %s", o.Error.Message)
			}
			return o.Actions, nil
		}
	}
	return nil, fmt.Errorf("lfs batch: oid %s not in response", oid)
}

func (d *HuggingFace) lfsUpload(ctx context.Context, action LFSAction, r io.Reader, size int64, up driver.UpdateProgress) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, action.Href, r)
	if err != nil {
		return err
	}
	req.ContentLength = size
	for k, v := range action.Header {
		req.Header.Set(k, v)
	}
	req.Header.Set("Content-Type", "application/octet-stream")
	req.Header.Set("User-Agent", userAgent)
	res, err := d.client.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if up != nil {
		up(1.0)
	}
	if res.StatusCode != 200 {
		rb, _ := io.ReadAll(res.Body)
		return fmt.Errorf("huggingface lfs upload failed: %s: %s", res.Status, string(rb))
	}
	return nil
}

// isMultipartAction reports whether the batch response selected the chunked
// upload protocol (files above the Hub's single-part limit).
func isMultipartAction(act LFSAction) bool {
	_, ok := act.Header["chunk_size"]
	return ok
}

// lfsUploadMultipart uploads a large object in presigned chunks, then
// completes the multipart session. Mirrors huggingface_hub's
// lfs-multipart-upload transfer agent (HfApi) — each part gets an ETag
// that is reported back in the completion call.
func (d *HuggingFace) lfsUploadMultipart(ctx context.Context, oid string, act LFSAction, tmp *os.File, size int64, up driver.UpdateProgress) error {
	chunkSize, err := strconv.ParseInt(act.Header["chunk_size"], 10, 64)
	if err != nil || chunkSize <= 0 {
		return fmt.Errorf("huggingface: invalid multipart chunk_size %q", act.Header["chunk_size"])
	}
	var partURLs []string
	for i := 1; ; i++ {
		u, ok := act.Header[fmt.Sprintf("%05d", i)]
		if !ok {
			break
		}
		partURLs = append(partURLs, u)
	}
	if len(partURLs) == 0 {
		return fmt.Errorf("huggingface: multipart batch returned no part urls")
	}
	parts := make([]map[string]any, 0, len(partURLs))
	for i, u := range partURLs {
		start := int64(i) * chunkSize
		if start >= size {
			break
		}
		end := start + chunkSize
		if end > size {
			end = size
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPut, u, io.NewSectionReader(tmp, start, end-start))
		if err != nil {
			return err
		}
		req.ContentLength = end - start
		req.Header.Set("User-Agent", userAgent)
		res, err := d.client.Do(req)
		if err != nil {
			return err
		}
		if res.StatusCode != 200 && res.StatusCode != 201 {
			rb, _ := io.ReadAll(res.Body)
			res.Body.Close()
			return fmt.Errorf("huggingface multipart part %d failed: %s: %s", i+1, res.Status, string(rb))
		}
		etag := res.Header.Get("ETag")
		res.Body.Close()
		parts = append(parts, map[string]any{"etag": etag, "partNumber": i + 1})
		if up != nil {
			up(float64(end) / float64(size))
		}
	}
	body, _ := json.Marshal(map[string]any{"oid": oid, "parts": parts})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, act.Href, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", userAgent)
	res, err := d.client.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if res.StatusCode != 200 && res.StatusCode != 201 {
		rb, _ := io.ReadAll(res.Body)
		return fmt.Errorf("huggingface multipart complete failed: %s: %s", res.Status, string(rb))
	}
	return nil
}

// pathsInfo fetches metadata (incl. LFS info) for one repo path.
func (d *HuggingFace) pathsInfo(ctx context.Context, path string) (*TreeEntry, error) {
	infos, err := d.pathsInfoMany(ctx, []string{path})
	if err != nil {
		return nil, err
	}
	for _, e := range infos {
		if e.Path == path {
			return e, nil
		}
	}
	// the path does not exist in this revision; callers decide the error
	return nil, nil
}

// pathsInfoMany fetches metadata (incl. LFS info) for many repo paths at once.
func (d *HuggingFace) pathsInfoMany(ctx context.Context, paths []string) ([]*TreeEntry, error) {
	if len(paths) == 0 {
		return nil, nil
	}
	payload := map[string]any{"paths": paths, "expand": true}
	body, _ := json.Marshal(payload)
	req, err := d.newRequest(ctx, "POST", d.apiURL()+"/paths-info/"+d.Revision, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := d.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	rb, _ := io.ReadAll(res.Body)
	if res.StatusCode == 404 {
		// the Hub reports 404 for paths that do not exist in this revision
		return nil, nil
	}
	if res.StatusCode != 200 {
		return nil, fmt.Errorf("huggingface paths-info failed: %s: %s", res.Status, string(rb))
	}
	var entries []TreeEntry
	if err := json.Unmarshal(rb, &entries); err != nil {
		return nil, err
	}
	out := make([]*TreeEntry, 0, len(entries))
	for i := range entries {
		out = append(out, &entries[i])
	}
	return out, nil
}
