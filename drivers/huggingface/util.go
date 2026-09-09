package huggingface

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	stdpath "path"
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

// lfsBatch requests git-lfs upload actions for a single object.
func (d *HuggingFace) lfsBatch(ctx context.Context, oid string, size int64) (map[string]LFSAction, error) {
	prefix := repoTypeURLPrefix[d.RepoType]
	url := fmt.Sprintf("%s/%s%s.git/info/lfs/objects/batch", Endpoint, prefix, d.RepoID)
	payload := map[string]any{
		"operation": "upload",
		"transfers": []string{"basic"},
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
	return nil, fmt.Errorf("path %q not found in paths-info response", path)
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
