package huggingface

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	stdpath "path"
	"strings"
	"time"

	"github.com/OpenListTeam/OpenList/v4/drivers/base"
	"github.com/OpenListTeam/OpenList/v4/internal/driver"
	"github.com/OpenListTeam/OpenList/v4/internal/model"
	"github.com/OpenListTeam/OpenList/v4/pkg/utils"
	"github.com/go-resty/resty/v2"
	"github.com/pkg/errors"
)

type HuggingFace struct {
	model.Storage
	Addition
	client *resty.Client
}

/* ---------- meta ---------- */

func (d *HuggingFace) Config() driver.Config {
	return config
}

func (d *HuggingFace) GetAddition() driver.Additional {
	return &d.Addition
}

func (d *HuggingFace) Init(ctx context.Context) error {
	d.RepoID = strings.TrimSpace(d.RepoID)
	if strings.Count(d.RepoID, "/") != 1 || strings.HasPrefix(d.RepoID, "/") || strings.HasSuffix(d.RepoID, "/") {
		return errors.New("repo_id must be in the form 'namespace/name'")
	}
	switch d.RepoType {
	case "model":
	case "dataset":
	case "space":
	default:
		return errors.New("repo_type must be one of model, dataset, space")
	}
	d.Revision = strings.TrimSpace(d.Revision)
	if d.Revision == "" {
		d.Revision = "main"
	}
	d.RootFolderPath = utils.FixAndCleanPath(d.RootFolderPath)
	d.client = base.NewRestyClient()
	if t := strings.TrimSpace(d.Token); t != "" {
		d.client.SetHeader("Authorization", "Bearer "+t)
	}

	// verify the repository is reachable (existence / token permission)
	res, err := d.client.R().SetContext(ctx).Get(d.apiURL())
	if err != nil {
		return errors.Wrap(err, "huggingface: cannot reach the Hub API")
	}
	switch res.StatusCode() {
	case http.StatusOK:
	case http.StatusNotFound:
		return fmt.Errorf("huggingface: repository %s (type %s) does not exist, or is private and no token was provided", d.RepoID, d.RepoType)
	case http.StatusUnauthorized, http.StatusForbidden:
		return fmt.Errorf("huggingface: access to %s denied (status %d): check token and repo visibility", d.RepoID, res.StatusCode())
	default:
		return fmt.Errorf("huggingface: unexpected status %d from the Hub API", res.StatusCode())
	}
	return nil
}

func (d *HuggingFace) Drop(ctx context.Context) error {
	return nil
}

func (d *HuggingFace) GetRoot(ctx context.Context) (model.Obj, error) {
	if d.RootFolderPath == "" {
		d.RootFolderPath = "/"
	}
	return &model.Object{
		Name:     d.RootFolderPath,
		IsFolder: true,
		Path:     d.RootFolderPath,
	}, nil
}

/* ---------- read ---------- */

func (d *HuggingFace) List(ctx context.Context, dir model.Obj, args model.ListArgs) ([]model.Obj, error) {
	entries, err := d.listTree(ctx, d.repoPath(dir.GetPath()), false)
	if err != nil {
		// an implicit Hf folder may not exist yet -> treat as empty
		if strings.Contains(err.Error(), "404") {
			return []model.Obj{}, nil
		}
		return nil, err
	}
	out := make([]model.Obj, 0, len(entries))
	for _, e := range entries {
		name := stdpath.Base(e.Path)
		if name == ".gitkeep" {
			continue
		}
		out = append(out, &model.Object{
			Name:     name,
			Size:     e.Size,
			Modified: time.Unix(0, 0).UTC(),
			Ctime:    time.Unix(0, 0).UTC(),
			IsFolder: e.Type == "directory",
			Path:     utils.FixAndCleanPath(stdpath.Join(dir.GetPath(), name)),
		})
	}
	return out, nil
}

func (d *HuggingFace) Link(ctx context.Context, file model.Obj, args model.LinkArgs) (*model.Link, error) {
	link := &model.Link{URL: d.resolveURL(d.repoPath(file.GetPath()))}
	if t := strings.TrimSpace(d.Token); t != "" {
		link.Header = http.Header{}
		link.Header.Set("Authorization", "Bearer "+t)
	}
	return link, nil
}

func (d *HuggingFace) Get(ctx context.Context, path string) (model.Obj, error) {
	info, err := d.pathsInfo(ctx, d.repoPath(path))
	if err != nil {
		return nil, err
	}
	return &model.Object{
		Name:     stdpath.Base(info.Path),
		Size:     info.Size,
		Modified: time.Unix(0, 0).UTC(),
		Ctime:    time.Unix(0, 0).UTC(),
		IsFolder: info.Type == "directory",
		Path:     utils.FixAndCleanPath(path),
	}, nil
}

/* ---------- write ---------- */

func (d *HuggingFace) MakeDir(ctx context.Context, parentDir model.Obj, dirName string) error {
	// Hub folders are implicit: write a .gitkeep placeholder so the folder
	// actually appears in listings (same trick as the GitHub driver).
	path := d.repoPath(stdpath.Join(parentDir.GetPath(), dirName, ".gitkeep"))
	return d.commit(ctx, []commitOp{{
		Key:   "file",
		Value: map[string]any{"path": path, "content": ""},
	}}, "OpenList mkdir "+path)
}

// progressReader wraps a stream and reports upload progress; safe with a nil
// callback (unlike stream.ReaderUpdatingProgress which panics on nil).
type progressReader struct {
	r    io.Reader
	size int64
	n    int64
	up   driver.UpdateProgress
}

func (p *progressReader) Read(b []byte) (int, error) {
	n, err := p.r.Read(b)
	p.n += int64(n)
	if p.up != nil && p.size > 0 {
		p.up(math.Min(100, float64(p.n)/float64(p.size)*100))
	}
	return n, err
}

func (p *progressReader) GetSize() int64 { return p.size }

// spoolToTemp streams src to a temp file, computing sha256 in the same pass.
func (d *HuggingFace) spoolToTemp(src io.Reader, size int64, up driver.UpdateProgress) (*os.File, string, int64, error) {
	tmp, err := os.CreateTemp("", "huggingface-put-*")
	if err != nil {
		return nil, "", 0, err
	}
	hasher := sha256.New()
	pr := &progressReader{r: src, size: size, up: up}
	n, err := io.Copy(tmp, io.TeeReader(pr, hasher))
	if err != nil {
		tmp.Close()
		os.Remove(tmp.Name())
		return nil, "", 0, err
	}
	if _, err := tmp.Seek(0, io.SeekStart); err != nil {
		tmp.Close()
		os.Remove(tmp.Name())
		return nil, "", 0, err
	}
	return tmp, hex.EncodeToString(hasher.Sum(nil)), n, nil
}

func (d *HuggingFace) Put(ctx context.Context, dstDir model.Obj, stream model.FileStreamer, up driver.UpdateProgress) error {
	path := d.repoPath(stdpath.Join(dstDir.GetPath(), stream.GetName()))

	tmp, oid, size, err := d.spoolToTemp(stream, stream.GetSize(), up)
	if err != nil {
		return err
	}
	defer func() { tmp.Close(); os.Remove(tmp.Name()) }()

	sample := make([]byte, 0, 512)
	if size > 0 {
		sample = make([]byte, 512)
		if _, err := io.ReadFull(tmp, sample); err != nil {
			return err
		}
		if _, err := tmp.Seek(0, io.SeekStart); err != nil {
			return err
		}
	}

	mode, err := d.preupload(ctx, path, size, sample)
	if err != nil {
		return err
	}

	switch mode {
	case "lfs":
		actions, err := d.lfsBatch(ctx, oid, size)
		if err != nil {
			return err
		}
		act, ok := actions["upload"]
		if !ok {
			return fmt.Errorf("huggingface lfs batch returned no upload action")
		}
		if _, err := tmp.Seek(0, io.SeekStart); err != nil {
			return err
		}
		if err := d.lfsUpload(ctx, act, io.NewSectionReader(tmp, 0, size), size, up); err != nil {
			return err
		}
		return d.commit(ctx, []commitOp{{
			Key: "lfsFile",
			Value: map[string]any{
				"path": path,
				"oid":  oid,
				"algo": "sha256",
				"size": size,
			},
		}}, "OpenList upload "+path)
	case "regular":
		content, err := io.ReadAll(tmp)
		if err != nil {
			return err
		}
		return d.commit(ctx, []commitOp{{
			Key: "file",
			Value: map[string]any{
				"path":     path,
				"content":  base64.StdEncoding.EncodeToString(content),
				"encoding": "base64",
			},
		}}, "OpenList upload "+path)
	default:
		return fmt.Errorf("huggingface preupload returned unknown mode %q", mode)
	}
}

func (d *HuggingFace) Remove(ctx context.Context, obj model.Obj) error {
	key := "deletedFile"
	if obj.IsDir() {
		key = "deletedFolder"
	}
	path := d.repoPath(obj.GetPath())
	return d.commit(ctx, []commitOp{{
		Key:   key,
		Value: map[string]any{"path": path},
	}}, "OpenList delete "+path)
}

func (d *HuggingFace) Rename(ctx context.Context, srcObj model.Obj, newName string) error {
	src := d.repoPath(srcObj.GetPath())
	dst := d.repoPath(stdpath.Join(stdpath.Dir(srcObj.GetPath()), newName))
	return d.moveOrRename(ctx, srcObj.IsDir(), src, dst)
}

func (d *HuggingFace) Move(ctx context.Context, srcObj, dstDir model.Obj) error {
	src := d.repoPath(srcObj.GetPath())
	dst := d.repoPath(stdpath.Join(dstDir.GetPath(), srcObj.GetName()))
	if strings.HasPrefix(dst, src+"/") {
		return errors.New("cannot move a folder into itself")
	}
	return d.moveOrRename(ctx, srcObj.IsDir(), src, dst)
}

// moveOrRename renames on the Hub via one commit; folders move every file
// within a single commit (oldPath semantics of the Hub commit API).
func (d *HuggingFace) moveOrRename(ctx context.Context, isDir bool, src, dst string) error {
	if src == dst {
		return nil
	}
	var infos []*TreeEntry
	if isDir {
		entries, err := d.listTree(ctx, src, true)
		if err != nil {
			return err
		}
		paths := make([]string, 0, len(entries))
		for _, e := range entries {
			if e.Type == "file" {
				paths = append(paths, e.Path)
			}
		}
		infos, err = d.pathsInfoMany(ctx, paths)
		if err != nil {
			return err
		}
	} else {
		info, err := d.pathsInfo(ctx, src)
		if err != nil {
			return err
		}
		if info.Type == "directory" {
			return d.moveOrRename(ctx, true, src, dst)
		}
		infos = []*TreeEntry{info}
	}

	ops := make([]commitOp, 0, len(infos))
	for _, info := range infos {
		rel := strings.TrimPrefix(info.Path, src)
		ops = append(ops, renameOp(info, dst+rel))
	}
	return d.commit(ctx, ops, "OpenList move "+src+" -> "+dst)
}

func renameOp(info *TreeEntry, newPath string) commitOp {
	if info.LFS != nil && info.LFS.OID != "" {
		return commitOp{Key: "lfsFile", Value: map[string]any{
			"path":    newPath,
			"oldPath": info.Path,
			"algo":    "sha256",
			"oid":     info.LFS.OID,
			"size":    info.LFS.Size,
		}}
	}
	return commitOp{Key: "file", Value: map[string]any{
		"path":    newPath,
		"oldPath": info.Path,
	}}
}

func (d *HuggingFace) Copy(ctx context.Context, srcObj, dstDir model.Obj) error {
	src := d.repoPath(srcObj.GetPath())
	dst := d.repoPath(stdpath.Join(dstDir.GetPath(), srcObj.GetName()))
	if strings.HasPrefix(dst, src+"/") {
		return errors.New("cannot copy a folder into itself")
	}

	var infos []*TreeEntry
	if srcObj.IsDir() {
		entries, err := d.listTree(ctx, src, true)
		if err != nil {
			return err
		}
		paths := make([]string, 0, len(entries))
		for _, e := range entries {
			if e.Type == "file" {
				paths = append(paths, e.Path)
			}
		}
		infos, err = d.pathsInfoMany(ctx, paths)
		if err != nil {
			return err
		}
	} else {
		info, err := d.pathsInfo(ctx, src)
		if err != nil {
			return err
		}
		infos = []*TreeEntry{info}
	}

	ops := make([]commitOp, 0, len(infos))
	for _, info := range infos {
		rel := strings.TrimPrefix(info.Path, src)
		target := dst + rel
		if info.LFS != nil && info.LFS.OID != "" {
			// LFS blobs are content-addressed: reuse the same oid server-side.
			ops = append(ops, commitOp{Key: "lfsFile", Value: map[string]any{
				"path": target,
				"algo": "sha256",
				"oid":  info.LFS.OID,
				"size": info.LFS.Size,
			}})
			continue
		}
		// small regular files: fetch the content and re-commit it
		content, err := d.downloadFileContent(ctx, info.Path)
		if err != nil {
			return err
		}
		ops = append(ops, commitOp{Key: "file", Value: map[string]any{
			"path":     target,
			"content":  base64.StdEncoding.EncodeToString(content),
			"encoding": "base64",
		}})
	}
	return d.commit(ctx, ops, "OpenList copy "+src+" -> "+dst)
}

// downloadFileContent fetches a file's content (used by Copy for small regular files).
func (d *HuggingFace) downloadFileContent(ctx context.Context, path string) ([]byte, error) {
	req, err := d.newRequest(ctx, http.MethodGet, d.resolveURL(path), nil)
	if err != nil {
		return nil, err
	}
	res, err := base.HttpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		return nil, fmt.Errorf("huggingface download failed: %s", res.Status)
	}
	return io.ReadAll(res.Body)
}

var (
	_ driver.Driver    = (*HuggingFace)(nil)
	_ driver.Getter    = (*HuggingFace)(nil)
	_ driver.Mkdir     = (*HuggingFace)(nil)
	_ driver.Put       = (*HuggingFace)(nil)
	_ driver.Remove    = (*HuggingFace)(nil)
	_ driver.Move      = (*HuggingFace)(nil)
	_ driver.Rename    = (*HuggingFace)(nil)
	_ driver.Copy      = (*HuggingFace)(nil)
	_ driver.GetRooter = (*HuggingFace)(nil)
)
