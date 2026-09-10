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

	"github.com/OpenListTeam/OpenList/v4/internal/driver"
	"github.com/OpenListTeam/OpenList/v4/internal/errs"
	"github.com/OpenListTeam/OpenList/v4/internal/model"
	"github.com/OpenListTeam/OpenList/v4/pkg/utils"
	"github.com/pkg/errors"
)

type HuggingFace struct {
	model.Storage
	Addition
	client *http.Client
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
	case "model", "dataset", "space":
	default:
		return errors.New("repo_type must be one of model, dataset, space")
	}
	d.Revision = strings.TrimSpace(d.Revision)
	if d.Revision == "" {
		d.Revision = "main"
	}
	d.RootFolderPath = utils.FixAndCleanPath(d.RootFolderPath)
	d.client = &http.Client{Timeout: 48 * time.Hour}

	// verify the repository is reachable (existence / token permission)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, d.apiURL(), nil)
	if err != nil {
		return errors.Wrap(err, "huggingface: cannot build request")
	}
	if t := strings.TrimSpace(d.Token); t != "" {
		req.Header.Set("Authorization", "Bearer "+t)
	}
	res, err := d.client.Do(req)
	if err != nil {
		return errors.Wrap(err, "huggingface: cannot reach the Hub API")
	}
	res.Body.Close()
	switch res.StatusCode {
	case http.StatusOK:
	case http.StatusNotFound:
		return fmt.Errorf("huggingface: repository %s (type %s) does not exist, or is private and no token was provided", d.RepoID, d.RepoType)
	case http.StatusUnauthorized, http.StatusForbidden:
		return fmt.Errorf("huggingface: access to %s denied (status %d): check token and repo visibility", d.RepoID, res.StatusCode)
	default:
		return fmt.Errorf("huggingface: unexpected status %d from the Hub API", res.StatusCode)
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
		Name:     "root",
		IsFolder: true,
		Path:     d.RootFolderPath,
	}, nil
}

/* ---------- read ---------- */

func (d *HuggingFace) List(ctx context.Context, dir model.Obj, args model.ListArgs) ([]model.Obj, error) {
	entries, err := d.listTree(ctx, d.repoPath(dir.GetPath()), false)
	if err != nil {
		// an implicit Hub folder may not exist yet -> treat as empty
		if strings.Contains(err.Error(), "404") {
			return []model.Obj{}, nil
		}
		return nil, err
	}
	out := make([]model.Obj, 0, len(entries))
	for _, e := range entries {
		if e.Type != "file" && e.Type != "directory" {
			continue
		}
		name := stdpath.Base(e.Path)
		if name == ".gitkeep" || name == ".gitattributes" {
			continue
		}
		out = append(out, &model.Object{
			Name:     name,
			Size:     e.Size,
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
	if info == nil {
		// the Hub answered 404: the path does not exist in this revision
		return nil, errs.ObjectNotFound
	}
	if info.Type == "directory" {
		return d.dirObj(path), nil
	}
	return &model.Object{
		Name:     stdpath.Base(info.Path),
		Size:     info.Size,
		IsFolder: false,
		Path:     utils.FixAndCleanPath(path),
	}, nil
}

// newObj builds a model.Obj; Modified/Ctime stay zero because the Hub does not
// expose per-file times without an extra expanded tree query.
func (d *HuggingFace) dirObj(path string) model.Obj {
	return &model.Object{
		Name:     stdpath.Base(path),
		IsFolder: true,
		Path:     utils.FixAndCleanPath(path),
	}
}

/* ---------- write ---------- */

func (d *HuggingFace) MakeDir(ctx context.Context, parentDir model.Obj, dirName string) (model.Obj, error) {
	// Hub folders are implicit: write a .gitkeep placeholder so the folder
	// actually appears in listings (same trick as the GitHub driver).
	path := d.repoPath(stdpath.Join(parentDir.GetPath(), dirName, ".gitkeep"))
	if err := d.commit(ctx, []commitOp{{
		Key:   "file",
		Value: map[string]any{"path": path, "content": ""},
	}}, "openlist mkdir "+path); err != nil {
		return nil, err
	}
	return d.dirObj(stdpath.Join(parentDir.GetPath(), dirName)), nil
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

func (d *HuggingFace) Put(ctx context.Context, dstDir model.Obj, stream model.FileStreamer, up driver.UpdateProgress) (model.Obj, error) {
	path := d.repoPath(stdpath.Join(dstDir.GetPath(), stream.GetName()))

	tmp, oid, size, err := d.spoolToTemp(stream, stream.GetSize(), up)
	if err != nil {
		return nil, err
	}
	defer func() { tmp.Close(); os.Remove(tmp.Name()) }()

	sample := make([]byte, 0, 512)
	if size > 0 {
		sample, err = io.ReadAll(io.LimitReader(tmp, 512))
		if err != nil {
			return nil, err
		}
		if _, err := tmp.Seek(0, io.SeekStart); err != nil {
			return nil, err
		}
	}

	mode, err := d.preupload(ctx, path, size, sample)
	if err != nil {
		return nil, err
	}

	switch mode {
	case "lfs":
		actions, err := d.lfsBatch(ctx, oid, size)
		if err != nil {
			return nil, err
		}
		act, ok := actions["upload"]
		if !ok {
			return nil, fmt.Errorf("huggingface: lfs batch returned no upload action")
		}
		if _, err := tmp.Seek(0, io.SeekStart); err != nil {
			return nil, err
		}
		if err := d.lfsUpload(ctx, act, io.NewSectionReader(tmp, 0, size), size, up); err != nil {
			return nil, err
		}
		err = d.commit(ctx, []commitOp{{
			Key: "lfsFile",
			Value: map[string]any{
				"path": path,
				"oid":  oid,
				"algo": "sha256",
				"size": size,
			},
		}}, "openlist upload "+path)
	case "regular":
		content, err := io.ReadAll(tmp)
		if err != nil {
			return nil, err
		}
		err = d.commit(ctx, []commitOp{{
			Key: "file",
			Value: map[string]any{
				"path":     path,
				"content":  base64.StdEncoding.EncodeToString(content),
				"encoding": "base64",
			},
		}}, "openlist upload "+path)
	default:
		return nil, fmt.Errorf("huggingface: preupload returned unknown mode %q", mode)
	}
	if err != nil {
		return nil, err
	}
	return &model.Object{
		Name:     stream.GetName(),
		Size:     size,
		IsFolder: false,
		Path:     utils.FixAndCleanPath(stdpath.Join(dstDir.GetPath(), stream.GetName())),
	}, nil
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
	}}, "openlist delete "+path)
}

func (d *HuggingFace) Rename(ctx context.Context, srcObj model.Obj, newName string) (model.Obj, error) {
	src := d.repoPath(srcObj.GetPath())
	dst := d.repoPath(stdpath.Join(stdpath.Dir(srcObj.GetPath()), newName))
	if err := d.moveOrRename(ctx, srcObj.IsDir(), src, dst); err != nil {
		return nil, err
	}
	return d.objByName(srcObj, stdpath.Join(stdpath.Dir(srcObj.GetPath()), newName)), nil
}

func (d *HuggingFace) Move(ctx context.Context, srcObj, dstDir model.Obj) (model.Obj, error) {
	src := d.repoPath(srcObj.GetPath())
	dst := d.repoPath(stdpath.Join(dstDir.GetPath(), srcObj.GetName()))
	if strings.HasPrefix(dst, src+"/") {
		return nil, errors.New("cannot move a folder into itself")
	}
	if err := d.moveOrRename(ctx, srcObj.IsDir(), src, dst); err != nil {
		return nil, err
	}
	return d.objByName(srcObj, stdpath.Join(dstDir.GetPath(), srcObj.GetName())), nil
}

// moveOrRename renames on the Hub via one commit; folders move every file
// within a single commit (oldPath semantics of the Hub commit API).
func (d *HuggingFace) moveOrRename(ctx context.Context, isDir bool, src, dst string) error {
	if src == dst {
		return nil
	}
	infos, err := d.collectEntries(ctx, isDir, src)
	if err != nil {
		return err
	}
	ops := make([]commitOp, 0, len(infos))
	for _, info := range infos {
		rel := strings.TrimPrefix(info.Path, src)
		ops = append(ops, renameOp(info, dst+rel))
	}
	return d.commit(ctx, ops, "openlist move "+src)
}

func (d *HuggingFace) Copy(ctx context.Context, srcObj, dstDir model.Obj) (model.Obj, error) {
	src := d.repoPath(srcObj.GetPath())
	dst := d.repoPath(stdpath.Join(dstDir.GetPath(), srcObj.GetName()))
	if strings.HasPrefix(dst, src+"/") {
		return nil, errors.New("cannot copy a folder into itself")
	}
	infos, err := d.collectEntries(ctx, srcObj.IsDir(), src)
	if err != nil {
		return nil, err
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
			return nil, err
		}
		ops = append(ops, commitOp{Key: "file", Value: map[string]any{
			"path":     target,
			"content":  base64.StdEncoding.EncodeToString(content),
			"encoding": "base64",
		}})
	}
	if err := d.commit(ctx, ops, "openlist copy "+src); err != nil {
		return nil, err
	}
	return d.objByName(srcObj, stdpath.Join(dstDir.GetPath(), srcObj.GetName())), nil
}

// collectEntries returns metadata for src, expanding a directory to the
// files it contains (with their LFS info).
func (d *HuggingFace) collectEntries(ctx context.Context, isDir bool, src string) ([]*TreeEntry, error) {
	if !isDir {
		info, err := d.pathsInfo(ctx, src)
		if err != nil {
			return nil, err
		}
		if info == nil {
			return nil, errs.ObjectNotFound
		}
		if info.Type == "directory" {
			return d.collectEntries(ctx, true, src)
		}
		return []*TreeEntry{info}, nil
	}
	entries, err := d.listTree(ctx, src, true)
	if err != nil {
		return nil, err
	}
	paths := make([]string, 0, len(entries))
	for _, e := range entries {
		if e.Type == "file" {
			paths = append(paths, e.Path)
		}
	}
	return d.pathsInfoMany(ctx, paths)
}

// objByName mirrors srcObj's shape at a new path.
func (d *HuggingFace) objByName(src model.Obj, path string) model.Obj {
	return &model.Object{
		Name:     src.GetName(),
		Size:     src.GetSize(),
		IsFolder: src.IsDir(),
		Path:     utils.FixAndCleanPath(path),
	}
}

// downloadFileContent fetches a file's content (used by Copy for small regular files).
func (d *HuggingFace) downloadFileContent(ctx context.Context, path string) ([]byte, error) {
	req, err := d.newRequest(ctx, http.MethodGet, d.resolveURL(path), nil)
	if err != nil {
		return nil, err
	}
	res, err := d.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		return nil, fmt.Errorf("huggingface: download failed: %s", res.Status)
	}
	return io.ReadAll(res.Body)
}

var (
	_ driver.Driver       = (*HuggingFace)(nil)
	_ driver.Getter       = (*HuggingFace)(nil)
	_ driver.MkdirResult  = (*HuggingFace)(nil)
	_ driver.PutResult    = (*HuggingFace)(nil)
	_ driver.Remove       = (*HuggingFace)(nil)
	_ driver.MoveResult   = (*HuggingFace)(nil)
	_ driver.RenameResult = (*HuggingFace)(nil)
	_ driver.CopyResult   = (*HuggingFace)(nil)
	_ driver.GetRooter    = (*HuggingFace)(nil)
)
