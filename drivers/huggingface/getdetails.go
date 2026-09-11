package huggingface

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	"github.com/OpenListTeam/OpenList/v4/internal/model"
)

// lfsBlob is one entry of the Hub's LFS accounting endpoint
// (GET .../lfs-files), which lists every object stored for the repo.
type lfsBlob struct {
	Size int64 `json:"size"`
}

// treeEntry mirrors the Hub's tree API item for size/lfs accounting.
type treeEntry struct {
	Type string `json:"type"`
	Size int64  `json:"size"`
	LFS  *struct {
		Size int64 `json:"size"`
	} `json:"lfs"`
}

// GetDetails implements driver.WithDetails so OpenList can render the
// storage usage bar for the mounted repo.
//
// The Hub has no public account-quota API, so the total is the documented
// free quota (100 GB per account, private repos); the used figure is the
// real byte count: every LFS blob stored for the repo plus the plain
// (non-LFS) blobs reachable from the current revision's tree.
func (d *HuggingFace) GetDetails(ctx context.Context) (*model.StorageDetails, error) {
	const freeQuota = int64(100) * 1000 * 1000 * 1000 // 100 GB, decimal (Hub's documented free tier)
	var used int64
	var err error
	if d.isBucket() {
		// Buckets report exact usage directly from the API.
		info, e := d.bucketInfoAPI(ctx)
		if e != nil {
			return nil, e
		}
		used = info.Size
	} else {
		used, err = d.usedStorage(ctx)
		if err != nil {
			return nil, err
		}
	}
	return &model.StorageDetails{
		DiskUsage: model.DiskUsage{
			TotalSpace: freeQuota,
			UsedSpace:  min(used, freeQuota),
		},
	}, nil
}

func (d *HuggingFace) usedStorage(ctx context.Context) (int64, error) {
	// 1) LFS blobs: the accounting endpoint returns every stored object.
	var blobs []lfsBlob
	if err := d.getJSON(ctx, d.apiURL()+"/lfs-files", &blobs); err != nil {
		return 0, err
	}
	used := int64(0)
	for _, b := range blobs {
		used += b.Size
	}
	// 2) Plain git blobs: sum the sizes of non-LFS files in the tree
	// (LFS pointer files themselves are ~130 bytes, negligible either way).
	var tree []treeEntry
	if err := d.getJSON(ctx, d.apiURL()+"/tree/"+d.Revision+"?recursive=true", &tree); err != nil {
		return 0, err
	}
	for _, e := range tree {
		if e.Type == "file" && e.LFS == nil {
			used += e.Size
		}
	}
	return used, nil
}

// getJSON performs an authenticated GET and decodes the JSON response.
func (d *HuggingFace) getJSON(ctx context.Context, url string, out any) error {
	req, err := d.newRequest(ctx, "GET", url, nil)
	if err != nil {
		return err
	}
	res, err := d.client.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		rb, _ := io.ReadAll(res.Body)
		return fmt.Errorf("huggingface get %s: %s: %s", url, res.Status, string(rb))
	}
	return json.NewDecoder(res.Body).Decode(out)
}
