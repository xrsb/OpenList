package huggingface

import (
	"context"

	"github.com/OpenListTeam/OpenList/v4/internal/model"
)

// repoInfo mirrors the Hub's repo metadata endpoint
// (GET /api/{model|dataset|space}/{namespace}/{name}), which reports the
// account's own storage accounting for this repo in `usedStorage` (bytes,
// LFS blobs only — the Hub bills storage by LFS bytes).
type repoInfo struct {
	UsedStorage int64 `json:"usedStorage"`
}

// GetDetails implements driver.WithDetails so OpenList can render the
// storage usage bar for a mounted repo.
//
// Usage comes straight from the Hub's own accounting — never computed
// locally:
//   - repo mounts: GET /api/{type}/{id} -> usedStorage
//   - bucket mounts: GET /api/buckets/{id} -> size
//
// The Hub exposes no quota endpoint (free tier is 100 GB per account,
// documented publicly), so TotalSpace keeps that documented value.
func (d *HuggingFace) GetDetails(ctx context.Context) (*model.StorageDetails, error) {
	const freeQuota = int64(100) * 1000 * 1000 * 1000 // 100 GB decimal, Hub's documented free tier
	used := int64(0)
	if d.isBucket() {
		info, err := d.bucketInfoAPI(ctx)
		if err != nil {
			return nil, err
		}
		used = info.Size
	} else {
		var ri repoInfo
		req, err := d.newRequest(ctx, "GET", d.apiURL(), nil)
		if err != nil {
			return nil, err
		}
		if err := doRequestJSON(d.client, req, &ri); err != nil {
			return nil, err
		}
		used = ri.UsedStorage
	}
	return &model.StorageDetails{
		DiskUsage: model.DiskUsage{
			TotalSpace: freeQuota,
			UsedSpace:  min(used, freeQuota),
		},
	}, nil
}
