package huggingface

// The Xet upload pipeline: obtain a short-lived CAS token, chunk + hash the
// file, upload xorbs, upload the shard, then commit the path with a batch
// addFile operation.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/OpenListTeam/OpenList/v4/internal/driver"
	"github.com/OpenListTeam/OpenList/v4/internal/model"
)

// xetWriteToken mirrors GET /api/buckets/{owner}/{name}/xet-write-token.
type xetWriteToken struct {
	CasURL      string `json:"casUrl"`
	AccessToken string `json:"accessToken"`
	Expiration  int64  `json:"exp"`
}

func (d *HuggingFace) xetWriteToken(ctx context.Context) (*xetWriteToken, error) {
	req, err := d.newRequest(ctx, http.MethodGet, d.bucketAPIURL()+"/xet-write-token", nil)
	if err != nil {
		return nil, err
	}
	var tok xetWriteToken
	if err := doRequestJSON(d.client, req, &tok); err != nil {
		return nil, err
	}
	if tok.CasURL == "" || tok.AccessToken == "" {
		return nil, fmt.Errorf("huggingface: xet-write-token response missing casUrl/accessToken")
	}
	return &tok, nil
}

// xetUpload reads tmp (a rewindable temp file holding the full content),
// chunks it, uploads the xorbs plus shard to the CAS endpoint, then registers
// dstPath with the bucket API. Progress contract: 0-55% local read/hash,
// 55-100% network upload.
func (d *HuggingFace) xetUpload(
	ctx context.Context,
	tmp io.Reader,
	size int64,
	dstPath string,
	up driver.UpdateProgress,
) error {
	if up == nil {
		up = func(float64) {}
	}
	tok, err := d.xetWriteToken(ctx)
	if err != nil {
		return err
	}

	// phase 1: read + chunk + hash (0-55)
	pr := &progressReader{r: tmp, size: size, up: model.UpdateProgressWithRange(up, 0, 55)}
	chunker := newXetChunker(pr)

	const xorbTarget = 60 << 20 // keep serialized xorbs under the 64MiB CAS cap
	var allPairs []xetHashSize
	var xorbMetas []xetXorbMeta
	var cur []xetChunkData
	var curBytes int

	flushXorb := func() error {
		if len(cur) == 0 {
			return nil
		}
		ser, hash, err := serializeXorb(cur)
		if err != nil {
			return err
		}
		if err := uploadCASXorb(ctx, d.client, tok.CasURL, tok.AccessToken, ser, hash); err != nil {
			return err
		}
		pairs := make([]xetHashSize, len(cur))
		for i, c := range cur {
			pairs[i] = xetHashSize{hash: c.hash, size: int64(len(c.data))}
		}
		xorbMetas = append(xorbMetas, xetXorbMeta{hash: hash, pairs: pairs, diskBytes: int64(len(ser))})
		allPairs = append(allPairs, pairs...)
		cur = cur[:0]
		curBytes = 0
		return nil
	}

	for {
		chunk, err := chunker.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("huggingface: chunking: %w", err)
		}
		if len(chunk) == 0 {
			continue
		}
		if curBytes+len(chunk) > xorbTarget {
			if err := flushXorb(); err != nil {
				return err
			}
		}
		cur = append(cur, xetChunkData{hash: xetChunkHash(chunk), data: chunk})
		curBytes += len(chunk)
	}
	if err := flushXorb(); err != nil {
		return err
	}
	if len(allPairs) == 0 {
		return fmt.Errorf("huggingface: empty file cannot be uploaded with the xet protocol")
	}

	// phase 2: shard + commit
	shard := buildShard(allPairs, xorbMetas)
	if err := uploadCASShard(ctx, d.client, tok.CasURL, tok.AccessToken, shard); err != nil {
		return err
	}
	fileHash := xetFileHash(allPairs)
	if err := d.bucketAddPath(ctx, dstPath, fileHash); err != nil {
		return err
	}
	up(100)
	return nil
}

// uploadCASXorb posts one full xorb to the CAS content store.
func uploadCASXorb(ctx context.Context, client *http.Client, casURL, token string, data []byte, hash [32]byte) error {
	url := casURL + "/v1/xorbs/default/" + xetHashString(hash)
	body := bytes.NewReader(data)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, body)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/octet-stream")
	res, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("huggingface: xorb upload: %w", err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(res.Body, 512))
		return fmt.Errorf("huggingface: xorb upload failed: %s: %s", res.Status, strings.TrimSpace(string(b)))
	}
	return nil
}

// uploadCASShard posts the assembled shard.
func uploadCASShard(ctx context.Context, client *http.Client, casURL, token string, shard []byte) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, casURL+"/v1/shards", bytes.NewReader(shard))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/octet-stream")
	res, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("huggingface: shard upload: %w", err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(res.Body, 512))
		return fmt.Errorf("huggingface: shard upload failed: %s: %s", res.Status, strings.TrimSpace(string(b)))
	}
	return nil
}

// isNotFoundErr reports whether an error means the object does not exist.
func isNotFoundErr(err error) bool {
	if err == nil {
		return false
	}
	s := strings.ToLower(err.Error())
	return strings.Contains(s, "404") || strings.Contains(s, "not found") || strings.Contains(s, "does not exist")
}

// doRequestJSON runs req against client and decodes a JSON response.
func doRequestJSON(client *http.Client, req *http.Request, out any) error {
	res, err := client.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(res.Body, 512))
		return fmt.Errorf("huggingface: %s %s: %s: %s", req.Method, req.URL.Path, res.Status, strings.TrimSpace(string(b)))
	}
	return json.NewDecoder(res.Body).Decode(out)
}
