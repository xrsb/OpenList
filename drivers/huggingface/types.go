package huggingface

const (
	Endpoint  = "https://huggingface.co"
	userAgent = "OpenList-HuggingFace-Driver/1.0"
)

// repoTypeURLPrefix maps a repo type to its URL prefix; models have none.
var repoTypeURLPrefix = map[string]string{
	"model":   "",
	"dataset": "datasets/",
	"space":   "spaces/",
}

type TreeEntry struct {
	Type string   `json:"type"` // "file" | "directory"
	Path string   `json:"path"`
	Size int64    `json:"size"`
	OID  string   `json:"oid"`
	LFS  *LFSInfo `json:"lfs,omitempty"`
}

// LFSInfo describes a git-lfs tracked file; the Hub exposes the blob sha256
// under "oid" (not "sha256" — that key does not exist).
type LFSInfo struct {
	OID  string `json:"oid"`
	Size int64  `json:"size"`
}

type LFSAction struct {
	Href   string            `json:"href"`
	Header map[string]string `json:"header,omitempty"`
}

type LFSBatchError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}
