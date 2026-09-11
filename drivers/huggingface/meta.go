package huggingface

import (
	"github.com/OpenListTeam/OpenList/v4/internal/driver"
	"github.com/OpenListTeam/OpenList/v4/internal/op"
)

type Addition struct {
	driver.RootPath
	RepoType string `json:"repo_type" type:"select" options:"model,dataset,space,bucket" default:"model" help:"Repository type on the Hub. Choose bucket for fast mutable object storage."`
	RepoID   string `json:"repo_id" type:"string" required:"true" help:"Namespace and repository name, e.g. username/my-repo"`
	Revision string `json:"revision" type:"string" default:"main" help:"Branch name, tag or commit SHA to mount."`
	Token    string `json:"token" type:"string" help:"User Access Token with write permission. Required for private repos and for any write operation."`
}

var config = driver.Config{
	Name:              "Hugging Face Hub",
	LocalSort:         true,
	OnlyProxy:         false,
	NoCache:           false,
	NoUpload:          false,
	NeedMs:            false,
	DefaultRoot:       "/",
	CheckStatus:       false,
	Alert:             "",
	NoOverwriteUpload: false,
	NoLinkURL:         false,
	// Link() exchanges tokens for pre-signed CDN URLs so direct downloads work
	// without relaying through the server (web_proxy can stay off).
	PreferProxy: false,
}

func init() {
	op.RegisterDriver(func() driver.Driver {
		return &HuggingFace{}
	})
}
