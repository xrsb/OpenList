package huggingface

import (
	"github.com/OpenListTeam/OpenList/v4/internal/driver"
	"github.com/OpenListTeam/OpenList/v4/internal/op"
)

type Addition struct {
	driver.RootPath
	RepoType string `json:"repo_type" type:"select" options:"model,dataset,space" default:"model" help:"Repository type on the Hub."`
	RepoID   string `json:"repo_id" type:"string" required:"true" help:"Namespace and repository name, e.g. username/my-repo"`
	Revision string `json:"revision" type:"string" default:"main" help:"Branch name, tag or commit SHA to mount."`
	Token    string `json:"token" type:"string" help:"User Access Token with write permission. Required for private repos and for any write operation."`
}

var config = driver.Config{
	Name:        "Hugging Face Hub",
	LocalSort:   true,
	DefaultRoot: "/",
	// downloads of private repos need the Authorization header, so prefer the
	// OpenList proxy to attach it; public links still work through it.
	PreferProxy: true,
}

func init() {
	op.RegisterDriver(func() driver.Driver {
		return &HuggingFace{}
	})
}
