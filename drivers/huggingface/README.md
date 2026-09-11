# Hugging Face Hub driver (OpenList)

Mount any Hugging Face repository — including your **personal** model /
dataset / space repos (public or private) — as a storage in OpenList.

## Configuration

| Field      | Required | Description                                                        |
|------------|----------|--------------------------------------------------------------------|
| Mount path | -       | Where the repo appears in OpenList                                 |
| Repo type  | no      | `model` (default), `dataset`, `space` or `bucket`                  |
| Repo ID    | **yes** | `username/repo-name` (namespace is mandatory)                      |
| Revision   | no      | Branch / tag / commit SHA, defaults to `main` (not used for bucket)|
| Token      | no      | HF User Access Token. Required for private repos/buckets **and for every write operation** (create, upload, rename, move, copy, delete) |

> Write access requires a token with the **write** permission; read-only
> browsing of public repos works without one.
>
> **Direct downloads**: The driver automatically exchanges authentication for pre-signed
> CDN URLs via 302 redirects, so clients download directly from the CDN without relaying
> through the OpenList server. `Web Proxy` can remain disabled.

## Capabilities

* List / browse folders (implicit folders, `.gitkeep` placeholders are filtered)
* Direct download links (pre-signed CloudFront CDN URLs via 302 redirect; direct zero-relay downloads without Web Proxy)
* Upload to Repositories (model/dataset/space): small files via the commit API (base64), large files through the git-lfs batch flow (server-signed PUT + LFS commit) — same protocol as `huggingface_hub`
* Upload to Storage Buckets: native Xet protocol with Content-Defined Chunking (Gearhash), BLAKE3 Merkle Tree hashing, CAS xorb chunk upload, and MDB Shard serialization
* Storage usage accounting via official Hub APIs (`usedStorage` for repositories, `size` for buckets)
* Make folder, delete (file & folder), rename, move (single commit / atomic server-side operations), copy (LFS blobs / objects referenced without re-uploading)

## Notes

* Folders are implicit on the Hub. `MakeDir` creates a `.gitkeep` placeholder
  so empty folders show up in listings; it is filtered from list results.
* This driver talks only to `https://huggingface.co`; no third-party
  dependencies were added.
* Integration tests: `go test ./drivers/huggingface -tags integration` —
  read-only tests need no token, the write lifecycle test needs `HF_TOKEN`
  (and creates + deletes a scratch private repo on your account).