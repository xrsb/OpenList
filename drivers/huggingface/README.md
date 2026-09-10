# Hugging Face Hub driver (OpenList)

Mount any Hugging Face repository — including your **personal** model /
dataset / space repos (public or private) — as a storage in OpenList.

## Configuration

| Field      | Required | Description                                                        |
|------------|----------|--------------------------------------------------------------------|
| Mount path | -       | Where the repo appears in OpenList                                 |
| Repo type  | no      | `model` (default), `dataset` or `space`                            |
| Repo ID    | **yes** | `username/repo-name` (namespace is mandatory)                      |
| Revision   | no      | Branch / tag / commit SHA, defaults to `main`                       |
| Token      | no      | HF User Access Token. Required for private repos **and for every write operation** (create, upload, rename, move, copy, delete) |

> Write access requires a token with the **write** permission; read-only
> browsing of public repos works without one.
>
> **Private repos**: enable the storage's **Web Proxy** option in OpenList. The
> driver always resolves private files with the `Authorization` header, and
> only the proxy path forwards that header — a direct 302 would leak an
> unauthenticated redirect to the client.

## Capabilities

* List / browse folders (implicit folders, `.gitkeep` placeholders are filtered)
* Direct download links (`resolve` URLs; private repos are streamed through the OpenList proxy with the `Authorization` header)
* Upload: small files via the commit API (base64), large files through the
  git-lfs batch flow (server-signed PUT + LFS commit) — same protocol as `huggingface_hub`
* Make folder, delete (file & folder), rename, move (single commit,
  server-side `oldPath` operations), copy (LFS blobs are referenced by their
  oid without re-uploading)

## Notes

* Folders are implicit on the Hub. `MakeDir` creates a `.gitkeep` placeholder
  so empty folders show up in listings; it is filtered from list results.
* This driver talks only to `https://huggingface.co`; no third-party
  dependencies were added.
* Integration tests: `go test ./drivers/huggingface -tags integration` —
  read-only tests need no token, the write lifecycle test needs `HF_TOKEN`
  (and creates + deletes a scratch private repo on your account).