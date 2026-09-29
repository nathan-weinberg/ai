# Run Codex or Claude Code through Praxis and vLLM

This guide runs a real coding client through a local Praxis gateway while vLLM
provides inference. The supported paths are:

```text
Codex       -> Praxis /v1/responses -> vLLM /v1/responses (native)
Codex       -> Praxis /v1/responses -> vLLM /v1/chat/completions (translated)
Claude Code -> Praxis /v1/messages  -> vLLM /v1/messages (native)
Claude Code -> Praxis /v1/messages  -> vLLM /v1/chat/completions (translated)
```

Use the native paths when vLLM exposes the corresponding Responses or
Anthropic-compatible API. For Codex, Praxis keeps `/v1/responses` end to end
and lowers only client-owned tool types that vLLM does not understand. Use a
translated path for a backend that only exposes OpenAI Chat Completions.

## 1. Start vLLM

### On-demand GPU endpoint

In GitHub Actions, open **vLLM Dev Endpoint**, choose **Run workflow**, and keep
the defaults for Qwen3-8B. The workflow provisions a temporary GPU, publishes a
Cloudflare URL in the run summary, and tears everything down at the requested
deadline or when the run is cancelled.

Record the two summary values:

```console
export VLLM_URL=https://example.trycloudflare.com
export VLLM_MODEL=qwen3-8b
```

The Quick Tunnel is an unauthenticated, best-effort development endpoint. Treat
its random URL as temporary sensitive data, do not use it for production, and
cancel the workflow as soon as testing is complete.

### Existing or local vLLM

If vLLM is already reachable, set the same variables to its URL and exact served
model name. A keyed local server can be started with:

```console
export VLLM_API_KEY="$(openssl rand -hex 32)"
vllm serve Qwen/Qwen3-8B \
  --served-model-name qwen3-8b \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser deepseek_r1 \
  --gpu-memory-utilization 0.97 \
  --enforce-eager \
  --api-key "$VLLM_API_KEY"
```

The same server can be run from the official
[`vllm/vllm-openai`](https://hub.docker.com/r/vllm/vllm-openai) image. Its
entrypoint already starts the server, so the arguments are the `vllm serve`
flags used above, and vLLM reads the backend key from `VLLM_API_KEY` in the
container environment instead of the command line:

```console
export VLLM_API_KEY="$(openssl rand -hex 32)"
docker run --rm --name vllm \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e VLLM_API_KEY \
  docker.io/vllm/vllm-openai:latest \
  --model Qwen/Qwen3-8B \
  --served-model-name qwen3-8b \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser deepseek_r1 \
  --gpu-memory-utilization 0.97 \
  --enforce-eager
```

With Podman, replace `--gpus all` with `--device nvidia.com/gpu=all` and leave
the rest unchanged. Pin a released tag rather than `latest` for reproducible
behavior; the on-demand GPU endpoint workflow uses `v0.29.0-cu129`. The first
run downloads a multi-gigabyte image plus the model weights, and the Hugging
Face cache mount keeps the weights for later runs.

Use `VLLM_URL=http://127.0.0.1:8000` for either local server.

The 32,768-token window is intentional for Claude Code auto mode. Its
client-initiated safety classifier reserves 2,112 output tokens independently
of the main Claude Code output-token setting and includes a large client-owned
prompt; 16K and 18K servers reject later classifier turns before inference. On
an A10G, eager execution reclaims CUDA-graph memory and the 0.97 utilization is
reserved for this single-user development workload. If you lower the window or
share the GPU, do not use auto mode unless the classifier request still fits.

The window is small enough that Claude Code's default output budget does not fit
beside a working prompt, so section 4 also sets `CLAUDE_CODE_MAX_OUTPUT_TOKENS`.

## 2. Point a Praxis example at vLLM

Choose one example and copy it outside `examples/`:

```console
# Codex, native Responses API (preferred for vLLM)
cp examples/configs/openai/responses/client-tool-compat.yaml praxis-vllm.yaml

# Codex, translated to Chat Completions
cp examples/configs/openai/responses/codex-http-chat-translation.yaml praxis-vllm.yaml

# Claude Code, native Anthropic API (preferred for vLLM)
cp examples/configs/anthropic/messages-native-vllm.yaml praxis-vllm.yaml

# Claude Code, translated to Chat Completions
cp examples/configs/anthropic/messages-to-openai-vllm.yaml praxis-vllm.yaml
```

Of the four, only the native Codex example persists server-side state, so it
alone needs a response store backend; see
[Choose a response store backend](#choose-a-response-store-backend) below.

The native Codex example uses `127.0.0.1:3001` as its fixture backend; change
that endpoint to `127.0.0.1:8000` for the local vLLM server. The other examples
already target port 8000. For the on-demand HTTPS endpoint, remove `https://`
from `VLLM_URL` and replace the selected example's vLLM backend endpoint with:

```yaml
endpoints:
  - "example.trycloudflare.com:443"
tls:
  sni: "example.trycloudflare.com"
```

For the translated Codex example, replace the fixed provider credential with
the same environment-backed form used by the Claude examples:

```yaml
- filter: credential_injection
  clusters:
    - name: codex-chat-provider
      header: Authorization
      env_var: VLLM_API_KEY
      header_prefix: "Bearer "
      strip_client_credential: true
```

Set a backend key. It must match `--api-key` for a keyed vLLM server; any random
value is sufficient for the unauthenticated dev endpoint:

```console
export VLLM_API_KEY="${VLLM_API_KEY:-$(openssl rand -hex 32)}"
export GATEWAY_AUTH_PASSWORD="$(openssl rand -hex 24)"
```

### Choose a response store backend

Only one of the four examples persists anything. `client-tool-compat.yaml` —
the preferred native Codex path — installs `openai_response_store` because the
Responses API keeps conversation state on the server: Codex sends
`store: true` and then refers back with `previous_response_id`, and
`GET /v1/responses/{id}`, `GET /v1/responses/{id}/input_items`, and
`DELETE /v1/responses/{id}` are answered from that store instead of being
forwarded to vLLM. The other three examples are stateless request translation
and never open a database, so the rest of this subsection does not apply to
them.

The filter supports two backends, and they are compile-time features rather
than pure configuration. A binary rejects a backend it was not built with,
during pipeline construction rather than at first use:

```text
openai_response_store: backend 'sqlite' is unavailable; rebuild with the 'store-sqlite' feature
```

| | SQLite | PostgreSQL |
| --- | --- | --- |
| Cargo feature | `store-sqlite` (opt-in) | `store-postgres` (included in `full`) |
| In `ghcr.io/praxis-proxy/ai` | No — needs a local image build | Yes |
| External service | None; one file on disk | A running PostgreSQL instance |
| Extra config keys | None | `allow_private_database_url`, usually `ssl_mode` |
| Connection TLS | Not applicable | `verify-full`; overrides URL `sslmode` |
| Shared by several proxy instances | No | Yes |
| State survives a proxy container restart | Only with a mounted path | Yes |

Neither backend needs schema setup: startup provisioning creates and validates
the schema-v4 tables before the proxy admits traffic. So the decision is really
about which one-time cost you would rather pay for this loop — rebuilding the
container image (SQLite) or running a database next to it (PostgreSQL). A
source build can compile in either or both, so the trade-off mostly bites the
published-image path; the container subsection below gives a worked setup for
each.

For the full backend feature matrix see
[Response Store](../architecture/response-store.md), and for the
recreate-only upgrade policy see
[Responses Store Schema Upgrade](../store/schema-migration.md).

### Run Praxis from source

Build and start Praxis. Running it in the background keeps the two generated
values in the shell that will launch the client. `store-sqlite` compiles in the
local response store used by the native Codex example and is harmless for the
other paths:

```console
cargo build -p praxis-ai-proxy --no-default-features \
  --features standard,openai-all,store-sqlite
./target/debug/praxis-ai -c praxis-vllm.yaml > /tmp/praxis-vllm.log 2>&1 &
export PRAXIS_PID=$!
```

A source build is the one place where the backend choice costs nothing: swap
the last feature to `store-postgres` for PostgreSQL only (the same selection
the default `full` build makes), or to `store-all` for a binary that accepts
either `backend:` value with no rebuild. Dropping both store features still
runs the three stateless examples and rejects `client-tool-compat.yaml` at
startup. Whichever backend is compiled in, the config must select it — see
[Choose a response store backend](#choose-a-response-store-backend) and, for
PostgreSQL, the two extra config keys described under
[Native Codex on the published image](#native-codex-on-the-published-image-postgresql).

Praxis listens at `http://127.0.0.1:8080`.

### Run Praxis from the official container image

Released images are published to
[`ghcr.io/praxis-proxy/ai`](https://ghcr.io/praxis-proxy/ai). The `latest` tag
tracks the most recent release; substitute a version tag such as `0.4` to pin
one:

```console
docker pull ghcr.io/praxis-proxy/ai:latest
```

The image entrypoint is `praxis-ai`, its working directory is `/etc/praxis`,
and it runs as the unprivileged `praxis` user (UID 100). Mounting the config at
`/etc/praxis/praxis.yaml` is enough — the binary falls back to `praxis.yaml` in
the working directory — but passing `-c` is clearer and works for any mount
path. `GATEWAY_AUTH_PASSWORD` and `VLLM_API_KEY` are read from the *proxy
process* environment at pipeline build time, so they must be forwarded into the
container, not merely exported on the host. The `z` mount suffix relabels the
config for SELinux hosts such as Fedora and RHEL; drop it elsewhere.

Host networking is the closest match to the source build and needs no config
edits at all, because `127.0.0.1` inside the container is the host loopback for
both the listener and the vLLM backend:

```console
docker run -d --rm --name praxis-vllm \
  --network host \
  -e VLLM_API_KEY -e GATEWAY_AUTH_PASSWORD \
  -v "$PWD/praxis-vllm.yaml:/etc/praxis/praxis.yaml:ro,z" \
  ghcr.io/praxis-proxy/ai:latest -c /etc/praxis/praxis.yaml
```

Praxis listens at `http://127.0.0.1:8080`, and sections 3 and 4 apply
unchanged. Docker's `--network host` is Linux-only.

To publish a port instead of sharing the host network stack, make three edits
to the copied config:

1. Change the listener to `address: "0.0.0.0:8080"`. A listener bound to
   `127.0.0.1` inside the container's own namespace is unreachable through a
   published port.
2. Change the local vLLM endpoint from `127.0.0.1:8000` to
   `host.docker.internal:8000`, and pass
   `--add-host=host.docker.internal:host-gateway` on the command line.
3. Add `allow_private_upstreams: true` beside the existing
   `allow_private_endpoints: true`. The existing option covers literal private
   endpoints written in the config; the new one is needed because that
   hostname *resolves* to a private address, and without it the request fails
   with `upstream hostname resolved to private/reserved IP address`.

```console
docker run -d --rm --name praxis-vllm \
  -p 8080:8080 \
  --add-host=host.docker.internal:host-gateway \
  -e VLLM_API_KEY -e GATEWAY_AUTH_PASSWORD \
  -v "$PWD/praxis-vllm.yaml:/etc/praxis/praxis.yaml:ro,z" \
  ghcr.io/praxis-proxy/ai:latest -c /etc/praxis/praxis.yaml
```

With Podman, both commands work unchanged. Podman also resolves
`host.containers.internal` on its own, so the published-port config can use
that name and drop `--add-host`.

The on-demand GPU endpoint needs no backend-specific container handling: the
tunnel hostname is public, so the `endpoints` and `tls.sni` edits above are the
only vLLM-side change with either networking mode.

Follow startup with `docker logs -f praxis-vllm`. For an explicit readiness
probe, add an admin listener to the copied config:

```yaml
admin:
  address: "127.0.0.1:9901"
```

The admin endpoint must bind loopback unless
`insecure_options.allow_public_admin: true` is set, so query it from inside the
container with
`docker exec praxis-vllm wget -qO- http://127.0.0.1:9901/healthy` (or directly
from the host under `--network host`). When finished, replace the `kill` in
section 4 with `docker rm -f praxis-vllm`.

#### The native Codex example needs a store backend

The published image is built with `PRAXIS_AI_FEATURES=full`, and `full` does
not include `store-sqlite` (it selects `store-postgres` as the store backend).
`client-tool-compat.yaml` — the preferred native Codex example — ships with
`backend: sqlite` and therefore cannot run on the stock image as written;
`--validate` rejects it up front:

```text
invalid configuration: openai_response_store: backend 'sqlite' is unavailable; rebuild with the 'store-sqlite' feature
```

The other three examples in this section have no store filter and run on the
published image unchanged. For the native Codex path, pick one of the two
options below. Nothing later in this guide depends on which store sits behind
`/v1/responses`; Codex behaves identically either way.

#### Native Codex on the published image: PostgreSQL

This keeps the stock image and adds a database container. Start PostgreSQL on
the host network so the proxy reaches it at `127.0.0.1:5432`:

```console
export PGPASSWORD="$(openssl rand -hex 16)"
docker run -d --rm --name praxis-pg \
  --network host \
  -e POSTGRES_USER=praxis \
  -e POSTGRES_DB=praxis \
  -e POSTGRES_PASSWORD="$PGPASSWORD" \
  docker.io/library/postgres:18
```

Then point the copied config's store filter at it:

```yaml
- filter: openai_response_store
  backend: postgres
  database_url: "postgres://praxis@127.0.0.1:5432/praxis"
  responses_table: openai_responses
  conversations_table: openai_conversations
  allow_private_database_url: true
  ssl_mode: disable
```

`database_url` is read verbatim from YAML — this field has no environment
expansion — so the URL deliberately omits the password. sqlx fills it from
`PGPASSWORD` in the *proxy process* environment, which keeps the secret out of
a config file you might copy around. Writing
`postgres://praxis:<password>@127.0.0.1:5432/praxis` inline works too and takes
precedence over the variable.

The two keys that have no SQLite counterpart are both deliberate opt-outs of a
fail-closed default, and each has a matching startup error when omitted:

- `allow_private_database_url: true` — the store's SSRF guard accepts only
  public IP literals by default, so it rejects `127.0.0.1`
  (`database_url host targets localhost`), every DNS name including a
  container name or `db.internal` (`host '...' is a DNS name`), and Unix
  sockets. Any local or in-cluster database needs this key.
- `ssl_mode: disable` — the filter always applies an explicit SSL mode and
  defaults to `verify-full`, overriding whatever `sslmode` the URL carries.
  The stock `postgres` image serves plaintext, so provisioning fails without
  this. Against a real database keep `verify-full` and add `ssl_root_cert`
  instead.

The proxy then runs with the ordinary host-network command plus the database
password — no rebuild, no writable mount, no image-local state. This example
installs neither `credential_injection` nor an auth filter, so it needs
neither `VLLM_API_KEY` nor `GATEWAY_AUTH_PASSWORD` forwarded:

```console
docker run -d --rm --name praxis-vllm \
  --network host \
  -e PGPASSWORD \
  -e PRAXIS_STORE_READINESS_ADDR=127.0.0.1:9200 \
  -v "$PWD/praxis-vllm.yaml:/etc/praxis/praxis.yaml:ro,z" \
  ghcr.io/praxis-proxy/ai:latest -c /etc/praxis/praxis.yaml
```

Unlike SQLite, a PostgreSQL store can be unreachable at startup, so this is
where the readiness endpoint earns its keep. It is opt-in: setting
`PRAXIS_STORE_READINESS_ADDR` starts it on its own port, and it reports store
provisioning composed with cluster health.

```console
curl -s http://127.0.0.1:9200/ready
{"status":"ready","store":"ready","clusters_ready":true}
```

A `"store":"pending"` or `"store":"failed"` verdict with HTTP 503 means the
database connection, not vLLM, is the problem. Tear both containers down with
`docker rm -f praxis-vllm praxis-pg` when finished.

Under the published-port networking mode described above, `127.0.0.1` in
`database_url` is the proxy container's own loopback rather than the host, so
the database host needs the same `host.docker.internal` treatment as the vLLM
endpoint. `allow_private_database_url: true` already covers that name.

For a non-development profile — TLS verification and client-certificate
authentication instead of a password — see
`examples/configs/openai/responses/response-store-postgres-mtls.yaml` and
[PostgreSQL cryptographic boundary](../architecture/postgres-cryptographic-boundary.md).

#### Native Codex with SQLite: rebuild the image

This keeps the example's `backend: sqlite` and rebuilds the image with SQLite
compiled in:

```console
docker build -f Containerfile \
  --build-arg PRAXIS_AI_FEATURES=full,store-sqlite \
  -t praxis-ai:sqlite .
```

Then give the store a writable path. The image's `/etc/praxis` working
directory is root-owned while the process runs as `praxis`, so the example's
relative `sqlite://responses.db?mode=rwc` cannot be created: the proxy starts,
and the first `/v1/responses` request fails with HTTP 500 and
`unable to open database file`. Point `database_url` at a mounted directory the
container user can write instead:

```yaml
- filter: openai_response_store
  backend: sqlite
  database_url: "sqlite:///data/responses.db?mode=rwc"
```

```console
mkdir -m 0777 -p "$PWD/praxis-responses"
docker run -d --rm --name praxis-vllm \
  --network host \
  -v "$PWD/praxis-vllm.yaml:/etc/praxis/praxis.yaml:ro,z" \
  -v "$PWD/praxis-responses:/data:z" \
  praxis-ai:sqlite -c /etc/praxis/praxis.yaml
```

The permissive mode is what lets UID 100 create the database in a host
directory owned by your user; a directory `chown`ed to UID 100 works too.
Podman can use a named volume instead, with the `:U` suffix asking it to chown
the volume to the container user: `-v praxis-responses:/data:U`.

`full,store-sqlite` keeps PostgreSQL compiled in alongside SQLite, so the same
image also runs the PostgreSQL config above without another build. Use
`standard,openai-all,store-sqlite` instead for a SQLite-only image.

Against the PostgreSQL option, this trades one image build plus a writable
mount for having no second service to run, and it keeps state in a file you
can delete between experiments. The database path is the better fit if a
PostgreSQL instance is already at hand, if you want to keep tracking the
published image, or if you plan to run more than one proxy against the same
store.

## 3. Connect Codex

Use an isolated Codex home so this test does not replace normal settings:

```console
export CODEX_HOME="$(mktemp -d)"
# Native /v1/responses passes this credential through to keyed vLLM.
export PRAXIS_API_KEY="$VLLM_API_KEY"
cat > "$CODEX_HOME/config.toml" <<EOF
model = "$VLLM_MODEL"
model_provider = "praxis"
web_search = "disabled"

[model_providers.praxis]
name = "Local Praxis"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
env_key = "PRAXIS_API_KEY"
EOF

codex exec --skip-git-repo-check \
  "Inspect this directory, create praxis-vllm-check.txt, then summarize the change."
```

With the preferred native example, Codex sends Responses API traffic to Praxis,
which forwards it to vLLM's `/v1/responses` endpoint. Praxis lowers and restores
Codex-specific client tool types but does not translate the inference request to
Chat Completions.

If you selected the translated Codex example instead, set
`PRAXIS_API_KEY=local-codex-client-key`. Praxis then translates Responses to
Chat Completions, injects `VLLM_API_KEY`, and streams the translated response
back.

## 4. Connect Claude Code

The Claude examples authenticate the client with Basic auth and independently
replace its Anthropic key with `VLLM_API_KEY` toward vLLM:

```console
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
export ANTHROPIC_API_KEY=local-claude-client-key
export ANTHROPIC_CUSTOM_HEADERS="Authorization: Basic $(printf 'gateway:%s' "$GATEWAY_AUTH_PASSWORD" | base64)"
export ANTHROPIC_MODEL="$VLLM_MODEL"
export ANTHROPIC_DEFAULT_MODEL="$VLLM_MODEL"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$VLLM_MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$VLLM_MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$VLLM_MODEL"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
# Keep the requested output budget inside the 32,768-token vLLM window.
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192
# Required if you select Claude Code's auto permission mode with this backend.
export CLAUDE_CODE_AUTO_MODE_SERVER=0

claude --model "$VLLM_MODEL"
# To exercise client-classified auto mode explicitly:
# claude --permission-mode auto --model "$VLLM_MODEL"
```

`GATEWAY_AUTH_PASSWORD` and `VLLM_API_KEY` must be present in the environment of
the Praxis process. The other variables configure Claude Code.

`CLAUDE_CODE_MAX_OUTPUT_TOKENS` is required for a 32,768-token server. Claude
Code does not know the served model's real window, so it sends the default
`max_tokens` for the Anthropic model name it believes it is calling — around
21,000 tokens. vLLM reserves `max_tokens` against `--max-model-len` before
inference, so that budget plus a modest prompt exceeds the window and the
request fails with HTTP 400 even though the prompt itself is small. Capping the
budget at 8192 leaves roughly 24K for input. Claude Code still sizes its own
auto-compaction against the window it assumes, not the real one, so run
`/compact` by hand if a long session creeps back into the limit. A larger window
is the alternative, but Qwen3-8B is natively 32,768 tokens and anything beyond
it needs `--rope-scaling` on `vllm serve`.

Praxis and vLLM do not implement Anthropic's server-side auto-mode classifier
protocol. `CLAUDE_CODE_AUTO_MODE_SERVER=0` makes Claude Code initiate the
classifier model requests through Praxis instead. This setting only affects
Claude Code's `auto` permission mode; the classifier still consumes model
inference and is not an on-device check. Do not set it to `1` for this setup.
See [Anthropic's auto-mode classifier documentation](https://code.claude.com/docs/en/auto-mode-classifier-billing).

When finished, stop local Praxis and cancel the on-demand endpoint workflow:

```console
kill "$PRAXIS_PID"
```

## Troubleshooting

- `401` from Praxis on the Claude path: verify the Basic authorization header.
- `401` from vLLM: `VLLM_API_KEY` does not match the key passed to vLLM.
- Model not found: use the exact slash-free served name, normally `qwen3-8b`.
- `400` with `maximum context length is 32768 tokens` and a requested output
  count near 21,000: Claude Code's default output budget does not fit the
  window. Set `CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192` as shown in section 4.
- `400` with `maximum context length is 32768 tokens` on the first turn, with a
  reported input count near 28,000 and a modest output count: the output cap
  from the previous entry is already in effect and the startup prompt itself is
  the problem. Claude Code loads MCP tool definitions, plugin skills, and
  `CLAUDE.md` before the session begins, so an empty conversation can consume
  most of the window. Run `/context` for the breakdown; it is a local command
  and still works while every request is failing. Start Claude Code with
  `--strict-mcp-config` to drop globally configured MCP servers, disable
  unneeded plugins with `/plugin`, and prefer a working directory whose
  `CLAUDE.md` is small or absent.
- TLS or connection failure: use only the tunnel hostname in the endpoint and
  `tls.sni`; do not include `https://` in Praxis's `endpoints` entry.
- Connection refused on 8080 from another machine, while vLLM on 8000 answers:
  every example in section 2 binds the listener to `127.0.0.1:8080`, so Praxis
  is reachable only from its own host. This is independent of how Praxis runs —
  `--network host` needs the same `address: "0.0.0.0:8080"` edit as the
  published-port path, not just the port mapping. After that edit, open 8080 on
  the host firewall; a loopback-only listener refuses the connection
  immediately, whereas a blocked port usually hangs or reports no route. Binding
  to `0.0.0.0` exposes the gateway to the network, so keep the authentication
  filter in place.
- Claude startup probes may call `/v1/messages/count_tokens`. Native vLLM
  supports it; the translated Chat path can return 404 and Claude degrades
  gracefully.
- `backend 'sqlite' is unavailable` or `backend 'postgres' is unavailable` at
  startup: the binary or image was not built with that store feature. Rebuild
  with `store-sqlite`, `store-postgres`, or `store-all`, or change the
  example's `backend:` to the one that is compiled in. Only the native Codex
  example has a store filter at all.
- `database_url host targets localhost` or `host '...' is a DNS name` from
  `openai_response_store`: the PostgreSQL SSRF guard accepts only public IP
  literals by default. Add `allow_private_database_url: true` to the store
  filter.
- The proxy cannot reach PostgreSQL even though `psql` connects from the same
  host: the filter defaults to `ssl_mode: verify-full` and overrides any
  `sslmode` in `database_url`, so a plaintext development server needs an
  explicit `ssl_mode: disable`.
- `password authentication failed for user "praxis"` when `database_url` has
  no password: `PGPASSWORD` must be in the *proxy process* environment, which
  for a container means forwarding it with `-e PGPASSWORD`, not just exporting
  it on the host.
- HTTP 500 with `unable to open database file` on the first `/v1/responses`
  request: the SQLite path is not writable by the container user (UID 100).
  Mount a writable directory and point `database_url` at it.
- Startup fails validating an existing store created by an earlier release:
  schema v4 is recreate-only, not an in-place migration. See
  [Responses Store Schema Upgrade](../store/schema-migration.md).
