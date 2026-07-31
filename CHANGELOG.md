# CHANGELOG

> **Duas linhagens neste arquivo.** As entradas `vX.Y.Z` abaixo da marca de
> divergência são do projeto original, `elvismdev/mem0-mcp-selfhosted`, geradas
> por semantic-release. As entradas `deepmem0-mcp-vX.Y.Z` são do fork usado no
> stack DeepMem0, que divergiu em **v0.3.2** e nunca foi republicado no trem de
> versões upstream. Numerar por cima da sequência dele misturaria dois autores
> como se fossem um só.


## deepmem0-mcp-v0.2.0 (2026-07-31)

Escopo de delete e readiness. Um tema só: valor errado que **não produz erro**.

### O escopo do delete não era normalizado

O filtro do vector store é casamento EXATO. `" alice"` e `"alice"` são escopos
diferentes, e a diferença não aparece como erro — aparece como resultado vazio.
Num delete isso é pior que um crash: nada casa, nada é removido, e a chamada
responde sucesso. Medido no corpus de produção: um `user_id` real casa 1159
pontos; o MESMO valor com um espaço à esquerda casa **0**.

`delete_all_memories` e `delete_entities` normalizam antes de montar o filtro, e
`safe_bulk_delete` normaliza como defesa em profundidade — só as quatro chaves de
ESCOPO, porque normalizar chave livre quebraria filtro de metadata legítimo. A
regra é **importada** do core (`normalize_scope_id`), nunca reimplementada: duas
definições de "escopo válido" foi o que deixou `importance='high'` entrar no
corpus.

### O escopo resolvido escapava por dois caminhos

Só o argumento do cliente passava pela regra. `get_default_user_id()` lê por
`env()`, que apara apenas as BORDAS, então `MEM0_USER_ID="a b"` chegava como
`'a b'` e `"   "` como `''` — os dois sem erro, em toda tool escopada. E o
`user_id` amarrado ao token vinha do cofre, que nunca aplicou a regra. O valor
RESOLVIDO agora normaliza, e o erro nomeia a ORIGEM: sem isso o operador procura
o defeito no cliente enquanto ele está no drop-in do systemd.

Isso também corrige a comparação de autorização, que rodava `!=` sobre a string
crua — um token amarrado a `"alice"` recusava `" alice"` como "token não pode
acessar esse escopo", mensagem errada para um erro de digitação.

### A sonda de readiness não podia reprovar

O `/health` passa a expor `entity_pipeline` e a responder **503** quando o
pipeline de entidade está degradado. O campo era documentado e não existia.

A versão anterior chamava `get_nlp_full()` sem argumento, o que inspeciona o
modelo do idioma DEFAULT: dizia `en_core_web_sm` num deployment português.
Carregar também dispara `spacy.cli.download`, e uma sonda de readiness que toca a
rede pendura. O idioma vem de `configured_language()`, a mesma função que o
`build_config()` usa — uma sonda que discorda do runtime é pior que sonda
nenhuma, porque ela afirma.

### Proveniência do próprio MCP no boot

O carimbo identificava só o FORK. É o servidor que decide escopo, autorização e
contrato de resposta, então um deploy dele era verificável apenas pelo
comportamento — bom para o smoke do dia, inútil no dia seguinte.
`boot_mcp_sha`, `boot_mcp_tree_dirty` e os hashes de `server.py`/`helpers.py`
entram no `/health`.

### Testes

A suíte de integração fixava `anthropic` na mão enquanto o deployment roda
`MEM0_PROVIDER=ollama` sem token Anthropic nenhum: exercitava um provedor que
produção não usa e morria em `429` no setup. Passou a resolver com a mesma
precedência do `build_config()`. Medido: 8 falhas antes, **15 passed / 4
skipped** depois.

## deepmem0-mcp-v0.1.0 (2026-07-31)

Primeira release do fork. Cobre os 62 commits desde a divergência em v0.3.2
(2026-03-13) — quatro meses de trabalho que nunca teve ponto de referência.
Ela existe para que "a produção roda X" seja uma frase verificável.

### Ingestão assíncrona (v0.4)

`add_memory` com `infer=true` deixou de ser síncrono: enfileira em SQLite WAL e
responde na hora com `{"status":"queued", "task_id", ...}`. Antes devolvia uma
lista crua e o cliente ficava esperando a extração do LLM, que domina o add.
Worker serial com claim FIFO, backoff exponencial até dead-letter, recuperação
de órfãos no boot, purge-on-retry e GC por retenção. `update_memory` seguiu o
mesmo caminho pelo mesmo motivo: o classificador de metadata é uma chamada lenta
que estourava o timeout de clientes MCP enquanto o update já tinha dado certo.

### Documentos e visão (v0.5a / v0.5b)

`add_document(file_path=...)`: validação no submit (allowlist, realpath, magic
bytes, cap de tamanho, páginas), spool content-addressed, chunker page-aware, e
um add por chunk com proveniência de documento e página. Transporte é só
`file_path` — base64 numa tool MCP é gerado token a token pelo modelo, o que
inviabiliza arquivo real.

Páginas escaneadas e imagens passam por um VLM local. VLM e extrator não cabem
juntos na GPU, então transcrição e extração são duas fases estritas com um swap
de modelo entre elas — dois swaps por documento, não por página.

### Temporalidade e ranking

`as_of` (âncora de record-time), metadata de supersedência, `memory_history`,
filtros do classificador expostos na busca, ranking ciente de `event_date`, e
opt-out de reforço — obrigatório para harness de medição, que senão reforça os
próprios alvos e infla a métrica que existe para proteger.

### Versionamento de update (v0.7)

Update passa a criar versão em vez de sobrescrever, com linhagem dedicada.
A v0.7.0 teve perda de dados no delete sobre topologia não-linear; contida por
env e corrigida na v0.7.1, com a v0.7.2 fechando o journal de intenção e
expondo `{deleted, remaining}` em falha parcial.

### Vault: autenticação por token (:8080 + gate no :8081)

Store SQLite WAL onde toda mutação e sua auditoria vivem na mesma transação;
gate ASGI puro em três modos (`off`/`shadow`/`on`) que não bufferiza SSE;
contrato de autorização declarando a decisão das 15 tools em quatro classes,
com teste de completude que quebra se uma tool nova subir sem decisão; sessão
MCP amarrada à credencial. Vários achados de revisão independente estão nos
`fix(vault)` — inclusive uma UI que **mentia sobre a postura de segurança**
porque lia o modo de auth do env errado.

### Contrato de metadata na fronteira de escrita

`importance`/`confidence` como float, `domain`/`memory_type` como enum, `tags`
como lista — validado no submit, com mensagem acionável. Antes gravava qualquer
coisa, e 17 memórias com `importance='high'` (str onde se esperava float)
derrubaram a recuperação de um cliente. Presença não é tipo.

### Contrato de `messages` (este ciclo)

`add_memory` aceitava `messages` cru. O core roda com visão desligada e
**descarta** partes `image_url`; mensagem só de imagem sumia inteira e o cliente
recebia `queued` assim mesmo. Agora o submit recusa, apontando `add_document`.
`MEM0_MESSAGE_CONTRACT=enforce|warn|off` resolvido no **boot** — o precedente da
metadata resolve por requisição, então um env inválido só explodia na primeira
escrita. Nasce em `warn`, com telemetria e critério de promoção datado; `off`
desliga por inteiro, porque kill switch que não desliga não é kill switch.

`/health` ganhou `boot_provenance`, carimbado na construção. O campo antigo
`fork_sha` é lido do disco a cada requisição: foi medido um processo iniciado às
13:32Z reportando um commit feito às 17:41Z, quatro horas mais novo que ele.
Uma sonda assim afirmaria que um restart pegou a mudança sem restart nenhum.

### Correções que valem menção

`MEM0_VLM_TIMEOUT` era calculado e nunca aplicado — o worker é serial e único,
então uma página travada no VLM prendia a fila inteira. A rota `/health` chegou
a ficar sem rota por um decorador mal posicionado, e agora tem teste. IPs de
exemplo nos fixtures passaram a usar RFC 5737 em vez da faixa da LAN real.

---

<!-- marca de divergencia: daqui para baixo, releases do upstream elvismdev -->


## v0.3.2 (2026-03-13)

### Bug Fixes

- Cache-bust Glama badge URL to force fresh camo proxy fetch
  ([`205ecf9`](https://github.com/elvismdev/mem0-mcp-selfhosted/commit/205ecf9a6d8d95f23fa0d8fa27826e3348ab0728))


## v0.3.1 (2026-03-12)

### Bug Fixes

- Add .python-version for Glama uv sync compatibility
  ([`e4d1f09`](https://github.com/elvismdev/mem0-mcp-selfhosted/commit/e4d1f09008652a84ed1340db9372f621b8ffa785))

Pin Python 3.12 so uv sync resolves the correct interpreter in Glama's Docker build environment
  instead of picking up Debian's externally-managed Python 3.11.

### Chores

- Remove Dockerfile (Glama generates its own)
  ([`33f2f1d`](https://github.com/elvismdev/mem0-mcp-selfhosted/commit/33f2f1d25bdb1e4c85617e90b21a72c48fc9c2a2))

Glama's admin page generates a Dockerfile from configuration fields rather than using the repo's
  Dockerfile. No other Docker deployment workflow exists, so the file is unused.


## v0.3.0 (2026-03-12)

### Features

- Lazy Memory init + Glama submission packaging
  ([`c6f2b76`](https://github.com/elvismdev/mem0-mcp-selfhosted/commit/c6f2b76aa7fc1f243c86fbcd941825ef7861b539))

Defer Memory.from_config() to the first tool call via _ensure_memory(), allowing the MCP server to
  respond to initialize/tools/list without live Qdrant/Neo4j/Ollama. This unblocks Glama's
  Docker-based inspection pipeline which builds and runs the container in an ephemeral sandbox.

Add LICENSE (MIT), glama.json, Dockerfile, and Glama badge in README.


## v0.2.1 (2026-02-28)

### Bug Fixes

- Update hooks to nested format for Claude Code schema compatibility
  ([`2f86dee`](https://github.com/elvismdev/mem0-mcp-selfhosted/commit/2f86dee99c3fa73220270b721c1621881beea655))

Migrate hook installer from the deprecated flat format to the current nested schema (matcher group
  -> hooks array -> handler objects). Add legacy format detection and auto-migration so existing
  users upgrading do not end up with duplicate or broken entries.

### Documentation

- Clarify hooks and CLAUDE.md as complementary layers
  ([`94f29dc`](https://github.com/elvismdev/mem0-mcp-selfhosted/commit/94f29dca52582ee18ce9ae256fc06d8cf1adab30))

Update README to explain that hooks (automated memory at session boundaries) and CLAUDE.md
  (behavioral instructions for mid-session engagement) work best together rather than as
  alternatives.


## v0.2.0 (2026-02-28)

### Features

- Add Claude Code session hooks for cross-session memory
  ([`113df26`](https://github.com/elvismdev/mem0-mcp-selfhosted/commit/113df2678b05091dd0acffa2776c755d4c380644))

Add SessionStart and Stop hooks that give Claude Code automatic cross-session memory without
  requiring CLAUDE.md rules or manual tool calls.

- SessionStart hook (mem0-hook-context): searches mem0 with multi-query strategy, deduplicates by
  ID, injects formatted memories as additionalContext on startup and compact events - Stop hook
  (mem0-hook-stop): reads last ~3 exchanges from JSONL transcript via bounded deque, saves session
  summary to mem0 with infer=True for atomic fact extraction - CLI installer (mem0-install-hooks):
  patches .claude/settings.json with idempotent hook entries, supports --global and --project-dir -
  Graph force-disabled in hooks to stay within 15s/30s timeout budgets - Atomic settings.json write
  via tempfile + os.replace - 43 unit tests covering protocol, edge cases, and error handling - 6
  integration tests against live Qdrant + Ollama infrastructure - README updated with hooks
  documentation, architecture diagram, and test structure


## v0.1.1 (2026-02-27)

### Bug Fixes

- Use NEO4J_DATABASE env var instead of config dict for non-default database
  ([`74e1188`](https://github.com/elvismdev/mem0-mcp-selfhosted/commit/74e1188d38154846ec8b12602fde1d757197873b))

mem0ai's graph_memory.py passes config as positional args to Neo4jGraph() where pos 3 is `token`,
  not `database`. Setting database in the config dict causes it to land in the token parameter,
  resulting in AuthenticationError. Use NEO4J_DATABASE env var which langchain_neo4j reads via
  get_from_dict_or_env().

Upstream: mem0ai #3906, #3981, #4085 (none merged)

Resolves: PAR-57


## v0.1.0 (2026-02-27)

### Bug Fixes

- **ci**: Use angular parser compatible with PSR v9.15.2
  ([`b5bc6ab`](https://github.com/elvismdev/mem0-mcp-selfhosted/commit/b5bc6ab45edff26f07fc73774c7e0c57d22cb40d))

The v9 GitHub Action does not recognize "conventional" parser name (v10+ only). Reverts to "angular"
  and changelog.changelog_file format.

### Continuous Integration

- Add python-semantic-release configuration and GitHub Actions workflow
  ([`2473ee4`](https://github.com/elvismdev/mem0-mcp-selfhosted/commit/2473ee4ec9c0db90b2bb412d3714caae7dc41498))

Automated versioning via Conventional Commits analysis, changelog generation, git tagging
  (v{version}), and GitHub Release creation on push to main.
