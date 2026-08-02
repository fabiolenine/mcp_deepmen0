"""UI strings, PT and EN (the mockup's ``L()`` dictionary, server-side).

The toggle is a cookie read per request, so a page render never mixes
languages. Templates get the active dict as ``t``.
"""

from __future__ import annotations

LANGS = ("pt", "en")
DEFAULT_LANG = "pt"
LANG_COOKIE = "vault_lang"

PT = {
    "loginTitle": "Entrar no cofre",
    "loginSub": "Gestão de credenciais do DeepMem0 MCP",
    "email": "E-mail",
    "password": "Senha",
    "signIn": "Entrar",
    "lanNote": "Acesso restrito à LAN · sessão de 12 h",
    "navDash": "Visão geral",
    "navAudit": "Auditoria",
    "logout": "Sair",
    "adminRole": "Administrador",
    "dashSub": "Usuários e tokens que protegem o MCP na porta 8081",
    "statMode": "Modo de auth",
    "statUsers": "Usuários ativos",
    "statTokens": "Tokens ativos",
    "statUsed24h": "Tokens vistos · 24 h",
    "mcpLabel": "MCP · :8081",
    "mcpDown": "sem resposta",
    "ofTotal": "de",
    "newUser": "Novo usuário",
    "create": "Criar",
    "nuNamePh": "Nome de exibição",
    "nuEmailPh": "email@dominio.com",
    "nuNote": "Sem auto-registro — apenas admin cria usuários. E-mail validado por formato.",
    "colUser": "Usuário",
    "colTokens": "Tokens",
    "colStatus": "Status",
    "colLast": "Último uso",
    "colLabel": "Rótulo",
    "colPrefix": "Prefixo",
    "colCreated": "Criado",
    "colExpires": "Expira",
    "colActions": "Ações",
    "colWhen": "Quando",
    "colActor": "Autor",
    "colAction": "Ação",
    "colSubject": "Alvo",
    "active": "Ativo",
    "disabled": "Desativado",
    "revoked": "Revogado",
    "expiring": "Expirando",
    "never": "Nunca",
    "back": "Voltar",
    "disable": "Desativar usuário",
    "enable": "Reativar usuário",
    "genToken": "Gerar token",
    "rotate": "Renovar",
    "revoke": "Revogar",
    "rotateNote": (
        "Renovar emite um token novo; o antigo permanece válido por {hours} h "
        "(janela de migração). Revogar mata na hora."
    ),
    "revokeNote": (
        "A revogação vale para novas requisições — e toda chamada de tool do MCP "
        "é uma requisição nova."
    ),
    "auditNote": "Registro append-only — tokens nunca aparecem em claro, apenas o prefixo.",
    "modalTitle": "Token gerado",
    "modalWarn": (
        "Este token é exibido apenas uma vez. Copie agora e guarde em local seguro — "
        "depois desta tela, só o prefixo fica visível."
    ),
    "copy": "Copiar",
    "copied": "Copiado",
    "copyManual": "Selecionado — use Ctrl+C",
    "done": "Já copiei, fechar",
    "modeHintOff": "Auth desligada — tokens não são exigidos",
    "modeHintShadow": "Valida e registra, sem bloquear",
    "modeHintOn": "Requisições sem token = 401",
    "tokenLabelPh": "rótulo do token (ex: claude-code-local)",
    "noTokens": "Nenhum token emitido.",
    "noUsers": "Nenhum usuário ainda.",
    "noAudit": "Sem eventos registrados.",
    "prev": "Anteriores",
    "next": "Próximos",
    "err_email_required": "Informe um e-mail.",
    "err_email_invalid": "E-mail inválido — verifique o formato.",
    "err_name_required": "Informe um nome de exibição.",
    "err_name_too_long": "Nome muito longo.",
    "err_label_invalid": "Rótulo inválido — use letras, números, espaço, . @ : + -",
    "err_label_too_long": "Rótulo muito longo (máx. 60).",
    "err_duplicate_email": "Este e-mail já está cadastrado.",
    "err_bad_credentials": "E-mail ou senha incorretos.",
    "err_csrf": "Sessão expirada — recarregue a página e tente de novo.",
    "err_disabled_user": "Usuário desativado — reative antes de emitir tokens.",
    "modeUnconfirmed": "não confirmado — o :8081 não respondeu",
    "readyTitle": "Prontidão para MEM0_REQUIRE_AUTH=on",
    "readyYes": "Pronto para ligar",
    "readyNo": "Ainda não",
    "readySeen": "tokens ativos vistos na janela",
    "readySilent": "sem uso na janela (tomariam 401)",
    "readyDenials": "requisições negadas na janela",
    "readyNote": "Silêncio não é evidência: token que ninguém usou é indistinguível de cliente que quebraria.",
    "readyWindow": "janela",
    "noAdminTitle": "Nenhum administrador cadastrado",
    "noAdminBody": (
        "Rode <code>deepmem0-vault bootstrap-admin</code> no servidor para criar "
        "o primeiro administrador."
    ),
    # -- memórias -----------------------------------------------------------
    "navMemories": "Memórias",
    "memSub": "Corpus indexado no escopo",
    "memUnit": "memórias",
    "colMemory": "Memória",
    "colClass": "Classificação",
    "colImportance": "Importância",
    "noMemories": "Nenhuma memória neste filtro.",
    "facet_domain": "Domínio",
    "facet_memory_type": "Tipo",
    "facet_project": "Projeto",
    "facet_attributed_to": "Proveniência",
    "facetFlags": "Marcadores",
    "facetClear": "Limpar filtros",
    "flagSuperseded": "Supersedidas",
    "flagEventDate": "Com data de evento",
    "flagDocument": "De documento",
    "badgeSuperseded": "supersedida",
    "badgeDoc": "documento",
    "pagOlder": "Mais antigas",
    "pagFirst": "Primeira página",
    "pagNote": "Ordenado por data de criação, mais recentes primeiro.",
    "srcUnavailable": "Fonte de leitura indisponível",
    "srcUnavailableHint": (
        "As telas de usuários e tokens seguem funcionando. Verifique a api-key do "
        "Qdrant no serviço do cofre."
    ),
    # -- detalhe da memória -------------------------------------------------
    "memDetail": "Memória",
    "memNotFound": "Memória não encontrada",
    "memNotFoundHint": "O identificador não existe neste escopo.",
    "memPartial": "Parte do detalhe não pôde ser carregada:",
    "fldConfidence": "Confiança",
    "fldProject": "Projeto",
    "fldEventDate": "Data do evento",
    "fldScope": "Escopo semântico",
    "fldActor": "locutor",
    "memActr": "Ativação (memória humana)",
    "actrNeutral": "Sem histórico de uso — a memória é neutra no ranking.",
    "actrActivation": "Ativação base",
    "actrBoost": "Boost",
    "actrAccess": "Encontros",
    "actrLast": "Último acesso",
    "actrNote": (
        "A ativação não é armazenada: é derivada da linha do tempo a cada leitura, "
        "por isso decai sozinha."
    ),
    "memChain": "Versões e supersedência",
    "chainNone": "Nenhuma versão anterior ou posterior.",
    "chainOlder": "Supersede",
    "chainNewer": "Supersedida por",
    "chainMissing": "registro ausente",
    "chainSupersededAt": "Supersedida em",
    "memProvenance": "Proveniência do documento",
    "provDoc": "Documento",
    "provPages": "Páginas",
    "provChunk": "Trecho",
    "provType": "Tipo",
    "memJob": "Trabalho de ingestão",
    "jobAttempts": "Tentativas",
    "memEntities": "Entidades vinculadas",
    "entNone": "Nenhuma entidade vinculada.",
    "histEvent": "Evento",
    "histActor": "Autor",
    "histChange": "Conteúdo",
    "histNone": "Sem histórico registrado para esta memória.",
    "histDeleteIntent": "Há intenção de exclusão registrada",
    "memPayload": "Payload bruto",
    "memPayloadNote": (
        "Campos que não têm lugar próprio acima — inclusive os que a busca do MCP "
        "não devolve."
    ),
    # -- busca --------------------------------------------------------------
    "navSearch": "Busca",
    "searchSub": "Busca semântica pelo caminho de produção: híbrido, reranker e ativação",
    "searchPh": "O que você quer encontrar?",
    "searchRun": "Buscar",
    "searchBusy": "buscando…",
    "searchSlowNote": (
        "A busca usa o mesmo pipeline dos clientes reais (denso + BM25 + reranker) e "
        "leva cerca de 8 s. Nada aqui reforça as memórias retornadas."
    ),
    "searchIdle": "Digite uma consulta para buscar.",
    "searchNoHits": "Nenhuma memória correspondeu.",
    "searchHits": "resultados",
    "searchFailed": "A busca falhou",
    "searchFailedHint": "O corpus não foi alterado. Verifique o serviço MCP e o token do cofre.",
    "searchUnavailable": "Busca indisponível",
    "searchIgnored": "Campos ignorados por valor inválido",
    "fldActorId": "Locutor (actor_id)",
    "fldActorHint": "v0.15 — ainda sem cobertura no corpus atual",
    "fldLimit": "Limite",
    "fldMinImportance": "Importância mínima",
    "fldAsOf": "Como estava em (as_of)",
    "fldAsOfHint": "Tempo de registro: o que se sabia naquela data",
    "fldEventFrom": "Evento de",
    "fldEventTo": "Evento até",
    "fldHistorical": "Recordação histórica",
    "fldHistoricalHint": "Exige as_of; não reforça e sinaliza versões mais novas",
    "resEventAnchor": "Âncora de evento",
    "resEventFilter": "Filtro de evento",
    "resPending": "Ingestão pendente",
    "resHistorical": "Recordação histórica",
    "resPenalty": "Penalidade",
    "resNewer": "Existe versão mais nova",
    "resProximity": "Proximidade",
    # -- fila de ingestão ---------------------------------------------------
    "navQueue": "Fila",
    "queueSub": "Ingestão assíncrona: conversas e documentos aguardando extração",
    "queueAutoRefresh": "atualiza a cada 10 s",
    "queueUnavailable": "Fila indisponível",
    "queueDepth": "Profundidade",
    "queueActive": "em andamento",
    "queuePending": "Aguardando",
    "queueOldest": "mais antigo há",
    "queueRetry": "Repetição",
    "queueProcessing": "em processamento",
    "queueDead": "Descartados",
    "queueDone": "concluídos",
    "queueActiveJobs": "Trabalhos em andamento",
    "queueIdle": "Nenhum trabalho em andamento.",
    "queueChunks": "trechos",
    "queueHeartbeat": "sinal de vida há",
    "queueWaiting": "na fila há",
    "queueError": "Erro",
    "queueDeadNote": (
        "Descartado é trabalho que esgotou as tentativas ou trazia payload inválido. "
        "Nada aqui é reprocessado pela interface."
    ),
    # -- entidades ----------------------------------------------------------
    "navEntities": "Entidades",
    "entSub": "Quem e o quê aparece nas memórias, e o que cada um conecta",
    "entUnit": "entidades",
    "entDetail": "Entidade",
    "entSearch": "Procurar",
    "entSearchPh": "Nome da entidade",
    "entName": "Entidade",
    "entKind": "Tipo",
    "entLinks": "Vínculos",
    "entNoMatch": "Nenhuma entidade corresponde.",
    "entMatched": "entidades correspondem",
    "entTruncated": "Mostrando",
    "entNotFound": "Entidade não encontrada",
    "entNoMemories": "Nenhuma memória vinculada foi encontrada.",
    "entIdentity": "Identidade",
    "entIdentityNote": (
        "A identidade é o texto normalizado: grafias diferentes da mesma entidade "
        "compartilham uma linha e somam seus vínculos."
    ),
    "entDangling": "Vínculos apontando para memórias que não existem",
    "entPayloadNote": "Sem as chaves de vínculo (uma por memória ligada).",
    "facetAll": "Todos",
    # -- painel e usuários --------------------------------------------------
    "navUsers": "Usuários",
    "usersSub": "Quem tem acesso ao MCP, e com quais tokens",
    "dashCorpus": "Corpus",
    "dashCredentials": "Credenciais",
    "statMemories": "Memórias",
    "statEntities": "Entidades",
    "dashAuditSub": "ver registro",
    "dashObserve": "Métricas e séries temporais:",
}

EN = {
    "loginTitle": "Sign in to the vault",
    "loginSub": "Credential management for the DeepMem0 MCP",
    "email": "Email",
    "password": "Password",
    "signIn": "Sign in",
    "lanNote": "LAN-only access · 12 h session",
    "navDash": "Overview",
    "navAudit": "Audit log",
    "logout": "Sign out",
    "adminRole": "Administrator",
    "dashSub": "Users and tokens protecting the MCP on port 8081",
    "statMode": "Auth mode",
    "statUsers": "Active users",
    "statTokens": "Active tokens",
    "statUsed24h": "Tokens seen · 24 h",
    "mcpLabel": "MCP · :8081",
    "mcpDown": "no answer",
    "ofTotal": "of",
    "newUser": "New user",
    "create": "Create",
    "nuNamePh": "Display name",
    "nuEmailPh": "email@domain.com",
    "nuNote": "No self-registration — only admins create users. Email is format-validated.",
    "colUser": "User",
    "colTokens": "Tokens",
    "colStatus": "Status",
    "colLast": "Last used",
    "colLabel": "Label",
    "colPrefix": "Prefix",
    "colCreated": "Created",
    "colExpires": "Expires",
    "colActions": "Actions",
    "colWhen": "When",
    "colActor": "Actor",
    "colAction": "Action",
    "colSubject": "Subject",
    "active": "Active",
    "disabled": "Disabled",
    "revoked": "Revoked",
    "expiring": "Expiring",
    "never": "Never",
    "back": "Back",
    "disable": "Disable user",
    "enable": "Re-enable user",
    "genToken": "Generate token",
    "rotate": "Rotate",
    "revoke": "Revoke",
    "rotateNote": (
        "Rotate issues a new token; the old one stays valid for {hours} h "
        "(migration window). Revoke kills it instantly."
    ),
    "revokeNote": (
        "Revocation applies to new requests — and every MCP tool call is a new request."
    ),
    "auditNote": "Append-only log — tokens never appear in plaintext, only the prefix.",
    "modalTitle": "Token generated",
    "modalWarn": (
        "This token is shown only once. Copy it now and store it safely — after this "
        "screen, only the prefix remains visible."
    ),
    "copy": "Copy",
    "copied": "Copied",
    "copyManual": "Selected — press Ctrl+C",
    "done": "Copied it, close",
    "modeHintOff": "Auth off — tokens not required",
    "modeHintShadow": "Validates and logs, never blocks",
    "modeHintOn": "Requests without a token = 401",
    "tokenLabelPh": "token label (e.g. claude-code-local)",
    "noTokens": "No tokens issued.",
    "noUsers": "No users yet.",
    "noAudit": "No events recorded.",
    "prev": "Previous",
    "next": "Next",
    "err_email_required": "Enter an email.",
    "err_email_invalid": "Invalid email — check the format.",
    "err_name_required": "Enter a display name.",
    "err_name_too_long": "Name is too long.",
    "err_label_invalid": "Invalid label — use letters, numbers, space, . @ : + -",
    "err_label_too_long": "Label is too long (max 60).",
    "err_duplicate_email": "That email is already registered.",
    "err_bad_credentials": "Wrong email or password.",
    "err_csrf": "Session expired — reload the page and try again.",
    "err_disabled_user": "User is disabled — re-enable before issuing tokens.",
    "modeUnconfirmed": "unconfirmed — :8081 did not answer",
    "readyTitle": "Readiness for MEM0_REQUIRE_AUTH=on",
    "readyYes": "Ready to flip",
    "readyNo": "Not yet",
    "readySeen": "active tokens seen in the window",
    "readySilent": "unused in the window (would take 401)",
    "readyDenials": "denied requests in the window",
    "readyNote": "Silence is not evidence: a token nobody used is indistinguishable from a client that would break.",
    "readyWindow": "window",
    "noAdminTitle": "No administrator registered",
    "noAdminBody": (
        "Run <code>deepmem0-vault bootstrap-admin</code> on the server to create "
        "the first administrator."
    ),
    # -- memories -----------------------------------------------------------
    "navMemories": "Memories",
    "memSub": "Indexed corpus in scope",
    "memUnit": "memories",
    "colMemory": "Memory",
    "colClass": "Classification",
    "colImportance": "Importance",
    "noMemories": "No memories match this filter.",
    "facet_domain": "Domain",
    "facet_memory_type": "Type",
    "facet_project": "Project",
    "facet_attributed_to": "Provenance",
    "facetFlags": "Flags",
    "facetClear": "Clear filters",
    "flagSuperseded": "Superseded",
    "flagEventDate": "With event date",
    "flagDocument": "From document",
    "badgeSuperseded": "superseded",
    "badgeDoc": "document",
    "pagOlder": "Older",
    "pagFirst": "First page",
    "pagNote": "Ordered by creation date, newest first.",
    "srcUnavailable": "Read source unavailable",
    "srcUnavailableHint": (
        "The users and tokens screens keep working. Check the Qdrant api-key on "
        "the vault service."
    ),
    # -- memory detail ------------------------------------------------------
    "memDetail": "Memory",
    "memNotFound": "Memory not found",
    "memNotFoundHint": "That identifier does not exist in this scope.",
    "memPartial": "Part of the detail could not be loaded:",
    "fldConfidence": "Confidence",
    "fldProject": "Project",
    "fldEventDate": "Event date",
    "fldScope": "Semantic scope",
    "fldActor": "speaker",
    "memActr": "Activation (human memory)",
    "actrNeutral": "No usage history — this memory is neutral in ranking.",
    "actrActivation": "Base activation",
    "actrBoost": "Boost",
    "actrAccess": "Encounters",
    "actrLast": "Last access",
    "actrNote": (
        "Activation is not stored: it is derived from the timeline on every read, "
        "which is how it decays on its own."
    ),
    "memChain": "Versions and supersedence",
    "chainNone": "No earlier or later version.",
    "chainOlder": "Supersedes",
    "chainNewer": "Superseded by",
    "chainMissing": "record missing",
    "chainSupersededAt": "Superseded at",
    "memProvenance": "Document provenance",
    "provDoc": "Document",
    "provPages": "Pages",
    "provChunk": "Chunk",
    "provType": "Type",
    "memJob": "Ingestion job",
    "jobAttempts": "Attempts",
    "memEntities": "Linked entities",
    "entNone": "No linked entities.",
    "histEvent": "Event",
    "histActor": "Actor",
    "histChange": "Content",
    "histNone": "No history recorded for this memory.",
    "histDeleteIntent": "A delete intent is recorded",
    "memPayload": "Raw payload",
    "memPayloadNote": (
        "Fields with no dedicated place above — including the ones the MCP search "
        "does not return."
    ),
    # -- search -------------------------------------------------------------
    "navSearch": "Search",
    "searchSub": "Semantic search through the production path: hybrid, reranker and activation",
    "searchPh": "What are you looking for?",
    "searchRun": "Search",
    "searchBusy": "searching…",
    "searchSlowNote": (
        "Search uses the same pipeline as real clients (dense + BM25 + reranker) and "
        "takes about 8 s. Nothing here reinforces the returned memories."
    ),
    "searchIdle": "Type a query to search.",
    "searchNoHits": "No memory matched.",
    "searchHits": "results",
    "searchFailed": "Search failed",
    "searchFailedHint": "The corpus was not changed. Check the MCP service and the vault token.",
    "searchUnavailable": "Search unavailable",
    "searchIgnored": "Fields ignored due to invalid value",
    "fldActorId": "Speaker (actor_id)",
    "fldActorHint": "v0.15 — no coverage in the current corpus yet",
    "fldLimit": "Limit",
    "fldMinImportance": "Minimum importance",
    "fldAsOf": "As of (record time)",
    "fldAsOfHint": "Record time: what was known on that date",
    "fldEventFrom": "Event from",
    "fldEventTo": "Event to",
    "fldHistorical": "Historical recollection",
    "fldHistoricalHint": "Requires as_of; never reinforces and flags newer versions",
    "resEventAnchor": "Event anchor",
    "resEventFilter": "Event filter",
    "resPending": "Pending ingestion",
    "resHistorical": "Historical recollection",
    "resPenalty": "Penalty",
    "resNewer": "A newer version exists",
    "resProximity": "Proximity",
    # -- ingestion queue ----------------------------------------------------
    "navQueue": "Queue",
    "queueSub": "Async ingestion: conversations and documents awaiting extraction",
    "queueAutoRefresh": "refreshes every 10 s",
    "queueUnavailable": "Queue unavailable",
    "queueDepth": "Depth",
    "queueActive": "in flight",
    "queuePending": "Pending",
    "queueOldest": "oldest for",
    "queueRetry": "Retrying",
    "queueProcessing": "processing",
    "queueDead": "Dead-lettered",
    "queueDone": "done",
    "queueActiveJobs": "Jobs in flight",
    "queueIdle": "No jobs in flight.",
    "queueChunks": "chunks",
    "queueHeartbeat": "heartbeat",
    "queueWaiting": "queued for",
    "queueError": "Error",
    "queueDeadNote": (
        "Dead-lettered means the job exhausted its retries or carried an invalid "
        "payload. Nothing here is reprocessed by the interface."
    ),
    # -- entities -----------------------------------------------------------
    "navEntities": "Entities",
    "entSub": "Who and what shows up in the memories, and what each one connects",
    "entUnit": "entities",
    "entDetail": "Entity",
    "entSearch": "Find",
    "entSearchPh": "Entity name",
    "entName": "Entity",
    "entKind": "Type",
    "entLinks": "Links",
    "entNoMatch": "No entity matches.",
    "entMatched": "entities match",
    "entTruncated": "Showing",
    "entNotFound": "Entity not found",
    "entNoMemories": "No linked memory was found.",
    "entIdentity": "Identity",
    "entIdentityNote": (
        "Identity is the normalized text: different spellings of the same entity "
        "share one row and pool their links."
    ),
    "entDangling": "Links pointing at memories that do not exist",
    "entPayloadNote": "Link keys omitted (there is one per linked memory).",
    "facetAll": "All",
    # -- dashboard and users ------------------------------------------------
    "navUsers": "Users",
    "usersSub": "Who can reach the MCP, and with which tokens",
    "dashCorpus": "Corpus",
    "dashCredentials": "Credentials",
    "statMemories": "Memories",
    "statEntities": "Entities",
    "dashAuditSub": "open log",
    "dashObserve": "Metrics and time series:",
}

class Strings:
    """Template-facing string table.

    A plain dict is the wrong thing to hand Jinja: ``{{ t.copy }}`` resolves
    ``dict.copy`` (the METHOD) before the key, and the page renders
    "<built-in method copy of dict object...>". Every key that happens to
    share a name with a dict method — copy, keys, values, items, get, update,
    pop, clear — is a landmine. This exposes only the strings, so attribute
    and subscript access both mean "look up the key".
    """

    __slots__ = ("_table",)

    def __init__(self, table: dict[str, str]):
        object.__setattr__(self, "_table", table)

    def __getattr__(self, key: str) -> str:
        try:
            return self._table[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __getitem__(self, key: str) -> str:
        return self._table[key]

    def __contains__(self, key: str) -> bool:
        return key in self._table

    def __iter__(self):
        return iter(self._table)

    def keys(self):  # noqa: D102 - mapping protocol for tests/debugging
        return self._table.keys()


_TABLES = {"pt": Strings(PT), "en": Strings(EN)}


def normalize_lang(raw: str | None) -> str:
    lang = (raw or "").strip().lower()
    return lang if lang in LANGS else DEFAULT_LANG


def strings(lang: str) -> Strings:
    return _TABLES.get(normalize_lang(lang), _TABLES["pt"])


def error_message(lang: str, code: str) -> str:
    """Map a validation code to a localized message (unknown → the code)."""
    key = f"err_{code}"
    table = strings(lang)
    return table[key] if key in table else code
