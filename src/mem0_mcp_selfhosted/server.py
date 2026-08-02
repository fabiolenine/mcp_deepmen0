"""FastMCP server for mem0-mcp-selfhosted.

Orchestrates: tool registration → transport → lazy Memory init on first call.
Memory initialization is deferred to the first tool invocation via _ensure_memory(),
allowing the server to respond to MCP initialize/tools/list without live infrastructure.
All 15 MCP tools + memory_assistant prompt.

Async ingest (v0.4): add_memory with infer=true enqueues into a durable SQLite
queue and acks immediately with a task_id envelope; a serial background worker
(ingest_worker.py) runs the 26-37s extraction pipeline off the client's clock.
Kill switch: MEM0_ASYNC_INGEST=false restores the synchronous path.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from mem0.memory.utils import normalize_scope_id, normalize_speaker_label
from mem0_mcp_selfhosted.config import ProviderInfo, build_config, configured_language
from mem0_mcp_selfhosted.document_source import resolve_and_spool
from mem0_mcp_selfhosted.env import bool_env, env
from mem0_mcp_selfhosted.graph_tools import get_entity, search_graph
from mem0_mcp_selfhosted.helpers import (
    _mem0_call,
    call_with_graph,
    get_default_user_id,
    list_entities_facet,
    patch_gemini_parse_response,
    patch_graph_sanitizer,
    safe_bulk_delete,
)
from mem0_mcp_selfhosted.image_extract import vision_enabled
from mem0_mcp_selfhosted.ingest_queue import IngestQueue, idempotency_key
from mem0_mcp_selfhosted.ingest_worker import IngestWorker
from mem0_mcp_selfhosted.pdf_extract import EncryptedPdf, pdf_info
from mem0_mcp_selfhosted.vault import middleware as vault_middleware
from mem0_mcp_selfhosted.vault import store as vault_store


def _bulk_delete_envelope(result, message: str) -> dict:
    """Report a bulk delete honestly.

    `count` alone could not distinguish "deleted everything" from "deleted the
    first page" — the tool used to answer `count: 100` for a 250-memory scope
    and look successful. `complete` is verified by a final scan; when it is
    False the caller is told to re-run instead of guessing.
    """
    out = {"message": message, "count": result.deleted,
           "targeted": result.targeted,
           # Narrow on purpose: the vector scope is empty. It does NOT claim the
           # entity links or graph went with it — over-reading a single
           # "complete" flag is the failure mode this whole contract exists for.
           "vector_scope_drained": result.vector_scope_drained}
    if result.graph_cleaned is not None:
        out["graph_cleaned"] = result.graph_cleaned
    if result.failed_ids:
        out["failed"] = len(result.failed_ids)
        out["failed_ids"] = result.failed_ids[:20]
    if not result.vector_scope_drained:
        n = len(result.remaining_ids)
        out["remaining"] = f">={n}" if result.remaining_is_partial else n
        out["warning"] = (
            f"scope NOT drained: {'at least ' if result.remaining_is_partial else ''}"
            f"{n} memories still match the filters — re-run to continue")
    return out


def _load_metadata_contract():
    """Contrato tipado de metadata — FONTE ÚNICA em mem0_patches/metadata_contract.py.

    Vive fora deste pacote de propósito: o mesmo arquivo é importado pelos patches
    (sitecustomize), pelos scripts offline e por aqui — foi a DIVERGÊNCIA entre
    definições de "válido" que deixou importance='high' entrar no corpus.
    Em produção o PYTHONPATH do unit systemd já inclui mem0_patches; fora dele
    (testes, checkout do repo) resolve pelo diretório irmão. Falha de resolução
    LEVANTA no import: enforcement que some em silêncio é o bug original.
    """
    try:
        import metadata_contract as _mc
        return _mc
    except ImportError:
        pass
    import importlib.util
    candidate = Path(__file__).resolve().parents[3] / "mem0_patches" / "metadata_contract.py"
    spec = importlib.util.spec_from_file_location("metadata_contract", candidate)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"metadata_contract não encontrado (nem no PYTHONPATH nem em {candidate}). "
            "Adicione mem0_patches ao PYTHONPATH — sem ele a validação de metadata "
            "ficaria desligada em silêncio.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metadata_contract = _load_metadata_contract()

logger = logging.getLogger(__name__)

# --- Contrato de MESSAGES na fronteira de escrita -------------------------
# Espelha MEM0_METADATA_CONTRACT em vocabulário, com UMA diferença deliberada:
# o modo é resolvido no BOOT, não por requisição. O precedente da metadata chama
# `metadata_contract.mode()` dentro do validador, então um valor de env inválido
# só explode na PRIMEIRA ESCRITA — o serviço sobe "saudável" e mente até alguém
# tentar gravar.
MESSAGE_CONTRACT_MODES = ("enforce", "warn", "off")
MESSAGE_CONTRACT_ENV = "MEM0_MESSAGE_CONTRACT"
# Nasce em "warn": hoje uma mensagem texto+imagem CONSERVA o texto, e estrear em
# "enforce" converteria isso em erro duro sem auditar chamadores. Promoção só com
# evidência medida (ver roadmap: contador `pass` + zero `warn_mixed` na janela).
_message_contract_mode: str = "warn"


# --- Proveniência CARIMBADA NO BOOT --------------------------------------
# `_provenance()` diz que lê "de DENTRO", mas `fork_sha` e os `sha_*` saem de
# `git rev-parse` e `open().read()` — do DISCO, em tempo de requisição. Medido em
# 30/07/2026: um processo iniciado 13:32:07Z reportando um commit de 17:41:42Z,
# 4h mais novo que ele. Um `/health` assim afirmaria que um restart "pegou a
# mudança" mesmo sem restart nenhum, porque o disco muda sozinho.
# Isto carimba na CONSTRUÇÃO. Os campos read-at-request continuam em
# `provenance` (rotulados por lá); estes são do processo.
_BOOT_PROVENANCE: dict = {}


def _stamp_boot_provenance() -> None:
    """Congela, no boot, o que ESTE processo carregou. Nunca levanta."""
    import hashlib
    import subprocess

    out: dict = {"stamped_at": datetime.now(timezone.utc).isoformat()}
    try:
        import mem0

        raiz = os.path.dirname(os.path.dirname(os.path.abspath(mem0.__file__)))
        out["mem0_root"] = raiz
        try:
            out["boot_fork_sha"] = subprocess.check_output(
                ["git", "-C", raiz, "rev-parse", "--short", "HEAD"],
                text=True, stderr=subprocess.DEVNULL).strip()
            out["boot_tree_dirty"] = bool(subprocess.check_output(
                ["git", "-C", raiz, "status", "--porcelain"],
                text=True, stderr=subprocess.DEVNULL).strip())
        except Exception:
            out["boot_fork_sha"] = None
            out["boot_tree_dirty"] = None
        # TODOS os hashes que `_provenance()` publica, não só main.py: um
        # carimbo parcial deixa o resto seguindo o disco sob nome de processo.
        # Caminhos IDÊNTICOS aos que `_provenance()` usa. Eu tinha escrito
        # "mem0/memory/entity_extraction.py", que não existe — o arquivo é
        # `mem0/utils/entity_extraction.py` — e o carimbo saía `null` em
        # silêncio, justamente no campo criado para acabar com silêncio.
        for rel in ("mem0/memory/main.py",
                    "mem0/utils/entity_extraction.py",
                    "mem0/utils/spacy_models.py"):
            nome = f"boot_sha_{os.path.basename(rel)}"
            try:
                with open(os.path.join(raiz, rel), "rb") as f:
                    out[nome] = hashlib.sha256(f.read()).hexdigest()[:12]
            except Exception:
                out[nome] = None
    except Exception as exc:
        out["error"] = str(exc)[:60]

    # ⚠️ O MCP também. Tudo acima identifica o FORK; nada identificava ESTE
    # pacote, então a sonda respondia "qual core subiu" e ficava muda sobre
    # "qual servidor subiu" — e é o servidor que decide escopo, autorização e
    # contrato de resposta. Um deploy do MCP era verificável só pelo
    # comportamento, o que serve para o smoke de hoje e para nada amanhã.
    try:
        raiz_mcp = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        out["mcp_root"] = raiz_mcp
        try:
            out["boot_mcp_sha"] = subprocess.check_output(
                ["git", "-C", raiz_mcp, "rev-parse", "--short", "HEAD"],
                text=True, stderr=subprocess.DEVNULL).strip()
            out["boot_mcp_tree_dirty"] = bool(subprocess.check_output(
                ["git", "-C", raiz_mcp, "status", "--porcelain"],
                text=True, stderr=subprocess.DEVNULL).strip())
        except Exception:
            out["boot_mcp_sha"] = None
            out["boot_mcp_tree_dirty"] = None
        for rel in ("server.py", "helpers.py"):
            nome = f"boot_sha_mcp_{rel}"
            try:
                with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       rel), "rb") as f:
                    out[nome] = hashlib.sha256(f.read()).hexdigest()[:12]
            except Exception:
                out[nome] = None
    except Exception as exc:
        out["mcp_error"] = str(exc)[:60]

    # `boot_tree_dirty` é o campo que importa para rollback: árvore suja no boot
    # significa que o processo NÃO corresponde a commit nenhum e é
    # irreconstituível — não existe last-known-good a apontar.
    _BOOT_PROVENANCE.clear()
    _BOOT_PROVENANCE.update(out)


_DISK_READ_AT_REQUEST = ("fork_sha", "sha_entity_extraction.py",
                         "sha_spacy_models.py", "sha_main.py")


def _relabel_disk_fields(prov: dict) -> dict:
    """ACRESCENTA `disk_*` para o que `_provenance()` lê do DISCO por requisição.

    O docstring de `_provenance()` diz "lido de DENTRO", mas estes campos saem de
    `git rev-parse` e `open().read()` NO MOMENTO DA CHAMADA — seguem o disco, não
    o processo. Medido: processo de 13:32:07Z reportando commit de 17:41:42Z.
    Deixá-los só com o nome antigo, ao lado de `boot_provenance`, manteria a
    leitura enganosa exatamente onde ela induziu erro.

    ADITIVO, não substituto, por duas razões concretas: `fork_sha` é asserido por
    `tests/unit/test_health_route.py` (contrato publicado da sonda), e esse mesmo
    arquivo está no stash de outra sessão — renomear quebraria o teste E criaria
    conflito no `stash pop`. O nome honesto passa a existir; o antigo continua
    válido e marcado.
    """
    out = dict(prov)
    for k in _DISK_READ_AT_REQUEST:
        if k in out:
            out[f"disk_{k}"] = out[k]
    out["_leitura"] = ("disk_* = lido do disco NESTA requisição (segue o disco, "
                       "não o processo); veja boot_provenance para o que ESTE "
                       "processo carregou")
    out["_deprecado"] = (
        f"OBSOLETOS, use os equivalentes disk_* ou boot_provenance: "
        f"{', '.join(_DISK_READ_AT_REQUEST)}. Mantidos porque "
        f"tests/unit/test_health_route.py os assere como contrato publicado da "
        f"sonda. NÃO são identidade do processo, apesar do nome: medido um "
        f"processo iniciado 13:32:07Z reportando um commit feito 17:41:42Z."
    )
    return out


def _parse_message_contract_mode() -> str:
    """Resolve o modo UMA vez, no boot. Valor inválido LEVANTA."""
    m = (env(MESSAGE_CONTRACT_ENV, "") or "warn").strip().lower()
    if m not in MESSAGE_CONTRACT_MODES:
        raise ValueError(
            f"{MESSAGE_CONTRACT_ENV}={m!r} inválido — use um de {list(MESSAGE_CONTRACT_MODES)}")
    return m


def _image_parts_in(content) -> int:
    """Quantas partes de imagem esta `content` carrega (lista OU dict solto)."""
    if isinstance(content, list):
        return sum(1 for p in content
                   if isinstance(p, dict) and p.get("type") == "image_url")
    if isinstance(content, dict) and content.get("type") == "image_url":
        return 1
    return 0


def _text_survives(content) -> bool:
    """Sobra algo utilizável depois que o core descartar as imagens?

    Espelha exatamente o que `parse_vision_messages` faz com visão desligada:
    junta as partes `text` e descarta o resto.
    """
    if isinstance(content, list):
        # `isinstance(text, str)` SEM exigir truthy: o core preserva a string
        # vazia (text_parts=[""] é lista não-vazia, então a mensagem sobrevive
        # com content=""). Exigir truthy aqui recusaria payload que o core
        # manteria — divergência medida contra o fork.
        return any(isinstance(p, dict) and p.get("type") == "text"
                   and isinstance(p.get("text"), str)
                   for p in content)
    return isinstance(content, str) and bool(content)


def _observe_message_contract(verdict: str) -> None:
    """Emite `stage=message_contract` no canal do Patch 6. Best-effort, nunca levanta.

    O `pass` NÃO é ruído: é o contador de LIVENESS. Sem ele, "zero warn_mixed na
    janela" não distingue "ninguém manda imagem" de "o validador nunca rodou" — e
    é essa distinção que autoriza a promoção warn -> enforce.
    """
    url = env("MEM0_OBSERVE_URL")
    if not url:
        return
    event = {
        "service": "mem0", "stage": "message_contract",
        "verdict": verdict, "mode": _message_contract_mode,
        "_timestamp": int(time.time() * 1_000_000),
    }

    def _push():
        try:
            import requests

            user, pw = env("MEM0_OBSERVE_USER"), env("MEM0_OBSERVE_PASS")
            requests.post(url, json=[event],
                          auth=(user, pw) if user else None, timeout=3)
        except BaseException:
            pass

    # EM THREAD: isto roda no caminho quente de todo add_memory, e o contrato da
    # tool é ack imediato. Um POST síncrono de até 3s (OpenObserve lento ou fora
    # do ar) atrasaria a resposta "queued" pelo tempo do timeout — observabilidade
    # não pode custar a garantia que ela existe para observar.
    threading.Thread(target=_push, daemon=True).start()


_MESSAGE_CONTRACT_ERROR = (
    "add_memory não ingere imagens: a parte image_url seria descartada em "
    "silêncio (a visão do core está desligada). Use add_document(file_path=...) "
    "para PNG/JPEG/PDF — a transcrição roda no VLM local."
)


def _validate_messages_shape(messages: list[dict] | None) -> str | None:
    """FRONTEIRA DE ESCRITA: recusa o que evaporaria em silêncio.

    O core (`parse_vision_messages` com `enable_vision=False`) DESCARTA partes de
    imagem; se a mensagem só tinha imagem, ela some inteira. O cliente recebia
    `{"status":"queued"}` e nunca ficava sabendo — o submit é o único ponto onde
    ainda dá para devolver um erro acionável.

    Regra que atravessa os modos, e o porquê da assimetria:

    ========================  enforce   warn                off
    texto + imagem            recusa    aceita + loga       aceita mudo
    IMAGEM-SÓ                 recusa    RECUSA              aceita mudo

    Imagem-só é recusada em `warn` porque aceitar ali reproduziria exatamente o
    bug que este contrato existe para consertar: `queued` para um payload que
    provadamente vira zero fato.

    Em `off`, NÃO: `off` é o KILL SWITCH do rollout e restaura o comportamento
    pré-contrato por inteiro. Um kill switch que não desliga não é kill switch —
    rollback parcial é pior que nenhum, porque quem o aciona num incidente espera
    o estado anterior e recebe um subconjunto dele. Quem não quer a mentira do
    ack não põe `off`.
    """
    mixed = image_only = 0
    for msg in (messages or []):
        if not isinstance(msg, dict):
            continue  # container malformado é problema do core, não deste contrato
        if msg.get("role") == "system":
            continue  # o core repassa system INTOCADA, imagem e tudo
        content = msg.get("content")
        n = _image_parts_in(content)
        if not n:
            continue
        if _text_survives(content):
            mixed += n
        else:
            image_only += n
    if _message_contract_mode == "off":
        # ANTES de qualquer veredito: com o contrato desligado nada foi avaliado,
        # e emitir `pass` aqui inflaria o piso de liveness do P7 com tráfego não
        # verificado — exatamente o motivo de `off` existir no enum. A versão
        # anterior emitia `pass` para add sem imagem antes de checar o modo.
        _observe_message_contract("off")
        return None
    if not (mixed or image_only):
        _observe_message_contract("pass")
        return None
    if image_only:
        _observe_message_contract("reject_image_only")
        return _MESSAGE_CONTRACT_ERROR
    if _message_contract_mode == "enforce":
        _observe_message_contract("reject_mixed")
        return _MESSAGE_CONTRACT_ERROR
    if _message_contract_mode == "warn":
        _observe_message_contract("warn_mixed")
        logger.warning(
            "message_contract (warn): %d parte(s) image_url serão descartadas pelo "
            "core; o texto sobrevive. %s", mixed, _MESSAGE_CONTRACT_ERROR)
    return None

# Instante de início DESTE processo. O harness lia `ActiveEnterTimestamp` do
# systemd, que prova idade do processo — não qual código ele carregou. Junto com
# os hashes de _provenance(), isto responde "o restart pegou a mudança?".
_STARTED_AT = datetime.now(timezone.utc).isoformat()

# --- Globals set during startup ---
memory = None
mcp: FastMCP | None = None
_enable_graph_default = False

# --- Lazy init state ---
_memory_init_lock = threading.Lock()
_last_init_failure: float = 0.0
_INIT_RETRY_COOLDOWN = 30.0  # seconds before retrying after a failed init

# --- Async ingest state ---
_ingest_queue: IngestQueue | None = None
_ingest_worker: IngestWorker | None = None
_ingest_lock = threading.Lock()


def _async_ingest_enabled() -> bool:
    return bool_env("MEM0_ASYNC_INGEST", "true")


def _queue_db_path() -> str:
    explicit = env("MEM0_QUEUE_DB_PATH")
    if explicit:
        return explicit
    history = env("MEM0_HISTORY_DB_PATH")
    if history:
        return str(Path(history).parent / "ingest_queue.db")
    return str(Path.home() / ".mem0" / "ingest_queue.db")


def _vault_db_path() -> str:
    """Vault database shared with the :8080 UI (same absolute path in both units)."""
    explicit = env("MEM0_VAULT_DB_PATH")
    if explicit:
        return explicit
    return str(Path.home() / ".mem0" / "vault.db")


def _auth_mode() -> str:
    """Effective MEM0_REQUIRE_AUTH mode; a typo raises instead of downgrading."""
    return vault_middleware.normalize_mode(env("MEM0_REQUIRE_AUTH"))


def _current_principal() -> dict[str, Any] | None:
    """The vault principal for the in-flight request, if the gate resolved one.

    Two lookups because the MCP SDK dispatches tool calls on a task of its own:
    the middleware's contextvar may not survive that hop, but the ASGI scope
    always does (the SDK hands the Starlette request to the request context).
    """
    principal = vault_middleware.current_principal.get()
    if principal is not None:
        return principal
    try:
        from mcp.server.lowlevel.server import request_ctx

        request = request_ctx.get().request
        if request is not None:
            return request.scope.get("state", {}).get("vault_user")
    except Exception:  # noqa: BLE001 - no HTTP request in flight (stdio/tests)
        return None
    return None


def _effective_user_id(user_id: str | None) -> tuple[str, str | None]:
    """Resolve the memory scope for a tool call. Returns ``(uid, error)``.

    Authorization contract: a token bound to a ``mem0_user_id`` WINS over
    whatever the client asks for, and a client that explicitly asks for a
    different scope gets an error — never a silent redirect to its own data.
    A token with an empty ``mem0_user_id`` (every token in the current phase)
    keeps today's behavior exactly: client value, else MEM0_USER_ID.

    NORMALIZA o valor do cliente, e é o lugar certo para isso porque é o funil
    ÚNICO de ``user_id``: toda tool escopada passa por aqui. Duas coisas
    dependem disso:

    * o filtro do vector store é casamento EXATO — `" alice"` não casa `"alice"`
      e a diferença não vira erro, vira resultado vazio (MEDIDO no corpus:
      um `user_id` real casa 1159 pontos e o MESMO com espaço à esquerda, 0);
    * a comparação de vínculo logo abaixo é `!=` sobre a string crua, então um
      `user_id` com espaço colado seria recusado como "token não pode acessar
      esse escopo" — mensagem errada para um defeito de digitação.
    """
    try:
        user_id = normalize_scope_id(user_id, "user_id")
    except ValueError as exc:
        return "", str(exc)

    principal = _current_principal()
    bound = (principal or {}).get("mem0_user_id") or ""
    if bound:
        if user_id and user_id != bound:
            return "", (
                f"token is bound to user_id '{bound}' and cannot access '{user_id}'"
            )
        resolvido, origem = bound, "o user_id amarrado ao token"
    elif user_id:
        resolvido, origem = user_id, "o argumento da tool"
    else:
        resolvido, origem = get_default_user_id(), "MEM0_USER_ID"

    # ⚠️ O valor RESOLVIDO também passa pela regra, não só o do cliente. Os dois
    # outros caminhos entravam crus: o default vem de `env()`, que apara só as
    # BORDAS (`MEM0_USER_ID="a b"` chegava como `'a b'`, e `"   "` como `''`,
    # ambos sem erro); e o valor amarrado ao token vem do cofre, que nunca
    # aplicou esta regra. Um escopo com espaço interno não casa nada no store e
    # não vira erro — vira resultado vazio em TODA tool escopada.
    # Normalizar é idempotente, então re-aplicar sobre o valor do cliente,
    # que já passou, é no-op.
    try:
        resolvido = normalize_scope_id(resolvido, "user_id")
    except ValueError as exc:
        return "", f"{exc} (origem: {origem})"
    return resolvido or "", None


def _normalized_scope(agent_id, run_id) -> tuple[str | None, str | None, str | None]:
    """Normaliza ``agent_id``/``run_id``. Retorna ``(agent_id, run_id, erro)``.

    ``user_id`` não entra aqui: ele tem funil próprio em `_effective_user_id`.
    Estes dois não têm, e são usados crus para montar o filtro de delete — que
    é onde escopo errado é mais caro, porque a resposta de um delete que não
    casou nada é indistinguível de um delete bem-sucedido de um escopo vazio.
    """
    try:
        return (normalize_scope_id(agent_id, "agent_id"),
                normalize_scope_id(run_id, "run_id"), None)
    except ValueError as exc:
        return None, None, str(exc)


#: How each tool decides what a request may touch. Every registered tool MUST
#: appear here — a completeness test fails the build otherwise, so tool #16
#: cannot quietly ship without an authorization decision.
#:
#:   scope-arg     — takes a user_id argument; resolved by _effective_user_id
#:   record-owner  — addresses a record by id; the record's owner is checked
#:   filtered      — enumerates; the answer is narrowed to the bound scope
#:   operator-only — global/operational; refused to a scope-bound token
#: Teto de `limit` do `get_memories`. O parâmetro esteve MORTO desde sempre
#: (caía no `**kwargs` do core e sumia), então nunca houve validação alguma —
#: ativá-lo é a primeira vez que um valor do chamador chega ao store por esse
#: caminho. O teto barra o pedido acidental de milhões, que faria o Qdrant
#: varrer a collection inteira.
#:
#: ⚠️ ISTO É UMA PRIMEIRA PÁGINA COM TETO, NÃO PAGINAÇÃO. `Memory.get_all` é
#: `(*, filters=None, top_k=20, **kwargs)` — o core NÃO expõe offset nem cursor,
#: então não há como alcançar o que fica além do teto por esta tool. MEDIDO em
#: 02/08/2026: um escopo de produção real tem 1202 memórias, ou seja, **o teto
#: morde de verdade** — 202 ficam inalcançáveis aqui.
#:
#: Isso NÃO é regressão: antes desta mudança o `limit` era descartado e a tool
#: devolvia 20 sempre, então o alcance era 20, não 1202. Mas também não é
#: "generoso" (uma versão anterior deste comentário afirmava isso, e era falso
#: nas duas pontas: não há paginação, e 1000 < corpus). Quem precisa varrer o
#: escopo inteiro deve usar o Qdrant direto ou ganhar uma tool com cursor —
#: decisão de projeto em aberto, não resolvida aqui.
MAX_GET_LIMIT = 1000


def _com_actor_id(filters: dict | None, actor_id: str | None) -> tuple[dict | None, str | None]:
    """Dobra `actor_id` dentro de `filters`. Devolve `(filters, erro)`.

    ⚠️ Vai no `filters`, não como kwarg de topo: `Memory.search` NÃO declara
    `actor_id`, e o lift de kwargs do core cobre só `user_id`/`agent_id`/`run_id`.
    Um `actor_id=` de topo seria ENGOLIDO em silêncio — a mesma classe do `limit`
    morto do `get_memories`.

    ⚠️ Precedência por `is not None`, NÃO por truthiness. Com
    `actor_id or filters["actor_id"]`, um `actor_id=""` explícito cairia
    silenciosamente no filtro do chamador em vez de falhar na validação — o
    chamador pediria uma coisa e receberia outra sem aviso.

    ⚠️ O `filters` do chamador é COPIADO. Mutar o dict que ele nos passou
    mudaria estado que não é nosso.

    A validação reusa `normalize_speaker_label` do core (mesmo precedente do
    `normalize_scope_id`): escrita e consulta canonizando diferente é filtro
    exato que erra em silêncio, porque o Qdrant casa por igualdade.
    """
    if actor_id is None:
        return filters, None
    rotulo = normalize_speaker_label(actor_id)
    if rotulo is None:
        return None, (
            f"actor_id inválido: {actor_id!r}. Um rótulo de locutor é texto não-vazio, "
            "sem quebra de linha, com letras/dígitos/espaço e `. - _ '`."
        )
    novo = dict(filters) if filters else {}
    novo["actor_id"] = rotulo
    return novo, None


TOOL_SCOPE_POLICY: dict[str, str] = {
    "add_memory": "scope-arg",
    "add_document": "scope-arg",
    "search_memories": "scope-arg",
    "get_memories": "scope-arg",
    "delete_all_memories": "scope-arg",
    "delete_entities": "scope-arg",
    "update_memory": "record-owner",
    "get_memory": "record-owner",
    "memory_history": "record-owner",
    "delete_memory": "record-owner",
    "memory_task_status": "record-owner",
    "list_entities": "filtered",
    "memory_queue_status": "operator-only",
    "mcp_search_graph": "operator-only",
    "mcp_get_entity": "operator-only",
}


def _bound_scope() -> str:
    """The mem0 scope this request is locked to; '' when the token is unbound.

    Every token issued today is unbound, so all the checks below are no-ops
    with zero extra I/O — they exist so binding a token is a complete change
    and not a half-enforced one.
    """
    return (_current_principal() or {}).get("mem0_user_id") or ""


def _authorize_owner(owner: str | None) -> str | None:
    """None when the in-flight principal may touch a record owned by ``owner``."""
    bound = _bound_scope()
    if not bound or (owner or "") == bound:
        return None
    return "not accessible with this token"


def _authorize_memory_id(mem: Any, memory_id: str) -> str | None:
    """Ownership guard for id-addressed tools. One extra read, bound tokens only.

    A record owned by someone else and a record that does not exist give the
    SAME answer: a scoped token must not be able to probe which ids are real.
    """
    if not _bound_scope():
        return None
    try:
        record = mem.get(memory_id)
    except Exception:  # noqa: BLE001 - treat a failed lookup as not-found
        record = None
    if not record or _authorize_owner(record.get("user_id")):
        return f"memory not found: {memory_id}"
    return None


def _operator_only(tool: str) -> str | None:
    """Refuse global/operational tools to a scope-bound token."""
    if not _bound_scope():
        return None
    return (
        f"{tool} reports on the whole server and is not available to a "
        f"scope-bound token"
    )


def _get_ingest() -> tuple[IngestQueue, IngestWorker]:
    """Lazy-create the queue + worker pair (thread-safe, idempotent).

    The worker thread autostarts unless MEM0_QUEUE_WORKER=false (tests disable
    it to exercise the queue without a live consumer racing them).
    """
    global _ingest_queue, _ingest_worker
    with _ingest_lock:
        if _ingest_queue is None:
            _ingest_queue = IngestQueue(_queue_db_path())
            _ingest_worker = IngestWorker(
                _ingest_queue,
                _ensure_memory,
                call_with_graph=call_with_graph,
            )
            if bool_env("MEM0_QUEUE_WORKER", "true"):
                _ingest_worker.start()
        return _ingest_queue, _ingest_worker


_VALID_MEMORY_SCOPES = {"user_fact", "system_meta", "eval_meta", "project_meta"}
_VALID_SCOPE_EVIDENCE = {"decisive", "strong", "weak", "conflicting", "none"}


def _validate_scope_metadata(metadata: dict | None) -> str | None:
    """Contrato memory_scope v1 (Passo 2 — campo PASSIVO, sem routing).

    memory_scope é enum de 4 valores ou null/ausente (abstention); evidência é
    NÍVEL, nunca decimal. Se o caller fornece memory_scope válido, carimba a
    proveniência default (version=1, source=manual) sem sobrescrever o que veio.
    Retorna mensagem de erro ou None se ok (metadata é mutada in-place).
    """
    if not metadata:
        return None
    if "memory_scope" in metadata:
        scope = metadata["memory_scope"]
        if scope is not None and scope not in _VALID_MEMORY_SCOPES:
            return (f"memory_scope inválido: {scope!r} — use um de "
                    f"{sorted(_VALID_MEMORY_SCOPES)} ou null (abstention)")
        if scope is None:
            metadata.pop("memory_scope")  # ausência == null; não persistir a chave
        else:
            metadata.setdefault("memory_scope_version", 1)
            metadata.setdefault("memory_scope_source", "manual")
    ev = metadata.get("memory_scope_evidence")
    if ev is not None and ev not in _VALID_SCOPE_EVIDENCE:
        return (f"memory_scope_evidence inválido: {ev!r} — use nível "
                f"{sorted(_VALID_SCOPE_EVIDENCE)} (nunca decimal não-calibrado)")
    return None


def _validate_metadata_contract(metadata: dict | None) -> str | None:
    """FRONTEIRA DE ESCRITA (26/07/2026): valida a metadata do caller antes de
    qualquer persistência — o submit é o único ponto onde ainda dá para devolver
    um erro ACIONÁVEL a quem escreveu.

    Cobre o contrato de scope (v1) + os campos TIPADOS
    (importance/confidence/domain/memory_type/tags). Chave LIVRE continua
    passando: 940 pontos do corpus usam campos livres (project, subcategory,
    topic...), restringi-los quebraria o padrão de escrita de quase tudo.

    Por que rejeitar em vez de coagir: coerção silenciosa esconde o bug do
    caller e foi assim que 17 memórias marcadas "alta importância" viraram as
    primeiras a sumir de um filtro de alta importância — o erro só apareceu
    quando o Open WebUI tentou recuperá-las. MEM0_METADATA_CONTRACT=warn aceita
    e loga; =off desliga todas as camadas.
    """
    scope_err = _validate_scope_metadata(metadata)
    if scope_err:
        return scope_err
    try:
        mode = metadata_contract.mode()
    except ValueError as exc:  # env com valor inválido: levanta, não vira "off"
        logger.error("metadata_contract: %s", exc)
        raise
    if mode == "off":
        return None
    typed_err = metadata_contract.validate(metadata)
    if typed_err and mode == "warn":
        logger.warning("metadata_contract (warn): %s", typed_err)
        return None
    return typed_err


def _estimate_wait_s(queue: IngestQueue) -> int:
    """Kind-aware drain estimate: conversations cost EST_ADD_S each, documents
    cost EST_CHUNK_S per remaining chunk — queue_depth × 40s lies by an order
    of magnitude once a document is in line."""
    # Defaults recalibrados no host de referência (jul/2026) por perda ASSIMÉTRICA:
    # subestimar faz o cliente MCP dar retry (quase gerou add duplicado no passado);
    # superestimar só faz o cliente esperar/pollar. Por isso p75, não p50.
    #  - ADD 180: service time real started->finished p75=184s (n=43, jul/2026;
    #    p50=64/p90=279/max=391; NÃO inclui queue-wait, medido p50=0).
    #  - CHUNK 120: digital-quente ~120s/chunk (n=39); OCR 75 (infer=false) a 135
    #    (infer=true+swaps de modelo) — valor conservador único, extração GPU-bound.
    #  - UPDATE 200 PROVISÓRIO: só n=2 (158/197s, classificador inline domina).
    # Ver docs/mem0-docs/cpu-upgrade-remedicao.md (registro autoritativo).
    est_add = int(env("MEM0_QUEUE_EST_ADD_S", "180"))
    est_chunk = int(env("MEM0_DOC_EST_CHUNK_S", "120"))
    est_update = int(env("MEM0_QUEUE_EST_UPDATE_S", "200"))
    try:
        by_kind = queue.queue_status().get("depth_by_kind", {})
        conversations = by_kind.get("conversation", 0)
        updates = by_kind.get("update", 0)
        doc_chunks = queue.pending_document_chunks()
        return conversations * est_add + updates * est_update + doc_chunks * est_chunk
    except Exception:
        return queue.depth() * est_add


def register_providers(providers_info: list[ProviderInfo]) -> None:
    """Register custom LLM providers with mem0ai's LlmFactory.

    Maps provider names to their config classes and registers each.
    Config classes are lazy-imported to avoid pulling in unnecessary
    dependencies (e.g. ``anthropic`` package in Ollama-only mode).
    Safe to call multiple times (LlmFactory.register_provider is idempotent).
    """
    if not providers_info:
        return

    from mem0.utils.factory import LlmFactory

    for pi in providers_info:
        config_class = _resolve_config_class(pi["name"])
        if config_class is None:
            logger.warning("No config class for provider %r, skipping", pi["name"])
            continue
        LlmFactory.register_provider(
            name=pi["name"],
            class_path=pi["class_path"],
            config_class=config_class,
        )


def _resolve_config_class(provider_name: str) -> type | None:
    """Lazy-resolve the config class for a provider name.

    Imports are deferred so that unnecessary packages (e.g. ``anthropic``)
    are never loaded in a pure-Ollama setup.
    """
    if provider_name == "ollama":
        from mem0.configs.llms.ollama import OllamaConfig

        return OllamaConfig
    if provider_name in ("anthropic", "anthropic_oat"):
        from mem0_mcp_selfhosted.llm_anthropic import AnthropicOATConfig

        return AnthropicOATConfig
    return None


def _init_memory() -> Any:
    """Initialize mem0ai Memory with config and registered providers."""
    global memory, _enable_graph_default

    config_dict, providers_info, split_config = build_config()

    register_providers(providers_info)

    # Patch mem0ai's relationship sanitizer before Memory init
    patch_graph_sanitizer()
    patch_gemini_parse_response()

    # Initialize Memory
    from mem0 import Memory

    memory = Memory.from_config(config_dict)

    # If split-model was requested, swap the graph LLM with the router
    if split_config and memory.graph is not None:
        from mem0_mcp_selfhosted.llm_router import SplitModelGraphLLM, SplitModelGraphLLMConfig

        router_config = SplitModelGraphLLMConfig(**split_config)
        memory.graph.llm = SplitModelGraphLLM(router_config)

    _enable_graph_default = bool_env("MEM0_ENABLE_GRAPH")
    return memory


def _ensure_memory() -> Any:
    """Lazy-initialize Memory on first tool call. Thread-safe with retry-after-delay.

    Returns the Memory instance, or None if initialization failed.
    After a failure, waits ``_INIT_RETRY_COOLDOWN`` seconds before retrying.
    Matches the lazy-init pattern used by ``graph_tools._get_driver()``.
    """
    global memory, _last_init_failure

    if memory is not None:
        return memory

    now = time.monotonic()
    if _last_init_failure and (now - _last_init_failure < _INIT_RETRY_COOLDOWN):
        return None  # Too soon to retry

    with _memory_init_lock:
        # Double-check after acquiring lock
        if memory is not None:
            return memory

        try:
            _init_memory()
            logger.info("mem0ai Memory initialized successfully (lazy)")
        except Exception as exc:
            _last_init_failure = time.monotonic()
            logger.error("Lazy Memory init failed: %s", exc)
            return None

    return memory


def _create_server() -> FastMCP:
    """Create and configure the FastMCP server with all tools and prompts."""
    global mcp, _message_contract_mode

    # Resolve o contrato de messages AQUI, não por requisição: valor inválido tem
    # que derrubar o boot, não a primeira escrita de um cliente desavisado.
    _message_contract_mode = _parse_message_contract_mode()
    _stamp_boot_provenance()

    host = env("MEM0_HOST", "0.0.0.0")
    port = int(env("MEM0_PORT", "8081"))

    mcp = FastMCP(
        "DeepMem0",
        host=host,
        port=port,
        instructions=(
            "DeepMem0 — memory tools for persistent cross-session memory. "
            "Use search_memories to find relevant context before starting work. "
            "Use add_memory to store important facts, preferences, and decisions; "
            "with infer=true (default) it is asynchronous — it acks immediately with a "
            "task_id while extraction runs in background (use memory_task_status to get "
            "the resulting memory_ids, memory_queue_status for queue health; a search "
            "response's pending_ingest field warns when queued facts are not searchable yet). "
            "Use add_document to ingest a PDF or image (PNG/JPEG) from a local path — "
            "facts are extracted per chunk with document/page provenance (scanned pages "
            "and images are read by a local vision model when MEM0_ENABLE_VISION is on); "
            "documents take minutes, poll memory_task_status for chunks_done progress. "
            "Use get_memories to browse stored memories with filters. "
            "Use search_graph to find relationships between entities. "
            "Use get_memory to retrieve a specific memory by ID. "
            "Use update_memory to modify existing memories. "
            "Use list_entities to see who/what has stored memories."
        ),
    )

    _register_tools(mcp)
    _register_prompts(mcp)
    _register_health(mcp)

    return mcp


def _register_health(mcp: FastMCP) -> None:
    """Liveness + readiness, exempt from the vault gate (see middleware).

    Readiness reports the vault database because ``MEM0_REQUIRE_AUTH=on`` with
    an unreadable vault denies every request — an operator must be able to see
    that from outside without reading logs.
    """
    from starlette.responses import JSONResponse

    def _provenance() -> dict:
        """O que ESTE processo carregou — não o que está no disco agora.

        O harness de eval rodava `git -C <fork> rev-parse` e chamava aquilo de
        proveniência. Não é: prova o estado do DISCO no momento em que o harness
        roda, e não que o serviço importou aquele código, nem que a árvore estava
        limpa, nem qual modelo spaCy foi carregado. Um restart no meio invalida a
        conclusão sem deixar rastro.

        Isto é lido de DENTRO: `mem0.__file__` diz de onde o módulo veio, os
        hashes dizem QUAL versão dos arquivos que importam está em memória, e as
        collections efetivas dizem em que dados o processo está mexendo.

        Cada campo falha em silêncio para o próprio valor de erro: /health é
        sonda de saúde e não pode virar 500 por causa de metadados.
        """
        import hashlib
        import subprocess

        out: dict = {"pid": os.getpid(), "started_at": _STARTED_AT}
        try:
            import mem0
            out["mem0_file"] = getattr(mem0, "__file__", None)
            out["is_deepmem0"] = bool(getattr(mem0, "__deepmem0__", False))
            raiz = os.path.dirname(os.path.dirname(os.path.abspath(mem0.__file__)))
            out["mem0_root"] = raiz
            try:
                sha = subprocess.check_output(
                    ["git", "-C", raiz, "rev-parse", "--short", "HEAD"],
                    text=True, stderr=subprocess.DEVNULL).strip()
                sujo = subprocess.check_output(
                    ["git", "-C", raiz, "status", "--porcelain"],
                    text=True, stderr=subprocess.DEVNULL).strip()
                out["fork_sha"] = f"{sha}-dirty" if sujo else sha
            except Exception as exc:
                out["fork_sha"] = f"erro: {str(exc)[:60]}"
            # Hash dos arquivos que decidem o vocabulário de entidades: é o que
            # distingue "o restart pegou a correção" de "não pegou".
            for rel in ("mem0/utils/entity_extraction.py", "mem0/utils/spacy_models.py",
                        "mem0/memory/main.py"):
                try:
                    with open(os.path.join(raiz, rel), "rb") as f:
                        out[f"sha_{os.path.basename(rel)}"] = \
                            hashlib.sha256(f.read()).hexdigest()[:12]
                except Exception:
                    out[f"sha_{os.path.basename(rel)}"] = None
        except Exception as exc:
            out["mem0_file"] = f"erro: {str(exc)[:60]}"

        # ⚠️ IDIOMA CONFIGURADO, e SEM CARREGAR. Três defeitos na versão que
        # chamava `get_nlp_full()` sem argumento:
        #   1. inspecionava o modelo do idioma DEFAULT (inglês) — a sonda dizia
        #      `en_core_web_sm` enquanto o deployment português rodava, o que
        #      não responde nada sobre o pipeline que importa;
        #   2. CARREGAR dispara `spacy.cli.download`, e sonda de readiness que
        #      toca a rede pendura;
        #   3. o `except` convertia a falha em METADADO, então readiness nunca
        #      reprovava por modelo ausente — o oposto do critério.
        # `entity_pipeline_status(idioma)` só pergunta "está instalado?" e
        # devolve `degraded`, que a readiness usa para responder 503.
        # Nasce degradado: se a checagem estourar, o silêncio não pode virar "ok".
        out["entity_pipeline"] = {"degraded": True, "erro": "não avaliado"}
        try:
            from mem0.utils.spacy_models import entity_pipeline_status
            st = entity_pipeline_status(configured_language())
            out["entity_pipeline"] = st
            out["spacy_model"] = st["model"]
            out["spacy_language"] = st["language"]
            try:
                import importlib.metadata as md
                out["spacy_version"] = md.version("spacy")
            except Exception:
                out["spacy_version"] = None
        except Exception as exc:
            out["spacy_model"] = f"erro: {str(exc)[:60]}"
            out["entity_pipeline"] = {"degraded": True, "erro": str(exc)[:80]}

        # Collections EFETIVAS: a de entidades é DERIVADA do nome da principal
        # (`_entity_collection_name`), não configurável — então tem que ser lida,
        # não presumida.
        try:
            # `memory` global, NÃO `_ensure_memory()`: a sonda de saúde não pode
            # DISPARAR a inicialização do Memory (Ollama + reranker, dezenas de
            # segundos) como efeito colateral. Se ainda não inicializou, o valor
            # honesto é "não inicializado", não uma inicialização forçada.
            mem = memory
            if mem is None:
                out["collection"] = "não inicializado"
                return out
            vs = getattr(mem, "vector_store", None)
            principal = getattr(vs, "collection_name", None)
            out["collection"] = principal
            try:
                from mem0.memory.main import _entity_collection_name
                out["entity_collection"] = _entity_collection_name(
                    mem.config.vector_store.provider, principal)
            except Exception:
                out["entity_collection"] = f"{principal}_entities" if principal else None
            out["language"] = getattr(mem.config, "language", None)
            out["reranker"] = bool(getattr(mem, "reranker", None))
        except Exception as exc:
            out["collection"] = f"erro: {str(exc)[:60]}"

        # Sanitiza ANTES de devolver: um valor não-serializável derruba o /health
        # com 500 no encoder JSON — falha exatamente onde a sonda deveria ser a
        # coisa mais robusta do serviço. Só tipos primitivos saem daqui.
        limpo: dict = {}
        for k, v in out.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                limpo[k] = v
            elif isinstance(v, dict):
                # `entity_pipeline` é dict e a readiness LÊ o campo `degraded`
                # dele. Achatar todo não-primitivo em string transformaria a
                # DECISÃO de readiness em texto, e `bool("False")` é True.
                limpo[k] = {
                    ik: (iv if isinstance(iv, (str, int, float, bool, type(None)))
                         else f"{type(iv).__name__}: {str(iv)[:40]}")
                    for ik, iv in v.items()
                }
            else:
                limpo[k] = f"{type(v).__name__}: {str(v)[:60]}"
        return limpo

    @mcp.custom_route("/health", methods=["GET"])
    async def health(request):  # noqa: ANN001, ARG001 - starlette signature
        try:
            mode = _auth_mode()
        except ValueError as exc:
            mode = f"invalid: {exc}"
        vault_db = vault_store.probe(_vault_db_path()) if mode != "off" else "not_required"
        prov = _relabel_disk_fields(_provenance())
        # ⚠️ READINESS REPROVA por pipeline de entidade degradado. Idioma
        # configurado sem o modelo dele não é "um pouco pior": o POS fica
        # inutilizável (verbo português volta PROPN), e a extração de entidade
        # segue rodando e gravando lixo em silêncio. `pt_core_news_sm` é
        # DEPENDÊNCIA DE RUNTIME, e a sonda é onde isso vira visível.
        # `.get(..., True)` fail-closed: campo ausente conta como degradado.
        pipeline_degradado = bool((prov.get("entity_pipeline") or {}).get("degraded", True))
        degraded = (mode == "on" and vault_db != "ok") or pipeline_degradado
        return JSONResponse(
            {
                "status": "degraded" if degraded else "ok",
                "entity_pipeline": prov.get("entity_pipeline"),
                # Informativo, NÃO um motivo de degradação: valor inválido derruba
                # a construção do servidor, e um processo que não constrói não
                # serve rota nenhuma para reportar o problema.
                "message_contract": _message_contract_mode,
                "boot_provenance": dict(_BOOT_PROVENANCE),
                "auth_mode": mode,
                "vault_db": vault_db,
                "provenance": prov,
            },
            status_code=503 if degraded else 200,
        )


# ============================================================
# Memory Tools (7 tools)
# ============================================================


def _register_tools(mcp: FastMCP) -> None:
    """Register all 15 MCP tools on the server."""

    @mcp.tool()
    def add_memory(
        text: Annotated[str, Field(description="Text to store as a memory. Converted to messages format internally.")],
        messages: Annotated[list[dict] | None, Field(description="Structured conversation history (role/content dicts). When provided, takes precedence over text.")] = None,
        user_id: Annotated[str | None, Field(description="User scope identifier. Defaults to MEM0_USER_ID.")] = None,
        agent_id: Annotated[str | None, Field(description="Agent scope identifier.")] = None,
        run_id: Annotated[str | None, Field(description="Run scope identifier.")] = None,
        metadata: Annotated[dict | None, Field(description="Arbitrary metadata JSON to store alongside the memory.")] = None,
        infer: Annotated[bool | None, Field(description="If true (default), LLM extracts key facts asynchronously: the call returns a queued envelope with a task_id immediately (use memory_task_status to fetch the resulting memory_ids). If false, stores raw text synchronously.")] = None,
        enable_graph: Annotated[bool | None, Field(description="Override default graph toggle for this call.")] = None,
    ) -> str:
        """Store a new memory. Requires at least one of user_id, agent_id, or run_id.

        Response contract (never a bare list):
        - {"status": "queued", "task_id", "submitted_at", "queue_depth", "estimated_wait_s"}
          — infer=true path; extraction runs in background, poll memory_task_status.
        - {"status": "stored", "memory_ids": [...], "results": [...]}
          — synchronous path; empty memory_ids carries "reason": "no_new_facts".
        - {"error": ...} — failure.
        """
        uid, auth_err = _effective_user_id(user_id)
        if auth_err:
            return json.dumps({"error": auth_err}, ensure_ascii=False)

        meta_err = _validate_metadata_contract(metadata)
        if meta_err:
            return json.dumps({"error": meta_err}, ensure_ascii=False)

        # ANTES de montar `msgs`, para cobrir os DOIS caminhos (async enfileirado
        # e síncrono infer=false) com uma checagem só.
        msg_err = _validate_messages_shape(messages)
        if msg_err:
            return json.dumps({"error": msg_err}, ensure_ascii=False)

        # Build messages for mem0ai
        if messages:
            msgs = messages
        else:
            msgs = [{"role": "user", "content": text}]

        eff_infer = True if infer is None else infer

        if eff_infer and _async_ingest_enabled():
            try:
                queue, worker = _get_ingest()
                params: dict[str, Any] = {}
                if metadata:
                    params["metadata"] = metadata
                # resolve the effective graph toggle NOW (the worker doesn't
                # know this server's default) and let it ride with the job
                params["enable_graph"] = enable_graph if enable_graph is not None else _enable_graph_default
                res = queue.enqueue(
                    user_id=uid, agent_id=agent_id or None, run_id=run_id or None,
                    messages=msgs, params=params,
                )
                worker.notify()
                envelope: dict[str, Any] = {
                    "status": "queued",
                    "task_id": res["task_id"],
                    "submitted_at": res["submitted_at"],
                    "queue_depth": res["queue_depth"],
                    "estimated_wait_s": _estimate_wait_s(queue),
                }
                if res["duplicate"]:
                    envelope["duplicate"] = True
                    envelope["note"] = "identical payload already queued; returning the existing task"
                return json.dumps(envelope, ensure_ascii=False)
            except Exception as exc:
                # A broken queue must not lose the fact — fall through to the
                # synchronous path and say so.
                logger.error("Async ingest enqueue failed, falling back to sync add: %s", exc)

        kwargs: dict[str, Any] = {"user_id": uid}
        if agent_id:
            kwargs["agent_id"] = agent_id
        if run_id:
            kwargs["run_id"] = run_id
        if metadata:
            kwargs["metadata"] = metadata
        if infer is not None:
            kwargs["infer"] = infer

        mem = _ensure_memory()

        def _do_add():
            return mem.add(msgs, **kwargs)

        def _do_add_enveloped():
            raw = call_with_graph(mem, enable_graph, _enable_graph_default, _do_add)
            results = raw.get("results", []) if isinstance(raw, dict) else (raw or [])
            memory_ids = [
                r["id"] for r in results
                if isinstance(r, dict) and r.get("event") in ("ADD", "UPDATE") and r.get("id")
            ]
            envelope: dict[str, Any] = {
                "status": "stored",
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "memory_ids": memory_ids,
                "results": results,
            }
            if isinstance(raw, dict) and raw.get("relations") is not None:
                envelope["relations"] = raw["relations"]
            if not memory_ids:
                envelope["reason"] = "no_new_facts"
            return envelope

        return _mem0_call(_do_add_enveloped)

    @mcp.tool()
    def add_document(
        file_path: Annotated[str, Field(description="Absolute path on the server host (must live under MEM0_DOC_PATH_ALLOWLIST, default $HOME) of a PDF or an image (PNG/JPEG). Scanned PDFs and images need vision on (MEM0_ENABLE_VISION).")],
        filename: Annotated[str | None, Field(description="Display name stored as source_doc provenance. Defaults to the file's basename.")] = None,
        user_id: Annotated[str | None, Field(description="User scope identifier. Defaults to MEM0_USER_ID.")] = None,
        agent_id: Annotated[str | None, Field(description="Agent scope identifier.")] = None,
        run_id: Annotated[str | None, Field(description="Run scope identifier.")] = None,
        metadata: Annotated[dict | None, Field(description="Extra metadata stored on every memory extracted from this document.")] = None,
        infer: Annotated[bool | None, Field(description="If true (default), the LLM extracts facts from each chunk; if false, raw chunks are stored as-is.")] = None,
        enable_graph: Annotated[bool | None, Field(description="Graph extraction per chunk. Defaults to FALSE for documents (expensive and noisy).")] = None,
        force: Annotated[bool | None, Field(description="Re-ingest even if this exact document (same bytes + scope) was already ingested.")] = None,
    ) -> str:
        """Ingest a PDF or image asynchronously: extract text per page (digital
        PDF via poppler; scanned pages and images via a local vision model),
        chunk, and extract memorable facts with document/page provenance.

        Returns immediately with {"status": "queued", task_id, pages,
        chunks_estimate, estimated_wait_s} — a large document takes many
        minutes; poll memory_task_status(task_id) for chunks_done progress and
        the final memory_ids. Re-submitting the same file returns
        {"status": "already_ingested"} unless force=true. There is NO
        synchronous fallback: if the queue is unavailable the call errors.
        """
        if not bool_env("MEM0_DOC_ENABLED", "true"):
            return json.dumps({"error": "document ingestion is disabled (MEM0_DOC_ENABLED=false)"}, ensure_ascii=False)
        uid, auth_err = _effective_user_id(user_id)
        if auth_err:
            return json.dumps({"error": auth_err}, ensure_ascii=False)

        meta_err = _validate_metadata_contract(metadata)
        if meta_err:
            return json.dumps({"error": meta_err}, ensure_ascii=False)

        def _do_submit():
            queue, worker = _get_ingest()
            info = resolve_and_spool(file_path)  # typed errors -> {"error": ...}
            is_image = info["content_type"].startswith("image/")
            if is_image:
                if not vision_enabled():
                    raise ValueError(
                        "image ingestion needs vision (set MEM0_ENABLE_VISION=true and MEM0_VLM_MODEL)"
                    )
                pages, chunks_estimate = 1, 1
            else:
                doc_meta = pdf_info(info["spool_path"])
                if doc_meta["encrypted"]:
                    raise EncryptedPdf("encrypted PDF — decrypt it before ingesting")
                max_pages = int(env("MEM0_DOC_MAX_PAGES", "50"))
                if doc_meta["pages"] > max_pages:
                    raise ValueError(
                        f"document has {doc_meta['pages']} pages; cap is {max_pages} "
                        f"(MEM0_DOC_MAX_PAGES) — split the file"
                    )
                pages = doc_meta["pages"]
                chunks_estimate = max(1, pages * 2)  # ~2 chunks/page at 1800 chars
            display_name = os.path.basename((filename or info["filename"]).strip()) or info["filename"]
            msgs = [{"role": "user", "content": f"[document sha256={info['doc_sha256']}]"}]

            if not force:
                done = queue.latest_done(idempotency_key(uid, agent_id or None, run_id or None, msgs))
                if done is not None:
                    return {
                        "status": "already_ingested",
                        "task_id": done["task_id"],
                        "finished_at": done.get("finished_at"),
                        "result": done.get("result"),
                        "note": "same bytes + scope already processed; resubmit with force=true to re-ingest",
                    }

            params: dict[str, Any] = {
                "spool_path": info["spool_path"],
                "doc_sha256": info["doc_sha256"],
                "content_type": info["content_type"],
                "filename": display_name,
                "pages": pages,
                "chunks_estimate": chunks_estimate,  # display only
                "enable_graph": enable_graph if enable_graph is not None else False,
            }
            if metadata:
                params["metadata"] = metadata
            if infer is not None:
                params["infer"] = infer

            res = queue.enqueue(
                user_id=uid, agent_id=agent_id or None, run_id=run_id or None,
                messages=msgs, params=params, kind="document",
            )
            worker.notify()
            envelope: dict[str, Any] = {
                "status": "queued",
                "task_id": res["task_id"],
                "submitted_at": res["submitted_at"],
                "source_doc": display_name,
                "content_type": info["content_type"],
                "pages": pages,
                "chunks_estimate": chunks_estimate,
                "queue_depth": res["queue_depth"],
                "estimated_wait_s": _estimate_wait_s(queue),
            }
            if res["duplicate"]:
                envelope["duplicate"] = True
                envelope["note"] = "this document is already queued; returning the existing task"
            return envelope

        return _mem0_call(_do_submit)

    @mcp.tool()
    def search_memories(
        query: Annotated[str, Field(description="Natural language description of what to find.")],
        user_id: Annotated[str | None, Field(description="User scope. Defaults to MEM0_USER_ID.")] = None,
        agent_id: Annotated[str | None, Field(description="Agent scope.")] = None,
        run_id: Annotated[str | None, Field(description="Run scope.")] = None,
        filters: Annotated[dict | None, Field(description="Additional structured filter clauses.")] = None,
        actor_id: Annotated[str | None, Field(description="Keep only memories SPOKEN BY this speaker (v0.15 attribution). Exact match on the canonical speaker label; wins over an actor_id given inside `filters`. Memories with no speaker are excluded while this is set.")] = None,
        limit: Annotated[int | None, Field(description="Maximum number of results (default 10).")] = None,
        threshold: Annotated[float | None, Field(description="Minimum relevance score (0.0-1.0).")] = None,
        rerank: Annotated[bool | None, Field(description="Whether to apply reranking. Defaults to the server's MEM0_ENABLE_RERANK.")] = None,
        enable_graph: Annotated[bool | None, Field(description="Override default graph toggle.")] = None,
        min_importance: Annotated[float | None, Field(description="Keep only memories whose classified importance is >= this value (0.0-1.0).")] = None,
        domain: Annotated[str | None, Field(description="Keep only memories whose classified domain matches (e.g. career, ai, data, software_engineering, finance, trading, health, education, personal, legal, business, infrastructure).")] = None,
        memory_type: Annotated[str | None, Field(description="Keep only memories of this classified type: semantic, episodic, or procedural.")] = None,
        sort_by_importance: Annotated[bool | None, Field(description="Sort results by classified importance descending.")] = None,
        as_of: Annotated[str | None, Field(description="Record-time anchor (ISO date or datetime): return what was known/current on that date — memories created later are excluded and facts superseded only after the anchor carry no demotion.")] = None,
        event_from: Annotated[str | None, Field(description="Event-time window start (inclusive). Full or partial ISO date: '2023' = whole year, '2023-10' = whole month, '2023-10-17' = that day. Filters on WHEN the fact happened (event_date), distinct from as_of's record-time. Memories without an event_date are EXCLUDED while the window is active. Either side alone = open interval. When neither event_from/event_to is given, a single date named in the query auto-anchors ranking without excluding anything.")] = None,
        event_to: Annotated[str | None, Field(description="Event-time window end (inclusive), same partial-date expansion as event_from.")] = None,
        reinforce: Annotated[bool | None, Field(description="Whether this search counts as a re-encounter for the returned memories (ACT-R usage timeline). Defaults to the server's MEM0_REINFORCE_ON_SEARCH. Measurement harnesses MUST pass false: a benchmark that runs its own queries would otherwise reinforce its own expected targets and inflate the metric it exists to protect.")] = None,
        historical: Annotated[bool | None, Field(description="RECORDAÇÃO HISTÓRICA (v0.10): 'o que eu sabia naquela época'. Requires as_of. A recollection NEVER reinforces the returned memories (even with reinforce=true) and usage-derived activation is fully inert in ranking; results with an explicitly linked newer fact (semantic correction or update version) come flagged has_newer_version=true and the response carries historical_recall.results_with_newer_version. It detects LINKED successors only — not arbitrary newer facts on the same subject. Plain as_of without this flag keeps the default search behavior.")] = None,
    ) -> str:
        """Semantic search across existing memories."""
        uid, auth_err = _effective_user_id(user_id)
        if auth_err:
            return json.dumps({"error": auth_err}, ensure_ascii=False)

        kwargs: dict[str, Any] = {"user_id": uid, "query": query}
        if agent_id:
            kwargs["agent_id"] = agent_id
        if run_id:
            kwargs["run_id"] = run_id
        filters, actor_err = _com_actor_id(filters, actor_id)
        if actor_err:
            return json.dumps({"error": actor_err}, ensure_ascii=False)
        if filters:
            kwargs["filters"] = filters
        # mem0ai 2.0.7: Memory.search recebe top_k (limit cairia no **kwargs e seria
        # ignorado) e rerank default False — aqui o default vem de MEM0_ENABLE_RERANK
        # (o reranker residente é a razão de ser do :8081).
        eff_limit = limit if limit is not None else 10
        do_rerank = rerank if rerank is not None else bool_env("MEM0_ENABLE_RERANK")
        kwargs["rerank"] = do_rerank
        # DeepMem0 já over-fetcha e corta no core (rerank_pool); duplicar aqui
        # dobraria o pool do cross-encoder (20 -> 40 = ~2x a latência à toa).
        import mem0 as _m0
        _core_overfetches = bool(getattr(_m0, "__deepmem0__", False))
        if do_rerank and not _core_overfetches:
            # Over-fetch manual só no runtime mem0ai upstream: o rerank de lá só
            # reordena o top_k fundido; um pool maior resgata alvos que a fusão
            # aditiva enterra (golden set 06/07/2026: hit@1 0.857→0.886 com pool
            # 20; pool 30 = igual, 3x o custo).
            kwargs["top_k"] = max(2 * eff_limit, int(env("MEM0_RERANK_POOL", "20")))
        else:
            kwargs["top_k"] = eff_limit
        if threshold is not None:
            kwargs["threshold"] = threshold
        if min_importance is not None:
            kwargs["min_importance"] = min_importance
        if domain:
            kwargs["domain"] = domain
        if memory_type:
            kwargs["memory_type"] = memory_type
        if sort_by_importance is not None:
            kwargs["sort_by_importance"] = sort_by_importance
        if reinforce is not None:
            # Repassado EXPLICITAMENTE (nunca via **kwargs, que o core engole em
            # silêncio): `False` aqui é o que impede um harness de medição de
            # reforçar os próprios alvos. No runtime upstream o parâmetro não
            # existe e o T3 também não — omitir é o comportamento correto.
            if _core_overfetches:
                kwargs["reinforce"] = reinforce
        if as_of:
            # DeepMem0 v0.3: âncora temporal. No runtime mem0ai upstream o
            # parâmetro não existe — erro claro em vez de TypeError críptico.
            if not _core_overfetches:
                return json.dumps(
                    {"error": "as_of requer o runtime DeepMem0 >= 0.3 (mem0ai upstream não suporta)"},
                    ensure_ascii=False,
                )
            kwargs["as_of"] = as_of
        if event_from or event_to:
            # DeepMem0 v0.6: janela event-time (event_date). Fork-only, mesmo guard
            # do as_of — mem0ai upstream engoliria os kwargs em silêncio.
            if not _core_overfetches:
                return json.dumps(
                    {"error": "event_from/event_to requerem o runtime DeepMem0 >= 0.6 (mem0ai upstream não suporta)"},
                    ensure_ascii=False,
                )
            if event_from:
                kwargs["event_from"] = event_from
            if event_to:
                kwargs["event_to"] = event_to
        if historical:
            # DeepMem0 v0.10: caminho EXPLÍCITO de recordação. Validação de
            # as_of/feature fica no core (fail-fast com mensagem clara).
            if not _core_overfetches:
                return json.dumps(
                    {"error": "historical requer o runtime DeepMem0 >= 0.10 (mem0ai upstream não suporta)"},
                    ensure_ascii=False,
                )
            kwargs["historical"] = True

        mem = _ensure_memory()

        # Classification keys clients actually filter/sort on; everything else in
        # metadata (text_lemmatized, entities, ...) only inflates client context.
        # v0.3: supersession/event-time fields are part of the contract clients
        # reason about (which fact is current, since when) — keep them visible.
        # task_id: provenance — which async submission a memory came from.
        # source_doc/page/chunk: document provenance (v0.5a) — which file and
        # page a fact was extracted from.
        # memory_scope*: ontologia v1 (Passo 2 — campo passivo; routing BLOQUEADO
        # até o Passo 4). O DeepMem0 promove memory_scope ao topo do resultado;
        # os campos auxiliares de proveniência ficam na metadata e precisam da
        # whitelist p/ serem visíveis aos clientes.
        _metadata_whitelist = {
            "importance", "domain", "tags", "memory_type",
            "superseded_by", "superseded_at", "supersedes", "event_date",
            "task_id", "source_doc", "page_start", "page_end", "chunk_index",
            "content_type",
            "memory_scope", "memory_scope_version", "memory_scope_source",
            "memory_scope_evidence", "memory_scope_reason",
        }

        def _do_search():
            res = mem.search(**kwargs)
            # v0.10 anti-degradação (achado do /critic-results): num runtime
            # DeepMem0 < 0.10 o kwarg `historical` cai no **kwargs e é ENGOLIDO
            # — o caller pediria recordação e receberia busca default sem
            # aviso. O echo `historical_recall` é o recibo do modo: ausente =
            # runtime sem suporte, erro claro (independe de checagem de versão).
            if historical and not (isinstance(res, dict) and res.get("historical_recall")):
                return {"error": "historical requer o runtime DeepMem0 >= 0.10 "
                                 "(o runtime ativo aceitou a busca mas não executou o modo)"}
            items = res.get("results") if isinstance(res, dict) else res
            if isinstance(items, list):
                # corta o over-fetch de volta ao limit pedido (pós-rerank/patch 4)
                if len(items) > eff_limit:
                    del items[eff_limit:]
                for r in items:
                    if isinstance(r, dict) and isinstance(r.get("metadata"), dict):
                        r["metadata"] = {
                            k: v for k, v in r["metadata"].items() if k in _metadata_whitelist
                        }
            # read-your-writes signal: facts still in the ingest queue are not
            # searchable yet — tell the caller the picture may be incomplete.
            if _async_ingest_enabled() and isinstance(res, dict):
                try:
                    queue, _ = _get_ingest()
                    res["pending_ingest"] = queue.pending_for_scope(uid)
                except Exception:
                    pass
            return res

        return _mem0_call(call_with_graph, mem, enable_graph, _enable_graph_default, _do_search)

    @mcp.tool()
    def get_memories(
        user_id: Annotated[str | None, Field(description="User scope. Defaults to MEM0_USER_ID.")] = None,
        agent_id: Annotated[str | None, Field(description="Agent scope.")] = None,
        run_id: Annotated[str | None, Field(description="Run scope.")] = None,
        limit: Annotated[int | None, Field(description="Maximum number of memories to return (1-1000, default 20). This is a CAPPED FIRST PAGE, not pagination: there is no offset/cursor, so memories beyond the cap are not reachable through this tool.")] = None,
        actor_id: Annotated[str | None, Field(description="Keep only memories SPOKEN BY this speaker (v0.15 attribution). Memories with no speaker are excluded while this is set.")] = None,
    ) -> str:
        """Page through memories using filters instead of search."""
        uid, auth_err = _effective_user_id(user_id)
        if auth_err:
            return json.dumps({"error": auth_err}, ensure_ascii=False)

        # `get_all` já aceita `filters=` e já canoniza `actor_id` lá dentro, então
        # não é preciso mudar o core: basta montar o dicionário aqui. As chaves de
        # escopo continuam indo como kwarg de topo — o core as levanta para dentro
        # do filtro sozinho.
        filtros, actor_err = _com_actor_id(None, actor_id)
        if actor_err:
            return json.dumps({"error": actor_err}, ensure_ascii=False)

        kwargs: dict[str, Any] = {"user_id": uid}
        if filtros:
            kwargs["filters"] = filtros
        if agent_id:
            kwargs["agent_id"] = agent_id
        if run_id:
            kwargs["run_id"] = run_id
        if limit is not None:
            # ⚠️ `top_k`, NÃO `limit`. A assinatura do core é
            # `get_all(*, filters=None, top_k=20, **kwargs)` e o corpo faz
            # `limit = top_k` — então `limit=` caía no `**kwargs` e SUMIA.
            # MEDIDO no caminho real: `get_all(limit=100)` devolvia **20**,
            # `get_all(top_k=100)` devolvia 100. Pedir 100 e receber 20, mudo.
            #
            # ⚠️ O teste unitário passava porque assere contra `MagicMock`, que
            # aceita qualquer kwarg. Mesma classe do teto de assinatura do
            # Patch 3b: mock não tem contrato, então não pode provar contrato.
            # A guarda agora é asserção sobre os argumentos REAIS da chamada.
            #
            # ⚠️ Limites: o campo esteve morto desde sempre, logo nunca houve
            # validação. Ativá-lo sem teto exporia 0, negativo e valor gigante
            # ao store.
            if not isinstance(limit, int) or isinstance(limit, bool) or not (1 <= limit <= MAX_GET_LIMIT):
                return json.dumps(
                    {"error": f"limit inválido: {limit!r}. Use um inteiro entre 1 e {MAX_GET_LIMIT}."},
                    ensure_ascii=False)
            kwargs["top_k"] = limit

        mem = _ensure_memory()
        if mem is None:
            return json.dumps({"error": "Memory not initialized", "detail": "Infrastructure may be unavailable."}, ensure_ascii=False)
        return _mem0_call(mem.get_all, **kwargs)

    @mcp.tool()
    def get_memory(
        memory_id: Annotated[str, Field(description="Exact memory UUID to fetch.")],
    ) -> str:
        """Fetch a single memory by its ID."""
        mem = _ensure_memory()
        if mem is None:
            return json.dumps({"error": "Memory not initialized", "detail": "Infrastructure may be unavailable."}, ensure_ascii=False)

        def _do_get():
            record = mem.get(memory_id)
            if not record:
                return {"error": f"memory not found: {memory_id}"}
            if _authorize_owner(record.get("user_id")):
                # indistinguishable from "does not exist", on purpose
                return {"error": f"memory not found: {memory_id}"}
            return record

        return _mem0_call(_do_get)

    @mcp.tool()
    def memory_history(
        memory_id: Annotated[str, Field(description="Exact memory UUID whose change history to fetch.")],
    ) -> str:
        """Full change timeline of a memory: ADD, UPDATE (old vs new text), SUPERSEDED (which fact replaced it) and DELETE events, oldest first."""
        mem = _ensure_memory()
        if mem is None:
            return json.dumps({"error": "Memory not initialized", "detail": "Infrastructure may be unavailable."}, ensure_ascii=False)

        denied = _authorize_memory_id(mem, memory_id)
        if denied:
            return json.dumps({"error": denied}, ensure_ascii=False)
        return _mem0_call(mem.history, memory_id)

    @mcp.tool()
    def memory_task_status(
        task_id: Annotated[str, Field(description="Task id returned by a queued add_memory call (tsk_...).")],
    ) -> str:
        """Status of an asynchronous add_memory task.

        States: pending | processing | done | failed_retryable | dead.
        When done, ``result.memory_ids`` lists the memories created/updated
        (fetch them with get_memory) and ``result.events`` includes any
        SUPERSEDED markings. ``last_error`` explains failed/dead tasks.
        """
        def _do_status():
            queue, _ = _get_ingest()
            row = queue.task_status(task_id)
            if row is None:
                return {"error": f"unknown task_id: {task_id}"}
            if _authorize_owner(row.get("user_id")):
                return {"error": f"unknown task_id: {task_id}"}
            return row

        return _mem0_call(_do_status)

    @mcp.tool()
    def memory_queue_status() -> str:
        """Health of the async ingest queue: depth (jobs waiting or running),
        per-status counts, age of the oldest pending job, estimated drain time,
        and whether the background worker is alive."""
        denied = _operator_only("memory_queue_status")
        if denied:
            return json.dumps({"error": denied}, ensure_ascii=False)

        def _do_queue_status():
            queue, worker = _get_ingest()
            status = queue.queue_status()
            status["worker_alive"] = worker.is_alive()
            status["async_ingest_enabled"] = _async_ingest_enabled()
            status["estimated_drain_s"] = _estimate_wait_s(queue)
            return status

        return _mem0_call(_do_queue_status)

    @mcp.tool()
    def update_memory(
        memory_id: Annotated[str, Field(description="Exact memory UUID to update.")],
        text: Annotated[str, Field(description="Replacement text for the memory.")],
    ) -> str:
        """Update an existing memory's text.

        ASYNCHRONOUS by default (when MEM0_ASYNC_INGEST != false): validates the
        memory exists, then returns {"status": "queued", "task_id", ...} immediately
        while the re-embed + metadata re-classification (a slow local-LLM call) run in
        the background worker — so the call never times out the client. Poll
        memory_task_status(task_id) for the result (memory_id / UPDATE event);
        memory_history(memory_id) shows the old-vs-new diff. An identical re-submit
        while the job is still active returns the same task_id (no double-apply). Set
        MEM0_ASYNC_INGEST=false for the synchronous path.

        DeepMem0 v0.7 (versioned update, when MEM0_VERSION_ON_UPDATE + temporality
        are on): the update MINTS A NEW CURRENT VERSION and marks the prior one
        superseded (kept, restorable via search(as_of=<before the edit>)). The result
        then carries a NEW current id: async → memory_task_status(task_id).memory_ids;
        sync → the "id" field (with "old_id"). The id you passed becomes the
        historical version; get_memory on it returns that older version, and search
        exposes superseded_by/superseded_at so you can follow the chain. Reusing the
        old id in a later update/delete resolves to the current head (no branching).
        """
        mem = _ensure_memory()
        if mem is None:
            return json.dumps({"error": "Memory not initialized", "detail": "Infrastructure may be unavailable."}, ensure_ascii=False)

        if _async_ingest_enabled():
            try:
                # Validate + resolve owner scope at submit: fail fast on a bad id
                # (never enqueue a doomed job) and scope the job to the memory's
                # owner so pending_ingest/purge stay correct.
                existing = mem.get(memory_id)
                if not existing:
                    return json.dumps({"error": "memory not found", "memory_id": memory_id}, ensure_ascii=False)
                uid, auth_err = _effective_user_id(existing.get("user_id"))
                if auth_err:
                    return json.dumps({"error": auth_err}, ensure_ascii=False)
                queue, worker = _get_ingest()
                # Sentinel messages encode memory_id+text so the idempotency key
                # (scope + messages) distinguishes distinct updates AND collapses an
                # identical retry onto the same task_id; the worker reads the real
                # memory_id/text from params.
                sentinel = [{"role": "user", "content": f"[update memory_id={memory_id}]\n{text}"}]
                res = queue.enqueue(
                    user_id=uid,
                    agent_id=existing.get("agent_id") or None,
                    run_id=existing.get("run_id") or None,
                    messages=sentinel,
                    params={"memory_id": memory_id, "text": text},
                    kind="update",
                )
                worker.notify()
                envelope: dict[str, Any] = {
                    "status": "queued",
                    "task_id": res["task_id"],
                    "submitted_at": res["submitted_at"],
                    "queue_depth": res["queue_depth"],
                    "estimated_wait_s": _estimate_wait_s(queue),
                }
                if res["duplicate"]:
                    envelope["duplicate"] = True
                    envelope["note"] = "identical update already queued; returning the existing task"
                return json.dumps(envelope, ensure_ascii=False)
            except Exception as exc:
                # A broken queue must not drop the update — fall through to sync.
                logger.error("Async update enqueue failed, falling back to sync update: %s", exc)

        denied = _authorize_memory_id(mem, memory_id)
        if denied:
            return json.dumps({"error": denied}, ensure_ascii=False)

        def _do_update():
            res = mem.update(memory_id, data=text)
            out = {"message": "Memory updated successfully!"}
            # DeepMem0 v0.7 (versioned update): surface the NEW current id (and the
            # now-historical old_id) so a synchronous client never loses the handle
            # to the current version. In-place/legacy updates return id == old_id.
            if isinstance(res, dict):
                if res.get("id"):
                    out["id"] = res["id"]
                if res.get("old_id"):
                    out["old_id"] = res["old_id"]
            return out

        return _mem0_call(_do_update)

    @mcp.tool()
    def delete_memory(
        memory_id: Annotated[str, Field(description="Exact memory UUID to delete.")],
    ) -> str:
        """Delete a single memory (and, when update-versioning is on, its whole version chain).

        Returns {"status":"deleted"} on success. On a PARTIAL chain delete (some
        versions removed, one or more failed) returns
        {"status":"partial","deleted":[...],"remaining":[...]} — retry delete_memory
        with ANY id in `remaining` to finish (idempotent: an already-absent id counts
        as success). `deleted` lists only what THIS call removed. A not-found id
        returns an error envelope.
        """
        mem = _ensure_memory()
        if mem is None:
            return json.dumps({"error": "Memory not initialized", "detail": "Infrastructure may be unavailable."}, ensure_ascii=False)

        denied = _authorize_memory_id(mem, memory_id)
        if denied:
            return json.dumps({"error": denied}, ensure_ascii=False)

        def _do_delete():
            # DeepMem0 v0.7.2: surface the REAL outcome, normalized to a stable MCP
            # schema (decoupled from fork internals). A partial version-chain delete
            # must NOT report success — the client needs `remaining` to retry.
            res = mem.delete(memory_id)
            if isinstance(res, dict) and res.get("remaining"):
                return {
                    "status": "partial",
                    "message": res.get("message", "Memory partially deleted; retry with a remaining id"),
                    "deleted": res.get("deleted", []),
                    "remaining": res["remaining"],
                }
            return {"status": "deleted", "message": "Memory deleted successfully!"}

        return _mem0_call(_do_delete)

    @mcp.tool()
    def delete_all_memories(
        user_id: Annotated[str | None, Field(description="User scope to delete.")] = None,
        agent_id: Annotated[str | None, Field(description="Agent scope to delete.")] = None,
        run_id: Annotated[str | None, Field(description="Run scope to delete.")] = None,
    ) -> str:
        """Bulk-delete all memories in the given scope. Requires at least one filter.

        NEVER calls memory.delete_all() — uses safe bulk-delete instead.
        """
        uid, auth_err = _effective_user_id(user_id)
        if auth_err:
            return json.dumps({"error": auth_err}, ensure_ascii=False)
        agent_id, run_id, scope_err = _normalized_scope(agent_id, run_id)
        if scope_err:
            return json.dumps({"error": scope_err}, ensure_ascii=False)
        if not any([uid, agent_id, run_id]):
            return json.dumps(
                {"error": "At least one scope (user_id, agent_id, or run_id) is required."},
                ensure_ascii=False,
            )

        filters: dict[str, Any] = {}
        if uid:
            filters["user_id"] = uid
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        mem = _ensure_memory()
        if mem is None:
            return json.dumps({"error": "Memory not initialized", "detail": "Infrastructure may be unavailable."}, ensure_ascii=False)

        def _do_bulk_delete():
            r = safe_bulk_delete(mem, filters, graph_enabled=_enable_graph_default)
            return _bulk_delete_envelope(r, f"Deleted {r.deleted} memories.")

        return _mem0_call(_do_bulk_delete)

    # ============================================================
    # Entity Tools (2 tools)
    # ============================================================

    @mcp.tool()
    def list_entities() -> str:
        """List which users/agents/runs currently hold memories.

        Uses Qdrant Facet API (v1.12+) for server-side aggregation,
        with scroll+dedupe fallback for older versions.
        """
        mem = _ensure_memory()
        if mem is None:
            return json.dumps({"error": "Memory not initialized", "detail": "Infrastructure may be unavailable."}, ensure_ascii=False)

        def _do_list():
            entities = list_entities_facet(mem)
            bound = _bound_scope()
            if bound and isinstance(entities, dict) and isinstance(entities.get("users"), list):
                # a scoped token learns that it exists, and nothing about anyone else
                entities = {**entities, "users": [u for u in entities["users"]
                                                  if (u.get("value") if isinstance(u, dict) else u) == bound]}
            return entities

        return _mem0_call(_do_list)

    @mcp.tool()
    def delete_entities(
        user_id: Annotated[str | None, Field(description="User entity to delete (cascades to all memories).")] = None,
        agent_id: Annotated[str | None, Field(description="Agent entity to delete.")] = None,
        run_id: Annotated[str | None, Field(description="Run entity to delete.")] = None,
    ) -> str:
        """Delete an entity and cascade-delete all its memories.

        Functionally equivalent to delete_all_memories in self-hosted mode.
        """
        if not any([user_id, agent_id, run_id]):
            return json.dumps(
                {"error": "At least one scope (user_id, agent_id, or run_id) is required."},
                ensure_ascii=False,
            )

        if user_id or _bound_scope():
            user_id, auth_err = _effective_user_id(user_id)
            if auth_err:
                return json.dumps({"error": auth_err}, ensure_ascii=False)
        agent_id, run_id, scope_err = _normalized_scope(agent_id, run_id)
        if scope_err:
            return json.dumps({"error": scope_err}, ensure_ascii=False)

        filters: dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        mem = _ensure_memory()
        if mem is None:
            return json.dumps({"error": "Memory not initialized", "detail": "Infrastructure may be unavailable."}, ensure_ascii=False)

        def _do_delete_entity():
            r = safe_bulk_delete(mem, filters, graph_enabled=_enable_graph_default)
            return _bulk_delete_envelope(
                r, f"Entity deleted. Removed {r.deleted} memories.")

        return _mem0_call(_do_delete_entity)

    # ============================================================
    # Direct Neo4j Graph Tools
    # ============================================================

    @mcp.tool()
    def mcp_search_graph(
        query: Annotated[str, Field(description="Entity or topic to search for (e.g., 'Python', 'TypeScript').")],
    ) -> str:
        """Search entities by name/id substring matching in Neo4j knowledge graph."""
        denied = _operator_only("mcp_search_graph")
        if denied:
            return json.dumps({"error": denied}, ensure_ascii=False)
        return search_graph(query)

    @mcp.tool()
    def mcp_get_entity(
        name: Annotated[str, Field(description="Exact entity name to look up.")],
    ) -> str:
        """Get all relationships for a specific entity (bidirectional)."""
        denied = _operator_only("mcp_get_entity")
        if denied:
            return json.dumps({"error": denied}, ensure_ascii=False)
        return get_entity(name)


# ============================================================
# MCP Prompt
# ============================================================


def _register_prompts(mcp: FastMCP) -> None:
    """Register MCP prompts."""

    @mcp.prompt()
    def memory_assistant() -> str:
        """Quick-start guide for using the mem0 memory server."""
        return (
            "You are using the mem0 MCP server for long-term memory management.\n\n"
            "Quick Start:\n"
            "1. Store memories: Use add_memory to save facts, preferences, or conversations\n"
            "2. Search memories: Use search_memories for semantic queries\n"
            "3. Browse memories: Use get_memories for filtered listing\n"
            "4. Update/Delete: Use update_memory and delete_memory for modifications\n"
            "5. Graph exploration: Use search_graph and get_entity for entity relationships\n\n"
            "Tips:\n"
            "- user_id is automatically injected from MEM0_USER_ID default\n"
            "- Set enable_graph=true to include knowledge graph results\n"
            "- Use infer=false to store raw text without LLM extraction\n"
            "- Use threshold on search_memories to filter by relevance score\n"
            "- Use filters for structured queries: {\"key\": {\"eq\": \"value\"}}\n"
        )


# ============================================================
# Server Runner
# ============================================================


def run_server() -> None:
    """Entry point: create server and run.

    Memory initialization is deferred to the first tool call via
    ``_ensure_memory()``, allowing the server to respond to MCP
    ``initialize`` and ``tools/list`` without live infrastructure.
    """
    # Configure logging
    log_level = env("MEM0_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(levelname)s %(name)s | %(message)s",
    )

    # Load .env file
    load_dotenv()

    # Create and run server (Memory init deferred to first tool call)
    server = _create_server()

    # Start the ingest worker at boot so jobs left over from a previous run
    # drain without waiting for the first tool call.
    if _async_ingest_enabled():
        try:
            _get_ingest()
        except Exception as exc:
            logger.error("Async ingest init failed (adds fall back to sync): %s", exc)

    transport = env("MEM0_TRANSPORT", "stdio").lower()

    if transport == "sse":
        # The legacy SSE app is NOT wrapped by the vault gate. Refusing to boot
        # turns a silent authentication bypass into an obvious failure.
        if _auth_mode() != vault_middleware.MODE_OFF:
            raise RuntimeError(
                "MEM0_REQUIRE_AUTH is set but the legacy 'sse' transport has no "
                "vault gate — use MEM0_TRANSPORT=streamable-http, or set "
                "MEM0_REQUIRE_AUTH=off and accept unauthenticated access"
            )
        server.run(transport="sse")
    elif transport == "streamable-http":
        _run_streamable_http(server)
    else:
        server.run(transport="stdio")


def _run_streamable_http(server: FastMCP) -> None:
    """Serve streamable-HTTP behind the vault gate.

    Single code path: build the ASGI app, wrap it, hand it to uvicorn. In
    ``off`` mode the wrapper delegates without opening the vault, so this is
    byte-for-byte today's behavior until an operator flips the drop-in.
    """
    import uvicorn

    mode = _auth_mode()  # raises on a typo — better at boot than at 3am
    db_path = _vault_db_path()
    app = vault_middleware.BearerTokenMiddleware(
        server.streamable_http_app(), db_path=db_path, mode=mode
    )
    if mode == vault_middleware.MODE_OFF:
        logger.info("Vault auth: off (no token required)")
    else:
        logger.info(
            "Vault auth: %s (db=%s, readiness=%s)", mode, db_path, vault_store.probe(db_path)
        )

    uvicorn.run(
        app,
        host=server.settings.host,
        port=server.settings.port,
        log_level=server.settings.log_level.lower(),
        # Pinned, not inherited: uvicorn already rewrites scope["client"] from
        # X-Forwarded-For when the peer is trusted, which is what keeps the
        # vault's denial counters and audit log honest behind a reverse proxy.
        # These are today's defaults — stated explicitly so a future uvicorn
        # cannot change them under us, and so the trust boundary is visible.
        proxy_headers=True,
        forwarded_allow_ips=env("MEM0_FORWARDED_ALLOW_IPS", "127.0.0.1"),
    )
