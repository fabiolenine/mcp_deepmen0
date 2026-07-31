"""Contrato de `messages` na fronteira de escrita do add_memory.

O core (`parse_vision_messages` com `enable_vision=False`) DESCARTA partes de
imagem; se a mensagem só tinha imagem, ela some inteira. Sem este contrato o
cliente recebia `{"status":"queued"}` e a imagem evaporava em silêncio — o
submit é o único ponto onde ainda dá para devolver um erro acionável.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

import mem0_mcp_selfhosted.server as server_mod

IMG = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
TXT = {"type": "text", "text": "o orçamento aprovado foi de R$ 1,25 milhão"}


@pytest.fixture(autouse=True)
def _env_defaults(monkeypatch):
    monkeypatch.setenv("MEM0_USER_ID", "test-user")
    monkeypatch.delenv("MEM0_OBSERVE_URL", raising=False)  # telemetria off nos testes


def _server(monkeypatch, tmp_path, *, mode=None, async_ingest="true"):
    if mode is None:
        monkeypatch.delenv("MEM0_MESSAGE_CONTRACT", raising=False)
    else:
        monkeypatch.setenv("MEM0_MESSAGE_CONTRACT", mode)
    monkeypatch.setenv("MEM0_ASYNC_INGEST", async_ingest)
    monkeypatch.setenv("MEM0_QUEUE_WORKER", "false")
    monkeypatch.setenv("MEM0_QUEUE_DB_PATH", str(tmp_path / "q.db"))
    server_mod._ingest_queue = None
    server_mod._ingest_worker = None
    server_mod.memory = MagicMock()
    return server_mod._create_server()


def _add(srv):
    return srv._tool_manager._tools["add_memory"].fn


def _queue_depth(srv):
    queue, _ = server_mod._get_ingest()
    return queue.claim_next()


class TestBootValidation:
    def test_invalid_value_aborts_construction(self, monkeypatch, tmp_path):
        """Falha no BOOT, não na primeira escrita.

        O precedente MEM0_METADATA_CONTRACT resolve o modo DENTRO do validador,
        então um env inválido só explode quando alguém tenta gravar — o serviço
        sobe "saudável" e mente até lá. Aqui o modo é resolvido na construção.
        """
        with pytest.raises(ValueError, match=r"MEM0_MESSAGE_CONTRACT=.*inválido"):
            _server(monkeypatch, tmp_path, mode="lixo")

    def test_default_is_warn_not_enforce(self, monkeypatch, tmp_path):
        # Nasce em warn: texto+imagem hoje CONSERVA o texto, e estrear em enforce
        # converteria isso em erro duro sem auditar chamadores.
        _server(monkeypatch, tmp_path, mode=None)
        assert server_mod._message_contract_mode == "warn"

    def test_health_route_reports_the_active_mode(self, monkeypatch, tmp_path):
        """Bate na ROTA. Asserção sobre a global só provaria que o parse rodou."""
        from starlette.testclient import TestClient

        srv = _server(monkeypatch, tmp_path, mode="enforce")
        resp = TestClient(srv.streamable_http_app()).get("/health")
        body = resp.json()
        assert body["message_contract"] == "enforce"
        assert body["status"] in ("ok", "degraded"), "modo inválido não pode degradar"
        assert "boot_provenance" in body


class TestImageOnlyIsRefusedInEveryMode:
    """Imagem-só é recusada em `enforce` E em `warn`.

    O core descarta a mensagem inteira, o job produz zero fato, e o cliente
    recebeu um ack — `queued` ali é mentira verificável (r1/MAJOR 6).

    Em `off`, NÃO: `off` é o kill switch do rollout e restaura o comportamento
    pré-P4 por inteiro (r2/MAJOR-8b). Um kill switch que não desliga não é kill
    switch — ver TestOffIsATrueKillSwitch.
    """

    @pytest.mark.parametrize("mode", ["enforce", "warn"])
    def test_image_only_list_refused(self, monkeypatch, tmp_path, mode):
        srv = _server(monkeypatch, tmp_path, mode=mode)
        parsed = json.loads(_add(srv)(text="", messages=[{"role": "user", "content": [IMG]}]))
        assert "error" in parsed
        assert "add_document" in parsed["error"]
        assert _queue_depth(srv) is None, "payload recusado NÃO pode ter sido enfileirado"
        server_mod.memory.add.assert_not_called()

    @pytest.mark.parametrize("mode", ["enforce", "warn"])
    def test_image_only_bare_dict_refused(self, monkeypatch, tmp_path, mode):
        srv = _server(monkeypatch, tmp_path, mode=mode)
        parsed = json.loads(_add(srv)(text="", messages=[{"role": "user", "content": IMG}]))
        assert "error" in parsed
        assert _queue_depth(srv) is None
        server_mod.memory.add.assert_not_called()


class TestMixedTextAndImage:
    def test_enforce_refuses(self, monkeypatch, tmp_path):
        srv = _server(monkeypatch, tmp_path, mode="enforce")
        parsed = json.loads(_add(srv)(text="", messages=[{"role": "user", "content": [TXT, IMG]}]))
        assert "error" in parsed and "add_document" in parsed["error"]
        assert _queue_depth(srv) is None
        server_mod.memory.add.assert_not_called()

    def test_warn_accepts_and_logs(self, monkeypatch, tmp_path, caplog):
        # O texto sobrevive ao core, então a resposta `queued` NÃO mente.
        srv = _server(monkeypatch, tmp_path, mode="warn")
        with caplog.at_level("WARNING", logger="mem0_mcp_selfhosted.server"):
            parsed = json.loads(
                _add(srv)(text="", messages=[{"role": "user", "content": [TXT, IMG]}]))
        assert parsed["status"] == "queued"
        assert "message_contract (warn)" in caplog.text
        assert _queue_depth(srv) is not None

    def test_off_accepts_silently(self, monkeypatch, tmp_path, caplog):
        srv = _server(monkeypatch, tmp_path, mode="off")
        with caplog.at_level("WARNING", logger="mem0_mcp_selfhosted.server"):
            parsed = json.loads(
                _add(srv)(text="", messages=[{"role": "user", "content": [TXT, IMG]}]))
        assert parsed["status"] == "queued"
        assert "message_contract" not in caplog.text


class TestOffIsATrueKillSwitch:
    """`off` tem que restaurar o comportamento pré-P4 para TODOS os payloads.

    Rollback parcial é pior que nenhum: o operador que põe `off` num incidente
    espera o estado anterior, não um subconjunto dele.
    """

    def test_off_accepts_image_only_list(self, monkeypatch, tmp_path):
        srv = _server(monkeypatch, tmp_path, mode="off")
        parsed = json.loads(_add(srv)(text="", messages=[{"role": "user", "content": [IMG]}]))
        assert parsed["status"] == "queued", "off não restaurou o comportamento pré-P4"

    def test_off_never_emits_pass(self, monkeypatch, tmp_path):
        """Com o contrato desligado NADA foi avaliado.

        Emitir `pass` aqui inflaria o piso de liveness do P7 com tráfego não
        verificado — e o piso é o que autoriza a promoção warn -> enforce. Era o
        que acontecia: o `pass` saía antes da checagem de modo.
        """
        vistos = []
        monkeypatch.setattr(server_mod, "_observe_message_contract", vistos.append)
        _server(monkeypatch, tmp_path, mode="off")
        for msgs in ([{"role": "user", "content": "texto puro"}],
                     [{"role": "user", "content": [TXT, IMG]}],
                     [{"role": "user", "content": [IMG]}]):
            server_mod._validate_messages_shape(msgs)
        assert vistos == ["off", "off", "off"], f"vazou veredito avaliado em off: {vistos}"

    def test_off_accepts_image_only_bare_dict(self, monkeypatch, tmp_path):
        srv = _server(monkeypatch, tmp_path, mode="off")
        parsed = json.loads(_add(srv)(text="", messages=[{"role": "user", "content": IMG}]))
        assert parsed["status"] == "queued"


class TestHotPathUntouched:
    """Anti-regressão: este validador roda em 100% dos add_memory."""

    def test_plain_text_passes(self, monkeypatch, tmp_path):
        srv = _server(monkeypatch, tmp_path, mode="enforce")
        parsed = json.loads(_add(srv)(text="uma memória normal de texto"))
        assert parsed["status"] == "queued"

    def test_plain_message_list_passes(self, monkeypatch, tmp_path):
        srv = _server(monkeypatch, tmp_path, mode="enforce")
        parsed = json.loads(_add(srv)(
            text="", messages=[{"role": "user", "content": "texto puro"}]))
        assert parsed["status"] == "queued"

    def test_non_dict_message_is_not_this_contract_s_problem(self, monkeypatch, tmp_path):
        # Container malformado é defeito do core (AttributeError, poison). Este
        # contrato valida partes de IMAGEM; não pode explodir na frente dele.
        srv = _server(monkeypatch, tmp_path, mode="enforce")
        assert server_mod._validate_messages_shape(["não é dict"]) is None


class TestSyncPathAlsoCovered:
    """r1/MAJOR 7: a checagem entra ANTES de montar `msgs`, então vale nos dois
    caminhos. Provar que `memory.add` também não foi chamado, não só a fila."""

    def test_infer_false_refused_and_memory_add_not_called(self, monkeypatch, tmp_path):
        srv = _server(monkeypatch, tmp_path, mode="enforce", async_ingest="false")
        parsed = json.loads(_add(srv)(
            text="", messages=[{"role": "user", "content": [IMG]}], infer=False))
        assert "error" in parsed
        server_mod.memory.add.assert_not_called()


class TestTelemetry:
    def test_pass_is_emitted_on_the_hot_path(self, monkeypatch, tmp_path):
        """O contador `pass` é o de LIVENESS.

        Sem ele, "zero warn_mixed na janela" não distingue "ninguém manda imagem"
        de "o validador nunca rodou" — e é essa distinção que autoriza promover
        warn -> enforce.
        """
        seen = []
        monkeypatch.setattr(server_mod, "_observe_message_contract", seen.append)
        srv = _server(monkeypatch, tmp_path, mode="warn")
        _add(srv)(text="", messages=[{"role": "user", "content": "texto puro"}])
        # E o caminho `text=` SEM messages, que é a maioria absoluta dos adds
        # reais: se ele não contar, o piso de liveness do P7 nunca é atingido e
        # a promoção warn->enforce fica impossível de justificar.
        _add(srv)(text="uma memória normal")
        assert seen == ["pass", "pass"]

    def test_real_emitter_shape_and_no_blocking(self, monkeypatch, tmp_path):
        """Exercita `_observe_message_contract` DE VERDADE: mocar o emissor
        inteiro apaga justamente o schema que o P7 vai consultar. E prova que
        ele não bloqueia — o ack de add_memory tem que ser imediato."""
        import time as _t

        posted = []

        class _FakeRequests:
            @staticmethod
            def post(url, json=None, auth=None, timeout=None):
                _t.sleep(0.4)          # OpenObserve lento
                posted.append((url, json))

        monkeypatch.setenv("MEM0_OBSERVE_URL", "http://observe.invalid/api/x/_json")
        monkeypatch.setitem(__import__("sys").modules, "requests", _FakeRequests)
        _server(monkeypatch, tmp_path, mode="warn")

        t0 = _t.monotonic()
        server_mod._observe_message_contract("pass")
        assert _t.monotonic() - t0 < 0.2, "emissor bloqueou o caminho quente"

        _t.sleep(1.0)
        assert len(posted) == 1
        _url, body = posted[0]
        ev = body[0]
        assert ev["stage"] == "message_contract"
        assert ev["verdict"] == "pass"
        assert ev["mode"] == "warn"
        assert isinstance(ev["_timestamp"], int)

    def test_verdicts_are_distinguishable(self, monkeypatch, tmp_path):
        seen = []
        monkeypatch.setattr(server_mod, "_observe_message_contract", seen.append)
        srv = _server(monkeypatch, tmp_path, mode="warn")
        _add(srv)(text="", messages=[{"role": "user", "content": [TXT, IMG]}])
        _add(srv)(text="", messages=[{"role": "user", "content": [IMG]}])
        assert seen == ["warn_mixed", "reject_image_only"]


class TestBootProvenance:
    def test_boot_provenance_is_stamped_at_construction(self, monkeypatch, tmp_path):
        _server(monkeypatch, tmp_path, mode="warn")
        prov = server_mod._BOOT_PROVENANCE
        assert prov.get("stamped_at"), "não carimbou no boot"
        assert "boot_tree_dirty" in prov, (
            "árvore suja no boot é o campo que decide se existe last-known-good: "
            "um processo iniciado com a árvore suja não corresponde a commit nenhum"
        )

    def test_boot_provenance_does_not_follow_the_disk(self, monkeypatch, tmp_path):
        """O defeito medido: /health reportava um commit 4h MAIS NOVO que o processo.

        `_provenance()` shella `git rev-parse` a cada requisição, então o valor
        segue o disco e um restart "verificado" por ele seria vácuo.
        """
        _server(monkeypatch, tmp_path, mode="warn")
        primeiro = dict(server_mod._BOOT_PROVENANCE)

        # MUDA O DISCO de verdade: constrói OUTRO repo, com outro HEAD, e
        # confere que ele difere do carimbado. `_provenance()` (read-at-request)
        # seguiria esse disco; o carimbo não.
        # O repo é criado aqui em vez de apontar para um caminho absoluto da
        # máquina: caminho fixo vaza o usuário num repo público E quebra o teste
        # em qualquer outro clone — a asserção passava a depender de uma árvore
        # que só existe numa máquina.
        import subprocess

        outro_repo = tmp_path / "outro_repo"
        outro_repo.mkdir()
        _git = ["git", "-C", str(outro_repo)]
        subprocess.run([*_git, "init", "-q"], check=True)
        subprocess.run([*_git, "-c", "user.email=t@t", "-c", "user.name=t",
                        "commit", "-q", "--allow-empty", "-m", "outro HEAD"],
                       check=True)
        outro = subprocess.run([*_git, "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True).stdout.strip()
        assert outro and outro != primeiro.get("boot_fork_sha")
        assert dict(server_mod._BOOT_PROVENANCE) == primeiro, (
            "o carimbo mudou sem nova construção -> está seguindo o disco"
        )

        _server(monkeypatch, tmp_path, mode="warn")
        assert server_mod._BOOT_PROVENANCE["stamped_at"] != primeiro["stamped_at"], (
            "uma construção nova tem que re-carimbar"
        )


class TestFidelityToTheCore:
    """DIFERENCIAL: a CLASSIFICAÇÃO do contrato tem que casar com o core.

    `_text_survives` afirma espelhar `parse_vision_messages` com visão off.
    Afirmação não basta — cada caso roda o core DE VERDADE, mensagem a mensagem.

    O oráculo é POR MENSAGEM, não pela lista: perguntar "sobrou algo na lista?"
    confunde "a mensagem do usuário evaporou" com "a mensagem de system passou".
    E compara CLASSIFICAÇÃO, não veredito final — recusar texto+imagem é política
    do modo `enforce`, não uma questão de fidelidade.

    Dois falsos positivos reais foram achados por aqui: mensagem `system` (o core
    a repassa INTOCADA, imagem e tudo) e parte de texto vazia (o core preserva
    `""`, produzindo content="").
    """

    CASOS = [
        ("texto puro", {"role": "user", "content": "oi"}),
        ("texto+imagem", {"role": "user", "content": [TXT, IMG]}),
        ("imagem-só lista", {"role": "user", "content": [IMG]}),
        ("imagem-só dict", {"role": "user", "content": IMG}),
        ("system com imagem", {"role": "system", "content": [IMG]}),
        ("texto VAZIO + imagem", {"role": "user", "content": [{"type": "text", "text": ""}, IMG]}),
        ("duas imagens sem texto", {"role": "user", "content": [IMG, IMG]}),
    ]

    def test_the_oracle_is_the_fork_not_upstream(self):
        """O oráculo depende de QUAL mem0 está importado.

        O pyproject deste pacote declara `mem0ai==2.0.7` (upstream); em produção
        o venv tem o fork por install editável. Uma instalação limpa, ou o
        rollback documentado (`uv pip install mem0ai==2.0.7`), trocaria o oráculo
        em SILÊNCIO e os diferenciais passariam a comparar o contrato contra o
        upstream — que é justamente o código com os quatro defeitos.
        """
        import mem0

        assert getattr(mem0, "__deepmem0__", False), (
            f"o mem0 importado ({getattr(mem0, '__file__', '?')}) NÃO é o fork "
            f"DeepMem0 — os diferenciais estariam comparando contra o upstream "
            f"2.0.7, que ainda tem os defeitos D1-D4"
        )

    @pytest.mark.parametrize("nome,msg", CASOS, ids=[c[0] for c in CASOS])
    def test_classification_agrees_with_the_core(self, monkeypatch, tmp_path, nome, msg):
        from mem0.memory.utils import parse_vision_messages

        _server(monkeypatch, tmp_path, mode="enforce")
        # o core preserva ESTA mensagem?
        preservada = bool(parse_vision_messages([msg], llm=None))
        # o contrato a trata como "nada utilizável sobra"?
        tratada_como_perdida = (
            server_mod._image_parts_in(msg.get("content"))
            and msg.get("role") != "system"
            and not server_mod._text_survives(msg.get("content"))
        )
        assert preservada != bool(tratada_como_perdida), (
            f"{nome}: core preserva={preservada}, contrato considera perdida="
            f"{bool(tratada_como_perdida)} -> divergência"
        )

    def test_image_only_message_is_refused_even_beside_a_system_message(
            self, monkeypatch, tmp_path):
        """A lista sobrevive (o system passa), mas o conteúdo do USUÁRIO evaporou.

        Um oráculo por-lista chamaria isto de 'preservado' e deixaria o bug
        original passar: o cliente receberia `queued` tendo perdido a única
        mensagem que ele de fato mandou.
        """
        srv = _server(monkeypatch, tmp_path, mode="warn")
        msgs = [{"role": "system", "content": "instruções"},
                {"role": "user", "content": [IMG]}]
        parsed = json.loads(_add(srv)(text="", messages=msgs))
        assert "error" in parsed, "a mensagem do usuário some; ack aqui seria mentira"


class TestVerdictEnumIsComplete:
    """O enum é o contrato que o P7 vai consultar para decidir a promoção.

    Um veredito emitido e não especificado (era o caso de `reject_mixed`) some
    da análise: o script de promoção filtraria por uma lista que não o contém e
    concluiria "zero rejeições" sobre um corpus que teve rejeições.
    """

    ESPERADOS = {"pass", "off", "reject_image_only", "reject_mixed", "warn_mixed"}

    def test_every_emitted_verdict_is_in_the_documented_enum(self, monkeypatch, tmp_path):
        import re
        import pathlib

        fonte = pathlib.Path(server_mod.__file__).read_text()
        emitidos = set(re.findall(r'_observe_message_contract\("([a-z_]+)"\)', fonte))
        assert emitidos == self.ESPERADOS, (
            f"emitidos={sorted(emitidos)} != documentados={sorted(self.ESPERADOS)} -> "
            f"veredito fora do contrato fica invisível para o gate de promoção do P7"
        )

    def test_each_mode_emits_its_own_verdict(self, monkeypatch, tmp_path):
        vistos = []
        monkeypatch.setattr(server_mod, "_observe_message_contract", vistos.append)
        for modo, msgs in (
            ("enforce", [{"role": "user", "content": [TXT, IMG]}]),
            ("warn", [{"role": "user", "content": [TXT, IMG]}]),
            ("off", [{"role": "user", "content": [IMG]}]),
            ("off", [{"role": "user", "content": "texto"}]),   # sem imagem, ainda assim `off`
            ("enforce", [{"role": "user", "content": [IMG]}]),
            ("enforce", [{"role": "user", "content": "texto"}]),
        ):
            _server(monkeypatch, tmp_path, mode=modo)
            server_mod._validate_messages_shape(msgs)
        assert vistos == ["reject_mixed", "warn_mixed", "off", "off",
                          "reject_image_only", "pass"]


class TestBootProvenanceHasNoSilentHoles:
    """Um campo de proveniência que sai `null` em silêncio é pior que ausente.

    Aconteceu: o carimbo apontava para `mem0/memory/entity_extraction.py`, que
    não existe (o arquivo é `mem0/utils/entity_extraction.py`), e o `except`
    convertia o erro em `None` — no campo criado justamente para acabar com
    leitura silenciosamente errada.
    """

    def test_every_stamped_hash_resolved(self, monkeypatch, tmp_path):
        _server(monkeypatch, tmp_path, mode="warn")
        prov = server_mod._BOOT_PROVENANCE
        hashes = {k: v for k, v in prov.items() if k.startswith("boot_sha_")}
        assert hashes, "nenhum hash carimbado"
        nulos = [k for k, v in hashes.items() if v is None]
        assert not nulos, f"caminho errado no carimbo -> {nulos} saíram null em silêncio"

    def test_stamped_paths_match_the_request_time_ones(self, monkeypatch, tmp_path):
        """Carimbo e leitura-por-requisição têm que cobrir os MESMOS arquivos,
        senão comparar os dois não significa nada."""
        import re
        import pathlib

        fonte = pathlib.Path(server_mod.__file__).read_text()
        todos = re.findall(r'"(mem0/[a-z_/]+\.py)"', fonte)
        assert todos.count("mem0/utils/entity_extraction.py") >= 2, (
            "carimbo e _provenance() divergem nos arquivos cobertos"
        )
