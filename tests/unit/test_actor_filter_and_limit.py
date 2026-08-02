"""Filtro por locutor nas tools de leitura, e o `limit` que estava MORTO.

⚠️ POR QUE ESTE ARQUIVO NÃO USA `MagicMock` PARA O QUE IMPORTA.

`get_memories(limit=N)` passava `limit=N` para `Memory.get_all`, cuja assinatura
é `get_all(*, filters=None, top_k=20, **kwargs)` e cujo corpo faz `limit = top_k`.
O `limit` caía no `**kwargs` e SUMIA. MEDIDO no caminho real:

    get_all(limit=100) -> 20 resultados
    get_all(top_k=100) -> 100 resultados

Pedir 100 e receber 20, em silêncio. O bug sobreviveu porque o teste existente
assere `mem.get_all.assert_called_once_with(..., limit=10)` contra um
`MagicMock`, **que aceita qualquer kwarg**. Mock não tem contrato, logo não pode
provar contrato — é a mesma classe do teto de assinatura do Patch 3b, onde um
wrapper com assinatura fixa engolia um kwarg novo e os testes com mock passavam.

A guarda aqui é dupla: (1) asserção sobre os argumentos REAIS da chamada, e
(2) um duplo que tem a assinatura VERDADEIRA do core e **levanta** se receber um
kwarg que o core descartaria.
"""
from unittest.mock import MagicMock

import pytest

from mem0_mcp_selfhosted import server as server_mod


#: Chaves que o core LEVANTA do `**kwargs` para dentro de `filters`
#: (`_extract_top_level_entity_params`). Tudo o mais que chegue por `**kwargs`
#: é descartado em silêncio — foi assim que o `limit` morreu.
_KWARGS_QUE_O_CORE_USA = {"user_id", "agent_id", "run_id"}

#: Teto do que o duplo MATERIALIZA. Ele existe para REGISTRAR a chamada, não
#: para simular o store — e essa distinção deixou de ser acadêmica em 01/08/2026.
#:
#: ⚠️ SEM ESTE TETO, RODAR ESTE ARQUIVO DERRUBA A MÁQUINA. O caso `mau=10**9`
#: de `test_limite_invalido_e_recusado` só alcança o duplo quando a guarda de
#: `limit` está desligada — que é precisamente o estado de uma rodada de
#: falsificação, o procedimento que este projeto EXIGE para toda guarda nova.
#: Sem teto o duplo obedecia e tentava empilhar um bilhão de dicionários
#: (~200 GB): os 62 GiB de RAM acabavam em minutos, o PSI da slice do usuário
#: passava de 50% e o `systemd-oomd` matava o maior cgroup — o scope do VS Code
#: e seus ~300 processos. A IDE não era a culpada, era só a maior.
#:
#: MEDIDO: guarda mutada = 24 GiB e SIGKILL; guarda no posto = 160 MB e 29
#: verdes; controle `mau=1001` (mesmo caminho, valor pequeno) = 152 MB.
#:
#: O teto corta o que é FABRICADO, nunca o que é ASSERIDO: `top_k` continua
#: gravado fiel em `self.chamadas`, que é onde os testes olham. Nenhum teste lê
#: o valor de retorno — conferido antes de pôr o teto.
_TETO_RESULTADOS_SINTETIZADOS = 50


class _GetAllComAssinaturaReal:
    """Duplo com a assinatura e o COMPORTAMENTO verdadeiros de `Memory.get_all`.

    `get_all(*, filters=None, top_k=20, **kwargs)`: o core levanta as três chaves
    de escopo do `**kwargs` para dentro de `filters` e **descarta o resto sem
    dizer nada** — foi exatamente aí que o `limit` sumiu.

    Aqui o descarte LEVANTA, para que o silêncio vire erro ruidoso.

    ⚠️ A primeira versão deste duplo rejeitava `user_id` também, e reprovou 7
    testes. O duplo é que estava errado: o core CONSOME `user_id` por esse
    caminho. Instrumento apertado demais acusa o código certo — mesma família do
    instrumento frouxo, e igualmente inútil.
    """

    def __init__(self):
        self.chamadas = []

    def __call__(self, *, filters=None, top_k=20, **kwargs):
        descartados = sorted(set(kwargs) - _KWARGS_QUE_O_CORE_USA)
        if descartados:
            raise TypeError(
                f"kwarg que o core DESCARTARIA em silêncio: {descartados}. "
                f"`Memory.get_all` declara só `filters`/`top_k` e levanta "
                f"{sorted(_KWARGS_QUE_O_CORE_USA)} do **kwargs."
            )
        escopo = {k: v for k, v in kwargs.items() if k in _KWARGS_QUE_O_CORE_USA}
        efetivo = {**escopo, **(filters or {})} if escopo else filters
        self.chamadas.append({"filters": filters, "top_k": top_k, "efetivo": efetivo})
        # ⚠️ NUNCA voltar a `range(top_k)` — ver _TETO_RESULTADOS_SINTETIZADOS.
        # O clamp é defensivo de propósito: numa rodada de falsificação chega
        # aqui exatamente o valor que a guarda recusaria (0, negativo, gigante,
        # ou nem inteiro), porque a guarda é o que está sendo desligado.
        n = top_k if isinstance(top_k, int) and not isinstance(top_k, bool) else 0
        n = max(0, min(n, _TETO_RESULTADOS_SINTETIZADOS))
        return {"results": [{"id": f"m{i}"} for i in range(n)]}


@pytest.fixture
def srv(mocker):
    mem = MagicMock()
    mem.graph = None
    mem.enable_graph = False
    mem.search.return_value = {"results": []}
    original = server_mod.memory
    server_mod.memory = mem
    server_mod._enable_graph_default = False
    s = server_mod._create_server()
    yield s, mem
    server_mod.memory = original


def _tool(s, nome):
    t = s._tool_manager._tools.get(nome)
    assert t is not None, f"tool {nome!r} não registrada"
    return t.fn


class TestLimitMorto:
    def test_limit_vira_top_k_nos_argumentos_REAIS(self, srv):
        """O teste que o bug teria reprovado. `assert_called_with(limit=...)`
        contra MagicMock passava com o código quebrado."""
        s, mem = srv
        mem.get_all = _GetAllComAssinaturaReal()
        _tool(s, "get_memories")(user_id="u", limit=100)
        c = mem.get_all.chamadas
        assert len(c) == 1 and c[0]["top_k"] == 100

    def test_o_duplo_realmente_pega_o_bug(self, srv):
        """CONTROLE POSITIVO do próprio instrumento: se o código voltasse a
        mandar `limit=`, o duplo tem de levantar. Sem isto, o teste acima
        poderia passar por não estar exercitando nada."""
        s, mem = srv
        duplo = _GetAllComAssinaturaReal()
        duplo(filters={"user_id": "u"}, top_k=5)          # forma certa: passa
        with pytest.raises(TypeError, match="DESCARTARIA"):
            duplo(filters={"user_id": "u"}, limit=100)    # a forma do bug: levanta

    def test_sem_limit_nao_manda_top_k(self, srv):
        s, mem = srv
        mem.get_all = _GetAllComAssinaturaReal()
        _tool(s, "get_memories")(user_id="u")
        c = mem.get_all.chamadas
        assert len(c) == 1 and c[0]["top_k"] == 20

    @pytest.mark.parametrize("mau", [0, -1, 10**9, 1001])
    def test_limite_invalido_e_recusado(self, srv, mau):
        """O campo esteve morto desde sempre, então nunca houve validação —
        ativá-lo sem teto exporia 0, negativo e gigante ao store."""
        s, mem = srv
        mem.get_all = _GetAllComAssinaturaReal()
        r = _tool(s, "get_memories")(user_id="u", limit=mau)
        assert "error" in r and "limit" in r
        assert mem.get_all.chamadas == [], "não pode chegar ao store"

    @pytest.mark.parametrize("bom", [1, 20, 1000])
    def test_fronteiras_validas_passam(self, srv, bom):
        s, mem = srv
        mem.get_all = _GetAllComAssinaturaReal()
        _tool(s, "get_memories")(user_id="u", limit=bom)
        assert mem.get_all.chamadas[0]["top_k"] == bom

    def test_bool_nao_e_inteiro_valido(self, srv):
        """`isinstance(True, int)` é True em Python; sem guarda própria,
        `limit=True` viraria `top_k=1` em silêncio."""
        s, mem = srv
        mem.get_all = _GetAllComAssinaturaReal()
        r = _tool(s, "get_memories")(user_id="u", limit=True)
        assert "error" in r


class TestActorIdNoSearch:
    def test_dobra_no_filters(self, srv):
        s, mem = srv
        _tool(s, "search_memories")(query="q", user_id="u", actor_id="Maria")
        assert mem.search.call_args.kwargs["filters"] == {"actor_id": "Maria"}

    def test_parametro_VENCE_o_filters_do_chamador(self, srv):
        s, mem = srv
        _tool(s, "search_memories")(query="q", user_id="u",
                                    filters={"actor_id": "Bruno", "domain": "ai"},
                                    actor_id="Maria")
        f = mem.search.call_args.kwargs["filters"]
        assert f["actor_id"] == "Maria", "o parâmetro explícito tem de vencer"
        assert f["domain"] == "ai", "o resto do filtro do chamador sobrevive"

    def test_filters_do_chamador_NAO_e_mutado(self, srv):
        """Mutar o dict do chamador mudaria estado que não é nosso."""
        s, mem = srv
        dele = {"actor_id": "Bruno", "domain": "ai"}
        _tool(s, "search_memories")(query="q", user_id="u", filters=dele, actor_id="Maria")
        assert dele == {"actor_id": "Bruno", "domain": "ai"}

    def test_canoniza_com_a_regra_do_core(self, srv):
        """O Qdrant casa por igualdade EXATA: escrita e consulta canonizando
        diferente é filtro que erra em silêncio."""
        s, mem = srv
        _tool(s, "search_memories")(query="q", user_id="u", actor_id="  Maria   Silva ")
        assert mem.search.call_args.kwargs["filters"]["actor_id"] == "Maria Silva"

    @pytest.mark.parametrize("mau", ["", "   ", "Maria\nX", 'Ana", "X', "Ana: fim"])
    def test_invalido_e_RECUSADO_nao_ignorado(self, srv, mau):
        """⚠️ Com precedência por truthiness (`actor_id or filters[...]`), um
        `""` cairia no filtro do chamador em silêncio. Por `is not None`, ele
        chega à validação e FALHA."""
        s, mem = srv
        r = _tool(s, "search_memories")(query="q", user_id="u", actor_id=mau)
        assert "error" in r and "actor_id" in r
        mem.search.assert_not_called()

    def test_vazio_nao_cai_no_filters_do_chamador(self, srv):
        """O caso exato que a truthiness escondia."""
        s, mem = srv
        r = _tool(s, "search_memories")(query="q", user_id="u",
                                        filters={"actor_id": "Bruno"}, actor_id="")
        assert "error" in r
        mem.search.assert_not_called()

    def test_ausente_nao_toca_no_filters(self, srv):
        s, mem = srv
        _tool(s, "search_memories")(query="q", user_id="u", filters={"domain": "ai"})
        assert mem.search.call_args.kwargs["filters"] == {"domain": "ai"}


class TestActorIdNoGetMemories:
    def test_monta_filters(self, srv):
        s, mem = srv
        mem.get_all = _GetAllComAssinaturaReal()
        _tool(s, "get_memories")(user_id="u", actor_id="Maria")
        assert mem.get_all.chamadas[0]["filters"] == {"actor_id": "Maria"}

    def test_convive_com_limit(self, srv):
        s, mem = srv
        mem.get_all = _GetAllComAssinaturaReal()
        _tool(s, "get_memories")(user_id="u", actor_id="Maria", limit=50)
        c = mem.get_all.chamadas
        assert c[0]["filters"] == {"actor_id": "Maria"} and c[0]["top_k"] == 50

    def test_invalido_recusado_antes_do_store(self, srv):
        s, mem = srv
        mem.get_all = _GetAllComAssinaturaReal()
        r = _tool(s, "get_memories")(user_id="u", actor_id="Maria\nX")
        assert "error" in r
        assert mem.get_all.chamadas == []


class TestSchemaExposto:
    """O ponto do E2: hoje dá para filtrar por locutor passando `filters` na
    mão, mas nenhum cliente DESCOBRE isso. O parâmetro tem de aparecer."""

    @pytest.mark.parametrize("tool", ["search_memories", "get_memories"])
    def test_actor_id_aparece_no_schema(self, srv, tool):
        s, _ = srv
        params = s._tool_manager._tools[tool].parameters
        assert "actor_id" in params.get("properties", {})

    @pytest.mark.parametrize("tool", ["search_memories", "get_memories"])
    def test_actor_id_e_opcional(self, srv, tool):
        s, _ = srv
        params = s._tool_manager._tools[tool].parameters
        assert "actor_id" not in params.get("required", [])
