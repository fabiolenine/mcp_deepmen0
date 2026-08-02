"""Presenters do read model: cursor, facetas, filtros, linha de listagem."""

from datetime import datetime, timedelta, timezone

from mem0_mcp_selfhosted.vault.memories import model


def _iso(days_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


class TestCursor:
    def test_first_page_is_empty(self):
        c = model.Cursor()
        assert c.is_first_page
        assert c.boundary_ts is None

    def test_roundtrip(self):
        c = model.Cursor("2026-07-23T03:31:28.904335+00:00", ["a", "b"])
        back = model.Cursor.decode(c.encode())
        assert back.boundary_ts == c.boundary_ts
        assert back.seen_ids == ("a", "b")

    def test_encoded_cursor_is_url_safe(self):
        c = model.Cursor("2026-07-23T03:31:28.904335+00:00", [f"id-{i}" for i in range(60)])
        encoded = c.encode()
        assert "+" not in encoded and "/" not in encoded and "=" not in encoded

    def test_garbage_decodes_to_first_page(self):
        for junk in ("", None, "!!!", "YWJj", "x" * 50):
            assert model.Cursor.decode(junk).is_first_page

    def test_advance_on_empty_page_ends_listing(self):
        assert model.Cursor().advance([]) is None

    def test_advance_collects_ids_of_the_boundary_instant(self):
        rows = [
            {"id": "a", "created_at": "2026-07-23T10:00:00+00:00"},
            {"id": "b", "created_at": "2026-07-23T09:00:00+00:00"},
            {"id": "c", "created_at": "2026-07-23T09:00:00+00:00"},
        ]
        nxt = model.Cursor().advance(rows)
        assert nxt.boundary_ts == "2026-07-23T09:00:00+00:00"
        # Só os do instante do boundary: "a" é de outro instante e nunca voltará.
        assert set(nxt.seen_ids) == {"b", "c"}

    def test_advance_accumulates_across_pages_inside_one_tie(self):
        """O caso que quebra o cursor ingênuo: um instante maior que a página.

        Chunks de documento nascem todos com o mesmo created_at (== submitted_at);
        no corpus real há um instante com 67 pontos e página de 25. Sem acumular
        os ids já entregues, `start_from` (que é inclusivo) devolveria o começo do
        grupo de novo, para sempre.
        """
        ts = "2026-07-23T03:31:28.904335+00:00"
        page1 = [{"id": f"p{i}", "created_at": ts} for i in range(25)]
        c1 = model.Cursor().advance(page1)
        assert len(c1.seen_ids) == 25

        page2 = [{"id": f"p{i}", "created_at": ts} for i in range(25, 50)]
        c2 = c1.advance(page2)
        assert c2.boundary_ts == ts
        assert len(c2.seen_ids) == 50
        assert set(c2.seen_ids) == {f"p{i}" for i in range(50)}

    def test_advance_resets_seen_when_the_instant_changes(self):
        ts_old = "2026-07-23T03:31:28+00:00"
        c = model.Cursor(ts_old, [f"p{i}" for i in range(40)])
        rows = [
            {"id": "p40", "created_at": ts_old},
            {"id": "z", "created_at": "2026-07-22T00:00:00+00:00"},
        ]
        nxt = c.advance(rows)
        assert nxt.boundary_ts == "2026-07-22T00:00:00+00:00"
        assert nxt.seen_ids == ("z",)

    def test_advance_without_timestamp_ends_listing(self):
        assert model.Cursor().advance([{"id": "a", "created_at": None}]) is None


class TestFacets:
    def test_counts_and_orders_by_frequency(self):
        payloads = [
            {"domain": "ai"}, {"domain": "ai"}, {"domain": "infrastructure"},
            {"memory_type": "semantic"},
        ]
        facets = model.facet_counts(payloads)
        assert facets["domain"] == [
            {"value": "ai", "count": 2},
            {"value": "infrastructure", "count": 1},
        ]
        assert facets["memory_type"] == [{"value": "semantic", "count": 1}]

    def test_ignores_missing_empty_and_list_values(self):
        facets = model.facet_counts([{"domain": None}, {"domain": ""}, {"domain": ["a"]}, {}])
        assert facets["domain"] == []

    def test_only_allowlisted_keys_appear(self):
        facets = model.facet_counts([{"domain": "ai", "board": "kanban", "tipo": "x"}])
        assert set(facets) == set(model.FACET_ALLOWLIST)


class TestFilters:
    def test_keeps_allowlisted_and_drops_the_rest(self):
        got = model.clean_filters({"domain": "ai", "board": "kanban", "data": "x"})
        assert got == {"domain": "ai"}

    def test_blank_values_are_not_filters(self):
        assert model.clean_filters({"domain": "  ", "project": ""}) == {}

    def test_flags_accept_truthy_forms_only(self):
        assert model.clean_filters({"only_superseded": "1"}) == {"only_superseded": True}
        assert model.clean_filters({"only_superseded": "0"}) == {}
        assert model.clean_filters({"has_event_date": "on"}) == {"has_event_date": True}

    def test_query_string_roundtrip_and_override(self):
        filters = {"domain": "ai", "only_superseded": True}
        assert model.filters_query(filters) == "domain=ai&only_superseded=1"
        assert model.filters_query(filters, domain=None) == "only_superseded=1"
        assert "memory_type=semantic" in model.filters_query(filters, memory_type="semantic")

    def test_query_string_escapes_values(self):
        assert model.filters_query({"project": "a b&c"}) == "project=a%20b%26c"


class TestPointId:
    def test_accepts_uuid(self):
        assert model.is_point_id("0006e359-7950-4779-8252-aa91c7702ae9")
        assert model.is_point_id("0003fd98-8c75-56f6-833f-f6fab6fc9a69")

    def test_rejects_path_traversal_and_junk(self):
        for bad in ("", "../etc/passwd", "1; DROP", "not-a-uuid", None):
            assert not model.is_point_id(bad)


class TestMemoryRow:
    def test_excerpt_collapses_whitespace_and_truncates(self):
        row = model.memory_row("id", {"data": "linha um\n\n  linha dois"})
        assert row["excerpt"] == "linha um linha dois"
        long_row = model.memory_row("id", {"data": "x" * 400})
        assert len(long_row["excerpt"]) == 240 and long_row["excerpt"].endswith("…")

    def test_superseded_is_true_for_either_marker(self):
        assert model.memory_row("i", {"superseded_at": "2026-01-01"})["superseded"]
        assert model.memory_row("i", {"superseded_by": "other-id"})["superseded"]
        assert not model.memory_row("i", {})["superseded"]

    def test_absent_fields_do_not_raise(self):
        row = model.memory_row("i", {})
        assert row["tags"] == [] and row["domain"] is None


class TestActrView:
    def test_memory_without_history_is_neutral(self):
        view = model.actr_view({"data": "x", "created_at": _iso(10)})
        assert view["has_history"] is False
        assert view["activation"] is None and view["boost"] == 0.0

    def test_activation_matches_the_forks_own_function(self, monkeypatch):
        """A tela não pode ter uma segunda fórmula de ativação.

        Se ela divergir da que o ranking usa, a UI passa a explicar o sistema
        errado — e a divergência seria silenciosa, porque os dois números são
        plausíveis.

        O relógio é congelado porque a ativação decai continuamente: sem isso as
        duas chamadas leem instantes diferentes e a igualdade falha por ~1e-10,
        escondendo o que o teste quer afirmar.
        """
        from mem0.utils import dynamics

        frozen = datetime.now(timezone.utc)
        monkeypatch.setattr(dynamics, "utcnow", lambda: frozen)

        payload = {
            "reinforced_at": [_iso(30), _iso(7), _iso(1)],
            "access_count": 3,
            "created_at": _iso(60),
        }
        view = model.actr_view(payload)
        assert view["activation"] == dynamics.base_level_activation(
            payload["reinforced_at"], 3, first_seen=payload["created_at"]
        )
        assert view["boost"] == dynamics.boost_from_payload(payload)

    def test_recent_reinforcement_activates_more_than_old(self):
        recent = model.actr_view({"reinforced_at": [_iso(1)], "access_count": 1})
        old = model.actr_view({"reinforced_at": [_iso(300)], "access_count": 1})
        assert recent["activation"] > old["activation"]
        assert 0.0 < old["boost"] < recent["boost"] < 1.0

    def test_malformed_timeline_does_not_raise(self):
        view = model.actr_view({"reinforced_at": ["não é data"], "access_count": 1})
        assert view["activation"] is None

    def test_reinforce_counts_are_exposed(self):
        view = model.actr_view(
            {"reinforced_at": [_iso(1)], "reinforce_counts": {"t2": 3, "t3": 1}}
        )
        assert view["reinforce_counts"] == {"t2": 3, "t3": 1}


class TestChainIds:
    def test_supersedes_accepts_string_or_list(self):
        assert model.chain_ids({"supersedes": "abc"})["supersedes"] == ["abc"]
        assert model.chain_ids({"supersedes": ["a", "b"]})["supersedes"] == ["a", "b"]

    def test_supersedes_of_wrong_type_is_ignored(self):
        assert model.chain_ids({"supersedes": 42})["supersedes"] == []

    def test_empty_payload_yields_no_links(self):
        links = model.chain_ids({})
        assert links["superseded_by"] is None and links["supersedes"] == []


class TestRawPayload:
    def test_hides_what_the_screen_already_renders(self):
        raw = model.raw_payload({"data": "x", "importance": 1, "coisa_estranha": "y"})
        assert raw == {"coisa_estranha": "y"}

    def test_keeps_fields_the_mcp_whitelist_would_drop(self):
        raw = model.raw_payload({"memory_scope_evidence": "decisive", "data": "x"})
        assert "memory_scope_evidence" in raw


class TestEntityRow:
    def test_normalizes_linked_ids(self):
        row = model.entity_row("e1", {"data": "DeepMem0", "linked_memory_ids": ["a" * 36]})
        assert row["link_count"] == 1

    def test_poisoned_string_does_not_become_a_list_of_letters(self):
        """Houve linhas gravadas com `set(str)`, que itera caractere a caractere."""
        row = model.entity_row("e1", {"data": "X", "linked_memory_ids": "abc"})
        assert row["link_count"] != 3

    def test_missing_links_yield_zero(self):
        assert model.entity_row("e1", {"data": "X"})["link_count"] == 0
