"""Toda string da UI existe nos dois idiomas, e todo `t.x` de template existe.

A UI é PT/EN em runtime: uma chave presente só em PT some da tela em inglês, e
`Strings` devolve o próprio nome da chave em vez de levantar — falha muda, que
só aparece para quem usa o outro idioma.
"""

import re
from pathlib import Path

import pytest

from mem0_mcp_selfhosted.vault import i18n

TEMPLATES = Path(i18n.__file__).parent / "templates"
KEY_RE = re.compile(r"\bt\.([A-Za-z_][A-Za-z0-9_]*)")
BRACKET_RE = re.compile(r"""\bt\[['"]([A-Za-z_][A-Za-z0-9_]*)['"]\]""")


def _template_files():
    return sorted(TEMPLATES.rglob("*.html"))


class TestTables:
    def test_pt_and_en_have_the_same_keys(self):
        assert set(i18n.PT) == set(i18n.EN)

    def test_no_value_is_empty(self):
        for table_name, table in (("PT", i18n.PT), ("EN", i18n.EN)):
            empty = [k for k, v in table.items() if not str(v).strip()]
            assert not empty, f"{table_name} tem valores vazios: {empty}"

    def test_translations_actually_differ(self):
        """Ao menos as strings de navegação têm de ser traduzidas de fato."""
        assert i18n.PT["navMemories"] != i18n.EN["navMemories"]
        assert i18n.PT["navQueue"] != i18n.EN["navQueue"]


class TestTemplates:
    def test_every_key_used_in_templates_exists(self):
        faltando = {}
        for path in _template_files():
            text = path.read_text(encoding="utf-8")
            used = set(KEY_RE.findall(text)) | set(BRACKET_RE.findall(text))
            ausentes = {k for k in used if k not in i18n.PT}
            if ausentes:
                faltando[path.name] = sorted(ausentes)
        assert not faltando, f"chaves inexistentes usadas em template: {faltando}"

    def test_dynamic_facet_keys_exist(self):
        """`t['facet_' ~ key]` é montado em runtime — o regex não o pega."""
        from mem0_mcp_selfhosted.vault.memories.model import FACET_ALLOWLIST

        for key in FACET_ALLOWLIST:
            assert f"facet_{key}" in i18n.PT
            assert f"facet_{key}" in i18n.EN

    def test_every_screen_has_a_template(self):
        nomes = {p.name for p in _template_files()}
        for esperado in (
            "memories.html", "memory.html", "search.html", "queue.html",
            "entities.html", "entity.html", "users.html", "dashboard.html",
        ):
            assert esperado in nomes


class TestStrings:
    def test_lookup_by_attribute_and_item(self):
        strings = i18n.strings("pt")
        assert strings.navMemories == i18n.PT["navMemories"]
        assert strings["navMemories"] == i18n.PT["navMemories"]

    @pytest.mark.parametrize("lang", ["pt", "en"])
    def test_every_key_is_reachable_in_both_languages(self, lang):
        strings = i18n.strings(lang)
        for key in i18n.PT:
            assert str(strings[key]).strip()
