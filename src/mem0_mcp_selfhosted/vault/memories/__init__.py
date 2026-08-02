"""Read model do corpus para a UI do cofre (navegação de memórias).

Este subpacote é ESTRITAMENTE DE LEITURA. Ele existe porque as superfícies que
a UI precisa mostrar não são todas alcançáveis pelo servidor MCP:

- ``get_memories`` é primeira-página-com-teto, não paginação (sem offset/cursor),
  e o corpus já passa do teto — navegar exige ``scroll`` direto no Qdrant;
- os campos de ACT-R (``reinforced_at``/``access_count``/...) não estão na
  whitelist de metadata do ``search_memories``, então não chegam pelo MCP;
- o entity store não tem caminho MCP nenhum (``list_entities`` faceta a
  collection PRINCIPAL, e as tools de grafo falam com um Neo4j que não existe).

A busca semântica é a exceção deliberada: ela vai pelo MCP, porque o valor está
justamente no pipeline de produção (híbrido + reranker + ACT-R + as_of/evento) —
reimplementá-lo aqui seria uma segunda verdade.

Fronteira de import: ``store.py`` e ``middleware.py`` do cofre são stdlib-only
por invariante testado; nada daqui é importado por eles.
"""

from __future__ import annotations

from mem0_mcp_selfhosted.vault.memories.sources import Sources, SourceUnavailable, build_sources

__all__ = ["Sources", "SourceUnavailable", "build_sources"]
