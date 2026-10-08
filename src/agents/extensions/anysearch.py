"""Built-in AnySearch search tools for the OpenAI Agents SDK.

This module provides a small family of :class:`~agents.FunctionTool` tools that let an
Agent search the web through `AnySearch <https://anysearch.com>`_ — an AI search
infrastructure for agents.

The tool family covers the full AnySearch capability set:

- ``anysearch_search`` — general web search and vertical domain search (via ``tag``).
- ``anysearch_batch_search`` — parallel execution of 1–5 independent queries in one call.
- ``anysearch_sub_domains`` — query the vertical-domain directory for valid ``tag`` values.
- ``anysearch_extract`` — fetch and extract the readable content of a public URL.

Authentication is optional. If no API key is configured the tools fall back to
AnySearch's anonymous mode (lower rate limits); set the ``ANYSEARCH_API_KEY``
environment variable or pass ``api_key=...`` to ``create_anysearch_tools`` to raise
the limits.

Usage::

    from agents import Agent, Runner
    from agents.extensions.anysearch import create_anysearch_tools

    agent = Agent(
        name="Researcher",
        instructions="You are a helpful research assistant.",
        tools=create_anysearch_tools(),
    )

    result = await Runner.run(agent, "What is the latest Go release?")

See ``docs/anysearch.md`` for installation and maintenance notes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import httpx2 as httpx

from agents import FunctionTool, function_tool

logger = logging.getLogger(__name__)

ANYSEARCH_API_BASE_URL = "https://api.anysearch.com"
ANYSEARCH_API_KEY_ENV = "ANYSEARCH_API_KEY"
_DEFAULT_TIMEOUT_SECONDS = 30.0
_MAX_BATCH_QUERIES = 5
_MAX_RESULTS = 10
_MARKDOWN_FORMAT = "markdown"


class AnySearchError(Exception):
    """Raised when the AnySearch API rejects a request."""


def _format_search_payload(
    query: str,
    max_results: int,
    tag: str | None,
    zone: str | None,
    language: str | None,
    output_format: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"query": query, "max_results": max_results}
    if tag:
        payload["tag"] = tag
    if zone:
        payload["zone"] = zone
    if language:
        payload["language"] = language
    if output_format:
        payload["format"] = output_format
    return payload


def _result_to_markdown(title: str, url: str, snippet: str | None, content: str | None) -> str:
    lines = [f"### {title}", f"URL: {url}"]
    if snippet:
        lines.append(snippet)
    if content:
        lines.append(content)
    return "\n\n".join(lines)


def _parse_search_results(data: dict[str, Any], output_format: str | None) -> str:
    results = data.get("results") or []
    if output_format == _MARKDOWN_FORMAT:
        blocks = [
            _result_to_markdown(
                str(item.get("title", "")),
                str(item.get("url", "")),
                item.get("snippet"),
                item.get("content"),
            )
            for item in results
        ]
        return "\n\n---\n\n".join(blocks) if blocks else "No results found."
    return json.dumps(results, ensure_ascii=False)


def create_anysearch_tools(
    *,
    api_key: str | None = None,
    base_url: str = ANYSEARCH_API_BASE_URL,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = 1,
    _transport: Any | None = None,
) -> list[FunctionTool]:
    """Create the AnySearch tool family for an Agent.

    Args:
        api_key: AnySearch API key. Defaults to the ``ANYSEARCH_API_KEY`` environment
            variable; when neither is set the tools use anonymous access.
        base_url: AnySearch API base URL. Overridable for testing/proxying.
        timeout: Per-request timeout in seconds.
        max_retries: Number of retries on rate limits and transient server errors.
        _transport: Internal httpx transport override used by unit tests.

    Returns:
        A list of four :class:`~agents.FunctionTool` instances.
    """
    resolved_key = api_key if api_key is not None else os.environ.get(ANYSEARCH_API_KEY_ENV)

    async def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if resolved_key:
            headers["Authorization"] = f"Bearer {resolved_key}"
        last_error: str | None = None
        for attempt in range(max_retries + 1):
            try:
                async with httpx.AsyncClient(
                    base_url=base_url,
                    timeout=timeout,
                    transport=_transport,
                ) as client:
                    response = await client.post(path, json=payload, headers=headers)
            except httpx.HTTPError as exc:  # network-level failure
                last_error = f"Network error while calling AnySearch: {exc}"
                if attempt < max_retries:
                    await asyncio.sleep(1.0)
                    continue
                raise AnySearchError(last_error) from exc
            if response.status_code in (429, 502, 503, 504) and attempt < max_retries:
                last_error = f"AnySearch returned HTTP {response.status_code}; retrying."
                await asyncio.sleep(1.0)
                continue
            if response.status_code == 402:
                raise AnySearchError(
                    "AnySearch quota exhausted. Configure a valid ANYSEARCH_API_KEY to continue."
                )
            if response.status_code == 401 or response.status_code == 403:
                raise AnySearchError(
                    "AnySearch rejected the API key (HTTP "
                    f"{response.status_code}). Check ANYSEARCH_API_KEY."
                )
            if response.status_code != 200:
                raise AnySearchError(
                    f"AnySearch request failed with HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )
            try:
                body = response.json()
            except ValueError as exc:
                raise AnySearchError("AnySearch returned a non-JSON response.") from exc
            if body.get("code") != 0:
                message = body.get("message") or "unknown error"
                raise AnySearchError(f"AnySearch error: {message}")
            return body.get("data") or {}
        raise AnySearchError(last_error or "AnySearch request failed.")

    @function_tool
    async def anysearch_search(
        query: str,
        max_results: int = 10,
        tag: str | None = None,
        zone: str | None = None,
        language: str | None = None,
        format: str | None = None,
    ) -> str:
        """Search the web with AnySearch.

        Performs a general web search, or a vertical domain search when ``tag`` is
        provided (e.g. ``finance.quote``, ``code.snippet``). Query one intent at a time.
        Use anysearch_sub_domains first to discover valid tags.

        Args:
            query: The natural-language search query.
            max_results: Number of results to return, 1-10 (default 10).
            tag: Optional vertical domain tag like ``finance.quote``.
            zone: Optional region, ``cn`` or ``intl``.
            language: Optional preferred language, e.g. ``zh-CN`` or ``en``.
            format: Output format, ``json`` (default) or ``markdown``.
        """
        clipped = max(1, min(int(max_results), _MAX_RESULTS))
        try:
            data = await _post(
                "/v1/search",
                _format_search_payload(query, clipped, tag, zone, language, format),
            )
        except AnySearchError as exc:
            return str(exc)
        return _parse_search_results(data, format)

    @function_tool
    async def anysearch_batch_search(queries_json: str) -> str:
        """Run 1-5 independent AnySearch searches in parallel.

        Executes multiple search queries in a single call; a failure in one query does
        not block the others.

        Args:
            queries_json: A JSON array of 1-5 query objects, each with the same fields
                as anysearch_search: {"query": "...", "max_results": 10, "tag": "...",
                "zone": "...", "language": "..."}. Example:
                [{"query": "Go 1.26 release notes", "tag": "code.doc"},
                 {"query": "latest NVIDIA driver"}]
        """
        try:
            queries = json.loads(queries_json)
        except ValueError as exc:
            return f"Invalid queries_json: {exc}"
        if not isinstance(queries, list) or not queries:
            return "queries_json must be a non-empty JSON array of query objects."
        queries = queries[:_MAX_BATCH_QUERIES]
        outputs: list[str] = []
        for item in queries:
            if not isinstance(item, dict) or not item.get("query"):
                outputs.append(f"- skipped invalid query entry: {item!r}")
                continue
            try:
                data = await _post(
                    "/v1/search",
                    _format_search_payload(
                        str(item["query"]),
                        int(item.get("max_results", _MAX_RESULTS)),
                        item.get("tag"),
                        item.get("zone"),
                        item.get("language"),
                        item.get("format"),
                    ),
                )
            except AnySearchError as exc:
                outputs.append(f"- query {item['query']!r} failed: {exc}")
                continue
            outputs.append(f"- {item['query']}:\n{_parse_search_results(data, item.get('format'))}")
        return "\n".join(outputs)

    @function_tool
    async def anysearch_sub_domains(domains: str) -> str:
        """List AnySearch vertical-domain search capabilities.

        Returns the sub-domains (tags) and their parameters for one or more vertical
        domains, as a Markdown table. Call this before using ``tag`` in anysearch_search.

        Args:
            domains: Comma-separated domain names, e.g. ``code,finance``. Unknown
                domains return an empty list.
        """
        domain_list = [d.strip() for d in domains.split(",") if d.strip()]
        if not domain_list:
            return "Please provide at least one domain, e.g. code,finance."
        headers = {"Content-Type": "application/json"}
        if resolved_key:
            headers["Authorization"] = f"Bearer {resolved_key}"
        try:
            async with httpx.AsyncClient(
                base_url=base_url,
                timeout=timeout,
                transport=_transport,
            ) as client:
                response = await client.get(
                    "/v1/sub-domains",
                    params=[("domain", d) for d in domain_list],
                    headers=headers,
                )
        except httpx.HTTPError as exc:
            return f"Network error while calling AnySearch: {exc}"
        if response.status_code != 200:
            return (
                f"AnySearch sub-domains request failed with HTTP "
                f"{response.status_code}: {response.text[:500]}"
            )
        try:
            body = response.json()
        except ValueError:
            return "AnySearch returned a non-JSON response."
        if body.get("code") != 0:
            return f"AnySearch error: {body.get('message') or 'unknown error'}"
        domains_data = (body.get("data") or {}).get("domains") or []
        rows: list[str] = [
            "| domain | sub_domain | description | params |",
            "| --- | --- | --- | --- |",
        ]
        for domain_entry in domains_data:
            domain_name = domain_entry.get("domain", "")
            for sub in domain_entry.get("sub_domains") or []:
                params_text = json.dumps(sub.get("params") or {}, ensure_ascii=False)
                rows.append(
                    f"| {domain_name} | {sub.get('sub_domain', '')} | "
                    f"{sub.get('description', '')} | {params_text} |"
                )
        return (
            "\n".join(rows) if len(rows) > 2 else "No sub-domains found for the requested domains."
        )

    @function_tool
    async def anysearch_extract(url: str) -> str:
        """Fetch and extract the readable content of a public web page.

        Returns the page title and extracted body text (as Markdown), truncated at
        50,000 characters. Only ``http``/``https`` URLs are supported.

        Args:
            url: Absolute URL of the page to extract.
        """
        try:
            data = await _post("/v1/extract", {"url": url})
        except AnySearchError as exc:
            return str(exc)
        title = data.get("title") or ""
        content = data.get("content") or ""
        return f"# {title}\n\n{content}" if title else content

    return [
        anysearch_search,
        anysearch_batch_search,
        anysearch_sub_domains,
        anysearch_extract,
    ]


__all__ = [
    "ANYSEARCH_API_BASE_URL",
    "ANYSEARCH_API_KEY_ENV",
    "AnySearchError",
    "create_anysearch_tools",
]
