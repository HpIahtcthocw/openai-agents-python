from __future__ import annotations

import json

import httpx2 as httpx  # type: ignore[import-not-found]
import pytest

from agents import FunctionTool
from agents.extensions.anysearch import create_anysearch_tools
from agents.tool_context import ToolContext


def _json_response(payload: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=payload,
        request=httpx.Request("POST", "https://api.anysearch.com"),
    )


def _search_ok() -> dict:
    return {
        "code": 0,
        "message": "success",
        "request_id": "req-1",
        "data": {
            "results": [
                {
                    "title": "Go 1.26 Release Notes",
                    "url": "https://go.dev/doc/go1.26",
                    "snippet": "Introduction to the changes in Go 1.26.",
                    "content": "Go 1.26 introduces changes...",
                }
            ],
            "metadata": {"total_results": 1, "search_time_ms": 312},
        },
    }


def _tools(handler: object) -> list[FunctionTool]:
    transport = httpx.MockTransport(handler)
    return create_anysearch_tools(_transport=transport)


async def _invoke(tool: FunctionTool, args: dict) -> str:
    context = ToolContext(
        None, tool_name=tool.name, tool_call_id="1", tool_arguments=json.dumps(args)
    )
    return await tool.on_invoke_tool(context, json.dumps(args))


class TestAnySearchSearch:
    @pytest.mark.asyncio
    async def test_search_returns_json_results(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/search"
            assert request.method == "POST"
            body = json.loads(request.content)
            assert body["query"] == "Go 1.26 release notes"
            return _json_response(_search_ok())

        tools = _tools(handler)
        search = next(t for t in tools if t.name == "anysearch_search")
        output = await _invoke(search, {"query": "Go 1.26 release notes"})
        parsed = json.loads(output)
        assert parsed[0]["title"] == "Go 1.26 Release Notes"
        assert parsed[0]["url"] == "https://go.dev/doc/go1.26"

    @pytest.mark.asyncio
    async def test_search_markdown_format(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response(_search_ok())

        tools = _tools(handler)
        search = next(t for t in tools if t.name == "anysearch_search")
        output = await _invoke(search, {"query": "go", "format": "markdown"})
        assert "### Go 1.26 Release Notes" in output
        assert "https://go.dev/doc/go1.26" in output

    @pytest.mark.asyncio
    async def test_search_clamps_max_results(self) -> None:
        seen: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            seen.append(int(body["max_results"]))
            return _json_response(_search_ok())

        tools = _tools(handler)
        search = next(t for t in tools if t.name == "anysearch_search")
        await _invoke(search, {"query": "q", "max_results": 99})
        await _invoke(search, {"query": "q", "max_results": 0})
        assert seen == [10, 1]

    @pytest.mark.asyncio
    async def test_search_api_error_returns_message(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"code": -1, "message": "Query is required."})

        tools = _tools(handler)
        search = next(t for t in tools if t.name == "anysearch_search")
        output = await _invoke(search, {"query": ""})
        assert "Query is required." in output

    @pytest.mark.asyncio
    async def test_search_quota_exhausted_402(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"code": -1, "message": "quota exhausted"}, status_code=402)

        tools = _tools(handler)
        search = next(t for t in tools if t.name == "anysearch_search")
        output = await _invoke(search, {"query": "q"})
        assert "ANYSEARCH_API_KEY" in output

    @pytest.mark.asyncio
    async def test_search_invalid_key_401(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"code": -1, "message": "Invalid API key."}, status_code=401)

        tools = _tools(handler)
        search = next(t for t in tools if t.name == "anysearch_search")
        output = await _invoke(search, {"query": "q"})
        assert "API key" in output

    @pytest.mark.asyncio
    async def test_search_sends_bearer_header(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers.get("Authorization") == "Bearer as_sk_test"
            return _json_response(_search_ok())

        transport = httpx.MockTransport(handler)
        tools = create_anysearch_tools(api_key="as_sk_test", _transport=transport)
        search = next(t for t in tools if t.name == "anysearch_search")
        await _invoke(search, {"query": "q"})

    @pytest.mark.asyncio
    async def test_search_retries_on_429(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return _json_response({}, status_code=429)
            return _json_response(_search_ok())

        transport = httpx.MockTransport(handler)
        tools = create_anysearch_tools(max_retries=1, _transport=transport)
        search = next(t for t in tools if t.name == "anysearch_search")
        output = await _invoke(search, {"query": "q"})
        assert calls["n"] == 2
        assert "Go 1.26" in output


class TestAnySearchBatchSearch:
    @pytest.mark.asyncio
    async def test_batch_search_runs_all_queries(self) -> None:
        paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            body = json.loads(request.content)
            query = body["query"]
            if query == "boom":
                return _json_response({"code": -1, "message": "failed"}, status_code=400)
            return _json_response(_search_ok())

        tools = _tools(handler)
        batch = next(t for t in tools if t.name == "anysearch_batch_search")
        output = await _invoke(
            batch,
            {"queries_json": '[{"query": "first"}, {"query": "boom"}, {"query": "third"}]'},
        )
        assert paths.count("/v1/search") == 3
        assert "first" in output
        assert "failed" in output  # failed query is reported, not fatal
        assert "third" in output

    @pytest.mark.asyncio
    async def test_batch_search_invalid_json(self) -> None:
        tools = _tools(lambda request: _json_response(_search_ok()))
        batch = next(t for t in tools if t.name == "anysearch_batch_search")
        output = await _invoke(batch, {"queries_json": "not-json"})
        assert "Invalid queries_json" in output

    @pytest.mark.asyncio
    async def test_batch_search_empty_array(self) -> None:
        tools = _tools(lambda request: _json_response(_search_ok()))
        batch = next(t for t in tools if t.name == "anysearch_batch_search")
        output = await _invoke(batch, {"queries_json": "[]"})
        assert "non-empty" in output


class TestAnySearchSubDomains:
    @pytest.mark.asyncio
    async def test_sub_domains_returns_markdown_table(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/sub-domains"
            assert request.method == "GET"
            assert ("domain", "code") in request.url.params.multi_items()
            return _json_response(
                {
                    "code": 0,
                    "message": "success",
                    "data": {
                        "domains": [
                            {
                                "domain": "code",
                                "description": "Developer documentation and code search",
                                "sub_domains": [
                                    {
                                        "sub_domain": "code.snippet",
                                        "description": "Search real code implementations",
                                        "params": {"lang": {"required": False}},
                                    }
                                ],
                            }
                        ]
                    },
                }
            )

        tools = _tools(handler)
        sub = next(t for t in tools if t.name == "anysearch_sub_domains")
        output = await _invoke(sub, {"domains": "code"})
        assert "| domain | sub_domain | description | params |" in output
        assert "code.snippet" in output

    @pytest.mark.asyncio
    async def test_sub_domains_empty_domains(self) -> None:
        tools = _tools(lambda request: _json_response({}))
        sub = next(t for t in tools if t.name == "anysearch_sub_domains")
        output = await _invoke(sub, {"domains": ""})
        assert "at least one domain" in output

    @pytest.mark.asyncio
    async def test_sub_domains_no_matches(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"code": 0, "message": "success", "data": {"domains": []}})

        tools = _tools(handler)
        sub = next(t for t in tools if t.name == "anysearch_sub_domains")
        output = await _invoke(sub, {"domains": "unknown"})
        assert "No sub-domains" in output


class TestAnySearchExtract:
    @pytest.mark.asyncio
    async def test_extract_returns_title_and_content(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/extract"
            body = json.loads(request.content)
            assert body["url"] == "https://example.com/article"
            return _json_response(
                {
                    "code": 0,
                    "message": "success",
                    "data": {
                        "url": "https://example.com/article",
                        "title": "Example Article",
                        "content": "Extracted page content...",
                    },
                }
            )

        tools = _tools(handler)
        extract = next(t for t in tools if t.name == "anysearch_extract")
        output = await _invoke(extract, {"url": "https://example.com/article"})
        assert "# Example Article" in output
        assert "Extracted page content" in output

    @pytest.mark.asyncio
    async def test_extract_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"code": -1, "message": "URL 无效。"}, status_code=400)

        tools = _tools(handler)
        extract = next(t for t in tools if t.name == "anysearch_extract")
        output = await _invoke(extract, {"url": "not-a-url"})
        assert "URL" in output


class TestToolSchema:
    def test_all_tools_have_strict_schema(self) -> None:
        tools = create_anysearch_tools(_transport=httpx.MockTransport(lambda r: _json_response({})))
        names = [t.name for t in tools]
        assert names == [
            "anysearch_search",
            "anysearch_batch_search",
            "anysearch_sub_domains",
            "anysearch_extract",
        ]
        for tool in tools:
            assert tool.strict_json_schema is True
            assert tool.params_json_schema["type"] == "object"


class TestAnySearchEdgeCases:
    @pytest.mark.asyncio
    async def test_search_retries_network_error(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ConnectError("boom", request=request)
            return _json_response(_search_ok())

        transport = httpx.MockTransport(handler)
        tools = create_anysearch_tools(max_retries=1, _transport=transport)
        search = next(t for t in tools if t.name == "anysearch_search")
        output = await _invoke(search, {"query": "q"})
        assert calls["n"] == 2
        assert "Go 1.26" in output

    @pytest.mark.asyncio
    async def test_search_network_error_exhausted(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom", request=request)

        transport = httpx.MockTransport(handler)
        tools = create_anysearch_tools(max_retries=0, _transport=transport)
        search = next(t for t in tools if t.name == "anysearch_search")
        output = await _invoke(search, {"query": "q"})
        assert "Network error" in output

    @pytest.mark.asyncio
    async def test_search_non_json_response(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>oops</html>", request=request)

        tools = _tools(handler)
        search = next(t for t in tools if t.name == "anysearch_search")
        output = await _invoke(search, {"query": "q"})
        assert "non-JSON" in output

    @pytest.mark.asyncio
    async def test_search_passes_optional_params(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return _json_response(_search_ok())

        tools = _tools(handler)
        search = next(t for t in tools if t.name == "anysearch_search")
        await _invoke(
            search,
            {
                "query": "q",
                "tag": "finance.quote",
                "zone": "cn",
                "language": "zh-CN",
                "format": "markdown",
            },
        )
        assert captured["tag"] == "finance.quote"
        assert captured["zone"] == "cn"
        assert captured["language"] == "zh-CN"
        assert captured["format"] == "markdown"

    @pytest.mark.asyncio
    async def test_batch_search_skips_invalid_entry(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response(_search_ok())

        tools = _tools(handler)
        batch = next(t for t in tools if t.name == "anysearch_batch_search")
        output = await _invoke(batch, {"queries_json": '[{"query": "good"}, {"foo": 1}, "nope"]'})
        assert "good" in output
        assert "skipped invalid" in output

    @pytest.mark.asyncio
    async def test_sub_domains_network_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom", request=request)

        tools = _tools(handler)
        sub = next(t for t in tools if t.name == "anysearch_sub_domains")
        output = await _invoke(sub, {"domains": "code"})
        assert "Network error" in output

    @pytest.mark.asyncio
    async def test_sub_domains_http_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({}, status_code=500)

        tools = _tools(handler)
        sub = next(t for t in tools if t.name == "anysearch_sub_domains")
        output = await _invoke(sub, {"domains": "code"})
        assert "HTTP 500" in output

    @pytest.mark.asyncio
    async def test_sub_domains_api_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _json_response({"code": -1, "message": "bad domain"})

        tools = _tools(handler)
        sub = next(t for t in tools if t.name == "anysearch_sub_domains")
        output = await _invoke(sub, {"domains": "code"})
        assert "bad domain" in output

    @pytest.mark.asyncio
    async def test_sub_domains_non_json(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="not-json", request=request)

        tools = _tools(handler)
        sub = next(t for t in tools if t.name == "anysearch_sub_domains")
        output = await _invoke(sub, {"domains": "code"})
        assert "non-JSON" in output
