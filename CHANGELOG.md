# Changelog

## 0.2.0

A rewrite. 0.1.x never reached Jev: it called `typesafe_sdk.Client(...).choice(...)`, which does not exist
in the official SDK, and without the SDK installed it silently used a hard-coded keyword matcher.

### Added
- `@jev_first`: Jev decides first, and your existing router becomes the fallback. Works on conditional-edge
  functions and on `Command`-returning nodes, sync and async.
- Zero-config route discovery from `-> Literal[...]` / `-> Command[Literal[...]]`, and `attach_routes(builder)`
  to use edge destinations and node docstrings as route descriptions.
- `install_shortcut(builder, before=...)` for routing you can't edit, including prebuilt `create_supervisor`.
- Shadow mode (your router decides, Jev is measured), with observers for `install_shortcut`.
- Reports: LLM calls avoided, latency, cost estimates, and a threshold recommendation from shadow data.
- `jev-router check` (verify key, network and model) and `jev-router report <log.jsonl>`.
- Safety defaults: fall back on low confidence, errors, unknown answers and mid-conversation turns; a
  5-second retry budget; `JEV_ROUTER_DISABLED` / `JEV_ROUTER_SHADOW` switches; decision cache.
- `FakeJev` for tests, used only when passed explicitly.
- Privacy: message text is not logged unless `log_inputs=True`.
- Tests (59), CI, LICENSE file, `pyproject.toml`, `typesafe-sdk` declared as a dependency.

### Changed
- Model pinned to `jev-1.13.0` by default instead of an alias.

### Deprecated
- `JevConditionalEdge`. It still works and now calls the real API, falling back to `fallback_route`
  (default: the first route) instead of crashing.
