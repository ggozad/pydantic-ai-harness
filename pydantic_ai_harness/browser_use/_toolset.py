"""The `browse_web` toolset and the factory contract for building browser-use agents."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from fnmatch import fnmatch
from typing import Literal, Protocol, overload
from urllib.parse import urlsplit

import anyio
from pydantic import BaseModel, ValidationError
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_harness.browser_use._model import resolve_chat_model
from pydantic_ai_harness.browser_use._settings import BrowserAgentSettings

try:
    from browser_use import Agent as _BrowserUseAgent
    from browser_use import Tools
    from browser_use.browser import BrowserProfile, BrowserSession
    from browser_use.llm.base import BaseChatModel
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'browser-use is required for BrowserUse. Install it with: pip install "pydantic-ai-harness[browser-use]"'
    ) from _import_error

logger = logging.getLogger(__name__)

_TOOL_NAME = 'browse_web'
_LOCAL_FILE_PATTERNS = ['file://*']
_LOCALHOST_PATTERNS = ['localhost', 'localhost.*', '*.localhost']
_LOCALHOST_HOSTS = ('localhost', 'localhost.example', 'example.localhost')

_HTTP_SCHEME_GLOB = 'http*'

_FILE_SCHEME_ALLOWLIST_ERROR = (
    'An `allowed_domains` entry that permits only `file://` URLs is not supported; '
    'local file navigation is always prohibited.'
)

# Teardown runs shielded from cancellation, so an unresponsive browser could otherwise hang the
# caller forever on exit. Bound it instead: a browser that will not close within this window is
# retained for a later cleanup attempt, which is strictly better than wedging the run.
_TEARDOWN_TIMEOUT = 30


def _strip_trailing_dot(hostname: str) -> str:
    """Drop a terminal DNS dot, which names the same host: `localhost.` is `localhost`.

    Dropping it makes an entry marginally stricter than the caller wrote, because
    browser-use matches hostnames literally and so will not match a `https://host./`
    URL against the canonical pattern. That is the safe direction: left in place, the
    dot lets an entry name a host without resembling it.
    """
    return hostname[:-1] if hostname.endswith('.') else hostname


def _restrict_to_http_schemes(domain: str) -> list[str]:
    """Scheme-qualify a host-only allowlist entry so it cannot admit a `file://` URL."""
    domain = _strip_trailing_dot(domain)
    restricted = [f'{_HTTP_SCHEME_GLOB}://{domain}']
    if '*' not in domain and domain.count('.') == 1:
        restricted.append(f'{_HTTP_SCHEME_GLOB}://www.{domain}')
    return restricted


def _normalize_allowed_domain(domain: str) -> list[str]:
    """Make an allowlist entry safe for browser-use's URL matching.

    browser-use consults `allowed_domains` or `prohibited_domains`, never both:
    a non-empty allowlist short-circuits the prohibition list entirely. A host-only
    entry matches on hostname regardless of scheme, so it would admit
    `file://<host>/...` and override the local-file prohibition. Qualifying such an
    entry to `http`/`https` keeps the caller's intent (including a local dev server
    on `http://localhost`) while leaving the file scheme unreachable.

    Entries already scoped to a scheme are left alone unless that scheme admits `file`:
    a scheme glob is narrowed to whichever of `http`/`https` it already matched, and an
    entry naming `file` alone is dropped. Narrowing emits the matched schemes explicitly
    rather than a single `http*`, because a glob can admit `file` and only one of the two
    (`????` matches `http` but not `https`), and widening to both would newly permit a
    scheme the caller never allowed. browser-use restricts bare `*.example.com` patterns
    to `http`/`https` itself, so those need no qualification.
    """
    if '://' in domain:
        scheme, rest = domain.split('://', maxsplit=1)
        if fnmatch('file', scheme.lower()):
            return [f'{matched}://{rest}' for matched in ('http', 'https') if fnmatch(matched, scheme.lower())]
        parsed = urlsplit(domain)
        if (
            parsed.scheme in ('http', 'https')
            and parsed.netloc
            and not parsed.path
            and not parsed.query
            and not parsed.fragment
            and '*' not in domain
        ):
            return [f'{domain}/*']
        return [domain]
    if domain.startswith('*.'):
        return [domain]
    return _restrict_to_http_schemes(domain)


def _normalize_allowed_domains(allowed_domains: list[str]) -> list[str]:
    """Normalize a capability-level allowlist, rejecting one that permits only `file://`."""
    normalized = [entry for domain in allowed_domains for entry in _normalize_allowed_domain(domain)]
    if allowed_domains and not normalized:
        raise ValueError(_FILE_SCHEME_ALLOWLIST_ERROR)
    return normalized


def _normalize_profile_allowed_domains(
    allowed_domains: list[str] | set[str] | None,
) -> list[str] | set[str] | None:
    """Normalize a browser profile's allowed domains without losing set semantics."""
    if allowed_domains is None:
        return None
    normalized: list[str] = [entry for domain in allowed_domains for entry in _normalize_allowed_domain(domain)]
    if allowed_domains and not normalized:
        raise ValueError(_FILE_SCHEME_ALLOWLIST_ERROR)
    return set(normalized) if isinstance(allowed_domains, set) else normalized


def _glob_hostname(pattern: str) -> str:
    """The hostname of an allowlist pattern, read as an fnmatch glob rather than a URL.

    `urlsplit` cannot be used here: it rejects a leading character class such as
    `[ab]*.example.com` as a malformed IPv6 literal, and it will not parse a glob
    scheme like `http*://`. The authority is split by hand instead. A trailing
    `:port` is dropped only when the colon falls outside a bracketed IPv6 literal,
    so `[::1]` keeps its address while `localhost:*` loses its port glob.
    """
    authority = pattern.split('://', maxsplit=1)[-1].split('/', maxsplit=1)[0]
    authority = authority.rpartition('@')[2]
    host, separator, port = authority.rpartition(':')
    if separator and ']' not in port:
        authority = host
    return _strip_trailing_dot(authority.lower())


def _pattern_allows_localhost(pattern: str) -> bool:
    """Whether a browser-use allowlist pattern would permit a localhost URL.

    Every allowlist reaching here is normalized first, so an entry is either
    scheme-qualified or a `*.`-prefixed domain glob.

    Matching representative hosts against the entry catches globs, but only for hosts
    the samples name. A concrete `<label>.localhost` is loopback under RFC 6761 without
    resembling any sample, so the suffix is tested directly as well.
    """
    if pattern.startswith('*.'):
        domain = _strip_trailing_dot(pattern[2:].lower())
        return any(host == domain or host.endswith(f'.{domain}') for host in _LOCALHOST_HOSTS)
    hostname = _glob_hostname(pattern)
    if hostname.endswith('.localhost') or hostname.startswith('localhost.'):
        return True
    return any(fnmatch(host, hostname) for host in _LOCALHOST_HOSTS)


@overload
def _exclude_localhost_allowlist_entries(allowed_domains: list[str]) -> list[str]: ...  # pragma: no cover


@overload
def _exclude_localhost_allowlist_entries(allowed_domains: set[str]) -> set[str]: ...  # pragma: no cover


@overload
def _exclude_localhost_allowlist_entries(allowed_domains: None) -> None: ...  # pragma: no cover


def _exclude_localhost_allowlist_entries(allowed_domains: list[str] | set[str] | None) -> list[str] | set[str] | None:
    """Keep a profile allowlist from overriding localhost prohibitions."""
    if allowed_domains is None:
        return None
    filtered = [domain for domain in allowed_domains if not _pattern_allows_localhost(domain)]
    if allowed_domains and not filtered:
        raise ValueError('An `allowed_domains` entry that permits only localhost requires `block_ip_addresses=False`.')
    return set(filtered) if isinstance(allowed_domains, set) else filtered


async def _kill(session: BrowserSession) -> bool:
    """Close a browser session, even while the caller is being cancelled.

    `BrowserSession.kill` is not a single round-trip: it saves storage state,
    dispatches a stop event, and drains the event bus, so it suspends several
    times over CDP. Unshielded, the first of those awaits inside a cancelled
    scope raises and leaves a live Chromium behind -- holding a lock on
    `user_data_dir` if the profile has one. The same shape as `ModalSandbox`'s
    teardown, for the same reason.

    Its failures are swallowed for that same reason. `kill()` can raise -- it awaits a
    storage-state save, a forced stop and an event-bus drain, none of which are wrapped
    upstream -- and it runs in a `finally`, where a raise replaces whatever was unwinding
    through it: a completed browse becomes a `TimeoutError` to the caller, a real error
    becomes a teardown error, and a cancellation stops propagating. The browser is left to
    the caller retains the session for another attempt, so nothing is gained by
    letting it through.
    """
    succeeded = True
    with anyio.CancelScope(shield=True):
        with anyio.move_on_after(_TEARDOWN_TIMEOUT) as timeout_scope:
            try:
                await session.kill()
            except (Exception, asyncio.CancelledError):
                # A `CancelledError` here is either the timeout below -- its scope
                # is inside the shield, so it lands on this await -- or one raised
                # by `kill()` itself; the shield rules out external cancellation.
                # Both retain the session, but a timeout reports itself, so do not
                # blame the teardown for failing on top of it.
                succeeded = False
                if not timeout_scope.cancel_called:
                    logger.warning(
                        'browser-use session teardown failed; retaining the session for retry', exc_info=True
                    )
        if timeout_scope.cancel_called:
            succeeded = False
            logger.warning(
                'browser-use session teardown timed out after %s seconds; retaining the session for retry',
                _TEARDOWN_TIMEOUT,
            )
    return succeeded


@dataclass(frozen=True)
class BrowserLease:
    """A remote browser leased for a browse: its CDP endpoint and how to release it.

    A `BrowserLeaseProvider` returns one per `browse_web` call (in `'call'`
    scope) or once per shared session (in `'agent'` scope), so each lease owns a
    freshly-provisioned remote browser (e.g. a Steel session) that is handed back
    when the browse -- or the shared session -- ends. `release` runs in a
    shielded, time-bounded teardown, on success, failure, and cancellation, and
    is retried if it fails or times out, so it must be idempotent. `cdp_url` is
    excluded from `repr` because remote endpoints often carry credentials.
    """

    cdp_url: str = field(repr=False)
    """The CDP endpoint of the leased browser, as `BrowserUse.cdp_url` would take it.

    Excluded from `repr` because a remote endpoint often carries a token.
    """

    release: Callable[[], Awaitable[None]]
    """Hand the browser back, called once the lease is finished with.

    Must be idempotent and scoped to this lease: a failed or timed-out release
    is retried later, by which time another call may own the backend, and it
    must not tear down a browser this lease no longer owns.
    """


class BrowserLeaseProvider(Protocol):
    """Leases a remote browser for `browse_web` to drive.

    Called once per `browse_web` call in `'call'` scope, or once for the shared
    session in `'agent'` scope. Returning an awaitable rather than being
    declared `async` keeps any callable that yields a `BrowserLease` eligible,
    an `async def` provider included. Set it via
    `BrowserUse.browser_lease_provider`, which is mutually exclusive with
    `BrowserUse.cdp_url`.
    """

    def __call__(self) -> Awaitable[BrowserLease]:
        """Provision a browser and return the lease that owns it."""
        ...  # pragma: no cover


@dataclass
class _PendingResource:
    """A leased browser awaiting cleanup: kill the client session, release the lease.

    Each step is retried until it succeeds. Releasing the lease frees the remote
    browser, so once it succeeds a client-side session kill is moot and dropped.
    """

    session: BrowserSession | None
    lease: BrowserLease | None

    @property
    def done(self) -> bool:
        return self.session is None and self.lease is None


async def _release(lease: BrowserLease) -> bool:
    """Release a lease, shielded and time-bounded like `_kill`. True when released.

    Swallows and logs failures and timeouts -- release does network I/O and runs
    in a `finally`, so a raise would replace whatever was unwinding through it --
    and reports the outcome so a failed release can be retained for retry.
    """
    succeeded = True
    with anyio.CancelScope(shield=True):
        with anyio.move_on_after(_TEARDOWN_TIMEOUT) as timeout_scope:
            try:
                await lease.release()
            except (Exception, asyncio.CancelledError):
                # A `CancelledError` here is either the timeout below -- its scope
                # is inside the shield, so it lands on this await -- or one raised
                # by `release()` itself; the shield rules out external
                # cancellation. Both retain the lease, but a timeout reports
                # itself, so do not blame the callback for failing on top of it.
                succeeded = False
                if not timeout_scope.cancel_called:
                    logger.warning('browser lease release failed; retaining for retry', exc_info=True)
        if timeout_scope.cancel_called:
            succeeded = False
            logger.warning(
                'browser lease release timed out after %s seconds; retaining for retry',
                _TEARDOWN_TIMEOUT,
            )
    return succeeded


async def _cleanup(resource: _PendingResource) -> bool:
    """Kill the client session, then release the lease; True once both are done.

    Kill first (a graceful client shutdown while the browser is alive), then
    release (the authoritative teardown that frees the remote browser). A
    successful release drops the session too: the remote browser is gone, so
    retrying its kill would be meaningless. Each step is cleared as it succeeds,
    so a retry only redoes what is left.
    """
    if resource.session is not None and await _kill(resource.session):
        resource.session = None
    if resource.lease is not None and await _release(resource.lease):
        resource.lease = None
        resource.session = None
    return resource.done


class BrowserAgentHistory(Protocol):
    """The subset of browser-use's `AgentHistoryList` that the `browse_web` tool reads.

    `final_result` is the text of the agent's final `done` action (or `None` when
    it never finished), `errors` collects per-step error messages, `is_successful`
    is the agent's own verdict on the finished task (`None` while not done), and
    `structured_output` is the final result parsed against the configured output
    schema (`None` when no schema was configured; raises a pydantic
    `ValidationError` when the result does not parse). A real `AgentHistoryList`
    satisfies this protocol as-is.
    """

    def final_result(self) -> None | str:
        """The text of the final result, or `None` when the agent never finished."""
        ...  # pragma: no cover

    def errors(self) -> list[str | None]:
        """One entry per step: the step's error message, or `None` for clean steps."""
        ...  # pragma: no cover

    def is_successful(self) -> bool | None:
        """The agent's own success verdict for a finished task; `None` while not done."""
        ...  # pragma: no cover

    @property
    def structured_output(self) -> BaseModel | None:
        """The final result parsed against the configured output schema, if any."""
        ...  # pragma: no cover


class BrowserAgent(Protocol):
    """A ready-to-run browser agent for one task, as built by a `BrowserAgentFactory`."""

    def run(self, max_steps: int = 500) -> Awaitable[BrowserAgentHistory]:
        """Run the agent's own loop until the task finishes or `max_steps` is reached.

        Declared as returning `Awaitable` (not `async def`) so that
        `browser_use.Agent.run`, whose tracing decorator types it as returning
        a plain `Coroutine`, satisfies the protocol; an `async def`
        implementation satisfies it too.
        """
        ...  # pragma: no cover


@dataclass
class BrowserTask:
    """Everything the `browse_web` tool passes to a `BrowserAgentFactory` for one call.

    A dataclass rather than keyword arguments so that new fields can be added
    without breaking existing factories: unpack what you forward, ignore the
    rest.
    """

    task: str
    """The natural-language goal for the browser agent."""

    llm: BaseChatModel | None
    """The resolved chat model; `None` means browser-use's own default."""

    browser_session: BrowserSession
    """The session to browse in. Owned by the tool: killed after the call in
    `'call'` scope, kept alive and reused in `'agent'` scope."""

    use_vision: bool | Literal['auto']
    """Whether to send page screenshots to the model (`'auto'` follows the model's capabilities)."""

    output_schema: type[BaseModel] | None
    """Schema the agent's final result must conform to, forwarded as browser-use's `output_model_schema`."""

    sensitive_data: dict[str, str | dict[str, str]] | None = field(repr=False)
    """Secret placeholders for browser-use to substitute without showing the values to the model.

    Kept out of `repr()`: a `BrowserTask` is what a factory receives, so it is the object most
    likely to end up in a log line or a traceback."""

    extend_system_message: str | None
    """Extra instructions appended to the browser agent's own system prompt."""

    settings: BrowserAgentSettings
    """The remaining browser-use `Agent` options, always a concrete instance.

    Its `*_llm` fields arrive resolved to browser-use chat models, so factories
    can forward them verbatim.
    """


class BrowserAgentFactory(Protocol):
    """Builds the browser agent that `browse_web` runs for one task.

    The default factory constructs a real `browser_use.Agent` from the
    `BrowserTask`, forwarding `BrowserTask.settings` in full. Pass a custom one
    via `BrowserUse.browser_agent` to intercept construction, or to substitute
    a fake in tests. Two rules: the factory must not start or stop the session
    itself (`browse_web` owns the session lifecycle), and it should keep
    browser-use's signal handling off (`enable_signal_handler=False`) -- the
    sub-agent must not install its own SIGINT handling inside a host
    application.
    """

    def __call__(self, request: BrowserTask) -> BrowserAgent:
        """Build a runnable browser agent for one `browse_web` call."""
        ...  # pragma: no cover


def _safe_tools(settings: BrowserAgentSettings) -> Tools[None]:
    """Return tools without unapproved file reads or uploads.

    browser-use 0.13.7's `read_file` action calls `FileSystem.read_file_structured`,
    which imports pypdf 6.10.2 for PDFs. Re-evaluate this restriction when
    browser-use publishes a release with pypdf 6.14.2 or later:
    https://github.com/browser-use/browser-use/commit/5405febce2d8834737bc7cd9afee9ad4604ec447
    """
    if settings.tools is None:
        return Tools(
            exclude_actions=['read_file', 'upload_file'],
            display_files_in_done_text=settings.display_files_in_done_text,
        )
    settings.tools.exclude_action('read_file')
    settings.tools.exclude_action('upload_file')
    return settings.tools


def default_browser_agent(request: BrowserTask) -> BrowserAgent:
    """Build a real `browser_use.Agent` (the default `BrowserAgentFactory`).

    The `resolve_chat_model` calls on the settings' `*_llm` fields narrow their
    static type; the toolset already resolved the values, so at runtime they
    pass through unchanged.
    """
    settings = request.settings
    # Explicit type arguments: `Agent`'s context and structured-output type
    # variables are unconstrained by this call, and neither is used here.
    # Signal handling stays off: the sub-agent must not install its own SIGINT
    # pause/resume handling inside a host application.
    return _BrowserUseAgent[None, BaseModel](
        task=request.task,
        llm=request.llm,
        browser_session=request.browser_session,
        use_vision=request.use_vision,
        output_model_schema=request.output_schema,
        sensitive_data=request.sensitive_data,
        extend_system_message=request.extend_system_message,
        enable_signal_handler=False,
        tools=_safe_tools(settings),
        override_system_message=settings.override_system_message,
        max_failures=settings.max_failures,
        max_actions_per_step=settings.max_actions_per_step,
        use_thinking=settings.use_thinking,
        flash_mode=settings.flash_mode,
        max_history_items=settings.max_history_items,
        page_extraction_llm=resolve_chat_model(settings.page_extraction_llm),
        fallback_llm=resolve_chat_model(settings.fallback_llm),
        use_judge=settings.use_judge,
        judge_llm=resolve_chat_model(settings.judge_llm),
        ground_truth=settings.ground_truth,
        calculate_cost=settings.calculate_cost,
        vision_detail_level=settings.vision_detail_level,
        llm_screenshot_size=settings.llm_screenshot_size,
        llm_timeout=settings.llm_timeout,
        step_timeout=settings.step_timeout,
        directly_open_url=settings.directly_open_url,
        include_recent_events=settings.include_recent_events,
        final_response_after_failure=settings.final_response_after_failure,
        enable_planning=settings.enable_planning,
        planning_replan_on_stall=settings.planning_replan_on_stall,
        planning_exploration_limit=settings.planning_exploration_limit,
        loop_detection_enabled=settings.loop_detection_enabled,
        loop_detection_window=settings.loop_detection_window,
        message_compaction=settings.message_compaction,
        max_clickable_elements_length=settings.max_clickable_elements_length,
        include_tool_call_examples=settings.include_tool_call_examples,
        initial_actions=settings.initial_actions,
        file_system_path=settings.file_system_path,
        display_files_in_done_text=settings.display_files_in_done_text,
        save_conversation_path=settings.save_conversation_path,
        save_conversation_path_encoding=settings.save_conversation_path_encoding,
        include_attributes=settings.include_attributes,
        extraction_schema=settings.extraction_schema,
        sample_images=settings.sample_images,
        skills=settings.skills,
        skill_ids=settings.skill_ids,
        pricing_url=settings.pricing_url,
        generate_gif=settings.generate_gif,
        demo_mode=settings.demo_mode,
    )


class BrowserUseToolset(FunctionToolset[AgentDepsT]):
    """Provides the `browse_web` tool: run an autonomous browser-use agent per task."""

    def __init__(
        self,
        *,
        browser_agent: BrowserAgentFactory,
        llm: BaseChatModel | None,
        browser_profile: BrowserProfile | None,
        allowed_domains: list[str] | None,
        block_ip_addresses: bool,
        headless: bool | None,
        max_steps: int,
        use_vision: bool | Literal['auto'],
        output_schema: type[BaseModel] | None,
        sensitive_data: dict[str, str | dict[str, str]] | None,
        extend_system_message: str | None,
        settings: BrowserAgentSettings,
        session_scope: Literal['call', 'agent'],
        cdp_url: str | None,
        browser_lease_provider: BrowserLeaseProvider | None = None,
    ) -> None:
        super().__init__()
        if cdp_url is not None and browser_lease_provider is not None:
            raise ValueError('Set either `cdp_url` or `browser_lease_provider`, not both.')
        self._browser_agent = browser_agent
        self._llm = llm
        self._browser_profile = browser_profile
        self._allowed_domains = allowed_domains
        self._block_ip_addresses = block_ip_addresses
        self._headless = headless
        self._max_steps = max_steps
        self._use_vision: bool | Literal['auto'] = use_vision
        self._output_schema = output_schema
        self._sensitive_data = sensitive_data
        self._extend_system_message = extend_system_message
        # Resolve the settings' chat models once, so every factory (custom
        # ones included) receives a `BrowserTask` with ready-to-use models.
        self._settings = replace(
            settings,
            page_extraction_llm=resolve_chat_model(settings.page_extraction_llm),
            fallback_llm=resolve_chat_model(settings.fallback_llm),
            judge_llm=resolve_chat_model(settings.judge_llm),
        )
        self._session_scope: Literal['call', 'agent'] = session_scope
        self._cdp_url = cdp_url
        self._browser_lease_provider = browser_lease_provider
        self._shared_session: BrowserSession | None = None
        self._shared_lease: BrowserLease | None = None
        self._pending_cleanup: list[_PendingResource] = []
        self._session_closed = False
        self._active_call_sessions = 0
        self._call_cleanup_in_progress = False
        self._call_condition = asyncio.Condition()
        self._cleanup_lock = asyncio.Lock()
        self._session_lock = asyncio.Lock()
        self.add_function(self.browse_web, name=_TOOL_NAME)

    def _build_session(self, cdp_url: str | None) -> BrowserSession:
        """A fresh session, merging the profile with the capability's overrides.

        `BrowserSession` itself merges a provided `browser_profile` with directly
        passed fields, letting the non-`None` direct fields win, so the
        capability's `headless`, `allowed_domains`, and `cdp_url` override the
        profile exactly like they would on a hand-built session. `headless`
        defaults to on only when no profile is given; a profile keeps its own
        setting.

        In `'agent'` scope the session is created with `keep_alive=True`:
        without it, `browser_use.Agent` kills the session at the end of each
        run, which would break reuse across calls. The toolset's own
        `kill()` (in `aclose` and on a failed run) is a force stop and closes
        the browser regardless.
        """
        headless = self._headless
        if headless is None and self._browser_profile is None:
            headless = True
        browser_profile = self._browser_profile
        # Normalization also runs in `BrowserUse.__post_init__`, which reports a bad
        # allowlist at construction. Repeating it here covers a `BrowserUseToolset`
        # built directly, which skips the capability entirely; the transform is
        # idempotent, so the capability path is unaffected.
        allowed_domains = None if self._allowed_domains is None else _normalize_allowed_domains(self._allowed_domains)
        if browser_profile is None:
            browser_profile = BrowserProfile(prohibited_domains=_LOCAL_FILE_PATTERNS)
        else:
            prohibited_domains = list(browser_profile.prohibited_domains or ())
            prohibited_domains.extend(pattern for pattern in _LOCAL_FILE_PATTERNS if pattern not in prohibited_domains)
            browser_profile = browser_profile.model_copy(update={'prohibited_domains': prohibited_domains})
        # `BrowserSession` lets a non-`None` `allowed_domains` replace the profile's own
        # list, so the profile allowlist only reaches the navigation guard when the
        # direct one is absent.
        if allowed_domains is None and browser_profile.allowed_domains is not None:
            browser_profile = browser_profile.model_copy(
                update={'allowed_domains': _normalize_profile_allowed_domains(browser_profile.allowed_domains)}
            )
        if self._sensitive_data is not None:
            browser_profile = browser_profile.model_copy(update={'cross_origin_iframes': False})
        if self._block_ip_addresses:
            allowed_domains = _exclude_localhost_allowlist_entries(allowed_domains)
            if allowed_domains is None:
                profile_allowed_domains = _exclude_localhost_allowlist_entries(browser_profile.allowed_domains)
            else:
                profile_allowed_domains = browser_profile.allowed_domains
            prohibited_domains = list(browser_profile.prohibited_domains or ())
            prohibited_domains.extend(pattern for pattern in _LOCALHOST_PATTERNS if pattern not in prohibited_domains)
            browser_profile = browser_profile.model_copy(
                update={
                    'allowed_domains': profile_allowed_domains,
                    'block_ip_addresses': True,
                    'prohibited_domains': prohibited_domains,
                }
            )
        else:
            browser_profile = browser_profile.model_copy(update={'block_ip_addresses': False})
        return BrowserSession(
            cdp_url=cdp_url,
            browser_profile=browser_profile,
            headless=headless,
            allowed_domains=allowed_domains,
            keep_alive=True if self._session_scope == 'agent' else None,
        )

    async def _acquire_lease(self) -> BrowserLease | None:
        """Lease a remote browser for one browse, or `None` for a static/local cdp_url."""
        if self._browser_lease_provider is not None:
            return await self._browser_lease_provider()
        return None

    def _lease_cdp_url(self, lease: BrowserLease | None) -> str | None:
        """The endpoint to connect to: the lease's when leased, else the static url."""
        return lease.cdp_url if lease is not None else self._cdp_url

    async def _teardown(self, session: BrowserSession | None, lease: BrowserLease | None) -> None:
        """Tear down a session and its lease, retaining them if cleanup needs a retry."""
        resource = _PendingResource(session=session, lease=lease)
        with anyio.CancelScope(shield=True):
            if not await _cleanup(resource):
                async with self._cleanup_lock:
                    self._pending_cleanup.append(resource)

    async def _run_agent(self, task: str, session: BrowserSession) -> BrowserAgentHistory:
        """Build the sub-agent for `task` against `session` and run its loop."""
        agent = self._browser_agent(
            BrowserTask(
                task=task,
                llm=self._llm,
                browser_session=session,
                use_vision=self._use_vision,
                output_schema=self._output_schema,
                sensitive_data=self._sensitive_data,
                extend_system_message=self._extend_system_message,
                settings=self._settings,
            )
        )
        return await agent.run(max_steps=self._max_steps)

    def _render_result(self, history: BrowserAgentHistory) -> str:
        """The tool result for a finished run: text, schema JSON, or a failure report."""
        result = history.final_result()
        if result is None:
            step_errors = [error for error in history.errors() if error]
            detail = '; '.join(step_errors) if step_errors else 'no further details'
            return f'The browser agent stopped without producing a result ({detail}).'
        answer = self._render_answer(result, history)
        # The verdict is applied to whatever the answer turned out to be, schema JSON included:
        # `structured_output` parses the final result whether or not the sub-agent called `done`
        # with `success=False`, so reading it alone would present a run it gave up on as a clean
        # answer.
        if history.is_successful() is False:
            return f'The browser agent could not fully complete the task. Its final result: {answer}'
        return answer

    def _render_answer(self, result: str, history: BrowserAgentHistory) -> str:
        """The answer itself: schema JSON when one is configured, otherwise the agent's own text."""
        if self._output_schema is None:
            return result
        try:
            structured = history.structured_output
        except ValidationError as error:
            raise ModelRetry(
                f'The browser agent finished, but its result did not match the configured output schema: {error}'
            ) from error
        # A `None` here is unreachable with browser-use's own history, which parses whenever there
        # is a final result and a schema -- both already true. Only a custom factory's history can
        # land here, and its prose is a better answer than an invented failure.
        return structured.model_dump_json() if structured is not None else result

    async def browse_web(self, task: str) -> str:
        """Have an autonomous browser agent carry out a web task and return its result.

        Args:
            task: One self-contained web goal in natural language, e.g.
                "find the price of the Pro plan on example.com and return it".

        Returns:
            The browser agent's final text result, or JSON conforming to the
            configured output schema when one is set.
        """
        if self._session_scope == 'call':
            history = await self._run_in_fresh_session(task)
        else:
            await self._retry_pending_cleanup()
            history = await self._run_in_shared_session(task)
        return self._render_result(history)

    async def _retry_pending_cleanup(self) -> None:
        """Retry leased browsers whose previous teardown failed or timed out."""
        async with self._cleanup_lock:
            pending, self._pending_cleanup = self._pending_cleanup, []
            for resource in pending:
                if not await _cleanup(resource):
                    self._pending_cleanup.append(resource)

    async def _run_in_fresh_session(self, task: str) -> BrowserAgentHistory:
        """One disposable session for one call, killed when the call ends, on success or failure."""
        async with self._call_condition:
            await self._call_condition.wait_for(lambda: not self._call_cleanup_in_progress)
            await self._retry_pending_cleanup()
            self._active_call_sessions += 1
        session: BrowserSession | None = None
        lease: BrowserLease | None = None
        try:
            lease = await self._acquire_lease()
            session = self._build_session(self._lease_cdp_url(lease))
            return await self._run_agent(task, session)
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await self._teardown(session, lease)
                finally:
                    # The slot must be returned even if teardown itself raises,
                    # or `aclose()` waits for a session that no longer exists.
                    async with self._call_condition:
                        self._active_call_sessions -= 1
                        self._call_condition.notify_all()

    async def _run_in_shared_session(self, task: str) -> BrowserAgentHistory:
        """The `'agent'`-scoped shared session; the lock serializes calls -- one browser, one driver at a time."""
        async with self._session_lock:
            if self._session_closed:
                # A call that was queued behind `aclose()` reaches the lock after the browser
                # is gone. Without this it would lazily start a fresh `keep_alive` session that
                # nothing is left to close, so the process would exit with a live Chromium.
                raise RuntimeError(
                    'The shared browser session is closed: `aclose()` was called, so `browse_web` '
                    'cannot open another one. Build a new capability to browse again.'
                )
            if self._shared_session is None:
                lease = await self._acquire_lease()
                try:
                    self._shared_session = self._build_session(self._lease_cdp_url(lease))
                except BaseException:
                    # Building the session failed after the lease was acquired;
                    # release the leased browser rather than leak it.
                    await self._teardown(None, lease)
                    raise
                self._shared_lease = lease
            try:
                return await self._run_agent(task, self._shared_session)
            except BaseException:
                # A failed or cancelled run can leave the shared browser in an
                # unknown state; tear it down so the next call starts fresh.
                session, self._shared_session = self._shared_session, None
                lease, self._shared_lease = self._shared_lease, None
                await self._teardown(session, lease)
                raise

    async def aclose(self) -> None:
        """Kill the shared browser session and refuse to open another.

        In `'agent'` session scope it closes for good, so a later `browse_web`
        raises rather than starting a browser nothing would close. In either
        scope it retries sessions retained after an earlier teardown failure or
        timeout. Safe to call multiple times.

        It coordinates with `browse_web`, so it waits for in-flight calls to
        finish before the final cleanup attempt -- and a call can run for
        `max_steps` steps of up to `BrowserAgentSettings.step_timeout` each.
        Cancel the run first if you need to close sooner.
        """
        if self._session_scope == 'call':
            async with self._call_condition:
                await self._call_condition.wait_for(lambda: not self._call_cleanup_in_progress)
                self._call_cleanup_in_progress = True
                try:
                    await self._call_condition.wait_for(lambda: self._active_call_sessions == 0)
                    await self._retry_pending_cleanup()
                finally:
                    self._call_cleanup_in_progress = False
                    self._call_condition.notify_all()
            return

        async with self._session_lock:
            self._session_closed = True
            if self._shared_session is not None:
                session, self._shared_session = self._shared_session, None
                lease, self._shared_lease = self._shared_lease, None
                await self._teardown(session, lease)
        await self._retry_pending_cleanup()
