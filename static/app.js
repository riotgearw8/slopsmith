// Demo analytics — real impl set by demo.js; no-op in normal builds
window.slopsmithDemoTrack = window.slopsmithDemoTrack ?? null;

// ── Screen Navigation ─────────────────────────────────────────────────────
function showScreen(id) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    document.getElementById(id).classList.add('active');
    if (id === 'home') loadLibrary();
    if (id === 'favorites') loadFavorites();
    if (id === 'settings') loadSettings();
    if (id !== 'player') {
        highway.stop();
        const audio = document.getElementById('audio');
        audio.pause();
        audio.src = '';
        isPlaying = false;
        document.getElementById('btn-play').textContent = '▶ Play';
    }
    window.scrollTo(0, 0);
    if (window.slopsmith) window.slopsmith.emit('screen:changed', { id });
}

// ── Library ──────────────────────────────────────────────────────────────

// Persist the view toggle (grid vs tree), sort selection, and format
// filter across reloads. Stored as separate keys (rather than one
// blob) so future controls can opt in independently and a corrupted
// single value doesn't wipe the rest. Validation lives at the read
// site — we coerce unknown values back to safe defaults rather than
// trusting whatever happens to be in localStorage.
const _LIB_VIEW_KEY = 'slopsmith.libView';
const _LIB_SORT_KEY = 'slopsmith.libSort';
const _LIB_FORMAT_KEY = 'slopsmith.libFormat';
const _LIB_VIEW_VALUES = new Set(['grid', 'tree']);
const _LIB_SORT_VALUES = new Set([
    'artist', 'artist-desc', 'title', 'title-desc',
    'recent', 'year-desc', 'year', 'tuning',
]);
const _LIB_FORMAT_VALUES = new Set(['', 'psarc', 'sloppak']);
// Tree-view expand/collapse persistence. Three states per tree:
//   '1'  → user asked to expand all
//   '0'  → user asked to collapse all
//   null → no explicit choice; renderTreeInto's existing heuristic
//          (auto-open when search active or few artists) wins
//
// Library and Favorites are separate trees with separate
// Expand/Collapse buttons, so each gets its own key — toggling one
// must not flip the other's persisted state.
const _LIB_TREE_EXPAND_KEY = 'slopsmith.libTreeExpand';
const _FAV_TREE_EXPAND_KEY = 'slopsmith.favTreeExpand';
const _LIB_TREE_EXPAND_VALUES = new Set(['1', '0']);

function _readPersistedChoice(key, allowed, fallback) {
    try {
        const v = localStorage.getItem(key);
        return v !== null && allowed.has(v) ? v : fallback;
    } catch {
        return fallback;
    }
}
function _writePersistedChoice(key, value) {
    try { localStorage.setItem(key, value); } catch { /* private mode / quota */ }
}

let libView = _readPersistedChoice(_LIB_VIEW_KEY, _LIB_VIEW_VALUES, 'grid');
let currentPage = 0;
const PAGE_SIZE = 24;
let _treeLetter = '';
let _treeStats = null;
let _debounceTimer = null;
let _loadingMore = false;
let _hasMore = true;
let _gridObserver = null;
// Bumped on filter/sort/view changes so in-flight page fetches can detect
// they've been superseded and skip rendering stale results.
let _libEpoch = 0;

// ── Library filters (slopsmith#129/#69) ────────────────────────────────
//
// Filter state lives in a single object so the active set can be
// serialized to localStorage as one key. Each axis is OR-within (Lead
// + Rhythm = "has Lead OR Rhythm"); cross-axis is AND. Tri-state pills
// translate to `_has` / `_lacks` lists on the wire so the server's
// SQL doesn't have to encode the third "any" state.
const _ARRANGEMENTS = ['Lead', 'Rhythm', 'Bass', 'Combo'];
// Stem ids match the bare strings sloppak manifests use ("drums",
// "bass", etc.). `full` is intentionally omitted from the filter UI:
// it's the fallback mix every sloppak ships with, so filtering by it
// would match all sloppaks and confuse users.
const _STEM_DEFS = [
    { id: 'drums', label: 'Drums' },
    { id: 'bass', label: 'Bass' },
    { id: 'vocals', label: 'Vocals' },
    { id: 'guitar', label: 'Guitar' },
    { id: 'piano', label: 'Piano' },
    { id: 'other', label: 'Other' },
];
const _LIB_FILTERS_KEY = 'slopsmith.libFilters';
let _libFilters = _loadLibFilters();
let _tuningNames = null;  // cached from /api/library/tuning-names

function _defaultLibFilters() {
    return {
        arrHas: [], arrLacks: [],
        stemsHas: [], stemsLacks: [],
        lyrics: null,             // null | 1 | 0
        tunings: [],
    };
}

function _normalizeStringArray(v) {
    return Array.isArray(v) ? v.filter(x => typeof x === 'string' && x) : [];
}

function _normalizeLibFilters(parsed) {
    // Defensive: a stale or hand-edited localStorage payload could have
    // any shape. Without normalization a later `.join` or `.includes`
    // on a non-array would throw at filter-apply time. Coerce each
    // field back to its expected type, dropping anything we don't
    // recognize. Slopsmith#134 review.
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        return _defaultLibFilters();
    }
    const lyrics = parsed.lyrics;
    return {
        arrHas: _normalizeStringArray(parsed.arrHas),
        arrLacks: _normalizeStringArray(parsed.arrLacks),
        stemsHas: _normalizeStringArray(parsed.stemsHas),
        stemsLacks: _normalizeStringArray(parsed.stemsLacks),
        lyrics: lyrics === 0 || lyrics === 1 ? lyrics : null,
        tunings: _normalizeStringArray(parsed.tunings),
    };
}

function _loadLibFilters() {
    try {
        const raw = localStorage.getItem(_LIB_FILTERS_KEY);
        if (!raw) return _defaultLibFilters();
        return _normalizeLibFilters(JSON.parse(raw));
    } catch {
        return _defaultLibFilters();
    }
}

function _saveLibFilters() {
    try { localStorage.setItem(_LIB_FILTERS_KEY, JSON.stringify(_libFilters)); }
    catch { /* private mode / quota — ignore, in-memory state still works */ }
}

function _libActiveCount() {
    let n = 0;
    if (_libFilters.arrHas.length) n++;
    if (_libFilters.arrLacks.length) n++;
    if (_libFilters.stemsHas.length) n++;
    if (_libFilters.stemsLacks.length) n++;
    if (_libFilters.lyrics !== null) n++;
    if (_libFilters.tunings.length) n++;
    return n;
}

function _applyLibFiltersToParams(params) {
    if (_libFilters.arrHas.length) params.set('arrangements_has', _libFilters.arrHas.join(','));
    if (_libFilters.arrLacks.length) params.set('arrangements_lacks', _libFilters.arrLacks.join(','));
    if (_libFilters.stemsHas.length) params.set('stems_has', _libFilters.stemsHas.join(','));
    if (_libFilters.stemsLacks.length) params.set('stems_lacks', _libFilters.stemsLacks.join(','));
    if (_libFilters.lyrics !== null) params.set('has_lyrics', String(_libFilters.lyrics));
    if (_libFilters.tunings.length) params.set('tunings', _libFilters.tunings.join(','));
    return params;
}

function _pillState(item, hasList, lacksList) {
    if (hasList.includes(item)) return 'require';
    if (lacksList.includes(item)) return 'exclude';
    return 'any';
}

function _cyclePill(item, hasKey, lacksKey) {
    // Cycle: any -> require -> exclude -> any. Mutates _libFilters in place.
    const hasList = _libFilters[hasKey];
    const lacksList = _libFilters[lacksKey];
    const inHas = hasList.indexOf(item);
    const inLacks = lacksList.indexOf(item);
    if (inHas === -1 && inLacks === -1) {
        hasList.push(item);
    } else if (inHas !== -1) {
        hasList.splice(inHas, 1);
        lacksList.push(item);
    } else {
        lacksList.splice(inLacks, 1);
    }
    _saveLibFilters();
    _renderLibFilterDrawer();
    _renderLibFilterChips();
    _libEpoch++;
    currentPage = 0;
    _treeStats = null;  // letter bar counts depend on filters now
    loadLibrary(0);
}

function _renderPillRow(containerId, items, hasKey, lacksKey, labelFor) {
    const c = document.getElementById(containerId);
    if (!c) return;
    c.innerHTML = '';
    for (const it of items) {
        const id = typeof it === 'string' ? it : it.id;
        const label = labelFor ? labelFor(it) : id;
        const state = _pillState(id, _libFilters[hasKey], _libFilters[lacksKey]);
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = `filter-pill state-${state}`;
        btn.textContent = label;
        btn.onclick = () => _cyclePill(id, hasKey, lacksKey);
        c.appendChild(btn);
    }
}

function _renderLyricsPill() {
    // Single tri-state pill matching the arrangement / stem pattern.
    // Cycle: any (null) -> require (1) -> exclude (0) -> any.
    const c = document.getElementById('filter-lyrics');
    if (!c) return;
    c.innerHTML = '';
    const v = _libFilters.lyrics;
    const state = v === 1 ? 'require' : v === 0 ? 'exclude' : 'any';
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `filter-pill state-${state}`;
    btn.textContent = 'Lyrics';
    btn.onclick = () => {
        _libFilters.lyrics = v === null ? 1 : v === 1 ? 0 : null;
        _saveLibFilters();
        _renderLyricsPill();
        _renderLibFilterChips();
        _libEpoch++;
        currentPage = 0;
        _treeStats = null;
        loadLibrary(0);
    };
    c.appendChild(btn);
}

async function _renderTuningList() {
    const c = document.getElementById('filter-tunings');
    if (!c) return;
    let fetchError = null;
    if (!_tuningNames) {
        c.innerHTML = '<div class="text-xs text-gray-500 px-2">Loading...</div>';
        try {
            const resp = await fetch('/api/library/tuning-names');
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            _tuningNames = Array.isArray(data.tunings) ? data.tunings : [];
        } catch (e) {
            // Distinguish a server / network failure from "the DB
            // genuinely has no tunings indexed". The latter wants a
            // Full Rescan; the former just wants a retry. Don't cache
            // the failure — leave _tuningNames null so reopening the
            // drawer triggers a fresh attempt.
            _tuningNames = null;
            fetchError = e.message || 'request failed';
        }
    }
    c.innerHTML = '';
    if (fetchError) {
        c.innerHTML = `<div class="text-xs text-red-400 px-2">Failed to load tunings (${esc(fetchError)}). Reopen the drawer to retry.</div>`;
        return;
    }
    if (!_tuningNames.length) {
        c.innerHTML = '<div class="text-xs text-gray-500 px-2">No tunings indexed yet — try Full Rescan.</div>';
        return;
    }
    for (const t of _tuningNames) {
        const checked = _libFilters.tunings.includes(t.name);
        const row = document.createElement('label');
        row.className = 'tuning-row';
        row.innerHTML =
            `<input type="checkbox" ${checked ? 'checked' : ''} class="rounded border-gray-600 bg-dark-700 text-accent">` +
            `<span class="flex-1">${esc(t.name)}</span>` +
            `<span class="tuning-count">${t.count}</span>`;
        const cb = row.querySelector('input');
        cb.onchange = () => {
            const i = _libFilters.tunings.indexOf(t.name);
            if (cb.checked && i === -1) _libFilters.tunings.push(t.name);
            else if (!cb.checked && i !== -1) _libFilters.tunings.splice(i, 1);
            _saveLibFilters();
            _updateLibFiltersBadge();
            _renderLibFilterChips();
            _renderTuningSummary();
            _libEpoch++;
            currentPage = 0;
            _treeStats = null;
            loadLibrary(0);
        };
        c.appendChild(row);
    }
    _renderTuningSummary();
}

function _renderTuningSummary() {
    const s = document.getElementById('filter-tunings-summary');
    if (!s) return;
    if (!_libFilters.tunings.length) { s.textContent = 'All tunings'; return; }
    if (_libFilters.tunings.length === 1) { s.textContent = _libFilters.tunings[0]; return; }
    s.textContent = `${_libFilters.tunings[0]} +${_libFilters.tunings.length - 1}`;
}

function _updateLibFiltersBadge() {
    const badge = document.getElementById('lib-filters-count');
    if (!badge) return;
    const n = _libActiveCount();
    badge.textContent = String(n);
    badge.classList.toggle('hidden', n === 0);
}

function _renderLibFilterDrawer() {
    _renderPillRow('filter-arrangements', _ARRANGEMENTS, 'arrHas', 'arrLacks');
    _renderPillRow('filter-stems', _STEM_DEFS, 'stemsHas', 'stemsLacks', s => s.label);
    _renderLyricsPill();
    // Stems section dimmed when format=psarc (no stems exist).
    const stemsSection = document.getElementById('filter-stems-section');
    if (stemsSection) {
        const fmt = (document.getElementById('lib-format') || {}).value || '';
        stemsSection.classList.toggle('opacity-40', fmt === 'psarc');
        stemsSection.classList.toggle('pointer-events-none', fmt === 'psarc');
    }
    _updateLibFiltersBadge();
}

function _renderLibFilterChips() {
    const row = document.getElementById('lib-filter-chips');
    if (!row) return;
    const chips = [];
    for (const a of _libFilters.arrHas) chips.push({ label: a, kind: 'require', remove: () => _libFilters.arrHas = _libFilters.arrHas.filter(x => x !== a) });
    for (const a of _libFilters.arrLacks) chips.push({ label: `no ${a}`, kind: 'exclude', remove: () => _libFilters.arrLacks = _libFilters.arrLacks.filter(x => x !== a) });
    for (const s of _libFilters.stemsHas) {
        const def = _STEM_DEFS.find(d => d.id === s);
        chips.push({ label: def ? def.label : s, kind: 'require', remove: () => _libFilters.stemsHas = _libFilters.stemsHas.filter(x => x !== s) });
    }
    for (const s of _libFilters.stemsLacks) {
        const def = _STEM_DEFS.find(d => d.id === s);
        chips.push({ label: `no ${def ? def.label : s}`, kind: 'exclude', remove: () => _libFilters.stemsLacks = _libFilters.stemsLacks.filter(x => x !== s) });
    }
    if (_libFilters.lyrics === 1) chips.push({ label: 'has lyrics', kind: 'require', remove: () => _libFilters.lyrics = null });
    if (_libFilters.lyrics === 0) chips.push({ label: 'no lyrics', kind: 'exclude', remove: () => _libFilters.lyrics = null });
    for (const t of _libFilters.tunings) chips.push({ label: t, kind: 'require', remove: () => _libFilters.tunings = _libFilters.tunings.filter(x => x !== t) });

    row.innerHTML = '';
    if (!chips.length) {
        row.classList.add('hidden');
        return;
    }
    row.classList.remove('hidden');
    for (const c of chips) {
        const el = document.createElement('span');
        el.className = `chip ${c.kind === 'exclude' ? 'chip-exclude' : ''}`;
        // The "×" glyph isn't a reliable accessible name; assistive tech
        // also can't depend on `title` alone. Spell out the action plus
        // the chip's label in `aria-label` so screen-reader users hear
        // "Remove filter: Lead" instead of "button" or just "×".
        const ariaLabel = `Remove filter: ${c.label}`;
        el.innerHTML =
            `${esc(c.label)}<button type="button" title="${esc(ariaLabel)}" aria-label="${esc(ariaLabel)}">×</button>`;
        el.querySelector('button').onclick = () => {
            c.remove();
            _saveLibFilters();
            _renderLibFilterDrawer();
            _renderLibFilterChips();
            _libEpoch++;
            currentPage = 0;
            _treeStats = null;
            loadLibrary(0);
        };
        row.appendChild(el);
    }
}

function toggleLibFilters(force) {
    const drawer = document.getElementById('lib-filter-drawer');
    const overlay = document.getElementById('lib-filter-overlay');
    if (!drawer) return;
    const open = force === undefined ? !drawer.classList.contains('open') : !!force;
    drawer.classList.toggle('open', open);
    overlay.classList.toggle('hidden', !open);
    if (open) {
        _renderLibFilterDrawer();
        _renderTuningList();
    }
}

function clearLibFilters() {
    _libFilters = _defaultLibFilters();
    _saveLibFilters();
    _renderLibFilterDrawer();
    _renderTuningList();
    _renderLibFilterChips();
    _libEpoch++;
    currentPage = 0;
    _treeStats = null;
    loadLibrary(0);
}

function setLibView(view) {
    libView = view;
    if (_LIB_VIEW_VALUES.has(view)) _writePersistedChoice(_LIB_VIEW_KEY, view);
    document.getElementById('lib-grid').classList.toggle('hidden', view !== 'grid');
    document.getElementById('lib-tree').classList.toggle('hidden', view !== 'tree');
    document.querySelectorAll('.lib-grid-ctrl').forEach(el => el.classList.toggle('hidden', view !== 'grid'));
    document.querySelectorAll('.lib-tree-ctrl').forEach(el => el.classList.toggle('hidden', view !== 'tree'));
    document.getElementById('view-grid-btn').className = `px-3 py-2.5 text-sm transition ${view === 'grid' ? 'text-accent-light' : 'text-gray-600 hover:text-gray-400'}`;
    document.getElementById('view-tree-btn').className = `px-3 py-2.5 text-sm transition ${view === 'tree' ? 'text-accent-light' : 'text-gray-600 hover:text-gray-400'}`;
    if (view !== 'grid') stopInfiniteScroll();
    _libEpoch++;
    loadLibrary();
}

async function loadLibrary(page) {
    if (libView === 'grid') {
        await loadGridPage(page !== undefined ? page : currentPage);
    } else {
        await loadTreeView();
    }
}

function filterLibrary() {
    clearTimeout(_debounceTimer);
    _debounceTimer = setTimeout(() => {
        _libEpoch++;
        currentPage = 0;
        _treeLetter = '';
        // Letter-bar counts depend on `q` and the active filter set —
        // any change to those must invalidate the tree-view stats
        // cache or the next switch to tree view will render stale
        // letter counts (slopsmith#134 review).
        _treeStats = null;
        loadLibrary(0);
    }, 250);
}

function sortLibrary() {
    // Persist whichever of the two dropdowns just changed so the next
    // page load can restore both. Both selects route through this
    // handler today; reading both is cheap and keeps the function
    // single-purpose.
    const sortEl = document.getElementById('lib-sort');
    if (sortEl && _LIB_SORT_VALUES.has(sortEl.value)) {
        _writePersistedChoice(_LIB_SORT_KEY, sortEl.value);
    }
    const fmtEl = document.getElementById('lib-format');
    if (fmtEl && _LIB_FORMAT_VALUES.has(fmtEl.value)) {
        _writePersistedChoice(_LIB_FORMAT_KEY, fmtEl.value);
    }
    _libEpoch++;
    currentPage = 0;
    // Same reason as filterLibrary: format dropdown changes the stats
    // payload, so the cache must drop too.
    _treeStats = null;
    loadLibrary(0);
}

// ── Grid View (server-side pagination, infinite scroll) ────────────────

async function loadGridPage(page = 0) {
    const myEpoch = _libEpoch;
    const q = document.getElementById('lib-filter').value.trim();
    const sort = document.getElementById('lib-sort').value;
    const format = (document.getElementById('lib-format') || {}).value || '';
    const params = new URLSearchParams({ q, page, size: PAGE_SIZE, sort });
    if (format) params.set('format', format);
    _applyLibFiltersToParams(params);
    const resp = await fetch(`/api/library?${params}`);
    const data = await resp.json();
    if (myEpoch !== _libEpoch) return; // filter/sort/view changed mid-fetch

    currentPage = page;
    const total = data.total || 0;
    document.getElementById('lib-count').textContent = `${total} songs`;

    renderGridCards(data.songs || [], 'lib-grid', page === 0 ? 'replace' : 'append');

    _hasMore = (page + 1) * PAGE_SIZE < total;
    setupInfiniteScroll();
}

function setupInfiniteScroll() {
    let sentinel = document.getElementById('lib-grid-sentinel');
    if (!sentinel) {
        sentinel = document.createElement('div');
        sentinel.id = 'lib-grid-sentinel';
        sentinel.style.height = '1px';
        document.getElementById('lib-grid').after(sentinel);
    }
    stopInfiniteScroll();
    if (!_hasMore) return;
    _gridObserver = new IntersectionObserver(async (entries) => {
        if (entries[0].isIntersecting && !_loadingMore && _hasMore) {
            _loadingMore = true;
            try { await loadGridPage(currentPage + 1); }
            finally { _loadingMore = false; }
        }
    }, { rootMargin: '400px' });
    _gridObserver.observe(sentinel);
}

function stopInfiniteScroll() {
    if (_gridObserver) {
        _gridObserver.disconnect();
        _gridObserver = null;
    }
}

function formatBadge(fmt, stemCount) {
    if (fmt === 'sloppak' && (stemCount || 0) > 1) {
        return `<span class="fmt-badge absolute top-2 right-2 px-1.5 py-0.5 rounded text-[10px] font-bold bg-purple-900/80 text-purple-200 border border-purple-700">STEMS</span>`;
    }
    if (fmt === 'sloppak') {
        return `<span class="fmt-badge absolute top-2 right-2 px-1.5 py-0.5 rounded text-[10px] font-bold bg-green-900/80 text-green-200 border border-green-700">SLOPPAK</span>`;
    }
    return `<span class="fmt-badge absolute top-2 right-2 px-1.5 py-0.5 rounded text-[10px] font-bold bg-blue-900/80 text-blue-200 border border-blue-700">PSARC</span>`;
}

function formatBadgeInline(fmt, stemCount) {
    if (fmt === 'sloppak' && (stemCount || 0) > 1) {
        return `<span class="px-1.5 py-0.5 rounded text-[10px] font-bold bg-purple-900/60 text-purple-300">STEMS</span>`;
    }
    if (fmt === 'sloppak') {
        return `<span class="px-1.5 py-0.5 rounded text-[10px] font-bold bg-green-900/60 text-green-300">SLOPPAK</span>`;
    }
    return `<span class="px-1.5 py-0.5 rounded text-[10px] font-bold bg-blue-900/60 text-blue-300">PSARC</span>`;
}

function renderGridCards(songs, containerId = 'lib-grid', mode = 'replace') {
    const grid = document.getElementById(containerId);
    const html = songs.map(s => {
        const title = s.title || s.filename.replace(/_p\.psarc$/i, '').replace(/_/g, ' ');
        const artist = s.artist || '';
        const duration = s.duration ? formatTime(s.duration) : '';
        const tuning = s.tuning || '';
        const artUrl = `/api/song/${encodeURIComponent(s.filename)}/art${s.mtime ? `?v=${Math.floor(s.mtime)}` : ''}`;
        const isSloppak = s.format === 'sloppak';
        const stdRetune = !isSloppak && tuning && !s.has_estd &&
            ['Eb Standard', 'D Standard', 'C# Standard', 'C Standard'].includes(tuning);
        const dropRetune = !isSloppak && tuning && !s.has_estd &&
            ['Drop C', 'Drop C#', 'Drop Bb', 'Drop A'].includes(tuning);
        const retuneBtn = stdRetune
            ? `<button data-retune="${encodeURIComponent(s.filename)}" data-title="${encodeURIComponent(title)}" data-tuning="${tuning}" data-target="E Standard"
                class="retune-btn mt-2 w-full px-2 py-1.5 bg-gold/10 hover:bg-gold/20 border border-gold/20 rounded-lg text-xs font-medium text-gold transition">
                ⬆ Convert to E Standard</button>`
            : dropRetune
            ? `<button data-retune="${encodeURIComponent(s.filename)}" data-title="${encodeURIComponent(title)}" data-tuning="${tuning}" data-target="Drop D"
                class="retune-btn mt-2 w-full px-2 py-1.5 bg-gold/10 hover:bg-gold/20 border border-gold/20 rounded-lg text-xs font-medium text-gold transition">
                ⬆ Convert to Drop D</button>`
            : '';
        const fmtBadge = formatBadge(s.format, s.stem_count);
        return `<div class="song-card group" data-play="${encodeURIComponent(s.filename)}">
            <div class="card-art">
                <img src="${artUrl}" alt="" loading="lazy" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
                <span class="placeholder" style="display:none">🎸</span>
                ${fmtBadge}
            </div>
            <div class="p-4">
                <div class="flex items-start justify-between gap-1">
                    <div class="min-w-0">
                        <h3 class="text-sm font-semibold text-white truncate group-hover:text-accent-light transition">${esc(title)}</h3>
                        <p class="text-xs text-gray-500 truncate mt-0.5">${esc(artist)}</p>
                    </div>
                    <div class="flex gap-1">
                        ${editBtn(s)}
                        ${heartBtn(s.filename, s.favorite)}
                    </div>
                </div>
                <div class="flex items-center flex-wrap gap-1.5 mt-3 text-xs">
                    ${(s.arrangements || []).map(a =>
                        `<span class="px-1.5 py-0.5 rounded ${
                            a.name === 'Lead' ? 'bg-red-900/40 text-red-300' :
                            a.name === 'Rhythm' ? 'bg-blue-900/40 text-blue-300' :
                            a.name === 'Bass' ? 'bg-green-900/40 text-green-300' :
                            'bg-dark-600 text-gray-400'
                        }">${a.name}</span>`
                    ).join('')}
                    ${tuning ? `<span class="px-1.5 py-0.5 rounded ${tuning === 'E Standard' ? 'bg-green-900/30 text-green-400' : 'bg-yellow-900/30 text-yellow-400'}">${tuning}</span>` : ''}
                    ${s.has_lyrics ? `<span class="px-1.5 py-0.5 bg-purple-900/30 rounded text-purple-300">Lyrics</span>` : ''}
                    ${duration ? `<span class="text-gray-600">${duration}</span>` : ''}
                </div>
                ${retuneBtn}
            </div>
        </div>`;
    }).join('');
    if (mode === 'append') {
        grid.insertAdjacentHTML('beforeend', html);
    } else {
        grid.innerHTML = html;
    }
}

// ── Tree View (server-side) ─────────────────────────────────────────────

async function loadTreeView() {
    if (!_treeStats) {
        const q = document.getElementById('lib-filter').value.trim();
        const format = (document.getElementById('lib-format') || {}).value || '';
        const sp = new URLSearchParams();
        if (q) sp.set('q', q);
        if (format) sp.set('format', format);
        _applyLibFiltersToParams(sp);
        const qs = sp.toString();
        const resp = await fetch(`/api/library/stats${qs ? '?' + qs : ''}`);
        _treeStats = await resp.json();
    }
    const q = document.getElementById('lib-filter').value.trim();
    await renderTreeInto('lib-tree', 'lib-count', _treeStats, _treeLetter, q, false);
}

let _treePage = 0;
const TREE_PAGE_SIZE = 50;

async function renderTreeInto(containerId, countId, stats, letter, q, favoritesOnly, page) {
    if (page === undefined) page = favoritesOnly ? _favTreePage || 0 : _treePage;
    const container = document.getElementById(containerId);
    const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ#'.split('');
    const chevron = `<svg class="chevron w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg>`;

    const letterFn = favoritesOnly ? 'filterFavTreeLetter' : 'filterTreeLetter';
    const pageFn = favoritesOnly ? 'goFavTreePage' : 'goTreePage';
    let html = '<div class="flex flex-wrap gap-1 mb-6">';
    html += `<button onclick="${letterFn}('')" class="px-2 py-1 rounded text-xs transition ${
        !letter ? 'bg-accent text-white' : 'bg-dark-700 text-gray-400 hover:text-white'
    }">All</button>`;
    for (const l of letters) {
        const count = stats.letters[l] || 0;
        const active = letter === l;
        html += `<button onclick="${letterFn}('${l}')" class="px-2 py-1 rounded text-xs transition ${
            active ? 'bg-accent text-white' :
            count ? 'bg-dark-700 text-gray-300 hover:text-white' :
            'bg-dark-700/50 text-gray-700 cursor-default'
        }" ${count ? '' : 'disabled'}>${l}</button>`;
    }
    html += '</div>';

    // Fetch artists for the selected letter/all
    const params = new URLSearchParams();
    if (letter) params.set('letter', letter);
    if (q) params.set('q', q);
    if (favoritesOnly) params.set('favorites', '1');
    const format = (document.getElementById('lib-format') || {}).value || '';
    if (format) params.set('format', format);
    if (!favoritesOnly) _applyLibFiltersToParams(params);
    params.set('page', page);
    params.set('size', TREE_PAGE_SIZE);
    const resp = await fetch(`/api/library/artists?${params}`);
    const data = await resp.json();
    const artists = data.artists || [];
    const totalArtists = data.total_artists || 0;
    const totalPages = Math.ceil(totalArtists / TREE_PAGE_SIZE);

    let songCount = 0, artistCount = artists.length;
    for (const a of artists) songCount += a.song_count;
    const pageInfo = totalPages > 1 ? ` · Page ${page + 1} of ${totalPages}` : '';
    document.getElementById(countId).textContent =
        `${totalArtists} artists (${songCount} songs on this page)${pageInfo}`;

    // A previous Expand/Collapse-All click is persisted as '1'/'0' and
    // overrides the auto-open heuristic for both artists and albums.
    // Library and Favorites have independent buttons and independent
    // keys (slopsmith.libTreeExpand vs slopsmith.favTreeExpand) — fed
    // off the favoritesOnly flag — so toggling one doesn't flip the
    // other's state. Falsy / unset key → fall back to the existing
    // heuristic (open when there's an active search or few rows).
    const expandKey = favoritesOnly ? _FAV_TREE_EXPAND_KEY : _LIB_TREE_EXPAND_KEY;
    const savedExpand = _readPersistedChoice(expandKey, _LIB_TREE_EXPAND_VALUES, null);
    const forceArtistOpen = savedExpand === '1';
    const forceArtistClosed = savedExpand === '0';

    for (const artist of artists) {
        const heuristicOpen = q || artists.length <= 5;
        const isOpen = forceArtistOpen ? true : forceArtistClosed ? false : heuristicOpen;
        const openClass = isOpen ? ' open' : '';
        html += `<div class="artist-row${openClass}">`;
        html += `<div class="artist-header" onclick="this.parentElement.classList.toggle('open')">`;
        html += chevron;
        html += `<span class="text-white font-semibold text-sm flex-1">${esc(artist.name)}</span>`;
        html += `<span class="text-xs text-gray-600">${artist.song_count} song${artist.song_count !== 1 ? 's' : ''} · ${artist.album_count} album${artist.album_count !== 1 ? 's' : ''}</span>`;
        html += `</div><div class="artist-body">`;

        for (const album of artist.albums) {
            const artUrl = `/api/song/${encodeURIComponent(album.songs[0].filename)}/art${album.songs[0].mtime ? `?v=${Math.floor(album.songs[0].mtime)}` : ''}`;
            const albumHeuristicOpen = q || artist.albums.length === 1;
            const albumIsOpen = forceArtistOpen ? true : forceArtistClosed ? false : albumHeuristicOpen;
            const albumOpen = albumIsOpen ? ' open' : '';
            html += `<div class="album-group${albumOpen}">`;
            html += `<div class="album-header" onclick="this.parentElement.classList.toggle('open')">`;
            html += chevron;
            html += `<img src="${artUrl}" alt="" class="album-art-sm" loading="lazy" onerror="this.style.display='none'">`;
            html += `<span class="text-gray-300 text-sm flex-1">${esc(album.name)}</span>`;
            html += `<span class="text-xs text-gray-600">${album.songs.length}</span>`;
            html += `</div><div class="album-body">`;

            for (const s of album.songs) {
                const title = s.title || s.filename;
                const duration = s.duration ? formatTime(s.duration) : '';
                const tuning = s.tuning || '';
                const isSloppak = s.format === 'sloppak';
                const stdRetune = !isSloppak && tuning && !s.has_estd &&
                    ['Eb Standard', 'D Standard', 'C# Standard', 'C Standard'].includes(tuning);
                const dropRetune = !isSloppak && tuning && !s.has_estd &&
                    ['Drop C', 'Drop C#', 'Drop Bb', 'Drop A'].includes(tuning);
                const canRetune = stdRetune || dropRetune;
                const retuneTarget = stdRetune ? 'E Standard' : 'Drop D';
                html += `<div class="song-row" data-play="${encodeURIComponent(s.filename)}">`;
                html += `<div class="flex-1 min-w-0 flex items-center gap-2"><span class="text-sm text-white truncate block">${esc(title)}</span>${formatBadgeInline(s.format, s.stem_count)}</div>`;
                html += `<div class="flex items-center gap-1.5 flex-shrink-0 text-xs">`;
                for (const a of (s.arrangements || [])) {
                    const cls = a.name === 'Lead' ? 'bg-red-900/40 text-red-300' :
                                a.name === 'Rhythm' ? 'bg-blue-900/40 text-blue-300' :
                                a.name === 'Bass' ? 'bg-green-900/40 text-green-300' :
                                'bg-dark-600 text-gray-400';
                    html += `<span class="px-1.5 py-0.5 rounded ${cls}">${a.name}</span>`;
                }
                if (tuning)
                    html += `<span class="px-1.5 py-0.5 rounded ${tuning === 'E Standard' ? 'bg-green-900/30 text-green-400' : 'bg-yellow-900/30 text-yellow-400'}">${tuning}</span>`;
                if (s.has_lyrics)
                    html += `<span class="px-1.5 py-0.5 bg-purple-900/30 rounded text-purple-300">Lyrics</span>`;
                if (duration)
                    html += `<span class="text-gray-600 w-10 text-right">${duration}</span>`;
                if (canRetune)
                    html += `<button data-retune="${encodeURIComponent(s.filename)}" data-title="${encodeURIComponent(title)}" data-tuning="${tuning}" data-target="${retuneTarget}"
                        class="retune-btn px-1.5 py-0.5 bg-gold/10 hover:bg-gold/20 border border-gold/20 rounded text-gold" title="Convert to ${retuneTarget}">${dropRetune ? 'D' : 'E'}</button>`;
                html += editBtn(s);
                html += heartBtn(s.filename, s.favorite);
                html += `</div></div>`;
            }
            html += `</div></div>`;
        }
        html += `</div></div>`;
    }

    // Pagination
    if (totalPages > 1) {
        html += '<div class="flex items-center justify-center gap-2 py-6">';
        html += `<button onclick="${pageFn}(0)" class="px-3 py-1.5 rounded-lg text-xs ${page === 0 ? 'text-gray-600' : 'bg-dark-600 text-gray-300 hover:bg-dark-500'}" ${page === 0 ? 'disabled' : ''}>« First</button>`;
        html += `<button onclick="${pageFn}(${page - 1})" class="px-3 py-1.5 rounded-lg text-xs ${page === 0 ? 'text-gray-600' : 'bg-dark-600 text-gray-300 hover:bg-dark-500'}" ${page === 0 ? 'disabled' : ''}>‹ Prev</button>`;
        const start = Math.max(0, page - 2);
        const end = Math.min(totalPages, start + 5);
        for (let i = start; i < end; i++) {
            html += `<button onclick="${pageFn}(${i})" class="px-3 py-1.5 rounded-lg text-xs ${i === page ? 'bg-accent text-white' : 'bg-dark-600 text-gray-300 hover:bg-dark-500'}">${i + 1}</button>`;
        }
        html += `<button onclick="${pageFn}(${page + 1})" class="px-3 py-1.5 rounded-lg text-xs ${page >= totalPages - 1 ? 'text-gray-600' : 'bg-dark-600 text-gray-300 hover:bg-dark-500'}" ${page >= totalPages - 1 ? 'disabled' : ''}>Next ›</button>`;
        html += `<button onclick="${pageFn}(${totalPages - 1})" class="px-3 py-1.5 rounded-lg text-xs ${page >= totalPages - 1 ? 'text-gray-600' : 'bg-dark-600 text-gray-300 hover:bg-dark-500'}" ${page >= totalPages - 1 ? 'disabled' : ''}>Last »</button>`;
        html += '</div>';
    }

    container.innerHTML = html;
}

function goTreePage(p) {
    _treePage = Math.max(0, p);
    loadTreeView();
    document.getElementById('library-section').scrollIntoView({ behavior: 'smooth' });
}

function filterTreeLetter(letter) {
    _treeLetter = (_treeLetter === letter) ? '' : letter;
    _treePage = 0;
    loadTreeView();
}

function _toggleAllInTree(containerId, expand, persistKey) {
    // Scope the open/close to the named tree's container so toggling
    // Library doesn't flip the (offscreen) Favorites DOM and vice
    // versa — they share `.artist-row` / `.album-group` classes.
    const container = document.getElementById(containerId);
    if (!container) return;
    container.querySelectorAll('.artist-row').forEach(el => el.classList.toggle('open', expand));
    container.querySelectorAll('.album-group').forEach(el => el.classList.toggle('open', expand));
    // Persist the explicit choice so the next page reload (or letter
    // change, which re-runs renderTreeInto) honors it instead of
    // falling back to the auto-open heuristic. Stored as '1'/'0' so a
    // missing key reliably means "no explicit choice".
    _writePersistedChoice(persistKey, expand ? '1' : '0');
}

function toggleAllArtists(expand) {
    _toggleAllInTree('lib-tree', expand, _LIB_TREE_EXPAND_KEY);
}

function toggleAllFavoriteArtists(expand) {
    _toggleAllInTree('fav-tree', expand, _FAV_TREE_EXPAND_KEY);
}

function esc(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

// ── Favorites ────────────────────────────────────────────────────────────
let favView = 'grid';
let favPage = 0;
let _favTreeLetter = '';
let _favTreePage = 0;
let _favTreeStats = null;
let _favDebounce = null;

function heartBtn(filename, isFav) {
    return `<button data-fav="${encodeURIComponent(filename)}" class="fav-btn text-lg leading-none transition ${isFav ? 'text-red-500' : 'text-gray-600 hover:text-red-400'}" title="Toggle favorite">${isFav ? '&#9829;' : '&#9825;'}</button>`;
}

function editBtn(song) {
    return `<button data-edit='${JSON.stringify({f:song.filename,t:song.title||'',a:song.artist||'',al:song.album||'',y:song.year||''}).replace(/'/g,"&#39;")}' class="edit-btn text-gray-600 hover:text-accent-light transition" title="Edit metadata"><svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"/></svg></button>`;
}

async function toggleFavorite(filename) {
    const resp = await fetch('/api/favorites/toggle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename }),
    });
    const data = await resp.json();
    // Refresh whichever view is active
    const activeScreen = document.querySelector('.screen.active');
    if (activeScreen?.id === 'favorites') loadFavorites();
    else loadLibrary();
    return data.favorite;
}

function setFavView(view) {
    favView = view;
    document.getElementById('fav-grid').classList.toggle('hidden', view !== 'grid');
    document.getElementById('fav-tree').classList.toggle('hidden', view !== 'tree');
    document.querySelectorAll('.fav-grid-ctrl').forEach(el => el.classList.toggle('hidden', view !== 'grid'));
    document.querySelectorAll('.fav-tree-ctrl').forEach(el => el.classList.toggle('hidden', view !== 'tree'));
    document.getElementById('fav-view-grid-btn').className = `px-3 py-2.5 text-sm transition ${view === 'grid' ? 'text-accent-light' : 'text-gray-600 hover:text-gray-400'}`;
    document.getElementById('fav-view-tree-btn').className = `px-3 py-2.5 text-sm transition ${view === 'tree' ? 'text-accent-light' : 'text-gray-600 hover:text-gray-400'}`;
    const pag = document.getElementById('fav-pagination');
    if (pag && view !== 'grid') pag.innerHTML = '';
    loadFavorites();
}

async function loadFavorites() {
    if (favView === 'grid') await loadFavGridPage(favPage);
    else await loadFavTreeView();
}

function filterFavorites() {
    clearTimeout(_favDebounce);
    _favDebounce = setTimeout(() => { favPage = 0; _favTreeLetter = ''; loadFavorites(); }, 250);
}

function sortFavorites() { favPage = 0; loadFavorites(); }

async function loadFavGridPage(page = 0) {
    const q = document.getElementById('fav-filter').value.trim();
    const sort = document.getElementById('fav-sort').value;
    favPage = page;
    const params = new URLSearchParams({ q, page, size: PAGE_SIZE, sort, favorites: 1 });
    const resp = await fetch(`/api/library?${params}`);
    const data = await resp.json();
    const totalPages = Math.ceil((data.total || 0) / PAGE_SIZE);
    document.getElementById('fav-count').textContent =
        `${data.total || 0} favorites · Page ${favPage + 1} of ${Math.max(1, totalPages)}`;
    renderGridCards(data.songs || [], 'fav-grid');
    renderFavPagination(totalPages);
}

function renderFavPagination(totalPages) {
    let pag = document.getElementById('fav-pagination');
    if (!pag) {
        pag = document.createElement('div');
        pag.id = 'fav-pagination';
        pag.className = 'flex items-center justify-center gap-2 py-6';
        document.getElementById('fav-grid').after(pag);
    }
    if (totalPages <= 1) { pag.innerHTML = ''; return; }
    let html = '';
    html += `<button onclick="goFavPage(0)" class="px-3 py-1.5 rounded-lg text-xs ${favPage === 0 ? 'text-gray-600 cursor-default' : 'bg-dark-600 text-gray-300 hover:bg-dark-500'}" ${favPage === 0 ? 'disabled' : ''}>« First</button>`;
    html += `<button onclick="goFavPage(${favPage - 1})" class="px-3 py-1.5 rounded-lg text-xs ${favPage === 0 ? 'text-gray-600 cursor-default' : 'bg-dark-600 text-gray-300 hover:bg-dark-500'}" ${favPage === 0 ? 'disabled' : ''}>‹ Prev</button>`;
    const start = Math.max(0, favPage - 2);
    const end = Math.min(totalPages, start + 5);
    for (let i = start; i < end; i++) {
        html += `<button onclick="goFavPage(${i})" class="px-3 py-1.5 rounded-lg text-xs ${i === favPage ? 'bg-accent text-white' : 'bg-dark-600 text-gray-300 hover:bg-dark-500'}">${i + 1}</button>`;
    }
    html += `<button onclick="goFavPage(${favPage + 1})" class="px-3 py-1.5 rounded-lg text-xs ${favPage >= totalPages - 1 ? 'text-gray-600 cursor-default' : 'bg-dark-600 text-gray-300 hover:bg-dark-500'}" ${favPage >= totalPages - 1 ? 'disabled' : ''}>Next ›</button>`;
    html += `<button onclick="goFavPage(${totalPages - 1})" class="px-3 py-1.5 rounded-lg text-xs ${favPage >= totalPages - 1 ? 'text-gray-600 cursor-default' : 'bg-dark-600 text-gray-300 hover:bg-dark-500'}" ${favPage >= totalPages - 1 ? 'disabled' : ''}>Last »</button>`;
    pag.innerHTML = html;
}

function goFavPage(p) { loadFavGridPage(Math.max(0, p)); }

async function loadFavTreeView() {
    if (!_favTreeStats) {
        const resp = await fetch('/api/library/stats?favorites=1');
        _favTreeStats = await resp.json();
    }
    const q = document.getElementById('fav-filter').value.trim();
    const letter = _favTreeLetter;
    // Reuse the tree renderer with fav-tree container and fav-count
    await renderTreeInto('fav-tree', 'fav-count', _favTreeStats, letter, q, true);
}

function filterFavTreeLetter(letter) {
    _favTreeLetter = (_favTreeLetter === letter) ? '' : letter;
    _favTreePage = 0;
    loadFavTreeView();
}

function goFavTreePage(p) {
    _favTreePage = Math.max(0, p);
    loadFavTreeView();
}

// ── Settings ─────────────────────────────────────────────────────────────
async function loadSettings() {
    const resp = await fetch('/api/settings');
    const data = await resp.json();
    document.getElementById('dlc-path').value = data.dlc_dir || '';
    document.getElementById('default-arrangement').value = data.default_arrangement || '';
    document.getElementById('demucs-server-url').value = data.demucs_server_url || '';
    const leftyEl = document.getElementById('setting-lefty');
    if (leftyEl) leftyEl.checked = highway.getLefty();
    // Restore master-difficulty slider from persisted value (defaults
    // to 100 when the key is absent — no behaviour change for users
    // who've never touched the slider).
    const masteryPct = typeof data.master_difficulty === 'number'
        ? Math.max(0, Math.min(100, data.master_difficulty))
        : 100;
    const masterySlider = document.getElementById('mastery-slider');
    const masteryLabel = document.getElementById('mastery-label');
    if (masterySlider) masterySlider.value = masteryPct;
    if (masteryLabel) masteryLabel.textContent = masteryPct + '%';
    highway.setMastery(masteryPct / 100);
    // Route the loaded value through setAvOffsetMs so the highway's
    // render clock, the Settings slider, the HUD readout, and the
    // module variable all pick it up consistently. Pass skipPersist
    // so we don't echo the loaded value back to the server.
    setAvOffsetMs(Number(data.av_offset_ms) || 0, /* skipPersist */ true);
    // Native folder picker — only present when running inside slopsmith-desktop.
    if (window.slopsmithDesktop && typeof window.slopsmithDesktop.pickDirectory === 'function') {
        document.getElementById('btn-pick-dlc')?.classList.remove('hidden');
    }
}

// A/V sync calibration. Positive = audio runs ahead of visuals; we
// add this to audio.currentTime when driving the highway so the
// visuals catch up. Persisted via /api/settings as av_offset_ms.
// Live-tunable from the player screen via [ / ] keys (Shift for
// ±50 ms) and from the Settings slider; both auto-save with the
// same debounced POST. loadSettings() seeds the value via
// setAvOffsetMs without saving (skipPersist=true) to avoid an
// echo-back round-trip.
let _avOffsetMs = 0;
let _avSaveDebounce = null;
function setAvOffsetMs(ms, skipPersist) {
    _avOffsetMs = Number(ms) || 0;
    // Drive the highway's render-time shift. getTime() still returns
    // the audio-aligned chart time so plugins (note detection, etc.)
    // keep scoring against the real chart clock regardless of visual
    // calibration.
    if (typeof highway !== 'undefined' && highway?.setAvOffset) highway.setAvOffset(_avOffsetMs);
    // Sync any visible Settings slider
    const avSlider = document.getElementById('setting-av-offset');
    if (avSlider) avSlider.value = _avOffsetMs;
    const avVal = document.getElementById('setting-av-offset-val');
    if (avVal) avVal.textContent = Math.round(_avOffsetMs);
    // Update the player HUD readout (hidden when offset = 0 to
    // avoid clutter; the keyboard shortcut is documented in the
    // Settings help text so it stays discoverable).
    const hud = document.getElementById('hud-avoffset');
    if (hud) {
        hud.textContent = `A/V ${_avOffsetMs >= 0 ? '+' : ''}${Math.round(_avOffsetMs)} ms`;
        hud.classList.toggle('hidden', _avOffsetMs === 0);
    }
    if (!skipPersist) _persistAvOffset();
}
function _persistAvOffset() {
    // Debounced persist — POST only the one field; the server merges.
    if (_avSaveDebounce) clearTimeout(_avSaveDebounce);
    _avSaveDebounce = setTimeout(async () => {
        _avSaveDebounce = null;
        try {
            await fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ av_offset_ms: _avOffsetMs }),
            });
        } catch (e) {
            console.warn('A/V offset save failed:', e);
        }
    }, 400);
}
function nudgeAvOffsetMs(delta) {
    setAvOffsetMs(Math.max(-1000, Math.min(1000, _avOffsetMs + delta)));
}

// Open a native OS folder picker via the Electron bridge (desktop only) and
// stash the chosen path into the DLC input. User still has to hit Save.
async function pickDlcFolder() {
    if (!window.slopsmithDesktop?.pickDirectory) return;
    const path = await window.slopsmithDesktop.pickDirectory();
    if (path) document.getElementById('dlc-path').value = path;
}

async function saveSettings() {
    const resp = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            dlc_dir: document.getElementById('dlc-path').value.trim(),
            default_arrangement: document.getElementById('default-arrangement').value,
            demucs_server_url: document.getElementById('demucs-server-url').value.trim(),
            av_offset_ms: _avOffsetMs,
        }),
    });
    const data = await resp.json();
    document.getElementById('settings-status').textContent = data.message || data.error;
}

// ── Settings export / import (slopsmith#113) ─────────────────────────────────
//
// Bundles server config + every localStorage key + opted-in plugin server
// files into a single JSON file.
//
// Apply semantics — phased, NOT all-or-nothing across the two stores:
//   1. Server first (/api/settings/import). Phase-1 validation guards
//      the whole bundle; phase-2 disk commit is per-file but ordered
//      so a mid-apply failure surfaces a `partial` field. A server
//      failure short-circuits before any localStorage write, so the
//      browser side stays untouched on validation refusals.
//   2. localStorage second, only after the server returns ok. Applied
//      as a MERGE (no clear): bundled keys overwrite, locally-present
//      keys absent from the bundle are preserved (so a plugin
//      installed after the export keeps its first-run defaults).
//      A localStorage exception here (quota / private mode) is
//      surfaced verbatim — server state is already committed and we
//      don't pretend the import was clean.
//
// In short: the server side is atomic in phase 1 and surface-partial in
// phase 2; the localStorage side is best-effort merge after server
// success. Failures are reported, never silenced.

async function exportSettings() {
    const status = document.getElementById('backup-status');
    status.textContent = 'Exporting...';
    try {
        const resp = await fetch('/api/settings/export');
        if (!resp.ok) {
            status.textContent = `Export failed (HTTP ${resp.status})`;
            return;
        }
        const bundle = await resp.json();
        // Layer in the browser's localStorage. Use the standard Storage
        // iteration API (length + key(i)) rather than Object.keys —
        // Object.keys on a Storage instance is not deterministic across
        // browsers and can both miss entries and include non-entry
        // properties depending on the implementation. Keys are preserved
        // verbatim as strings; that's how localStorage stores them, and
        // round-trip fidelity matters more than re-typing values that
        // were never typed in the first place.
        const localStorageData = {};
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key === null) continue;
            const value = localStorage.getItem(key);
            if (value !== null) localStorageData[key] = value;
        }
        bundle.local_storage = localStorageData;

        // Trigger download via blob + temporary <a download>. We honor the
        // server's Content-Disposition filename when present, otherwise
        // fall back to a date-stamped default.
        let filename = 'slopsmith-settings.json';
        const disposition = resp.headers.get('Content-Disposition');
        if (disposition) {
            const match = /filename="([^"]+)"/.exec(disposition);
            if (match) filename = match[1];
        }
        const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        status.textContent = `Exported ${filename}`;
    } catch (e) {
        status.textContent = `Export failed: ${e.message}`;
    }
}

async function importSettings(file) {
    if (!file) return;
    const status = document.getElementById('backup-status');
    if (!confirm('Import will overwrite settings present in the bundle (server config, browser preferences, and opted-in plugin data) and reload the page. Settings not in the bundle (e.g. from plugins installed after the export) are preserved. Continue?')) {
        status.textContent = 'Import cancelled';
        return;
    }
    let bundle;
    try {
        bundle = JSON.parse(await file.text());
    } catch (e) {
        status.textContent = `Import failed: not valid JSON (${e.message})`;
        return;
    }

    status.textContent = 'Importing...';
    let resp, data;
    try {
        resp = await fetch('/api/settings/import', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(bundle),
        });
        data = await resp.json();
    } catch (e) {
        status.textContent = `Import failed: ${e.message}`;
        return;
    }
    // Two failure shapes to surface: our own validation handler
    // returns `{ok: false, error: "..."}`, but if the body fails
    // FastAPI's request-level validation (e.g. top-level value is
    // an array, not an object), the response is the framework's
    // `{detail: ...}` shape with no `ok` key. `resp.ok` distinguishes
    // both from success without depending on which path produced
    // the failure.
    if (!resp.ok || data.ok === false) {
        let msg = data.error;
        if (!msg && data.detail) {
            msg = typeof data.detail === 'string'
                ? data.detail
                : JSON.stringify(data.detail);
        }
        status.textContent = `Import failed: ${msg || `HTTP ${resp.status}`}`;
        return;
    }

    // Server applied successfully. Now apply the localStorage portion as
    // a MERGE (not clear+restore): keys in the bundle overwrite, keys
    // present locally but absent from the bundle are preserved. This
    // matters when a plugin was installed *after* the export — wiping
    // its localStorage would erase first-run defaults the plugin set on
    // load, leaving it in a worse state than before the import. The
    // tradeoff is that orphan keys from removed plugins or renamed key
    // schemes also linger; cleaning those up is the user's job.
    const ls = bundle.local_storage;
    if (ls && typeof ls === 'object') {
        try {
            for (const [key, value] of Object.entries(ls)) {
                if (typeof value === 'string') localStorage.setItem(key, value);
            }
        } catch (e) {
            // Quota exceeded / private mode etc. Server side already
            // committed, so we surface the partial state rather than
            // pretending it succeeded.
            status.textContent = `Server applied, but localStorage write failed: ${e.message}`;
            return;
        }
    }

    const warnings = (data.warnings || []).join('; ');
    status.textContent = warnings ? `Imported with warnings: ${warnings}. Reloading...` : 'Imported. Reloading...';
    setTimeout(() => location.reload(), 800);
}

async function rescanLibrary() {
    const btn = document.getElementById('btn-rescan');
    const status = document.getElementById('rescan-status');
    btn.disabled = true;
    btn.textContent = 'Scanning...';
    status.textContent = '';
    const resp = await fetch('/api/rescan', { method: 'POST' });
    const data = await resp.json();
    status.textContent = data.message;
    // Poll until done
    const poll = setInterval(async () => {
        const sr = await fetch('/api/scan-status');
        const sd = await sr.json();
        if (sd.running) {
            const cur = sd.current ? ` · ${sd.current}` : '';
            status.textContent = `${sd.done} / ${sd.total} scanned${cur}...`;
        } else {
            clearInterval(poll);
            btn.disabled = false;
            btn.textContent = 'Rescan Library';
            status.textContent = sd.error ? `Error: ${sd.error}` : 'Done!';
            _treeStats = null;
            _tuningNames = null;  // re-fetch on next drawer open
            loadLibrary();
        }
    }, 1000);
}

async function fullRescanLibrary() {
    if (!confirm('This will clear the entire library cache and re-scan all songs. This can take a long time with large libraries. Continue?')) return;
    const btn = document.getElementById('btn-full-rescan');
    const status = document.getElementById('rescan-status');
    btn.disabled = true;
    btn.textContent = 'Clearing...';
    const resp = await fetch('/api/rescan/full', { method: 'POST' });
    const data = await resp.json();
    btn.textContent = 'Scanning...';
    status.textContent = data.message;
    const poll = setInterval(async () => {
        const sr = await fetch('/api/scan-status');
        const sd = await sr.json();
        if (sd.running) {
            const cur = sd.current ? ` · ${sd.current}` : '';
            status.textContent = `${sd.done} / ${sd.total} scanned${cur}...`;
        } else {
            clearInterval(poll);
            btn.disabled = false;
            btn.textContent = 'Full Rescan';
            status.textContent = sd.error ? `Error: ${sd.error}` : 'Done!';
            _treeStats = null;
            _tuningNames = null;  // re-fetch on next drawer open
            loadLibrary();
        }
    }, 1000);
}

// ── Plugin Updates ───────────────────────────────────────────────────────
async function checkPluginUpdates() {
    const btn = document.getElementById('btn-check-updates');
    const status = document.getElementById('updates-status');
    const list = document.getElementById('plugin-updates-list');
    btn.disabled = true;
    btn.textContent = 'Checking...';
    status.textContent = '';
    list.innerHTML = '';
    try {
        const resp = await fetch('/api/plugins/updates');
        const data = await resp.json();
        const updates = data.updates || {};
        const keys = Object.keys(updates);
        if (keys.length === 0) {
            status.textContent = 'All plugins are up to date.';
        } else {
            status.textContent = `${keys.length} update${keys.length > 1 ? 's' : ''} available`;
            for (const id of keys) {
                const u = updates[id];
                const row = document.createElement('div');
                row.className = 'flex items-center gap-3 bg-dark-700 rounded-lg px-4 py-2';
                row.innerHTML = `
                    <span class="text-sm text-gray-300 flex-1">${u.name} <span class="text-xs text-gray-500">(${u.behind} commit${u.behind > 1 ? 's' : ''} behind — ${u.local} → ${u.remote})</span></span>
                    <button onclick="updatePlugin('${id}', this)" class="bg-accent/20 hover:bg-accent/30 text-accent-light px-3 py-1 rounded-lg text-xs transition">Update</button>`;
                list.appendChild(row);
            }
        }
    } catch (e) {
        status.textContent = 'Failed to check for updates.';
    }
    btn.disabled = false;
    btn.textContent = 'Check for Updates';
}

async function updatePlugin(pluginId, btn) {
    btn.disabled = true;
    btn.textContent = 'Updating...';
    try {
        const resp = await fetch(`/api/plugins/${pluginId}/update`, { method: 'POST' });
        const data = await resp.json();
        if (data.ok) {
            btn.textContent = 'Updated — restart to apply';
            btn.className = 'bg-green-900/30 text-green-400 px-3 py-1 rounded-lg text-xs';
        } else {
            btn.textContent = 'Failed';
            btn.title = data.error || '';
        }
    } catch (e) {
        btn.textContent = 'Error';
    }
}

// ── Plugin functions loaded dynamically from plugin screen.js files ──────
// (searchCF, installCF, loginCF, searchUG, buildFromUG, etc.)

// ── Retune ───────────────────────────────────────────────────────────────
function retuneSong(filename, title, tuning, target) {
    target = target || 'E Standard';
    if (!confirm(`Convert "${title}" from ${tuning} to ${target}?`)) return;

    // Show modal overlay
    const modal = document.createElement('div');
    modal.id = 'retune-modal';
    modal.className = 'fixed inset-0 z-[200] flex items-center justify-center bg-black/70 backdrop-blur-sm';
    modal.innerHTML = `
        <div class="bg-dark-700 border border-gray-700 rounded-2xl p-8 w-full max-w-md mx-4 shadow-2xl">
            <h3 class="text-lg font-bold text-white mb-1">Converting to ${target}</h3>
            <p class="text-sm text-gray-400 mb-5">${title}</p>
            <div class="progress-bar mb-3"><div class="fill" id="retune-bar" style="width:0%"></div></div>
            <p class="text-xs text-gray-500" id="retune-stage">Connecting...</p>
        </div>`;
    document.body.appendChild(modal);

    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/retune?filename=${encodeURIComponent(decodeURIComponent(filename))}&target=${encodeURIComponent(target)}`);
    ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.progress !== undefined) {
            document.getElementById('retune-bar').style.width = msg.progress + '%';
        }
        if (msg.stage) {
            document.getElementById('retune-stage').textContent = msg.stage;
        }
        if (msg.done) {
            modal.querySelector('.bg-dark-700').innerHTML = `
                <div class="text-center">
                    <div class="text-3xl mb-3">✓</div>
                    <h3 class="text-lg font-bold text-white mb-1">Done!</h3>
                    <p class="text-sm text-gray-400 mb-5">${msg.filename}</p>
                    <button onclick="document.getElementById('retune-modal').remove();loadLibrary()"
                        class="bg-accent hover:bg-accent-light px-6 py-2 rounded-xl text-sm font-semibold text-white transition">OK</button>
                </div>`;
        }
        if (msg.error) {
            modal.querySelector('.bg-dark-700').innerHTML = `
                <div class="text-center">
                    <div class="text-3xl mb-3">✕</div>
                    <h3 class="text-lg font-bold text-red-400 mb-1">Failed</h3>
                    <p class="text-sm text-gray-400 mb-5">${msg.error}</p>
                    <button onclick="document.getElementById('retune-modal').remove()"
                        class="bg-dark-600 hover:bg-dark-500 px-6 py-2 rounded-xl text-sm text-gray-300 transition">Close</button>
                </div>`;
        }
    };
    ws.onerror = () => {
        modal.querySelector('.bg-dark-700').innerHTML = `
            <div class="text-center">
                <p class="text-red-400 mb-4">Connection lost</p>
                <button onclick="document.getElementById('retune-modal').remove()"
                    class="bg-dark-600 px-6 py-2 rounded-xl text-sm text-gray-300">Close</button>
            </div>`;
    };
}

// ── Player ───────────────────────────────────────────────────────────────
const audio = document.getElementById('audio');
let isPlaying = false;
let currentFilename = '';
// Plugin context API — lightweight event bus for plugin integration
window.slopsmith = Object.assign(new EventTarget(), {
    currentSong: null,
    isPlaying: false,
    _navParams: {},
    navigate(screenId, params) {
        this._navParams = params || {};
        showScreen(screenId);
    },
    getNavParams() {
        const p = this._navParams;
        this._navParams = {};
        return p;
    },
    emit(event, detail) {
        this.dispatchEvent(new CustomEvent(event, { detail }));
    },
    on(event, fn) { this.addEventListener(event, fn); },
    off(event, fn) { this.removeEventListener(event, fn); }
});

// Initialise volume from persisted preference (matches lefty / invertHighway /
// renderScale / showLyrics convention). The mixer popover (audio-mixer.js)
// owns the UI surface; this just hydrates audio.volume on boot.
function _readSongVolume() {
    try {
        const stored = parseFloat(localStorage.getItem('volume'));
        return Number.isFinite(stored) ? Math.min(100, Math.max(0, stored)) : 80;
    } catch (e) {
        return 80;
    }
}
audio.volume = _readSongVolume() / 100;

// Re-sync audio.volume from the persisted setting whenever a new source
// finishes loading metadata. Belt + suspenders — some combinations of plugin
// audio-graph routing and media-element swaps reset audio.volume to 1.0
// (slopsmith#54). Delegates to audio-mixer's readSongVolume when loaded so
// the in-memory fallback (for storage-blocked contexts) is authoritative.
audio.addEventListener('loadedmetadata', () => {
    audio.volume = (window.slopsmith?.audio?.readSongVolume?.() ?? _readSongVolume()) / 100;
});

// Debug audio issues
audio.addEventListener('pause', () => { if (isPlaying) console.log('Audio paused unexpectedly at', audio.currentTime.toFixed(1)); });
audio.addEventListener('error', (e) => {
    // Ignore errors from empty src (happens during song switch cleanup)
    if (!audio.src || audio.src === window.location.href) return;
    console.error('Audio error:', audio.error?.code, audio.error?.message);
});
audio.addEventListener('stalled', () => console.log('Audio stalled at', audio.currentTime.toFixed(1)));
audio.addEventListener('waiting', () => console.log('Audio waiting/buffering at', audio.currentTime.toFixed(1)));
audio.addEventListener('ended', () => {
    console.log('Audio ended'); isPlaying = false;
    document.getElementById('btn-play').textContent = '▶ Play';
    window.slopsmith.isPlaying = false;
    window.slopsmith.emit('song:ended', { time: audio.currentTime });
});
audio.addEventListener('play', () => {
    window.slopsmith.isPlaying = true;
    window.slopsmith.emit('song:play', { time: audio.currentTime });
});
audio.addEventListener('pause', () => {
    if (!isPlaying) return;
    window.slopsmith.isPlaying = false;
    window.slopsmith.emit('song:pause', { time: audio.currentTime });
});

// Abort controller for cancelling pending requests when entering player
let artAbortController = null;

async function playSong(filename, arrangement) {
    console.log('playSong called:', filename);

    // Cancel any pending art/metadata requests
    if (artAbortController) artAbortController.abort();
    artAbortController = null;

    highway.stop();
    audio.pause();
    audio.src = '';
    isPlaying = false;
    document.getElementById('btn-play').textContent = '▶ Play';
    document.getElementById('speed-slider').value = 100;
    document.getElementById('speed-label').textContent = '1.0x';
    clearLoop();

    currentFilename = filename;
    showScreen('player');

    // Wait for previous WebSocket to fully close before opening new one
    await new Promise(r => setTimeout(r, 500));
    highway.init(document.getElementById('highway'));

    const arrParam = arrangement !== undefined ? `?arrangement=${arrangement}` : '';
    const wsUrl = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/highway/${decodeURIComponent(filename)}${arrParam}`;
    highway.connect(wsUrl);
    loadSavedLoops();
    document.getElementById('quality-select').value = highway.getRenderScale();
}

function changeArrangement(index) {
    if (currentFilename) {
        const wasPlaying = isPlaying;
        const time = audio.currentTime;
        if (isPlaying) { audio.pause(); isPlaying = false; }

        // Show loading overlay
        let overlay = document.getElementById('arr-loading');
        if (overlay) overlay.remove();
        overlay = document.createElement('div');
        overlay.id = 'arr-loading';
        overlay.className = 'fixed inset-0 z-[200] flex items-center justify-center bg-black/60 backdrop-blur-sm';
        overlay.innerHTML = `
            <div class="bg-dark-700 border border-gray-700 rounded-2xl p-6 w-72 text-center shadow-2xl">
                <div class="text-sm text-gray-300 mb-3">Loading arrangement...</div>
                <div class="progress-bar"><div class="fill" style="width:30%;animation:pulse 1s infinite"></div></div>
            </div>`;
        document.body.appendChild(overlay);

        // Set callback for when data is ready
        highway._onReady = () => {
            const ol = document.getElementById('arr-loading');
            if (ol) ol.remove();
            audio.currentTime = time;
            if (wasPlaying) {
                audio.play().then(() => { isPlaying = true; }).catch(() => {});
            }
            highway._onReady = null;
        };

        highway.reconnect(currentFilename, index);
        window.slopsmith.emit('arrangement:changed', { index, filename: currentFilename });
    }
}

function togglePlay() {
    if (isPlaying) {
        audio.pause(); isPlaying = false;
        document.getElementById('btn-play').textContent = '▶ Play';
    } else {
        audio.play(); isPlaying = true;
        document.getElementById('btn-play').textContent = '⏸ Pause';
    }
}

function seekBy(s) { audio.currentTime = Math.max(0, audio.currentTime + s); }
function setSpeed(v) {
    audio.playbackRate = parseFloat(v);
    document.getElementById('speed-label').textContent = parseFloat(v).toFixed(2) + 'x';
}
// Master-difficulty slider (slopsmith#48). Persists partial via
// /api/settings — the POST handler merges only the keys present, so
// this fire-and-forget call doesn't clobber dlc_dir or other settings.
//
// Debounced trailing-edge (300ms) so dragging the slider — which fires
// oninput per pixel — doesn't flood the server with concurrent writes
// to config.json. highway.setMastery() still fires every oninput so
// the chart re-filters in real time; only disk persistence waits.
let _masteryPersistTimer = null;
function _persistMastery(pct) {
    if (_masteryPersistTimer) clearTimeout(_masteryPersistTimer);
    _masteryPersistTimer = setTimeout(() => {
        _masteryPersistTimer = null;
        fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ master_difficulty: pct }),
        }).catch(() => { /* best-effort — next setMastery() will retry */ });
    }, 300);
}
function setMastery(v) {
    // Guard + clamp: v might be a slider string, a programmatic call
    // from a plugin, or a restored settings value with a bad shape.
    // Don't let NaN hit the label (would show "NaN%") or the POST.
    const parsed = parseInt(v, 10);
    if (!Number.isFinite(parsed)) return;
    const pct = Math.max(0, Math.min(100, parsed));
    document.getElementById('mastery-label').textContent = pct + '%';
    highway.setMastery(pct / 100);
    _persistMastery(pct);
}
// Reflect phrase-data availability on the slider after every `ready`.
// The server omits the `phrases` message entirely for single-level
// sources (GP imports, legacy sloppak), so hasPhraseData() is the
// right signal to enable/disable the slider.
function _applyMasteryAvailability(hasPhraseData) {
    const slider = document.getElementById('mastery-slider');
    if (!slider) return;
    if (hasPhraseData) {
        slider.disabled = false;
        slider.title = 'Master difficulty — low = simpler chart, high = full';
    } else {
        slider.disabled = true;
        slider.title = 'Source chart has a single difficulty level — slider disabled';
    }
}
if (window.slopsmith) {
    // slopsmith's event bus dispatches CustomEvent with the payload in
    // event.detail (see EventTarget setup around line 699), so the
    // handler receives an Event, not the raw payload.
    window.slopsmith.on('song:ready', (e) => {
        _applyMasteryAvailability(!!e.detail?.hasPhraseData);
        // Auto mode: re-evaluate the active renderer against the
        // newly-loaded song. The picker's current <option> value is the
        // source of truth here — localStorage is a persistence mirror
        // that can throw in private / sandboxed contexts, and the
        // picker already reflects fresh-install / post-cleanup
        // fallthroughs to 'auto' even when writes failed.
        const sel = document.getElementById('viz-picker');
        if (sel && sel.value === 'auto') _autoMatchViz();
    });
    // Highway signals when it's auto-reverted to the default renderer
    // after a broken plugin (init failure or repeated draw failures).
    // Sync the picker + persisted selection so the UI stops advertising
    // the broken choice and the user doesn't hit the same failure on
    // next reload.
    window.slopsmith.on('viz:reverted', (e) => {
        const sel = document.getElementById('viz-picker');
        if (sel) sel.value = 'default';
        try { localStorage.setItem('vizSelection', 'default'); } catch (_) {}
        console.warn(
            `viz picker: reverted to default renderer (${e.detail?.reason || 'unknown'}).`
        );
    });
}

// ── Visualization picker (slopsmith#36) ─────────────────────────────────
//
// Discovers viz plugins via /api/plugins and adds them to the #viz-picker
// dropdown. A viz plugin declares itself by setting `"type": "visualization"`
// in its plugin.json AND exposing a factory function on
// window.slopsmithViz_<id> that returns an object matching the setRenderer
// contract ({init, draw, resize, destroy}).
//
// The "default" option in the dropdown is the built-in 2D highway that
// lives inside createHighway(); selecting it calls setRenderer(null) which
// restores the default renderer.
async function _populateVizPicker(plugins) {
    const sel = document.getElementById('viz-picker');
    if (!sel) return;
    // Clear any previously-appended plugin options so calling this
    // function more than once (e.g. from DevTools, or a hot-reloaded
    // plugin) doesn't produce duplicates. The built-in "auto" and
    // "default" options are static markup — preserve them.
    const BUILTIN_OPT_VALUES = new Set(['auto', 'default']);
    Array.from(sel.options).forEach(opt => {
        if (!BUILTIN_OPT_VALUES.has(opt.value)) sel.removeChild(opt);
    });
    // Accept a pre-fetched plugins array (normal startup path reuses
    // loadPlugins' fetch). Fall back to our own fetch if called
    // standalone — e.g. from the DevTools console for debugging.
    if (!Array.isArray(plugins)) {
        plugins = [];
        try {
            const resp = await fetch('/api/plugins');
            if (resp.ok) plugins = await resp.json();
        } catch (e) {
            console.warn('viz picker: /api/plugins fetch failed', e);
        }
    }
    const vizPlugins = plugins.filter(p => p && p.type === 'visualization');
    // "default" is reserved for the built-in 2D renderer option and
    // "auto" is reserved for the Auto-mode entry — both already in the
    // <select>. A plugin with either id would collide: the
    // restore-from-localStorage lookup would find the built-in entry,
    // dragging the plugin into never-selected land silently. Fail
    // loudly instead.
    const RESERVED_IDS = new Set(['default', 'auto']);
    for (const p of vizPlugins) {
        if (RESERVED_IDS.has(p.id)) {
            console.error(`viz picker: plugin id '${p.id}' collides with a reserved built-in picker entry ('auto' = Auto mode, 'default' = built-in 2D highway); rename the plugin's id in plugin.json to include it in the picker.`);
            continue;
        }
        // Skip entries where the plugin script hasn't exposed a factory —
        // likely means the script failed to load, or the plugin declared
        // itself as a viz without shipping the factory yet.
        const factoryName = 'slopsmithViz_' + p.id;
        if (typeof window[factoryName] !== 'function') {
            console.warn(`viz picker: plugin '${p.id}' has type=visualization but ${factoryName} is not a function; skipping`);
            continue;
        }
        const opt = document.createElement('option');
        opt.value = p.id;
        opt.textContent = p.name || p.id;
        sel.appendChild(opt);
    }
    // Restore previous selection if still available. Direct option
    // scan instead of a CSS-selector lookup so we don't depend on
    // CSS.escape (missing in some test environments / older runtimes)
    // and so a weird saved string (e.g. with a quote) can't throw.
    // localStorage.getItem can itself throw when storage is blocked
    // (private mode, sandboxed iframes, some strict test runners);
    // fall back to null so the startup chain doesn't abort.
    let saved = null;
    try { saved = localStorage.getItem('vizSelection'); }
    catch (e) { console.warn('viz picker: unable to read vizSelection', e); }
    const savedMatches = saved && Array.from(sel.options).some(opt => opt.value === saved);
    if (savedMatches) {
        sel.value = saved;
        // 'default' needs no setViz — the highway already starts with
        // the built-in renderer. 'auto' runs setViz so _autoMatchViz
        // fires, though it's a no-op before the first song_info frame.
        if (saved !== 'default') setViz(saved);
    } else if (saved) {
        // Saved selection references an option that no longer exists —
        // plugin uninstalled since last session, renamed, or the plugin
        // script failed to register its factory this time. Clear the
        // stale value so we don't keep trying the same missing viz on
        // every reload, and fall through to the fresh-install default
        // below.
        try { localStorage.removeItem('vizSelection'); }
        catch (_) { /* storage blocked; ignore */ }
        saved = null;
    }
    if (!saved) {
        // Fresh install (or post-cleanup fallthrough): default to Auto
        // so the arrangement-matching plugins (piano on Keys songs,
        // drums on Drums songs, ...) take over without a manual pick.
        // Users who actively selected 'default' keep 'default' —
        // savedMatches above handles that.
        sel.value = 'auto';
        try { localStorage.setItem('vizSelection', 'auto'); } catch (_) {}
    }
    // Close a startup race: if playback began before loadPlugins
    // finished, song:ready already fired while the picker had no
    // plugin options — _autoMatchViz saw no candidates and left the
    // default active. Now that plugins are registered, re-evaluate
    // against whatever song is currently loaded (a no-op when no song
    // has been loaded yet, since highway.getSongInfo() returns {}).
    if (sel.value === 'auto') _autoMatchViz();
}

function setViz(id) {
    // Helper: reset the UI and persisted selection to the built-in
    // "default" entry. Called whenever the requested viz can't be
    // applied (missing factory, factory threw, factory returned a
    // non-conforming renderer) so the picker, localStorage, and the
    // highway's active renderer stay in sync.
    const fallbackToDefault = () => {
        try { localStorage.setItem('vizSelection', 'default'); } catch (_) {}
        const sel = document.getElementById('viz-picker');
        if (sel) sel.value = 'default';
        highway.setRenderer(null);
    };

    if (id === 'default' || !id) {
        try { localStorage.setItem('vizSelection', id || 'default'); } catch (_) {}
        highway.setRenderer(null);
        return;
    }
    if (id === 'auto') {
        try { localStorage.setItem('vizSelection', 'auto'); } catch (_) {}
        _autoMatchViz();
        return;
    }
    const factory = window['slopsmithViz_' + id];
    if (typeof factory !== 'function') {
        console.error(`viz picker: factory slopsmithViz_${id} not available`);
        fallbackToDefault();
        return;
    }
    let renderer;
    try { renderer = factory(); }
    catch (e) {
        console.error(`viz picker: factory slopsmithViz_${id} threw`, e);
        fallbackToDefault();
        return;
    }
    // Validate shape — highway.setRenderer will itself fall back to
    // default on a bad renderer, but without this check the UI and
    // localStorage would still advertise the broken selection.
    if (!renderer || typeof renderer.draw !== 'function') {
        console.error(`viz picker: factory slopsmithViz_${id} returned an invalid renderer (missing draw)`);
        fallbackToDefault();
        return;
    }
    // Persist only once we know the renderer is valid.
    try { localStorage.setItem('vizSelection', id); } catch (_) {}
    highway.setRenderer(renderer);
}

// Auto mode: evaluate each registered viz factory's static
// `matchesArrangement(songInfo)` predicate and install the first
// matching renderer. No match → fall back to the built-in 2D highway.
//
// vizSelection stays 'auto' across invocations so the next song:ready
// re-evaluates. An explicit picker choice overrides Auto by persisting
// a different vizSelection.
//
// Enumerates viz plugins by walking the picker's own <option> list —
// that's the canonical set built by _populateVizPicker above and keeps
// us from needing a second module-level registry.
function _autoMatchViz() {
    const sel = document.getElementById('viz-picker');
    if (!sel) return;
    const songInfo = (typeof highway !== 'undefined' && typeof highway.getSongInfo === 'function')
        ? (highway.getSongInfo() || {}) : {};
    // Options are stable in DOM order, which matches what users see in
    // the picker. The underlying order comes from /api/plugins →
    // _populateVizPicker, and /api/plugins reflects the order the
    // plugin loader discovered plugins in — plugins/__init__.py walks
    // `sorted(plugins_base_dir.iterdir())`, i.e. sorted by the on-disk
    // PLUGIN DIRECTORY name (e.g. "slopsmith-plugin-drums" sorts
    // before "slopsmith-plugin-piano"), not by the plugin id declared
    // in plugin.json. Two consequences worth noting:
    //   1. First match wins among registered viz plugins — keep each
    //      plugin's matchesArrangement predicate narrow to avoid
    //      stealing songs from more specialized viz.
    //   2. If you need a strict priority when multiple plugins match
    //      the same song, name the higher-priority plugin's directory
    //      earlier alphabetically. The picker dropdown reveals the
    //      actual tiebreaker at a glance.
    const candidateIds = Array.from(sel.options)
        .map(o => o.value)
        .filter(v => v !== 'auto' && v !== 'default');
    for (const id of candidateIds) {
        const factory = window['slopsmithViz_' + id];
        if (typeof factory !== 'function') continue;
        const predicate = factory.matchesArrangement;
        if (typeof predicate !== 'function') continue;
        let matched = false;
        try { matched = !!predicate(songInfo); }
        catch (err) {
            console.error(`viz auto: matchesArrangement for ${id} threw`, err);
            continue;
        }
        if (!matched) continue;
        let renderer;
        try { renderer = factory(); }
        catch (err) {
            console.error(`viz auto: factory slopsmithViz_${id} threw`, err);
            continue;
        }
        if (!renderer || typeof renderer.draw !== 'function') {
            console.error(`viz auto: factory slopsmithViz_${id} returned an invalid renderer (missing draw)`);
            continue;
        }
        // Deliberately NOT persisting id — vizSelection stays 'auto' so
        // the next song:ready re-evaluates against the new arrangement.
        highway.setRenderer(renderer);
        return;
    }
    // No match — restore the built-in 2D highway. setRenderer(null) is
    // a no-op when the default is already active. KNOWN LIMITATION:
    // when the previous Auto pick was a WebGL renderer, the canvas has
    // been locked to 'webgl' by that renderer's init; reverting to the
    // default 2D renderer will fail silently (see CLAUDE.md "first
    // context wins"). That's the same limitation manual picker swaps
    // already have — a future wave will teach highway to recreate the
    // canvas on context-type change.
    highway.setRenderer(null);
}

function formatTime(s) { return `${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`; }

// ── A-B Loop ────────────────────────────────────────────────────────────
let loopA = null;
let loopB = null;

function setLoopStart() {
    loopA = audio.currentTime;
    document.getElementById('btn-loop-a').className = 'px-3 py-1.5 bg-green-900/50 rounded-lg text-xs text-green-300 transition';
    updateLoopUI();
}

function setLoopEnd() {
    if (loopA === null) return;
    loopB = audio.currentTime;
    if (loopB <= loopA) { loopB = null; return; }
    document.getElementById('btn-loop-b').className = 'px-3 py-1.5 bg-green-900/50 rounded-lg text-xs text-green-300 transition';
    updateLoopUI();
}

function clearLoop() {
    loopA = null;
    loopB = null;
    document.getElementById('btn-loop-a').className = 'px-3 py-1.5 bg-dark-600 hover:bg-dark-500 rounded-lg text-xs text-gray-300 transition';
    document.getElementById('btn-loop-b').className = 'px-3 py-1.5 bg-dark-600 hover:bg-dark-500 rounded-lg text-xs text-gray-300 transition';
    document.getElementById('btn-loop-clear').classList.add('hidden');
    document.getElementById('btn-loop-save').classList.add('hidden');
    document.getElementById('loop-label').textContent = '';
    document.getElementById('saved-loops').value = '';
}

function updateLoopUI() {
    const label = document.getElementById('loop-label');
    const hasLoop = loopA !== null && loopB !== null;
    if (hasLoop) {
        label.textContent = `${formatTime(loopA)} → ${formatTime(loopB)}`;
        document.getElementById('btn-loop-clear').classList.remove('hidden');
        document.getElementById('btn-loop-save').classList.remove('hidden');
    } else if (loopA !== null) {
        label.textContent = `${formatTime(loopA)} → ?`;
        document.getElementById('btn-loop-clear').classList.add('hidden');
        document.getElementById('btn-loop-save').classList.add('hidden');
    } else {
        label.textContent = '';
    }
}

async function loadSavedLoops() {
    const sel = document.getElementById('saved-loops');
    const delBtn = document.getElementById('btn-loop-delete');
    if (!currentFilename) { sel.classList.add('hidden'); delBtn.classList.add('hidden'); return; }

    const resp = await fetch(`/api/loops?filename=${encodeURIComponent(decodeURIComponent(currentFilename))}`);
    const loops = await resp.json();

    sel.innerHTML = '<option value="">Saved Loops</option>';
    for (const l of loops) {
        sel.innerHTML += `<option value="${l.id}" data-start="${l.start}" data-end="${l.end}">${esc(l.name)} (${formatTime(l.start)}→${formatTime(l.end)})</option>`;
    }
    if (loops.length > 0) {
        sel.classList.remove('hidden');
    } else {
        sel.classList.add('hidden');
    }
    delBtn.classList.add('hidden');
}

function loadSavedLoop(loopId) {
    const sel = document.getElementById('saved-loops');
    const opt = sel.selectedOptions[0];
    const delBtn = document.getElementById('btn-loop-delete');
    if (!loopId || !opt?.dataset.start) {
        delBtn.classList.add('hidden');
        return;
    }
    loopA = parseFloat(opt.dataset.start);
    loopB = parseFloat(opt.dataset.end);
    audio.currentTime = loopA;
    document.getElementById('btn-loop-a').className = 'px-3 py-1.5 bg-green-900/50 rounded-lg text-xs text-green-300 transition';
    document.getElementById('btn-loop-b').className = 'px-3 py-1.5 bg-green-900/50 rounded-lg text-xs text-green-300 transition';
    updateLoopUI();
    delBtn.classList.remove('hidden');
}

async function saveCurrentLoop() {
    if (loopA === null || loopB === null || !currentFilename) return;
    const name = prompt('Loop name:', `Loop`);
    if (name === null) return;
    await fetch('/api/loops', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            filename: decodeURIComponent(currentFilename),
            name: name,
            start: loopA,
            end: loopB,
        }),
    });
    await loadSavedLoops();
    document.getElementById('btn-loop-save').classList.add('hidden');
}

async function deleteSelectedLoop() {
    const sel = document.getElementById('saved-loops');
    const loopId = sel.value;
    if (!loopId) return;
    await fetch(`/api/loops/${loopId}`, { method: 'DELETE' });
    clearLoop();
    await loadSavedLoops();
}

// ── Count-in click sound (Web Audio API) ────────────────────────────────
let _audioCtx = null;
function playClick(high = false) {
    if (!_audioCtx) _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = _audioCtx.createOscillator();
    const gain = _audioCtx.createGain();
    osc.connect(gain);
    gain.connect(_audioCtx.destination);
    osc.frequency.value = high ? 1200 : 800;
    osc.type = 'sine';
    gain.gain.setValueAtTime(0.5, _audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, _audioCtx.currentTime + 0.08);
    osc.start(_audioCtx.currentTime);
    osc.stop(_audioCtx.currentTime + 0.08);
}

let _countingIn = false;
let _countOverlay = null;

function showCountOverlay(n) {
    if (!_countOverlay) {
        _countOverlay = document.createElement('div');
        _countOverlay.className = 'fixed inset-0 z-[100] flex items-center justify-center pointer-events-none';
        document.body.appendChild(_countOverlay);
    }
    _countOverlay.innerHTML = `<span class="text-9xl font-black text-white/30">${n}</span>`;
}

function hideCountOverlay() {
    if (_countOverlay) { _countOverlay.remove(); _countOverlay = null; }
}

function startCountIn() {
    if (_countingIn) return;
    _countingIn = true;
    audio.pause();

    // Rewind animation: sweep highway time from B to A
    const rewindDuration = 400; // ms
    const rewindStart = performance.now();
    const fromTime = loopB;
    const toTime = loopA;

    function rewindStep(now) {
        const elapsed = now - rewindStart;
        const t = Math.min(elapsed / rewindDuration, 1);
        // Ease out quad
        const eased = 1 - (1 - t) * (1 - t);
        const currentT = fromTime + (toTime - fromTime) * eased;
        highway.setTime(currentT);
        if (t < 1) {
            requestAnimationFrame(rewindStep);
        } else {
            // Rewind done — set final position and start count
            audio.currentTime = loopA;
            lastAudioTime = loopA;
            highway.setTime(loopA);
            beginCount();
        }
    }
    requestAnimationFrame(rewindStep);

    function beginCount() {
        const bpm = highway.getBPM(loopA);
        const beatInterval = 60 / bpm;
        let count = 0;

        function tick() {
            count++;
            if (count > 4) {
                hideCountOverlay();
                _countingIn = false;
                audio.play();
                isPlaying = true;
                document.getElementById('btn-play').textContent = '⏸ Pause';
                return;
            }
            showCountOverlay(count);
            playClick(count === 1);
            setTimeout(tick, beatInterval * 1000);
        }
        setTimeout(tick, 500);
    }
}

// Time display + highway sync
let lastAudioTime = 0;
setInterval(() => {
    if (audio.duration && !_countingIn) {
        // A-B loop: count-in then seek back to A
        if (loopA !== null && loopB !== null && audio.currentTime >= loopB) {
            lastAudioTime = loopB;
            startCountIn();
        }
        // Detect and fix audio time jumps (browser seeking bug)
        else if (isPlaying && Math.abs(audio.currentTime - lastAudioTime) > 30 && lastAudioTime > 0) {
            console.warn(`Audio time jumped from ${lastAudioTime.toFixed(1)} to ${audio.currentTime.toFixed(1)}, resetting`);
            audio.currentTime = lastAudioTime;
        }
        lastAudioTime = audio.currentTime;
        document.getElementById('hud-time').textContent = `${formatTime(audio.currentTime)} / ${formatTime(audio.duration)}`;
    }
    if (!_countingIn) highway.setTime(audio.currentTime);
}, 1000 / 60);

// Keyboard shortcuts (player only)
document.addEventListener('keydown', e => {
    if (!document.getElementById('player').classList.contains('active')) return;
    if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
    else if (e.code === 'ArrowLeft') seekBy(-5);
    else if (e.code === 'ArrowRight') seekBy(5);
    else if (e.code === 'Escape') showScreen('home');
    // A/V offset live-calibration — watch the highway and listen to
    // the audio while tuning. Shift for coarse ±50 ms, bare key for
    // fine ±10 ms. Match on e.key (the produced character) rather
    // than e.code (physical-key position) so layouts where `[`/`]`
    // are AltGr combinations (QWERTZ, AZERTY) still fire correctly.
    else if (e.key === '[') { e.preventDefault(); nudgeAvOffsetMs(e.shiftKey ? -50 : -10); }
    else if (e.key === ']') { e.preventDefault(); nudgeAvOffsetMs(e.shiftKey ?  50 :  10); }
});

// ── Edit metadata modal ─────────────────────────────────────────────────
function openEditModal(songData) {
    const artUrl = `/api/song/${encodeURIComponent(songData.f)}/art?t=${Date.now()}`;
    const modal = document.createElement('div');
    modal.id = 'edit-modal';
    modal.className = 'fixed inset-0 z-[200] flex items-center justify-center bg-black/70 backdrop-blur-sm';
    modal.innerHTML = `
        <div class="bg-dark-700 border border-gray-700 rounded-2xl p-6 w-full max-w-md mx-4 shadow-2xl">
            <h3 class="text-lg font-bold text-white mb-4">Edit Song</h3>
            <div class="space-y-3">
                <div class="flex items-center gap-4 mb-2">
                    <div class="relative group cursor-pointer" id="edit-art-wrapper">
                        <img src="${artUrl}" alt="" class="w-20 h-20 rounded-lg object-cover bg-dark-600" id="edit-art-preview">
                        <div class="absolute inset-0 bg-black/50 rounded-lg flex items-center justify-center opacity-0 group-hover:opacity-100 transition">
                            <span class="text-white text-xs">Change</span>
                        </div>
                        <input type="file" accept="image/*" id="edit-art-file" class="hidden" onchange="previewEditArt(this)">
                    </div>
                    <p class="text-xs text-gray-500 flex-1">Click image to change album art</p>
                </div>
                <div>
                    <label class="text-xs text-gray-400 mb-1 block">Title</label>
                    <input type="text" id="edit-title" value="${esc(songData.t)}"
                        class="w-full bg-dark-600 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 outline-none focus:border-accent/50">
                </div>
                <div>
                    <label class="text-xs text-gray-400 mb-1 block">Artist</label>
                    <input type="text" id="edit-artist" value="${esc(songData.a)}"
                        class="w-full bg-dark-600 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 outline-none focus:border-accent/50">
                </div>
                <div>
                    <label class="text-xs text-gray-400 mb-1 block">Album</label>
                    <input type="text" id="edit-album" value="${esc(songData.al)}"
                        class="w-full bg-dark-600 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 outline-none focus:border-accent/50">
                </div>
            </div>
            <div class="flex gap-3 mt-5">
                <button onclick="saveEditModal('${encodeURIComponent(songData.f)}')"
                    class="flex-1 bg-accent hover:bg-accent-light px-4 py-2 rounded-xl text-sm font-semibold text-white transition">Save</button>
                <button onclick="document.getElementById('edit-modal').remove()"
                    class="px-4 py-2 bg-dark-600 hover:bg-dark-500 rounded-xl text-sm text-gray-300 transition">Cancel</button>
            </div>
        </div>`;
    document.body.appendChild(modal);

    // Click on art triggers file input
    document.getElementById('edit-art-wrapper').addEventListener('click', () => {
        document.getElementById('edit-art-file').click();
    });

    // Close on backdrop click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.remove();
    });
}

function previewEditArt(input) {
    if (!input.files || !input.files[0]) return;
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('edit-art-preview').src = e.target.result;
    };
    reader.readAsDataURL(input.files[0]);
}

async function saveEditModal(encodedFilename) {
    const filename = decodeURIComponent(encodedFilename);

    // Save metadata
    await fetch(`/api/song/${encodeURIComponent(filename)}/meta`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            title: document.getElementById('edit-title').value.trim(),
            artist: document.getElementById('edit-artist').value.trim(),
            album: document.getElementById('edit-album').value.trim(),
        }),
    });

    // Upload art if changed
    const fileInput = document.getElementById('edit-art-file');
    if (fileInput.files && fileInput.files[0]) {
        const reader = new FileReader();
        reader.onload = async (e) => {
            await fetch(`/api/song/${encodeURIComponent(filename)}/art/upload`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: e.target.result }),
            });
        };
        reader.readAsDataURL(fileInput.files[0]);
    }

    document.getElementById('edit-modal').remove();
    // Refresh current view
    const activeScreen = document.querySelector('.screen.active');
    if (activeScreen?.id === 'favorites') loadFavorites();
    else loadLibrary();
}

// Delegated click handlers
document.addEventListener('click', e => {
    // Edit button
    const edit = e.target.closest('.edit-btn');
    if (edit) {
        e.stopPropagation();
        openEditModal(JSON.parse(edit.dataset.edit));
        return;
    }
    // Favorite button
    const fav = e.target.closest('.fav-btn');
    if (fav) {
        e.stopPropagation();
        toggleFavorite(decodeURIComponent(fav.dataset.fav));
        return;
    }
    // Retune button
    const btn = e.target.closest('.retune-btn');
    if (btn) {
        e.stopPropagation();
        retuneSong(btn.dataset.retune, decodeURIComponent(btn.dataset.title), btn.dataset.tuning, btn.dataset.target || 'E Standard');
        return;
    }
    // Song card
    const card = e.target.closest('[data-play]');
    if (card) {
        playSong(card.dataset.play);
    }
});

// ── Scan banner (non-blocking) ──────────────────────────────────────────
function showScanBanner() {
    if (document.getElementById('scan-banner')) return;
    const el = document.createElement('div');
    el.id = 'scan-banner';
    el.className = 'fixed bottom-0 left-0 right-0 z-50 bg-dark-700/95 backdrop-blur border-t border-gray-700 px-6 py-3 flex items-center gap-4';
    el.innerHTML = `
        <div class="flex-1">
            <div class="flex items-center gap-3 mb-1">
                <span class="text-sm font-semibold text-white">Importing Library</span>
                <span class="text-xs text-gray-400" id="scan-progress">0 / 0</span>
            </div>
            <div class="progress-bar"><div class="fill" id="scan-bar" style="width:0%"></div></div>
            <p class="text-xs text-gray-500 mt-1 truncate" id="scan-file">Starting...</p>
        </div>
        <button onclick="hideScanBanner()" class="px-3 py-1.5 bg-dark-600 hover:bg-dark-500 rounded-lg text-xs text-gray-400 transition flex-shrink-0">Dismiss</button>`;
    document.body.appendChild(el);
}

function hideScanBanner() {
    const el = document.getElementById('scan-banner');
    if (el) el.remove();
}

let _scanPollId = null;

async function pollScanStatus() {
    try {
        const resp = await fetch('/api/scan-status');
        const data = await resp.json();
        if (data.stage === 'error' && data.error) {
            // Surface the error in the banner and stop polling.
            showScanBanner();
            const file = document.getElementById('scan-file');
            const prog = document.getElementById('scan-progress');
            if (file) { file.textContent = 'Scan failed: ' + data.error; file.classList.add('text-red-400'); }
            if (prog) prog.textContent = 'Error';
            clearInterval(_scanPollId);
            _scanPollId = null;
            return;
        }
        if (data.running) {
            showScanBanner();
            const pct = data.total > 0 ? Math.round(data.done / data.total * 100) : 0;
            const bar = document.getElementById('scan-bar');
            const prog = document.getElementById('scan-progress');
            const file = document.getElementById('scan-file');
            if (bar) bar.style.width = pct + '%';
            if (prog) prog.textContent = `${data.done} / ${data.total} (${pct}%)`;
            if (file) {
                const name = (data.current || '').replace(/_p\.psarc$/i, '').replace(/_/g, ' ');
                file.textContent = name || (data.stage === 'listing' ? 'Listing DLC folder...' : 'Processing...');
            }
        } else {
            if (document.getElementById('scan-banner')) {
                hideScanBanner();
                _treeStats = null;  // Refresh stats
                loadLibrary();
            }
            clearInterval(_scanPollId);
            _scanPollId = null;
        }
    } catch (e) { /* ignore */ }
}

async function checkScanAndLoad() {
    const resp = await fetch('/api/scan-status');
    const data = await resp.json();
    if (data.running) {
        showScanBanner();
        _scanPollId = setInterval(pollScanStatus, 1000);
    }
    loadLibrary();
}

// ── Plugin loader ───────────────────────────────────────────────────────
function setPluginLoadingState(loading, message) {
    const navContainer = document.getElementById('nav-plugins');
    const mobileNavContainer = document.getElementById('mobile-nav-plugins');
    const settingsArea = document.getElementById('plugin-settings-area');
    if (!navContainer || !mobileNavContainer) return;

    if (loading) {
        navContainer.innerHTML = `<span class="text-xs text-gray-500 animate-pulse">${esc(message || 'Loading plugins...')}</span>`;
        mobileNavContainer.innerHTML = `
            <span class="text-xs text-gray-600 uppercase tracking-wider">Plugins</span>
            <span class="text-xs text-gray-500 animate-pulse">${esc(message || 'Loading plugins...')}</span>`;
        if (settingsArea) settingsArea.classList.add('hidden');
        return;
    }

    navContainer.innerHTML = '';
    mobileNavContainer.innerHTML = '<span class="text-xs text-gray-600 uppercase tracking-wider">Plugins</span>';
}

async function waitForPluginStartupComplete(timeoutMs = 180000) {
    const start = Date.now();
    let last = null;
    let failCount = 0;
    const MAX_CONSECUTIVE_FAILURES = 5;
    while (Date.now() - start < timeoutMs) {
        try {
            const resp = await fetch('/api/startup-status');
            if (resp.ok) {
                failCount = 0;
                const status = await resp.json();
                last = status;
                const phase = (status.phase || '').trim();
                const msg = (status.message || '').trim() || 'Loading plugins...';
                const countMsg = status.total > 0 ? ` (${status.loaded || 0}/${status.total})` : '';
                setPluginLoadingState(Boolean(status.running), `${msg}${countMsg}`);
                if (!status.running && (phase === 'complete' || phase === 'error')) return status;
            } else {
                failCount++;
                if (failCount >= MAX_CONSECUTIVE_FAILURES) {
                    setPluginLoadingState(false);
                    return last || { running: false, phase: 'error', message: 'Startup status unavailable', error: null, current_plugin: '', loaded: 0, total: 0 };
                }
            }
        } catch (e) {
            failCount++;
            if (failCount >= MAX_CONSECUTIVE_FAILURES) {
                setPluginLoadingState(false);
                return last || { running: false, phase: 'error', message: 'Startup status unavailable', error: null, current_plugin: '', loaded: 0, total: 0 };
            }
        }
        await new Promise((r) => setTimeout(r, 800));
    }
    setPluginLoadingState(false);
    return { running: false, phase: 'timeout', message: 'Plugin startup timed out', error: null, current_plugin: '', loaded: 0, total: 0 };
}

async function loadPlugins() {
    let plugins;
    try {
        const resp = await fetch('/api/plugins');
        plugins = await resp.json();

        const navContainer = document.getElementById('nav-plugins');
        const mobileNavContainer = document.getElementById('mobile-nav-plugins');
        const settingsContainer = document.getElementById('plugin-settings');

        // One-shot hydration guard: always clear plugin-owned containers first.
        navContainer.innerHTML = '';
        mobileNavContainer.innerHTML = '<span class="text-xs text-gray-600 uppercase tracking-wider">Plugins</span>';
        if (settingsContainer) settingsContainer.innerHTML = '';
        document.querySelectorAll('.screen[id^="plugin-"]').forEach((el) => el.remove());

        // Plugin settings area hosts both "Plugin Updates" and per-plugin
        // collapsibles. Reveal it whenever any plugins are installed —
        // updates are relevant even for plugins that contribute no settings.
        if (plugins.length > 0) {
            const area = document.getElementById('plugin-settings-area');
            if (area) area.classList.remove('hidden');
        }

        // Build plugin dropdown for desktop nav
        const navPlugins = plugins.filter(p => p.nav);
        if (navPlugins.length > 0) {
            const dropdown = document.createElement('div');
            dropdown.className = 'relative';
            dropdown.innerHTML = `
                <button class="text-sm text-gray-400 hover:text-white transition flex items-center gap-1" onclick="this.nextElementSibling.classList.toggle('hidden')">
                    Plugins
                    <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
                </button>
                <div class="hidden absolute top-full left-0 mt-2 bg-dark-800 border border-gray-700 rounded-xl shadow-xl py-2 min-w-[180px] z-50" id="plugin-dropdown"></div>`;
            navContainer.appendChild(dropdown);
            const ddMenu = dropdown.querySelector('#plugin-dropdown');

            // Close dropdown when clicking outside
            document.addEventListener('click', (e) => {
                if (!dropdown.contains(e.target)) ddMenu.classList.add('hidden');
            });

            for (const plugin of navPlugins) {
                const screenId = `plugin-${plugin.id}`;
                const item = document.createElement('a');
                item.href = '#';
                item.className = 'block px-4 py-2 text-sm text-gray-400 hover:text-white hover:bg-dark-700 transition';
                item.textContent = plugin.nav.label;
                item.onclick = (e) => { e.preventDefault(); ddMenu.classList.add('hidden'); showScreen(screenId); window.slopsmithDemoTrack?.('event/plugin-open/' + plugin.id); };
                ddMenu.appendChild(item);

                // Mobile nav — flat list
                const ma = document.createElement('a');
                ma.href = '#';
                ma.className = 'text-gray-400 hover:text-white pl-4 text-sm';
                ma.textContent = plugin.nav.label;
                ma.onclick = (e) => { e.preventDefault(); showScreen(screenId); ma.closest('#mobile-menu').classList.add('hidden'); window.slopsmithDemoTrack?.('event/plugin-open/' + plugin.id); };
                mobileNavContainer.appendChild(ma);
            }
        }

        for (const plugin of plugins) {
            try {
            const screenId = `plugin-${plugin.id}`;

            // Inject screen container
            if (plugin.has_screen) {
                const screenDiv = document.createElement('div');
                screenDiv.id = screenId;
                screenDiv.className = 'screen';
                // Insert before the player screen
                const player = document.getElementById('player');
                player.parentNode.insertBefore(screenDiv, player);

                const htmlResp = await fetch(`/api/plugins/${plugin.id}/screen.html`);
                screenDiv.innerHTML = await htmlResp.text();
            }

            // Inject settings section — wrapped in a collapsible <details>
            // per plugin so the page stays scannable as plugins accumulate.
            // Collapsed by default; <details>/<summary> handles state natively.
            if (plugin.has_settings && settingsContainer) {
                const details = document.createElement('details');
                details.className = 'bg-dark-700/40 border border-gray-800 rounded-xl overflow-hidden group';

                const summary = document.createElement('summary');
                // .plugin-settings-summary class hides the browser's native
                // disclosure triangle (see style.css) so only our chevron shows.
                summary.className = 'plugin-settings-summary cursor-pointer select-none px-4 py-3 text-sm font-medium text-gray-300 hover:bg-dark-700/70 transition flex items-center justify-between';
                const labelSpan = document.createElement('span');
                labelSpan.textContent = plugin.name || plugin.id;
                summary.appendChild(labelSpan);
                // Chevron icon — built via setAttributeNS so the SVG sits in
                // the SVG namespace and renders correctly. Plugin label is
                // appended as text above so manifest values can't inject HTML.
                const svgNS = 'http://www.w3.org/2000/svg';
                const svg = document.createElementNS(svgNS, 'svg');
                svg.setAttribute('class', 'w-4 h-4 text-gray-500 transition-transform group-open:rotate-180');
                svg.setAttribute('fill', 'none');
                svg.setAttribute('stroke', 'currentColor');
                svg.setAttribute('viewBox', '0 0 24 24');
                const path = document.createElementNS(svgNS, 'path');
                path.setAttribute('stroke-linecap', 'round');
                path.setAttribute('stroke-linejoin', 'round');
                path.setAttribute('stroke-width', '2');
                path.setAttribute('d', 'M19 9l-7 7-7-7');
                svg.appendChild(path);
                summary.appendChild(svg);
                details.appendChild(summary);

                const body = document.createElement('div');
                body.id = `plugin-settings-${plugin.id}`;
                body.className = 'px-4 py-4 border-t border-gray-800 space-y-4';
                details.appendChild(body);

                settingsContainer.appendChild(details);

                const settingsResp = await fetch(`/api/plugins/${plugin.id}/settings.html`);
                body.innerHTML = await settingsResp.text();
                // <script> tags inserted via innerHTML are intentionally
                // inert per the HTML5 spec — the browser parses them as
                // DOM nodes but never runs the body. That silently breaks
                // any plugin settings.html that wires event handlers via
                // addEventListener (e.g. file pickers, anything that
                // can't be expressed as an inline onclick=… attribute),
                // and any inline IIFE that hydrates form values from
                // localStorage. Re-create each script node — script
                // elements created via document.createElement DO execute
                // when appended — so plugins get the script behavior
                // they'd expect from a normal HTML document.
                body.querySelectorAll('script').forEach(oldScript => {
                    const newScript = document.createElement('script');
                    for (const attr of oldScript.attributes) {
                        newScript.setAttribute(attr.name, attr.value);
                    }
                    newScript.textContent = oldScript.textContent;
                    oldScript.parentNode.replaceChild(newScript, oldScript);
                });
            }

            // Load plugin JS
            if (plugin.has_script) {
                await new Promise((resolve, reject) => {
                    const script = document.createElement('script');
                    script.src = `/api/plugins/${plugin.id}/screen.js`;
                    script.onload = resolve;
                    script.onerror = reject;
                    document.body.appendChild(script);
                });
            }
            } catch (e) {
                console.warn(`Plugin '${plugin.id}' failed to load, skipping:`, e);
            }
        }
    } catch (e) {
        console.error('Failed to load plugins:', e);
        return null;
    }
    return plugins;
}

async function _scheduleStartupRehydration() {
    // Continue polling until the backend startup completes (or a long deadline).
    // Used when the initial waitForPluginStartupComplete() window expired before
    // LOADED_PLUGINS was populated — re-hydrates plugins + viz picker once done.
    const REHYDRATE_TIMEOUT_MS = 10 * 60 * 1000; // 10 min additional window
    const start = Date.now();
    while (Date.now() - start < REHYDRATE_TIMEOUT_MS) {
        await new Promise((r) => setTimeout(r, 5000));
        try {
            const resp = await fetch('/api/startup-status');
            if (!resp.ok) continue;
            const status = await resp.json();
            if (!status.running) {
                if (status.phase === 'complete') {
                    console.log('[slopsmith] Background startup complete — re-hydrating plugins');
                    const plugins = await loadPlugins();
                    _populateVizPicker(plugins);
                    setPluginLoadingState(false, '');
                } else {
                    console.warn('[slopsmith] Backend startup ended without completing — skipping re-hydration');
                }
                return;
            }
        } catch (_e) { /* network error — keep trying */ }
    }
}

async function bootstrapPluginsAndUi() {
    setPluginLoadingState(true, 'Loading plugins...');
    const startup = await waitForPluginStartupComplete();
    if (startup && (startup.phase === 'error' || startup.phase === 'timeout')) {
        const msg = startup.error || startup.message || 'Plugin startup failed';
        setPluginLoadingState(false, '');
        console.warn('Plugin startup reported error:', msg);
        // On timeout the backend may still be loading. Continue polling in the
        // background so plugins are hydrated once startup eventually completes.
        if (startup.phase === 'timeout') {
            _scheduleStartupRehydration();
        }
    }
    const plugins = await loadPlugins();
    return plugins;
}

// Load library on start. loadSettings is awaited alongside so persisted
// values (A/V offset, mastery, etc.) are applied to the highway + HUD
// before any playSong runs — otherwise a fast click could start
// playback with stale settings before /api/settings returned.
(async () => {
    // Restore library-filter UI state from localStorage before the first
    // grid fetch so the badge/chips are accurate immediately
    // (slopsmith#129).
    _renderLibFilterChips();
    _updateLibFiltersBadge();
    // Restore the persisted sort and format-filter dropdowns BEFORE
    // the first setLibView() call — setLibView triggers loadLibrary,
    // which reads `lib-sort` / `lib-format` to build the API query
    // string. Without this, the first page would always load with
    // "Artist A-Z" / "All formats" regardless of what the user had
    // picked previously.
    const savedSort = _readPersistedChoice(_LIB_SORT_KEY, _LIB_SORT_VALUES, 'artist');
    const savedFormat = _readPersistedChoice(_LIB_FORMAT_KEY, _LIB_FORMAT_VALUES, '');
    const sortEl = document.getElementById('lib-sort');
    const fmtEl = document.getElementById('lib-format');
    if (sortEl) sortEl.value = savedSort;
    if (fmtEl) fmtEl.value = savedFormat;
    // `libView` was already initialized from localStorage at module
    // load; passing it through setLibView replays the visibility
    // toggling and triggers the initial load.
    setLibView(libView);
    try { await loadSettings(); } catch (e) { console.warn('initial loadSettings failed:', e); }
    checkScanAndLoad();

    const plugins = await bootstrapPluginsAndUi();
    // Viz picker depends on plugin scripts having loaded (to find
    // window.slopsmithViz_<id> factories), so run it after loadPlugins.
    // Reuse the plugin list loadPlugins just fetched — no need to
    // round-trip /api/plugins a second time.
    _populateVizPicker(plugins);
    fetch('/api/version')
        .then(r => { if (!r.ok) throw new Error(); return r.json(); })
        .then(d => {
            const el = document.getElementById('app-version');
            const v = typeof d.version === 'string' ? d.version.trim() : '';
            if (el && v && v.toLowerCase() !== 'unknown') el.textContent = 'v' + v;
        })
        .catch(() => {});
})();
