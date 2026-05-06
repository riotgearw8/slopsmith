// Demo analytics — real impl set by demo.js; no-op in normal builds
window.slopsmithDemoTrack = window.slopsmithDemoTrack ?? null;

// ── Global keyboard shortcuts ─────────────────────────────────────────────
//
// `/` focuses the active screen's search input (Library / Favorites);
// `Esc` while focused blurs and clears it. Mirrors the GitHub / Gmail
// convention. The listener bails when the user is already typing in
// any text-accepting element so it can't intercept normal typing —
// including inputs inside the filters drawer, plugin settings, or
// modal dialogs.
function _isTextInput(el) {
    if (!el) return false;
    const tag = el.tagName;
    if (tag === 'INPUT') {
        // Some <input> types (button, checkbox, radio, range, ...) don't
        // accept text; only intercept the ones that do.
        const t = (el.type || 'text').toLowerCase();
        return ['text', 'search', 'email', 'url', 'tel', 'password', 'number'].includes(t);
    }
    if (tag === 'TEXTAREA') return true;
    if (tag === 'SELECT') return true;
    if (el.isContentEditable) return true;
    return false;
}

function _activeSearchInput() {
    // Pick the search field for whichever screen is currently active.
    // No match (e.g. on the player or settings screen) means `/` does
    // nothing — the shortcut only fires where a search box exists.
    const active = document.querySelector('.screen.active');
    if (!active) return null;
    if (active.id === 'home') return document.getElementById('lib-filter');
    if (active.id === 'favorites') return document.getElementById('fav-filter');
    return null;
}

// ── Library keyboard navigation ──────────────────────────────────────────
//
// Arrow keys move a single "selected" item among the visible cards
// (grid view) or song rows (tree view). Enter plays the selected
// song. The selected element gets:
//   - native keyboard focus via .focus() so :focus-visible draws the
//     accessible ring (announced by screen readers, follows scroll)
//   - a `.selected` class that persists when focus drifts elsewhere
//     so the user can glance back and still see their place.
//
// Grid columns are inferred from the live computed grid template at
// the moment of navigation, so up/down works correctly across all
// breakpoints (1 / 2 / 3 / 4 cols depending on viewport).

function _isElementVisible(el) {
    // Walk ancestors looking for display:none. Handles collapsed
    // `.album-body` / `.artist-body` subtrees (hidden via CSS class
    // rules). Using a DOM walk rather than `offsetParent` avoids the
    // false-negative for `position:fixed` elements whose offsetParent
    // is null even when they are perfectly visible.
    if (!el) return false;
    let node = el;
    while (node && node !== document.body) {
        if (getComputedStyle(node).display === 'none') return false;
        node = node.parentElement;
    }
    return true;
}

// `_libNavItems` is consulted on every arrow / Enter / Space / Home /
// End / activation press, including during autorepeat. Re-running
// `querySelectorAll` + visibility filtering on every keypress is the
// dominant cost on large libraries (hundreds of nodes × per-keypress
// layout reads), so the result is memoised against a generation
// counter that's bumped only when the underlying DOM actually
// changes shape: render functions and `_toggleHeader` bump
// `_libNavGeneration`. Cache misses fall through to a fresh query.
let _libNavGeneration = 0;
let _libNavItemsCache = { gen: -1, items: [], container: null, mode: null, scope: null };
function _bumpLibNavGeneration() { _libNavGeneration++; }

function _libNavItems() {
    const active = document.querySelector('.screen.active');
    if (!active) return { items: [], container: null, mode: null };
    let tree, grid;
    if (active.id === 'home') {
        tree = document.getElementById('lib-tree');
        grid = document.getElementById('lib-grid');
    } else if (active.id === 'favorites') {
        tree = document.getElementById('fav-tree');
        grid = document.getElementById('fav-grid');
    } else {
        return { items: [], container: null, mode: null };
    }
    const treeMode = tree && !tree.classList.contains('hidden');
    const scope = treeMode ? tree : grid;
    // Cache key includes the active container — switching grid↔tree or
    // home↔favorites must miss even if the generation hasn't ticked.
    if (
        _libNavItemsCache.gen === _libNavGeneration &&
        _libNavItemsCache.scope === scope &&
        scope && document.body.contains(scope)
    ) {
        return {
            items: _libNavItemsCache.items,
            container: _libNavItemsCache.container,
            mode: _libNavItemsCache.mode,
        };
    }
    let items, container, mode;
    if (treeMode) {
        // List mode — include artist headers, album headers, and song
        // rows so arrow nav still works when artists/albums are
        // collapsed (only the headers are visible then). Filter to
        // the currently-displayed nodes so collapsed children don't
        // count as targets the keyboard can land on.
        const all = Array.from(tree.querySelectorAll(
            '.artist-header, .album-header, .song-row[data-play]'
        ));
        items = all.filter(_isElementVisible);
        container = tree;
        mode = 'list';
    } else {
        items = Array.from((grid || document).querySelectorAll('.song-card[data-play]'));
        container = grid;
        mode = 'grid';
    }
    _libNavItemsCache = { gen: _libNavGeneration, items, container, mode, scope };
    return { items, container, mode };
}

function _gridColumns(container) {
    // Count columns by grouping the first row of children by their
    // top coordinate. Robust against any grid-template-columns syntax
    // (`repeat(...)`, `auto-fit`, named lines, etc.) where naively
    // splitting `getComputedStyle().gridTemplateColumns` on whitespace
    // would miscount because of spaces inside `repeat(...)` /
    // `minmax(...)`. Falls back to 1 when the container is empty
    // so callers' max(1, ...) clamps stay valid.
    if (!container) return 1;
    const children = Array.from(container.children).filter(
        c => c && c.offsetParent !== null
    );
    if (!children.length) return 1;
    const firstTop = children[0].getBoundingClientRect().top;
    let cols = 0;
    for (const c of children) {
        // Allow ~1px slop for sub-pixel rounding so two children that
        // would visually align still group together.
        if (Math.abs(c.getBoundingClientRect().top - firstTop) < 1.5) cols++;
        else break;
    }
    return Math.max(1, cols);
}

// Tracked separately from `document.activeElement` so the persistent
// `.selected` highlight survives focus drifting elsewhere (clicks
// outside the grid, drawer opening, etc). Also lets us avoid a global
// `querySelectorAll('.selected')` on every arrow press — large
// libraries make that a noticeable hot path.
let _lastLibSelected = null;

// Tracks which list screen launched the player so Esc-from-player
// returns the user to that screen instead of always defaulting to
// the Library (slopsmith#126). Reset on every `playSong` call so a
// song launched from a deep-link / plugin screen still gets a sane
// fallback ('home').
let _playerOriginScreen = 'home';

// One-shot flag set in `showScreen` when the user enters Home or
// Favorites. Consumed by the very next library render so the
// restored selection scrolls into view exactly once on screen entry
// (player → home, hard reload). Routine re-renders driven by
// search / sort / filter changes leave the user's scroll position
// alone — the highlight still re-applies, but they aren't yanked.
const _libScrollOnNextRender = { home: false, favorites: false };

// localStorage keys for "remember the last selection across reloads
// and after returning from the player". One key per screen so the
// Library and Favorites trees don't fight over the same slot. Only
// song-row / song-card selections are persisted — header selections
// in the tree are ephemeral by design (re-derived from arrow nav).
const _LIB_SELECTED_KEY = 'slopsmith.libLastSelected';
const _FAV_SELECTED_KEY = 'slopsmith.favLastSelected';
function _selectedKeyForActiveScreen() {
    const active = document.querySelector('.screen.active');
    if (!active) return null;
    if (active.id === 'home') return _LIB_SELECTED_KEY;
    if (active.id === 'favorites') return _FAV_SELECTED_KEY;
    return null;
}
function _persistLibSelection(el) {
    if (!el || !el.dataset || !el.dataset.play) return;
    const key = _selectedKeyForActiveScreen();
    if (!key) return;
    // Stored as JSON `{f, a}` — `f` (filename) drives the
    // restore-by-attribute lookup; `a` (artist) is recorded for
    // future use (e.g. cross-page restore that needs to fetch the
    // saved artist's letter bucket) but currently unread. The bare-
    // string filename format older builds wrote is still tolerated
    // in `_loadPersistedLibSelection`.
    const artist = el.dataset.artist || '';
    try {
        localStorage.setItem(key, JSON.stringify({ f: el.dataset.play, a: artist }));
    } catch { /* private mode / quota */ }
}

function _loadPersistedLibSelection(key) {
    let raw = null;
    try { raw = localStorage.getItem(key); } catch { return null; }
    if (!raw) return null;
    // Tolerate the older bare-string format (just the encoded
    // filename) — older builds wrote that and we'd rather upgrade
    // silently than orphan the user's saved selection.
    if (raw[0] !== '{') return { f: raw, a: '' };
    try {
        const o = JSON.parse(raw);
        return (o && typeof o === 'object') ? { f: o.f || '', a: o.a || '' } : null;
    } catch { return null; }
}


function _setLibSelection(el, { focus = true } = {}) {
    if (!el) return;
    // Only the previously-tracked element needs its `.selected` class
    // cleared. classList.remove on an element that no longer carries
    // the class is a no-op, so a stale `_lastLibSelected` from a
    // re-render is harmless. Avoids the global `querySelectorAll`
    // pass that the earlier implementation ran on every keypress.
    if (_lastLibSelected && _lastLibSelected !== el) {
        _lastLibSelected.classList.remove('selected');
    }
    el.classList.add('selected');
    _lastLibSelected = el;
    // Save song selections to localStorage so a reload (or returning
    // from the player) can restore the highlight. Headers don't get
    // persisted — they don't carry a stable id and the tree's auto-
    // open heuristic re-derives them on each render anyway.
    _persistLibSelection(el);
    if (focus) {
        // `preventScroll: true` skips the browser's native focus-scroll,
        // then we run a single `scrollIntoView` so we don't double-jank
        // when the element is partially in view. The browser's default
        // focus scroll uses `block: 'nearest'` too but isn't smoothable
        // and can interact poorly with sticky headers.
        el.focus({ preventScroll: true });
    }
    _scrollSelectionIntoView(el);
}

// Scroll the selected element to keep it inside a margin from the
// viewport edges. Plain `scrollIntoView({block:'nearest'})` only
// reacts when the element is fully off-screen, so during arrow nav
// the selection drifts to the edge and stays partially visible
// until it falls off — feels laggy. Centering when the row enters
// the buffer zone keeps it comfortably on-screen as the user holds
// the arrow keys.
const _SCROLL_EDGE_MARGIN = 96;
function _scrollSelectionIntoView(el) {
    if (!el) return;
    const r = el.getBoundingClientRect();
    const vh = window.innerHeight || document.documentElement.clientHeight;
    if (r.top < _SCROLL_EDGE_MARGIN || r.bottom > vh - _SCROLL_EDGE_MARGIN) {
        el.scrollIntoView({ block: 'center', inline: 'nearest' });
    }
}

function _restoreLibSelection(scopeEl, screen, { scroll = true } = {}) {
    // Re-apply the persistent `.selected` class to whichever song
    // matches the saved filename. For the tree we also walk up and
    // open every collapsed ancestor so the restored row is actually
    // visible — the user shouldn't have to hunt for their place
    // inside a collapsed artist node.
    if (!scopeEl) return null;
    const key = screen === 'favorites' ? _FAV_SELECTED_KEY : _LIB_SELECTED_KEY;
    const saved = _loadPersistedLibSelection(key);
    if (!saved || !saved.f) return null;
    // Match `data-play` exactly — both are the encoded form, so no
    // decoding needed. Avoid interpolating persisted data into a CSS
    // selector so malformed localStorage cannot make querySelector
    // throw and break rendering.
    const candidates = scopeEl.querySelectorAll('.song-card[data-play], .song-row[data-play]');
    const el = Array.from(candidates).find((node) => node.dataset.play === saved.f);
    if (!el) return null;
    // Open every collapsed ancestor in the tree so the restored row
    // is on-screen; harmless on the grid since cards have no such
    // ancestors. Sync `aria-expanded` on the matching header inside
    // each ancestor too — bypassing `_toggleHeader` here would leave
    // assistive tech reporting "collapsed" while the visual is open.
    let n = el.parentElement;
    while (n && n !== scopeEl) {
        if (n.classList.contains('artist-row') || n.classList.contains('album-group')) {
            n.classList.add('open');
            const header = Array.from(n.children).find(c => c.classList.contains('artist-header') || c.classList.contains('album-header'));
            if (header) header.setAttribute('aria-expanded', 'true');
        }
        n = n.parentElement;
    }
    if (_lastLibSelected && _lastLibSelected !== el) {
        _lastLibSelected.classList.remove('selected');
    }
    el.classList.add('selected');
    _lastLibSelected = el;
    // Center the restored element in the viewport so the user's eye
    // lands on it instead of having to scan up from the bottom edge.
    // `block: 'center'` is forgiving of items already on-screen — the
    // browser only scrolls when needed to bring the requested
    // alignment into view.
    // Skip when the caller opts out (e.g. during search/filter/sort
    // re-renders, where the user's scroll position should be left
    // alone and only the `.selected` class is re-applied).
    if (scroll) {
        el.scrollIntoView({ block: 'center', inline: 'nearest' });
    }
    return el;
}

function _moveSelectionInItems(items, deltaIdx) {
    // Items are passed in by the caller so we don't re-query the DOM
    // twice per keypress (handler queries `_libNavItems`, then we'd
    // query it again).
    if (!items.length) return false;
    const current = document.activeElement && items.includes(document.activeElement)
        ? document.activeElement
        : (_lastLibSelected && items.includes(_lastLibSelected) ? _lastLibSelected : null);
    let idx = current ? items.indexOf(current) : -1;
    let next;
    if (idx === -1) {
        // No current selection — first arrow lands on the first item
        // regardless of direction. Saves a press.
        next = items[0];
    } else {
        next = items[Math.max(0, Math.min(items.length - 1, idx + deltaIdx))];
    }
    _setLibSelection(next);
    return true;
}

function _isInsideInteractiveControl(el) {
    // Bail when the user is interacting with anything that has its
    // own keyboard semantics — form controls (checkbox / select /
    // button) consume arrow keys for their own behavior, and the
    // filters drawer is a focus trap of those. Without this guard the
    // library's arrow nav would steal arrow presses from a focused
    // tuning checkbox or sort dropdown.
    if (!el) return false;
    const tag = el.tagName;
    if (['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(tag)) return true;
    if (el.isContentEditable) return true;
    if (el.closest && el.closest('#lib-filter-drawer, [role="dialog"], #edit-modal')) return true;
    return false;
}

function _handleLibArrowNav(e) {
    // Space (' ') is the standard activation key for focusable
    // elements alongside Enter — without it, a screen-reader user
    // hitting Space on a focused card would just scroll the page
    // instead of activating it. We treat Space identically to Enter
    // inside this handler.
    const isActivate = e.key === 'Enter' || e.key === ' ' || e.key === 'Spacebar';
    if (!isActivate &&
        !['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(e.key)) {
        return false;
    }
    if (_isInsideInteractiveControl(document.activeElement)) return false;
    const { items, container, mode } = _libNavItems();
    if (!items.length) return false;

    const currentTarget = (document.activeElement && items.includes(document.activeElement))
        ? document.activeElement
        : (_lastLibSelected && items.includes(_lastLibSelected) ? _lastLibSelected : null);

    if (isActivate) {
        if (!currentTarget) return false;
        e.preventDefault();
        // Sync persistent selection before activating so Tab-then-Enter
        // (no prior arrow nav or mouse click) still lights up the `.selected`
        // ring and updates `_lastLibSelected`/localStorage — consistent with
        // the click delegate at the bottom of this file.
        _setLibSelection(currentTarget, { focus: false });
        if (currentTarget.classList.contains('song-row') ||
            currentTarget.classList.contains('song-card')) {
            // Song row OR card → play it. Pass `dataset.play` raw to
            // match the click delegate; `playSong` handles decoding
            // internally so decoding here would double-decode and
            // throw `URIError` on filenames containing `%`.
            playSong(currentTarget.dataset.play);
        } else if (currentTarget.classList.contains('artist-header') ||
                   currentTarget.classList.contains('album-header')) {
            // Header row → toggle the parent open/closed and re-derive
            // visible items so the next arrow press lands correctly.
            // `_toggleHeader` keeps `aria-expanded` in sync for
            // assistive tech.
            _toggleHeader(currentTarget);
            // Keep keyboard focus on the header we just toggled —
            // browsers sometimes drop focus to body when the
            // surrounding subtree changes display.
            currentTarget.focus({ preventScroll: true });
        }
        return true;
    }

    if (e.key === 'Home') { e.preventDefault(); _setLibSelection(items[0]); return true; }
    if (e.key === 'End')  { e.preventDefault(); _setLibSelection(items[items.length - 1]); return true; }

    if (mode === 'list') {
        if (e.key === 'ArrowDown') { e.preventDefault(); _moveSelectionInItems(items, 1); return true; }
        if (e.key === 'ArrowUp')   { e.preventDefault(); _moveSelectionInItems(items, -1); return true; }
        // Right/Left expand and collapse the artist/album under focus,
        // file-manager style. With nothing selected yet, both keys
        // initialize selection on the first visible item (matches
        // Up/Down behavior in `_moveSelectionInItems`) so the first
        // press doesn't fall through to native scroll.
        if (!currentTarget && (e.key === 'ArrowRight' || e.key === 'ArrowLeft')) {
            e.preventDefault();
            _setLibSelection(items[0]);
            return true;
        }
        if (e.key === 'ArrowRight' && currentTarget) {
            const parent = (currentTarget.classList.contains('artist-header') ||
                            currentTarget.classList.contains('album-header'))
                ? currentTarget.parentElement : null;
            if (parent && !parent.classList.contains('open')) {
                e.preventDefault();
                // Use the shared toggle path so aria-expanded stays
                // synced with the visual state for screen readers.
                _toggleHeader(currentTarget);
                currentTarget.focus({ preventScroll: true });
                return true;
            }
            // Already open — step to the next visible item (which is
            // the first child of this header).
            e.preventDefault();
            _moveSelectionInItems(items, 1);
            return true;
        }
        if (e.key === 'ArrowLeft' && currentTarget) {
            // If on an open header, collapse it. If on a song row or
            // closed header, jump to the nearest enclosing header.
            const isHeader = currentTarget.classList.contains('artist-header') ||
                             currentTarget.classList.contains('album-header');
            const headerParent = isHeader ? currentTarget.parentElement : null;
            if (headerParent && headerParent.classList.contains('open')) {
                e.preventDefault();
                _toggleHeader(currentTarget);
                currentTarget.focus({ preventScroll: true });
                return true;
            }
            // Walk up to the nearest .album-header / .artist-header
            // ancestor's sibling header. Closest album-group → its
            // header; otherwise closest artist-row → its header.
            const albumGroup = currentTarget.closest('.album-group');
            if (albumGroup && albumGroup.contains(currentTarget) &&
                !currentTarget.classList.contains('album-header')) {
                e.preventDefault();
                _setLibSelection(albumGroup.querySelector('.album-header'));
                return true;
            }
            const artistRow = currentTarget.closest('.artist-row');
            if (artistRow && !currentTarget.classList.contains('artist-header')) {
                e.preventDefault();
                _setLibSelection(artistRow.querySelector('.artist-header'));
                return true;
            }
            return false;
        }
        return false;
    }
    // Grid mode: 2D nav. Columns are read from the live CSS grid so
    // we follow the responsive breakpoints automatically.
    const cols = _gridColumns(container);
    if (e.key === 'ArrowRight') { e.preventDefault(); _moveSelectionInItems(items, 1); return true; }
    if (e.key === 'ArrowLeft')  { e.preventDefault(); _moveSelectionInItems(items, -1); return true; }
    if (e.key === 'ArrowDown')  { e.preventDefault(); _moveSelectionInItems(items, cols); return true; }
    if (e.key === 'ArrowUp')    { e.preventDefault(); _moveSelectionInItems(items, -cols); return true; }
    return false;
}

// Focus trap: keep Tab / Shift+Tab cycling inside `modal` so focus
// can't escape to the content underneath while the overlay is open.
// Call this once after the modal is in the DOM and initial focus is set.
function _trapFocusInModal(modal) {
    const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';
    modal.addEventListener('keydown', (e) => {
        if (e.key !== 'Tab') return;
        const els = Array.from(modal.querySelectorAll(FOCUSABLE)).filter(el => {
            if (!_isElementVisible(el)) return false;
            if (getComputedStyle(el).visibility === 'hidden') return false;
            if (el.disabled) return false;
            return true;
        });
        if (!els.length) return;
        const first = els[0];
        const last = els[els.length - 1];
        if (e.shiftKey) {
            if (document.activeElement === first) { e.preventDefault(); last.focus(); }
        } else {
            if (document.activeElement === last) { e.preventDefault(); first.focus(); }
        }
    });
}

// Shortcut cheat-sheet overlay. Opens on `?` (Shift+/), closes on
// Esc (handled by the generic modal close path) or on backdrop /
// close-button click. The list mirrors the canonical shortcut table
// in this file's keydown handler — when a shortcut changes here, the
// table below should change too. We keep it inline rather than
// fetching a separate file so the cheat sheet can never disagree
// with the version of app.js the user actually loaded.
function _openShortcutsModal() {
    if (document.getElementById('shortcuts-modal')) return;
    const SHORTCUTS = [
        ['Library', [
            ['/',          'Focus search'],
            ['↑ ↓ ← →',    'Move selection (grid 2D, tree vertical)'],
            ['→',          'Tree: expand header / step into open one'],
            ['←',          'Tree: collapse header / jump to parent'],
            ['Home / End', 'Jump to first / last item'],
            ['Enter / Space', 'Play song; toggle artist / album header'],
            ['c',          'Convert PSARC entry to .sloppak'],
            ['f',          'Toggle favorite'],
            ['e',          'Edit metadata'],
            ['?',          'Show this cheat sheet'],
        ]],
        ['Modals', [
            ['Esc',        'Close the open modal (edit metadata, this overlay)'],
            ['Esc',        'Otherwise: clear + blur the focused search box'],
        ]],
    ];

    const modal = document.createElement('div');
    modal.id = 'shortcuts-modal';
    modal.className = 'slopsmith-modal fixed inset-0 z-[200] flex items-center justify-center bg-black/70 backdrop-blur-sm';
    modal.setAttribute('role', 'dialog');
    modal.setAttribute('aria-modal', 'true');
    modal.setAttribute('aria-label', 'Keyboard shortcuts');
    // Record the element that triggered the modal so Esc / close can
    // return focus to the correct entry even if _lastLibSelected drifts.
    // Scope to the active screen so a stale _lastLibSelected from a
    // different screen (e.g. Library vs Favorites) doesn't receive focus.
    const _scModal = document.querySelector('.screen.active');
    modal._opener = (_lastLibSelected && document.body.contains(_lastLibSelected)
        && _scModal && _scModal.contains(_lastLibSelected))
        ? _lastLibSelected : null;

    const sections = SHORTCUTS.map(([heading, rows]) => {
        const items = rows.map(([keys, desc]) => `
            <div class="flex items-baseline justify-between gap-4 py-1.5">
                <span class="text-sm text-gray-300">${esc(desc)}</span>
                <kbd class="text-xs font-mono px-2 py-0.5 rounded bg-dark-600 border border-gray-700 text-gray-200 whitespace-nowrap">${esc(keys)}</kbd>
            </div>
        `).join('');
        return `
            <section class="mb-4 last:mb-0">
                <h4 class="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-2">${esc(heading)}</h4>
                ${items}
            </section>
        `;
    }).join('');

    modal.innerHTML = `
        <div class="bg-dark-700 border border-gray-700 rounded-2xl p-6 w-full max-w-md mx-4 shadow-2xl">
            <div class="flex items-center justify-between mb-4">
                <h3 class="text-lg font-bold text-white">Keyboard shortcuts</h3>
                <button type="button" data-shortcuts-close
                        class="text-gray-500 hover:text-white transition" aria-label="Close shortcuts">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
                </button>
            </div>
            ${sections}
            <p class="text-[11px] text-gray-500 mt-4">Tip: shortcuts bail out while you're typing in inputs, so you can always type the literal keys.</p>
        </div>
    `;

    // Click outside the inner panel (i.e. on the backdrop) closes the
    // modal — matches the conventional dialog UX.
    modal.addEventListener('click', (ev) => {
        if (ev.target === modal || ev.target.closest('[data-shortcuts-close]')) {
            const opener = modal._opener;
            modal.remove();
            const focusTarget = (opener && document.body.contains(opener)) ? opener
                : (_lastLibSelected && document.body.contains(_lastLibSelected) ? _lastLibSelected : null);
            if (focusTarget) focusTarget.focus({ preventScroll: true });
        }
    });

    document.body.appendChild(modal);
    // Move focus into the dialog so background shortcuts (and arrow
    // nav) can't fire on the underlying library entry while the
    // overlay is open. Close button is the safe default — there's no
    // primary input to focus on a read-only cheat sheet.
    const closeBtn = modal.querySelector('[data-shortcuts-close]');
    if (closeBtn) closeBtn.focus({ preventScroll: true });
    // Trap Tab / Shift+Tab inside the modal so focus can't escape to
    // the library content underneath while the overlay is open.
    _trapFocusInModal(modal);
}

document.addEventListener('keydown', (e) => {
    // Modifier-key combos belong to the browser / OS shortcuts; never
    // intercept those.
    if (e.ctrlKey || e.metaKey || e.altKey) return;

    if (_handleLibArrowNav(e)) return;

    if (e.key === '/') {
        if (_isTextInput(document.activeElement)) return;
        // Also bail when focus is inside the filter drawer, a dialog, or
        // any other interactive region — those contexts have their own
        // keyboard semantics and shouldn't be hijacked by the search
        // shortcut (e.g. a focused checkbox inside the filters drawer).
        if (_isInsideInteractiveControl(document.activeElement)) return;
        const search = _activeSearchInput();
        if (!search) return;
        e.preventDefault();  // suppress the literal '/' the input would receive
        search.focus();
        // Move caret to end without mutating .value — round-tripping
        // the value resets the browser's undo stack and can fire
        // unexpected input events on some engines. setSelectionRange
        // is the no-side-effects path.
        try {
            const len = search.value.length;
            search.setSelectionRange(len, len);
        } catch {
            // Some input types (search/email/tel) don't support
            // selection APIs in older browsers; the focus alone is
            // still useful, just no caret-end guarantee.
        }
        return;
    }

    // `?` (Shift+/) opens the keyboard-shortcuts cheat sheet. Same
    // bail rules as the other shortcuts so typing a literal `?` in
    // any input or drawer still works.
    if (e.key === '?') {
        if (_isInsideInteractiveControl(document.activeElement)) return;
        e.preventDefault();
        _openShortcutsModal();
        return;
    }

    // Single-letter shortcuts that act on the focused / selected
    // library entry — works on both grid cards and tree rows. Each
    // dispatches to a button class that the entry markup already
    // exposes, so plugins can keep owning the actual behavior:
    //   c → .sloppak-convert-btn  (Sloppak Converter plugin)
    //   f → .fav-btn              (favorite heart toggle)
    //   e → .edit-btn             (edit metadata modal)
    // No-op when no entry is currently focused / selected, when the
    // entry doesn't expose the requested button (e.g. a sloppak
    // entry has no convert button), or when the button is disabled.
    // Bails on text input / drawer focus so single-letter typing in
    // inputs still works.
    const entryShortcut = { c: 'button.sloppak-convert-btn', f: 'button.fav-btn', e: 'button.edit-btn' }[e.key.toLowerCase()];
    if (entryShortcut) {
        if (_isInsideInteractiveControl(document.activeElement)) return;
        const ae = document.activeElement;
        const activeScreen = document.querySelector('.screen.active');
        const isEntry = el => el && el.classList && (el.classList.contains('song-card') || el.classList.contains('song-row'));
        // Scope both candidates to the active screen so that a stale
        // _lastLibSelected from Library doesn't fire when the user is
        // on Favorites (or vice-versa), and so pressing f/e/c on a
        // hidden screen can't accidentally persist that filename into
        // the current screen's localStorage key.
        const inActiveScreen = el => activeScreen && activeScreen.contains(el);
        const target = (isEntry(ae) && inActiveScreen(ae)) ? ae
            : (isEntry(_lastLibSelected) && inActiveScreen(_lastLibSelected) ? _lastLibSelected : null);
        if (!target) return;
        const btn = target.querySelector(entryShortcut);
        if (!btn || btn.disabled) return;
        e.preventDefault();
        // Sync the persistent selection to the acted-on entry so that
        // Esc-to-close-modal returns focus to the correct element and
        // the `.selected` highlight stays consistent with the action.
        _setLibSelection(target, { focus: false });
        btn.click();
        return;
    }

    if (e.key === 'Escape') {
        // Modal-first: close the topmost open modal (edit-metadata,
        // shortcuts cheat sheet, future modals) so Esc dismisses
        // from anywhere — including when keyboard focus is inside
        // a form field within the modal. Restores focus to the
        // element that opened the modal (tracked in modal._opener)
        // so arrow nav resumes without an extra Tab; falls back to
        // _lastLibSelected when the opener is no longer in the DOM.
        const modals = document.querySelectorAll('[role="dialog"][aria-modal="true"].slopsmith-modal');
        if (modals.length) {
            e.preventDefault();
            e.stopImmediatePropagation();
            const modal = modals[modals.length - 1];
            const opener = modal._opener;
            modal.remove();
            const focusTarget = (opener && document.body.contains(opener)) ? opener
                : (_lastLibSelected && document.body.contains(_lastLibSelected) ? _lastLibSelected : null);
            if (focusTarget) focusTarget.focus({ preventScroll: true });
            return;
        }
        // Esc while typing in either search box clears + blurs. Other Esc
        // semantics (drawer close, screen back) are handled elsewhere; we
        // only act when a search box is the focused element.
        const ae = document.activeElement;
        if (ae && (ae.id === 'lib-filter' || ae.id === 'fav-filter')) {
            if (ae.value) {
                ae.value = '';
                ae.dispatchEvent(new Event('input', { bubbles: true }));
            }
            ae.blur();
        }
    }
});

// ── Screen Navigation ─────────────────────────────────────────────────────
async function showScreen(id) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    document.getElementById(id).classList.add('active');
    // Mark the next render as a screen-entry so it scrolls the
    // restored selection into view exactly once. Routine renders
    // (search / sort / filter typing) won't have this flag set and
    // so won't yank the viewport. Also bump the nav-items
    // generation so the next keypress doesn't reuse a cache built
    // against a now-hidden screen's container.
    _bumpLibNavGeneration();
    if (id === 'home') { _libScrollOnNextRender.home = true; loadLibrary(); }
    if (id === 'favorites') { _libScrollOnNextRender.favorites = true; loadFavorites(); }
    if (id === 'settings') loadSettings();
    if (id !== 'player') {
        highway.stop();
        if (window._juceMode) {
            await jucePlayer.stop().catch(() => {});
            window._juceMode = false;
            window._juceAudioUrl = null;
        }
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
// Tree letter selection persists across reloads / coming back from
// the player so the user lands on the same alphabet group they
// picked. Validation: any single uppercase letter, or `#` for
// non-alphabetical artists, or `''` for the All bucket.
const _LIB_TREE_LETTER_KEY = 'slopsmith.libTreeLetter';
const _FAV_TREE_LETTER_KEY = 'slopsmith.favTreeLetter';
function _readPersistedLetter(key) {
    let v = null;
    try { v = localStorage.getItem(key); } catch { return ''; }
    if (v === null) return '';
    return (v === '' || v === '#' || /^[A-Z]$/.test(v)) ? v : '';
}
function _writePersistedLetter(key, value) {
    try { localStorage.setItem(key, value || ''); } catch { /* private mode / quota */ }
}
let _treeLetter = _readPersistedLetter(_LIB_TREE_LETTER_KEY);
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
    // View toggle changes which container `_libNavItems` resolves
    // to (tree vs grid) — drop the cache so the next keypress
    // re-derives.
    _bumpLibNavGeneration();
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
        const retuneBtn = stdRetune
            ? `<button data-retune="${encodeURIComponent(s.filename)}" data-title="${encodeURIComponent(title)}" data-tuning="${tuning}" data-target="E Standard"
                class="retune-btn mt-2 w-full px-2 py-1.5 bg-gold/10 hover:bg-gold/20 border border-gold/20 rounded-lg text-xs font-medium text-gold transition">
                ⬆ Convert to E Standard</button>`
            : '';
        const fmtBadge = formatBadge(s.format, s.stem_count);
        const ariaLabel = `Play ${title || s.filename}${artist ? ' by ' + artist : ''}`;
        return `<div class="song-card group" data-play="${encodeURIComponent(s.filename)}" data-artist="${_escAttr(artist || '')}" tabindex="0" role="button" aria-label="${_escAttr(ariaLabel)}">
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
    // Items list invalidation: any DOM mutation to the grid changes
    // the result of the next `_libNavItems` call.
    _bumpLibNavGeneration();
    // Re-apply the persistent selection after a fresh render so the
    // user's last picked card stays highlighted across reloads / a
    // round-trip through the player. Skip this during `append` mode
    // (infinite scroll) so restoring selection can't re-center the
    // viewport and yank the user away from the newly loaded page.
    // When a search input is focused the user is actively filtering —
    // re-apply the highlight but don't move the viewport (they didn't
    // leave the page and their scroll position should be preserved).
    if (mode !== 'append') {
        const screen = containerId.startsWith('fav') ? 'favorites' : 'home';
        // Scroll only on the first render after a screen entry —
        // routine search / sort / filter renders re-apply the
        // highlight without moving the viewport. The flag is
        // one-shot and consumed here.
        const scroll = _libScrollOnNextRender[screen];
        if (scroll) _libScrollOnNextRender[screen] = false;
        _restoreLibSelection(grid, screen, { scroll });
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
        const artistAria = _escAttr(`Toggle artist ${artist.name}`);
        html += `<div class="artist-row${openClass}">`;
        html += `<div class="artist-header" tabindex="0" role="button" aria-expanded="${isOpen ? 'true' : 'false'}" aria-label="${artistAria}" onclick="_onHeaderClick(this)">`;
        html += chevron;
        html += `<span class="text-white font-semibold text-sm flex-1">${esc(artist.name)}</span>`;
        html += `<span class="text-xs text-gray-600">${artist.song_count} song${artist.song_count !== 1 ? 's' : ''} · ${artist.album_count} album${artist.album_count !== 1 ? 's' : ''}</span>`;
        html += `</div><div class="artist-body">`;

        for (const album of artist.albums) {
            const artUrl = `/api/song/${encodeURIComponent(album.songs[0].filename)}/art${album.songs[0].mtime ? `?v=${Math.floor(album.songs[0].mtime)}` : ''}`;
            const albumHeuristicOpen = q || artist.albums.length === 1;
            const albumIsOpen = forceArtistOpen ? true : forceArtistClosed ? false : albumHeuristicOpen;
            const albumOpen = albumIsOpen ? ' open' : '';
            const albumAria = _escAttr(`Toggle album ${album.name}`);
            html += `<div class="album-group${albumOpen}">`;
            html += `<div class="album-header" tabindex="0" role="button" aria-expanded="${albumIsOpen ? 'true' : 'false'}" aria-label="${albumAria}" onclick="_onHeaderClick(this)">`;
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
                const rowAria = _escAttr(`Play ${title}${artist.name ? ' by ' + artist.name : ''}`);
                html += `<div class="song-row" data-play="${encodeURIComponent(s.filename)}" data-artist="${_escAttr(artist.name || '')}" tabindex="0" role="button" aria-label="${rowAria}">`;
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
                if (stdRetune)
                    html += `<button data-retune="${encodeURIComponent(s.filename)}" data-title="${encodeURIComponent(title)}" data-tuning="${tuning}" data-target="E Standard"
                        class="retune-btn px-1.5 py-0.5 bg-gold/10 hover:bg-gold/20 border border-gold/20 rounded text-gold" title="Convert to E Standard">E</button>`;
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
    // Items list invalidation — see grid render counterpart.
    _bumpLibNavGeneration();
    // Re-apply the persisted selection. For the tree we also expand
    // every collapsed ancestor of the saved row so the highlight is
    // actually visible — see _restoreLibSelection. Scroll only on
    // the first render after a screen entry (one-shot flag set in
    // showScreen) so routine renders don't yank the viewport.
    const screen = favoritesOnly ? 'favorites' : 'home';
    const scroll = _libScrollOnNextRender[screen];
    if (scroll) _libScrollOnNextRender[screen] = false;
    _restoreLibSelection(container, screen, { scroll });
}

function goTreePage(p) {
    _treePage = Math.max(0, p);
    loadTreeView();
    document.getElementById('library-section').scrollIntoView({ behavior: 'smooth' });
}

function filterTreeLetter(letter) {
    _treeLetter = (_treeLetter === letter) ? '' : letter;
    _treePage = 0;
    _writePersistedLetter(_LIB_TREE_LETTER_KEY, _treeLetter);
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
    // Bulk open/close changes which song-rows pass the visibility
    // filter in `_libNavItems` — same reason `_toggleHeader` bumps
    // the generation. Without this, a stale cached items list from
    // before the toggle would let arrow nav step into now-hidden
    // rows.
    _bumpLibNavGeneration();
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

// `esc()` escapes the HTML-content metacharacters (<, >, &) but not
// quotes — fine for text-node interpolation but unsafe when the
// result is used as an attribute value, where a literal `"` ends the
// attribute early. Use `_escAttr` for any `attr="${...}"` site.
function _escAttr(s) {
    return esc(s == null ? '' : String(s))
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

// Toggle an artist/album header's parent `.open` state and keep
// `aria-expanded` on the header itself in sync so screen readers
// announce the collapsed/expanded transition correctly. Used by
// both the inline onclick (mouse) and the keyboard handlers.
function _toggleHeader(headerEl) {
    if (!headerEl) return;
    const parent = headerEl.parentElement;
    if (!parent) return;
    parent.classList.toggle('open');
    headerEl.setAttribute('aria-expanded', parent.classList.contains('open') ? 'true' : 'false');
    // Toggling open/closed changes which song-rows pass the
    // visibility filter in `_libNavItems`, so the cached items list
    // is now stale.
    _bumpLibNavGeneration();
}

// Called by the inline onclick on artist- and album-headers so the
// mouse-click path also syncs the persistent `.selected` state —
// keeps arrow-nav resuming from the last-clicked header rather than
// from a stale highlight on a different element.
function _onHeaderClick(el) {
    _toggleHeader(el);
    _setLibSelection(el, { focus: false });
}

// ── Favorites ────────────────────────────────────────────────────────────
let favView = 'grid';
let favPage = 0;
let _favTreeLetter = _readPersistedLetter(_FAV_TREE_LETTER_KEY);
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
    // Same reason as setLibView: dropping the items cache so the
    // next keypress re-derives against the now-active container.
    _bumpLibNavGeneration();
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
    _writePersistedLetter(_FAV_TREE_LETTER_KEY, _favTreeLetter);
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
    const psarcPlatformEl = document.getElementById('psarc-platform');
    if (psarcPlatformEl) psarcPlatformEl.value = data.psarc_platform || 'both';
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
            psarc_platform: document.getElementById('psarc-platform')?.value || 'both',
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

// ── Diagnostics export (slopsmith#166) ───────────────────────────────────
//
// Companion to Settings export but for troubleshooting bug reports.
// Bundle layout + schemas: docs/diagnostics-bundle-spec.md.
//
// Frontend's job is to:
//   1. Snapshot the browser-only state (console ring buffer, hardware
//      probe, localStorage, ua) via window.slopsmith.diagnostics.
//   2. POST it to /api/diagnostics/export with the user's include /
//      redact toggles.
//   3. Stream the returned zip to disk.

function _diagIncludeFromUI() {
    const v = (id) => document.getElementById(id)?.checked !== false;
    return {
        system: v('diag-incl-system'),
        hardware: v('diag-incl-hardware'),
        logs: v('diag-incl-logs'),
        console: v('diag-incl-console'),
        plugins: v('diag-incl-plugins'),
    };
}

function _diagRedactFromUI() {
    const el = document.getElementById('diag-redact');
    return el ? !!el.checked : true;
}

// Map raw file paths inside the bundle to plain-English labels +
// descriptions for the preview UI. Only paths that show up in
// previews need entries — unknown paths fall back to the path itself.
const _DIAG_FILE_LABELS = {
    'system/version.json':   { label: 'App version',    desc: 'Slopsmith version, Python, OS' },
    'system/env.json':       { label: 'Environment',    desc: 'Allowlisted env vars (LOG_LEVEL, etc.). No secrets.' },
    'system/hardware.json':  { label: 'Hardware (server-side)', desc: 'CPU, RAM, GPU. In Docker this reflects the container, not the host.' },
    'system/plugins.json':   { label: 'Plugins',        desc: 'Loaded plugins + git commit + orphan detection.' },
    'logs/server.log':       { label: 'Server log',     desc: 'Tail of LOG_FILE (last ~5 MB).' },
    'logs/server.log.meta.json': { label: 'Log metadata', desc: 'Log file path, size, rotation info.' },
    'client/console.json':   { label: 'Browser console', desc: 'console.log/warn/error transcript + window errors.' },
    'client/hardware.json':  { label: 'Hardware (browser)', desc: 'WebGL/WebGPU adapter, host OS via userAgent.' },
    'client/local_storage.json': { label: 'Browser storage', desc: 'localStorage contents (preferences).' },
    'client/ua.json':        { label: 'User agent',     desc: 'Browser, screen, page URL.' },
};

function _formatBytes(n) {
    if (!n || n < 1024) return (n || 0) + ' B';
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
    return (n / (1024 * 1024)).toFixed(1) + ' MB';
}

function _escapeHtml(s) {
    return String(s || '').replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
}

function _renderDiagPreview(data) {
    const m = data.manifest || {};
    const files = m.files || [];
    const groups = { system: [], logs: [], client: [], plugins: [], other: [] };
    for (const f of files) {
        const top = (f.path || '').split('/')[0];
        (groups[top] || groups.other).push(f);
    }
    const totalBytes = files.reduce((s, f) => s + (f.size || 0), 0);
    const include = _diagIncludeFromUI();
    const redact = _diagRedactFromUI();

    const sections = [];
    // Per-file `summary` (server-derived) → human one-liner.
    function _summaryLine(path, summary) {
        if (!summary || typeof summary !== 'object') return '';
        if (path === 'system/plugins.json') {
            const loaded = summary.loaded_count || 0;
            const orphans = summary.orphan_count || 0;
            const orphPart = orphans ? ` · <span class="text-amber-400">${orphans} orphan${orphans === 1 ? '' : 's'}</span>` : '';
            return `${loaded} plugin${loaded === 1 ? '' : 's'} loaded${orphPart}`;
        }
        if (path === 'client/console.json') {
            const total = summary.entry_count || 0;
            const lvl = summary.by_level || {};
            const parts = [];
            for (const k of ['error','warn','info','log','debug']) {
                if (lvl[k]) parts.push(`${lvl[k]} ${k}`);
            }
            return `${total} entries${parts.length ? ' (' + parts.join(', ') + ')' : ''}`;
        }
        if (path === 'system/hardware.json') {
            const bits = [];
            if (summary.cpu_brand) bits.push(summary.cpu_brand);
            if (summary.cores_logical) bits.push(`${summary.cores_logical} cores`);
            if (summary.gpu_count) bits.push(`${summary.gpu_count} GPU`);
            if (summary.runtime) bits.push(`runtime: ${summary.runtime}`);
            return bits.join(' · ');
        }
        if (path === 'client/hardware.json') {
            const bits = [];
            if (summary.runtime) bits.push(summary.runtime);
            if (summary.webgl_renderer) bits.push(summary.webgl_renderer);
            return bits.join(' · ');
        }
        if (path === 'client/local_storage.json') {
            return `${summary.key_count || 0} keys`;
        }
        if (path === 'system/version.json') {
            const bits = [];
            if (summary.slopsmith) bits.push(`slopsmith ${summary.slopsmith}`);
            if (summary.python) bits.push(`python ${summary.python}`);
            if (summary.os) bits.push(summary.os);
            return bits.join(' · ');
        }
        return '';
    }

    function pushSection(title, list, emptyHint) {
        if (!list.length) {
            if (emptyHint) {
                sections.push(`<div class="mb-3"><div class="text-gray-300 font-semibold mb-1">${_escapeHtml(title)}</div><div class="text-gray-500">${_escapeHtml(emptyHint)}</div></div>`);
            }
            return;
        }
        const rows = list.map(f => {
            const meta = _DIAG_FILE_LABELS[f.path] || { label: f.path, desc: '' };
            const summary = _summaryLine(f.path, f.summary);
            const summaryHtml = summary
                ? `<div class="text-accent-light text-[10px] mt-0.5">${summary}</div>`
                : '';
            return `<div class="flex justify-between gap-4 py-1 border-b border-dark-600 last:border-0">
                <div class="min-w-0">
                    <div class="text-gray-200">${_escapeHtml(meta.label)}</div>
                    <div class="text-gray-500 text-[10px]">${_escapeHtml(meta.desc)}</div>
                    ${summaryHtml}
                </div>
                <div class="text-gray-400 text-right whitespace-nowrap">${_escapeHtml(_formatBytes(f.size))}</div>
            </div>`;
        }).join('');
        sections.push(`<div class="mb-3"><div class="text-gray-300 font-semibold mb-1">${_escapeHtml(title)}</div>${rows}</div>`);
    }

    pushSection('System', groups.system, include.system ? '' : 'Skipped (toggle off)');
    pushSection('Server logs', groups.logs, include.logs
        ? 'No log file configured — set LOG_FILE env var to include server logs.'
        : 'Skipped (toggle off)');
    pushSection('Plugin diagnostics', groups.plugins, include.plugins
        ? 'No plugins have opted in to diagnostics.'
        : 'Skipped (toggle off)');

    // Client section preview is a server-side estimate only — actual
    // client/* payloads are added at Export time after the browser
    // snapshots. Show what WILL be added, not file sizes.
    const clientLines = [];
    if (include.console) clientLines.push({ label: 'Browser console', desc: 'console.log/warn/error transcript + window errors.' });
    if (include.hardware) clientLines.push({ label: 'Hardware (browser)', desc: 'WebGL/WebGPU adapter, host OS via userAgent.' });
    clientLines.push({ label: 'Browser storage', desc: 'localStorage contents (preferences).' });
    clientLines.push({ label: 'User agent', desc: 'Browser, screen, page URL.' });
    const clientHtml = clientLines.map(c => `<div class="flex justify-between gap-4 py-1 border-b border-dark-600 last:border-0">
        <div><div class="text-gray-200">${_escapeHtml(c.label)}</div><div class="text-gray-500 text-[10px]">${_escapeHtml(c.desc)}</div></div>
        <div class="text-gray-500 text-right whitespace-nowrap">added on export</div>
    </div>`).join('');
    sections.push(`<div class="mb-3"><div class="text-gray-300 font-semibold mb-1">Browser data</div>${clientHtml}</div>`);

    const notesHtml = (m.notes || []).length
        ? `<div class="mb-3 bg-dark-600 border border-amber-500/30 rounded-lg p-2">
              <div class="text-amber-400 text-[10px] font-semibold uppercase mb-1">Notes</div>
              ${(m.notes).map(n => `<div class="text-gray-300 text-[11px]">• ${_escapeHtml(n)}</div>`).join('')}
           </div>`
        : '';

    const privacyHtml = redact
        ? `<div class="text-emerald-400 text-[11px]">🔒 Redaction enabled — paths, song names, IPs, and secrets will be replaced with stable hash tokens.</div>`
        : `<div class="text-amber-400 text-[11px]">⚠ Redaction OFF — bundle will contain raw paths, song names, and IPs. Only share with people you trust.</div>`;

    return `
        <div class="text-[11px]">
            <div class="flex justify-between items-baseline mb-2">
                <div class="text-gray-200 font-semibold">${_escapeHtml(data.filename)}</div>
                <div class="text-gray-400">${_escapeHtml(_formatBytes(totalBytes))}<span class="text-gray-600"> server-side</span></div>
            </div>
            <div class="text-gray-500 text-[10px] mb-3">runtime: ${_escapeHtml(m.runtime || 'unknown')} · exported_at: ${_escapeHtml(m.exported_at || '')}</div>
            ${notesHtml}
            ${sections.join('')}
            ${privacyHtml}
        </div>`;
}

async function previewDiagnostics() {
    const status = document.getElementById('diag-status');
    const preview = document.getElementById('diag-preview');
    if (!status || !preview) return;
    status.textContent = 'Building preview…';
    preview.classList.add('hidden');
    const include = _diagIncludeFromUI();
    const params = new URLSearchParams({
        redact: String(_diagRedactFromUI()),
        system: String(include.system),
        hardware: String(include.hardware),
        logs: String(include.logs),
        console: String(include.console),
        plugins: String(include.plugins),
    });
    try {
        const resp = await fetch(`/api/diagnostics/preview?${params.toString()}`);
        if (!resp.ok) {
            status.textContent = `Preview failed (HTTP ${resp.status})`;
            return;
        }
        const data = await resp.json();
        preview.innerHTML = _renderDiagPreview(data);
        preview.classList.remove('hidden');
        status.textContent = 'Preview ready.';
    } catch (e) {
        status.textContent = `Preview failed: ${e.message}`;
    }
}

async function exportDiagnostics() {
    const status = document.getElementById('diag-status');
    if (!status) return;
    status.textContent = 'Building bundle…';
    const include = _diagIncludeFromUI();
    const redact = _diagRedactFromUI();

    const diag = window.slopsmith && window.slopsmith.diagnostics;
    const body = {
        redact,
        include,
        client_console: include.console && diag ? diag.snapshotConsole() : null,
        client_hardware: include.hardware && diag ? await diag.snapshotHardware() : null,
        client_ua: diag ? diag.snapshotUa() : null,
        local_storage: diag ? diag.snapshotLocalStorage() : null,
        client_contributions: diag ? diag.snapshotContributions() : null,
    };

    let resp;
    try {
        resp = await fetch('/api/diagnostics/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
    } catch (e) {
        status.textContent = `Export failed: ${e.message}`;
        return;
    }
    if (!resp.ok) {
        status.textContent = `Export failed (HTTP ${resp.status})`;
        return;
    }
    let filename = 'slopsmith-diag.zip';
    const disp = resp.headers.get('Content-Disposition');
    if (disp) {
        const m = /filename="([^"]+)"/.exec(disp);
        if (m) filename = m[1];
    }
    try {
        const blob = await resp.blob();
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
        status.textContent = `Export failed during download: ${e.message}`;
    }
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

// In Slopsmith Desktop, WASAPI Exclusive Mode locks the audio device so Chromium
// cannot play through it. When window._juceMode is true, song audio is routed
// through the JUCE backing track player instead of the HTML5 <audio> element.
window._juceMode = false;
window._juceAudioUrl = null;
const jucePlayer = {
    _timer: null,
    _pos: 0,
    _dur: 0,
    _pollAt: 0,    // performance.now() when _pos was last set
    _polling: false,
    get currentTime() {
        if (!this._polling) return this._pos;
        // Interpolate between IPC polls so highway motion is smooth at 60fps
        const elapsed = (performance.now() - this._pollAt) / 1000;
        return Math.min(this._pos + elapsed, this._dur > 0 ? this._dur : Infinity);
    },
    get duration() { return this._dur; },
    async play() {
        try {
            await window.slopsmithDesktop.audio.startBacking();
        } catch (err) {
            console.warn('[jucePlayer] startBacking failed:', err);
            return false;
        }
        this._startPolling();
        return true;
    },
    async pause() {
        // Snapshot the interpolated position before stopping the poll so
        // _pos stays at the visible pause point rather than jumping back
        // to the last raw IPC sample (which can be up to 100ms behind).
        this._pos = this.currentTime;
        this._pollAt = performance.now();
        this._stopPolling();
        try {
            await window.slopsmithDesktop.audio.stopBacking();
        } catch (err) {
            console.warn('[jucePlayer] stopBacking failed:', err);
        }
    },
    async seek(s) {
        const prev = this._pos;
        this._pos = s;
        this._pollAt = performance.now();
        try {
            await window.slopsmithDesktop.audio.seekBacking(s);
        } catch (err) {
            console.warn('[jucePlayer] seekBacking failed:', err);
            this._pos = prev;
            this._pollAt = performance.now();
        }
    },
    _startPolling() {
        this._stopPolling();
        this._polling = true;
        this._pollAt = performance.now();
        const self = this;
        function scheduleNext() {
            self._timer = setTimeout(async () => {
                if (!self._polling) return;
                try {
                    self._pos = await window.slopsmithDesktop.audio.getBackingPosition();
                    self._pollAt = performance.now();
                } catch (err) {
                    console.warn('[jucePlayer] position poll failed:', err);
                } finally {
                    if (self._polling) scheduleNext();
                }
            }, 100);
        }
        scheduleNext();
    },
    _stopPolling() {
        this._polling = false;
        if (this._timer) { clearTimeout(this._timer); this._timer = null; }
    },
    async stop() {
        await this.pause();
        this._pos = 0;
        this._dur = 0;
        this._pollAt = 0;
    },
};
window.jucePlayer = jucePlayer;

function _audioTime() { return window._juceMode ? jucePlayer.currentTime : audio.currentTime; }
function _audioDuration() { return window._juceMode ? jucePlayer.duration : audio.duration; }
async function _audioSeek(s) {
    if (window._juceMode) return jucePlayer.seek(s);
    else audio.currentTime = s;
}
let currentFilename = '';
// Plugin context API — lightweight event bus for plugin integration
// Preserve any namespace attached by earlier-loaded scripts (e.g.
// diagnostics.js, slopsmith#166) so reassigning the root doesn't drop
// their public APIs. Only `slopsmith.diagnostics` exists today, but
// the snapshot pattern is intentional: it keeps app.js the
// authoritative owner of the EventTarget while letting other modules
// hang their surfaces off the same namespace without coordinating
// load order.
const _slopsmithExisting = (typeof window.slopsmith === 'object' && window.slopsmith !== null) ? window.slopsmith : null;
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
if (_slopsmithExisting) {
    for (const key of Object.keys(_slopsmithExisting)) {
        if (!(key in window.slopsmith)) {
            window.slopsmith[key] = _slopsmithExisting[key];
        }
    }
}

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
    if (window._juceMode) {
        await jucePlayer.stop().catch(() => {});
        window._juceMode = false;
        window._juceAudioUrl = null;
    }
    audio.pause();
    audio.src = '';
    isPlaying = false;
    document.getElementById('btn-play').textContent = '▶ Play';
    document.getElementById('speed-slider').value = 100;
    document.getElementById('speed-label').textContent = '1.0x';
    clearLoop();

    currentFilename = filename;
    // Remember which screen the player was launched from so Esc /
    // navigation back from the player returns the user there
    // (slopsmith#126). Falls back to 'home' if launched from
    // somewhere unexpected (settings, a plugin screen, etc.).
    const _launchFrom = document.querySelector('.screen.active');
    _playerOriginScreen = (_launchFrom && (_launchFrom.id === 'home' || _launchFrom.id === 'favorites'))
        ? _launchFrom.id : 'home';
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

async function changeArrangement(index) {
    if (currentFilename) {
        const wasPlaying = isPlaying;
        const time = _audioTime();
        if (isPlaying) {
            if (window._juceMode) await jucePlayer.pause();
            else audio.pause();
            isPlaying = false;
        }

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
        highway._onReady = async () => {
            const ol = document.getElementById('arr-loading');
            if (ol) ol.remove();
            await _audioSeek(time);
            if (wasPlaying) {
                if (window._juceMode) {
                    const started = await jucePlayer.play();
                    if (started) {
                        isPlaying = true;
                        window.slopsmith.isPlaying = true;
                        window.slopsmith.emit('song:play', { time: jucePlayer.currentTime });
                    }
                } else audio.play().then(() => { isPlaying = true; }).catch(() => {});
            }
            highway._onReady = null;
        };

        highway.reconnect(currentFilename, index);
        window.slopsmith.emit('arrangement:changed', { index, filename: currentFilename });
    }
}

async function togglePlay() {
    if (window._juceMode) {
        if (isPlaying) {
            await jucePlayer.pause();
            isPlaying = false;
            document.getElementById('btn-play').textContent = '▶ Play';
            window.slopsmith.isPlaying = false;
            window.slopsmith.emit('song:pause', { time: jucePlayer.currentTime });
        } else {
            const started = await jucePlayer.play();
            if (!started) return; // startBacking() failed — IPC error already logged
            isPlaying = true;
            document.getElementById('btn-play').textContent = '⏸ Pause';
            window.slopsmith.isPlaying = true;
            window.slopsmith.emit('song:play', { time: jucePlayer.currentTime });
        }
        return;
    }
    if (isPlaying) {
        audio.pause(); isPlaying = false;
        document.getElementById('btn-play').textContent = '▶ Play';
    } else {
        audio.play(); isPlaying = true;
        document.getElementById('btn-play').textContent = '⏸ Pause';
    }
}

async function seekBy(s) {
    if (window._juceMode) { await jucePlayer.seek(Math.max(0, jucePlayer.currentTime + s)); return; }
    audio.currentTime = Math.max(0, audio.currentTime + s);
}
function setSpeed(v) {
    if (window._juceMode) {
        audio.playbackRate = 1.0;
        document.getElementById('speed-slider').value = 100;
        document.getElementById('speed-label').textContent = '1.0x';
        return;
    }
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
// restores the default renderer. The bundled 3D Highway plugin
// (plugins/highway_3d/) registers as id `highway_3d` and is the new
// fresh-install default per slopsmith#160 PR 3.

// ── WebGL2 detection (one-shot probe) ────────────────────────────────────
// 3D Highway requires WebGL2. On environments where it's unavailable
// (older browsers, some embedded webviews, software-only contexts), we
// silently fall back to the Classic 2D Highway and flash a single toast
// so the user knows why their highway looks different. Cached so we don't
// thrash the GPU with repeat throwaway-canvas creations.
let _webgl2Probe = null;
function _canRun3D() {
    if (_webgl2Probe !== null) return _webgl2Probe;
    try {
        const c = document.createElement('canvas');
        const gl = c.getContext('webgl2');
        _webgl2Probe = !!gl;
        // Lose the context immediately — the probe canvas is never reused.
        if (gl && gl.getExtension) {
            const ext = gl.getExtension('WEBGL_lose_context');
            if (ext && ext.loseContext) ext.loseContext();
        }
    } catch (_) { _webgl2Probe = false; }
    return _webgl2Probe;
}

// ── Migration / nag flags ────────────────────────────────────────────────
// `slopsmith_3d_promoted_v1` is set the first time we auto-flip an existing
// `vizSelection='default'` user to `'highway_3d'`. Persistence ensures we
// don't re-nag on every reload — and ensures the WebGL2 fallback path
// doesn't ping-pong (one fallback toast, not one per page load).
const _3D_PROMOTED_FLAG_KEY = 'slopsmith_3d_promoted_v1';
function _markPromoted() {
    try { localStorage.setItem(_3D_PROMOTED_FLAG_KEY, '1'); } catch (_) {}
}
function _hasPromotedFlag() {
    try { return localStorage.getItem(_3D_PROMOTED_FLAG_KEY) === '1'; }
    catch (_) { return false; }
}

// Pending nag: queued during _populateVizPicker, fired on the first
// `song:ready` (so the toast lands when the user actually opens the
// player, not at page load when they're still in the library).
// `song:ready` is emitted by highway.js via window.slopsmith.emit(), so
// subscribe through the same EventTarget. window.slopsmith is created in
// this same file before _populateVizPicker is reachable, so the global
// is guaranteed to exist by the time this listener registers — but guard
// anyway in case this module is ever loaded standalone for tests.
let _pendingPromotionNag = false;
if (window.slopsmith && typeof window.slopsmith.on === 'function') {
    window.slopsmith.on('song:ready', () => {
        if (!_pendingPromotionNag) return;
        _pendingPromotionNag = false;
        _showPromotionNag();
    });
}

function _showPromotionNag() {
    // Lightweight toast — no dependency on a generic toast helper, since
    // app.js doesn't currently have one. Fixed bottom-center, dismissed
    // by clicking either action button or the × close.
    const existing = document.getElementById('slopsmith-3d-nag');
    if (existing) existing.remove();
    const wrap = document.createElement('div');
    wrap.id = 'slopsmith-3d-nag';
    wrap.setAttribute('role', 'dialog');
    wrap.setAttribute('aria-modal', 'false');
    wrap.setAttribute('aria-label', '3D Highway upgrade notification');
    wrap.style.cssText = `
        position: fixed; left: 50%; bottom: 24px; transform: translateX(-50%);
        background: linear-gradient(145deg, #1a1a30 0%, #0d0d18 100%);
        border: 1px solid rgba(64,128,224,0.4);
        border-radius: 12px; padding: 12px 16px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(64,128,224,0.15);
        font-size: 13px; color: #e2e8f0; z-index: 10000;
        max-width: 480px; display: flex; align-items: center; gap: 12px;
    `;
    wrap.innerHTML = `
        <span aria-live="polite" style="flex:1;">Your highway was upgraded to <strong>3D</strong>.</span>
        <button type="button" data-act="tour" style="background:rgba(64,128,224,0.25);color:#e2e8f0;border:1px solid rgba(64,128,224,0.5);padding:6px 12px;border-radius:8px;font-size:12px;cursor:pointer;">Try the tour</button>
        <button type="button" data-act="back" style="background:transparent;color:#cbd5e1;border:1px solid rgba(255,255,255,0.1);padding:6px 12px;border-radius:8px;font-size:12px;cursor:pointer;">Switch back to 2D</button>
        <button type="button" data-act="dismiss" aria-label="Dismiss" style="background:transparent;color:#6b7280;border:none;font-size:18px;cursor:pointer;padding:0 4px;line-height:1;">×</button>
    `;
    wrap.addEventListener('click', (ev) => {
        const btn = ev.target.closest('button[data-act]');
        if (!btn) return;
        const act = btn.dataset.act;
        if (act === 'tour') {
            try {
                if (window.slopsmithTour && typeof window.slopsmithTour.start === 'function') {
                    window.slopsmithTour.start('highway_3d');
                }
            } catch (_) {}
        } else if (act === 'back') {
            setViz('default');
        }
        wrap.remove();
    });
    document.body.appendChild(wrap);
}

function _showWebGL2FallbackToast() {
    // One-time fallback notice. Same lightweight DOM as the nag, simpler
    // copy and only a dismiss button.
    if (document.getElementById('slopsmith-3d-fallback')) return;
    const wrap = document.createElement('div');
    wrap.id = 'slopsmith-3d-fallback';
    wrap.setAttribute('role', 'dialog');
    wrap.setAttribute('aria-modal', 'false');
    wrap.setAttribute('aria-label', 'WebGL2 not available');
    wrap.style.cssText = `
        position: fixed; left: 50%; bottom: 24px; transform: translateX(-50%);
        background: #181830; border: 1px solid rgba(255,180,80,0.4);
        border-radius: 12px; padding: 10px 14px;
        font-size: 12px; color: #e2e8f0; z-index: 10000;
        display: flex; align-items: center; gap: 10px;
    `;
    wrap.innerHTML = `
        <span aria-live="polite">3D Highway needs WebGL2 — falling back to Classic 2D.</span>
        <button type="button" data-act="dismiss" aria-label="Dismiss" style="background:transparent;color:#6b7280;border:none;font-size:16px;cursor:pointer;padding:0 4px;line-height:1;">×</button>
    `;
    wrap.addEventListener('click', (ev) => {
        if (ev.target.closest('button[data-act]')) wrap.remove();
    });
    document.body.appendChild(wrap);
    setTimeout(() => { try { wrap.remove(); } catch (_) {} }, 8000);
}

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

    // ── 3D promotion migration (slopsmith#160 PR 3) ──────────────────────
    // Existing users with `vizSelection='default'` (the old built-in 2D
    // highway) are auto-flipped to the bundled 3D Highway exactly once,
    // and a non-modal nag toast offers them "Try the tour" / "Switch
    // back to 2D" the first time they open the player. Users on `auto`
    // are left alone (auto-pick semantics unchanged). Users on a custom
    // viz plugin are left alone. WebGL2 absence falls back via setViz.
    if (saved === 'default' && !_hasPromotedFlag()) {
        const has3D = Array.from(sel.options).some(o => o.value === 'highway_3d');
        if (has3D && _canRun3D()) {
            saved = 'highway_3d';
            try { localStorage.setItem('vizSelection', 'highway_3d'); } catch (_) {}
            _markPromoted();
            _pendingPromotionNag = true;
            // Race guard: if song:ready already fired before _populateVizPicker
            // ran (e.g. a deeplink or a fast-loading song), getSongInfo() will
            // already be non-empty and we'll never receive another song:ready
            // in this session. Show the nag immediately in that case.
            const _si = window.highway && window.highway.getSongInfo();
            if (_si && _si.title) {
                _pendingPromotionNag = false;
                _showPromotionNag();
            }
        } else if (has3D && !_canRun3D()) {
            // 3D registered but WebGL2 absent — promote in name but
            // immediately fall back so we don't ping-pong on every load.
            // Set the flag so we don't try again next reload.
            _markPromoted();
            _showWebGL2FallbackToast();
        }
        // No `highway_3d` option (plugin unloaded?) → leave saved as
        // 'default'. We'll retry the migration once the plugin is back.
    }

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
        // Fresh install (or post-cleanup fallthrough): default to the
        // bundled 3D Highway when available + WebGL2-capable, falling
        // back to Auto otherwise so the arrangement-matching plugins
        // (piano on Keys songs, drums on Drums songs, ...) still take
        // over for non-3D arrangements.
        const has3D = Array.from(sel.options).some(o => o.value === 'highway_3d');
        if (has3D && _canRun3D()) {
            sel.value = 'highway_3d';
            try { localStorage.setItem('vizSelection', 'highway_3d'); } catch (_) {}
            setViz('highway_3d');
        } else {
            sel.value = 'auto';
            try { localStorage.setItem('vizSelection', 'auto'); } catch (_) {}
            if (has3D && !_canRun3D()) { _markPromoted(); _showWebGL2FallbackToast(); }
        }
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
        const _sel = document.getElementById('viz-picker');
        if (_sel) _sel.value = 'default';
        highway.setRenderer(null);
        return;
    }
    if (id === 'auto') {
        try { localStorage.setItem('vizSelection', 'auto'); } catch (_) {}
        _autoMatchViz();
        return;
    }
    // 3D Highway specifically gates on WebGL2. Any future WebGL viz
    // plugin should declare its own probe — for now the bundled 3D
    // Highway is the only viz with this requirement, so the gate is
    // hardcoded. Falling back to 'default' (Classic 2D) keeps the
    // picker in sync; toast informs the user.
    if (id === 'highway_3d' && !_canRun3D()) {
        console.warn('viz picker: WebGL2 unavailable, falling back to Classic 2D Highway');
        _markPromoted();
        _showWebGL2FallbackToast();
        fallbackToDefault();
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
        // 3D Highway gates on WebGL2; if the browser can't run it, skip
        // it during Auto evaluation so the next candidate (or the 2D
        // fallback at the end of the loop) takes over instead of
        // installing a renderer that'll fail at init.
        if (id === 'highway_3d' && !_canRun3D()) continue;
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
    loopA = _audioTime();
    document.getElementById('btn-loop-a').className = 'px-3 py-1.5 bg-green-900/50 rounded-lg text-xs text-green-300 transition';
    updateLoopUI();
}

function setLoopEnd() {
    if (loopA === null) return;
    loopB = _audioTime();
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

async function loadSavedLoop(loopId) {
    const sel = document.getElementById('saved-loops');
    const opt = sel.selectedOptions[0];
    const delBtn = document.getElementById('btn-loop-delete');
    if (!loopId || !opt?.dataset.start) {
        delBtn.classList.add('hidden');
        return;
    }
    loopA = parseFloat(opt.dataset.start);
    loopB = parseFloat(opt.dataset.end);
    await _audioSeek(loopA);
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

async function startCountIn() {
    if (_countingIn) return;
    _countingIn = true;
    if (window._juceMode) {
        await jucePlayer.pause().catch((err) => console.error('[app] jucePlayer.pause error in count-in:', err));
    } else {
        audio.pause();
    }

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
            // Rewind done — set final position and start count.
            // Await the JUCE seek so the engine has repositioned before
            // we start the click track (HTML5 path is synchronous).
            _audioSeek(loopA).then(() => {
                lastAudioTime = loopA;
                highway.setTime(loopA);
                beginCount();
            });
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
                if (window._juceMode) {
                    jucePlayer.play().then((started) => {
                        if (!started) return;
                        isPlaying = true;
                        document.getElementById('btn-play').textContent = '⏸ Pause';
                        window.slopsmith.isPlaying = true;
                        window.slopsmith.emit('song:play', { time: jucePlayer.currentTime });
                    }).catch((err) => console.error('[app] jucePlayer.play error:', err));
                } else {
                    audio.play();
                    isPlaying = true;
                    document.getElementById('btn-play').textContent = '⏸ Pause';
                }
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
    const ct = _audioTime();
    const dur = _audioDuration();
    if (dur && !_countingIn) {
        // JUCE end-of-track: HTML5 fires 'ended'; JUCE needs a manual check
        if (window._juceMode && isPlaying && ct >= dur) {
            isPlaying = false;
            document.getElementById('btn-play').textContent = '▶ Play';
            window.slopsmith.isPlaying = false;
            window.slopsmith.emit('song:ended', { time: ct });
            jucePlayer.pause().catch((err) => console.warn('[app] end-of-track pause error:', err));
        }
        // A-B loop: count-in then seek back to A
        else if (loopA !== null && loopB !== null && ct >= loopB) {
            lastAudioTime = loopB;
            startCountIn();
        }
        // Detect and fix audio time jumps (browser seeking bug; skip for JUCE — position is polled)
        else if (!window._juceMode && isPlaying && Math.abs(ct - lastAudioTime) > 30 && lastAudioTime > 0) {
            console.warn(`Audio time jumped from ${lastAudioTime.toFixed(1)} to ${ct.toFixed(1)}, resetting`);
            audio.currentTime = lastAudioTime;
        }
        lastAudioTime = ct;
        document.getElementById('hud-time').textContent = `${formatTime(ct)} / ${formatTime(dur)}`;
    }
    if (!_countingIn) highway.setTime(ct);
}, 1000 / 60);

// Keyboard shortcuts (player only)
document.addEventListener('keydown', e => {
    if (!document.getElementById('player').classList.contains('active')) return;
    if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
    else if (e.code === 'ArrowLeft') seekBy(-5);
    else if (e.code === 'ArrowRight') seekBy(5);
    else if (e.code === 'Escape') showScreen(_playerOriginScreen || 'home');
    // A/V offset live-calibration — watch the highway and listen to
    // the audio while tuning. Shift for coarse ±50 ms, bare key for
    // fine ±10 ms. Match on e.key (the produced character) rather
    // than e.code (physical-key position) so layouts where `[`/`]`
    // are AltGr combinations (QWERTZ, AZERTY) still fire correctly.
    else if (e.key === '[') { e.preventDefault(); nudgeAvOffsetMs(e.shiftKey ? -50 : -10); }
    else if (e.key === ']') { e.preventDefault(); nudgeAvOffsetMs(e.shiftKey ?  50 :  10); }
});

// ── Edit metadata modal ─────────────────────────────────────────────────
function openEditModal(songData, openerEl) {
    const artUrl = `/api/song/${encodeURIComponent(songData.f)}/art?t=${Date.now()}`;
    const modal = document.createElement('div');
    modal.id = 'edit-modal';
    modal.className = 'slopsmith-modal fixed inset-0 z-[200] flex items-center justify-center bg-black/70 backdrop-blur-sm';
    // role=dialog: assistive tech announces it as a modal; also lets
    // the global keyboard listener's `_isInsideInteractiveControl`
    // bail when typing inside the modal so Library shortcuts don't
    // hijack keys from the edit form.
    modal.setAttribute('role', 'dialog');
    modal.setAttribute('aria-modal', 'true');
    modal.setAttribute('aria-label', 'Edit song metadata');
    // Record the element that triggered the modal so Esc / Cancel can
    // return focus to the exact entry the user was on, even if
    // _lastLibSelected changes before the modal closes.
    // Prefer the explicitly-passed openerEl (from the edit-btn click
    // handler, which has the exact [data-play] parent) over
    // _lastLibSelected, which may not have been updated when the
    // click's stopPropagation() prevented the card-click handler.
    const _emActive = document.querySelector('.screen.active');
    const _emLast = (_lastLibSelected && document.body.contains(_lastLibSelected)
        && _emActive && _emActive.contains(_lastLibSelected)) ? _lastLibSelected : null;
    modal._opener = (openerEl && document.body.contains(openerEl)) ? openerEl : _emLast;
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
                    <input type="text" id="edit-title" value="${_escAttr(songData.t)}"
                        class="w-full bg-dark-600 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 outline-none focus:border-accent/50">
                </div>
                <div>
                    <label class="text-xs text-gray-400 mb-1 block">Artist</label>
                    <input type="text" id="edit-artist" value="${_escAttr(songData.a)}"
                        class="w-full bg-dark-600 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 outline-none focus:border-accent/50">
                </div>
                <div>
                    <label class="text-xs text-gray-400 mb-1 block">Album</label>
                    <input type="text" id="edit-album" value="${_escAttr(songData.al)}"
                        class="w-full bg-dark-600 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-200 outline-none focus:border-accent/50">
                </div>
            </div>
            <div class="flex gap-3 mt-5">
                <button onclick="saveEditModal('${encodeURIComponent(songData.f)}')"
                    class="flex-1 bg-accent hover:bg-accent-light px-4 py-2 rounded-xl text-sm font-semibold text-white transition">Save</button>
                <button data-edit-close
                    class="px-4 py-2 bg-dark-600 hover:bg-dark-500 rounded-xl text-sm text-gray-300 transition">Cancel</button>
            </div>
        </div>`;
    document.body.appendChild(modal);

    // Move focus into the dialog's first text input so background
    // shortcuts (and arrow nav) can't fire on the underlying library
    // entry while the edit form is open. Title is the natural primary
    // field — most edits are correcting spelling there. Caret-end
    // selection so the user can keep typing rather than overtype the
    // current value.
    const titleInput = document.getElementById('edit-title');
    if (titleInput) {
        titleInput.focus({ preventScroll: true });
        try {
            const len = titleInput.value.length;
            titleInput.setSelectionRange(len, len);
        } catch { /* some browsers reject selection on certain input types */ }
    }

    // Trap Tab / Shift+Tab inside the modal so focus can't escape to
    // the library content underneath while the edit form is open.
    _trapFocusInModal(modal);

    // Click on art triggers file input
    document.getElementById('edit-art-wrapper').addEventListener('click', () => {
        document.getElementById('edit-art-file').click();
    });

    // Close on backdrop click or Cancel button; restore focus to opener.
    modal.addEventListener('click', (e) => {
        if (e.target === modal || e.target.closest('[data-edit-close]')) {
            const opener = modal._opener;
            modal.remove();
            const focusTarget = (opener && document.body.contains(opener)) ? opener
                : (_lastLibSelected && document.body.contains(_lastLibSelected) ? _lastLibSelected : null);
            if (focusTarget) focusTarget.focus({ preventScroll: true });
        }
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

    const modal = document.getElementById('edit-modal');
    const opener = modal ? modal._opener : null;
    if (modal) modal.remove();
    // Restore focus to the entry the modal was opened from so subsequent
    // keyboard navigation resumes correctly (same as Esc / Cancel paths).
    const focusTarget = (opener && document.body.contains(opener)) ? opener
        : (_lastLibSelected && document.body.contains(_lastLibSelected) ? _lastLibSelected : null);
    if (focusTarget) focusTarget.focus({ preventScroll: true });
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
        const entry = edit.closest('[data-play]');
        openEditModal(JSON.parse(edit.dataset.edit), entry);
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
    // Song card / row — keep persistent selection in sync with mouse
    // clicks so arrow-keying after a click resumes from where the
    // user clicked, not from a stale highlight.
    // Guard: if the click originated from any <button> inside the
    // entry (e.g. a plugin-provided .sloppak-convert-btn that has no
    // own stopPropagation handler above), don't treat it as a play
    // action. Known action buttons (.fav-btn, .edit-btn, .retune-btn)
    // already return early via stopPropagation() above; this catches
    // any remaining button that bubbles through.
    const card = e.target.closest('[data-play]');
    if (card && !e.target.closest('button')) {
        _setLibSelection(card, { focus: false });
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
            <p class="text-xs text-blue-400/70 mt-1 hidden" id="scan-first-note">First-time import — results are cached for future launches</p>
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
            const firstNote = document.getElementById('scan-first-note');
            if (file) { file.textContent = 'Scan failed: ' + data.error; file.classList.add('text-red-400'); }
            if (prog) prog.textContent = 'Error';
            if (firstNote) firstNote.classList.add('hidden');
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
            const firstNote = document.getElementById('scan-first-note');
            if (bar) bar.style.width = pct + '%';
            if (prog) prog.textContent = `${data.done} / ${data.total} (${pct}%)`;
            if (file) {
                const name = (data.current || '').replace(/_p\.psarc$/i, '').replace(/_/g, ' ');
                file.textContent = name || (data.stage === 'listing' ? 'Listing DLC folder...' : 'Processing...');
            }
            if (firstNote) firstNote.classList.toggle('hidden', !data.is_first_scan);
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
        const firstNote = document.getElementById('scan-first-note');
        if (firstNote) firstNote.classList.toggle('hidden', !data.is_first_scan);
        _scanPollId = setInterval(pollScanStatus, 1000);
    }
    loadLibrary();
}

// ── Plugin loader ───────────────────────────────────────────────────────
function setPluginLoadingState(loading, message) {
    console.log('[slopsmith] setPluginLoadingState', loading, message, new Error().stack.split('\n')[2]);
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

function _makeTimeoutStatus() {
    return { running: false, phase: 'timeout', message: 'Plugin startup timed out', error: null, current_plugin: '', loaded: 0, total: 0 };
}

async function _waitViaSSE(timeoutMs) {
    if (typeof EventSource === 'undefined') return null;
    // Must exceed the server's _SSE_KA_INTERVAL (15 s) — keepalives are data
    // events so onmessage re-arms this timer.  A gap > MSG_DEADLINE_MS means
    // the proxy is buffering or dropping; fall back to polling.  Update this
    // if _SSE_KA_INTERVAL in server.py changes.
    const MSG_DEADLINE_MS = 20000;
    return new Promise((resolve) => {
        let resolved = false;
        let msgDeadlineTimer = null;
        const armDeadline = () => {
            clearTimeout(msgDeadlineTimer);
            msgDeadlineTimer = setTimeout(() => done(null), MSG_DEADLINE_MS);
        };
        const done = (result) => {
            if (resolved) return;
            resolved = true;
            clearTimeout(timer);
            clearTimeout(msgDeadlineTimer);
            es.close();
            resolve(result);
        };
        const timer = setTimeout(() => {
            done(_makeTimeoutStatus());
        }, timeoutMs);
        armDeadline(); // deadline for the first message
        const es = new EventSource('/api/startup-status/stream');
        // Startup is a one-shot connection; treat any error (including transient
        // reconnects that EventSource would normally retry) as a signal to fall
        // back to polling rather than waiting for the browser to retry.
        es.onerror = () => done(null);
        es.onmessage = (event) => {
            armDeadline(); // re-arm on every message — server sends data-level keepalives
            let status;
            try { status = JSON.parse(event.data); } catch { return; }
            if (!status || status.type === 'keepalive') return; // keepalive — deadline re-armed, nothing else to do
            if (!status.phase) return;
            const phase = (status.phase || '').trim();
            const msg = (status.message || '').trim() || 'Loading plugins...';
            const countMsg = status.total > 0 ? ` (${status.loaded || 0}/${status.total})` : '';
            setPluginLoadingState(Boolean(status.running), `${msg}${countMsg}`);
            if (!status.running && (phase === 'complete' || phase === 'error')) done(status);
        };
    });
}

async function _waitViaPolling(timeoutMs) {
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
    return _makeTimeoutStatus();
}

async function waitForPluginStartupComplete(timeoutMs = 180000) {
    const start = Date.now();
    const sseResult = await _waitViaSSE(timeoutMs);
    if (sseResult !== null) return sseResult;
    const remaining = Math.max(0, timeoutMs - (Date.now() - start));
    if (remaining <= 0) {
        return _makeTimeoutStatus();
    }
    return _waitViaPolling(remaining);
}

let _loadPluginsInFlight = false;

async function loadPlugins() {
    if (_loadPluginsInFlight) { console.log('[slopsmith] loadPlugins: in-flight, skipping'); return null; }
    _loadPluginsInFlight = true;
    console.log('[slopsmith] loadPlugins: start');
    let plugins;
    const navContainer = document.getElementById('nav-plugins');
    const mobileNavContainer = document.getElementById('mobile-nav-plugins');
    // Snapshot current nav so we can restore it if the fetch fails.
    const _savedNav = navContainer ? navContainer.innerHTML : null;
    const _savedMobileNav = mobileNavContainer ? mobileNavContainer.innerHTML : null;
    try {
        const resp = await fetch('/api/plugins');
        plugins = await resp.json();
        console.log('[slopsmith] loadPlugins: got', plugins.length, 'plugins');

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
                // flex-col allows the fallback explanation note to appear below
                // the name/badges row when plugin.fallback is set.
                summary.className = 'plugin-settings-summary cursor-pointer select-none px-4 py-3 text-sm font-medium text-gray-300 hover:bg-dark-700/70 transition flex flex-col';
                // Inner row: plugin name/badges (left) + chevron (right).
                const headerRow = document.createElement('span');
                headerRow.className = 'flex items-center justify-between';
                const labelWrap = document.createElement('span');
                labelWrap.className = 'flex items-center gap-2';
                const labelSpan = document.createElement('span');
                labelSpan.textContent = plugin.name || plugin.id;
                labelWrap.appendChild(labelSpan);
                // "Bundled" marker (slopsmith#160). Visually distinguishes
                // plugins that ship with the default container image from
                // user-installed ones so users don't try to remove a core
                // plugin via the manage-plugin flow and brick a feature
                // that's expected to "just work".
                if (plugin.bundled) {
                    const bundledDesc = 'This plugin ships with Slopsmith core and is expected to be present.';
                    const badge = document.createElement('span');
                    badge.className = 'inline-flex items-center gap-1 text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded border border-purple-400/30 bg-purple-500/10 text-purple-300';
                    badge.title = bundledDesc;
                    badge.setAttribute('aria-label', 'Bundled — ' + bundledDesc);
                    badge.setAttribute('role', 'img');
                    badge.innerHTML = `
                        <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                                  d="M12 11c1.657 0 3-1.343 3-3V6a3 3 0 10-6 0v2c0 1.657 1.343 3 3 3zM6 11h12a2 2 0 012 2v6a2 2 0 01-2 2H6a2 2 0 01-2-2v-6a2 2 0 012-2z"/>
                        </svg>
                        Bundled
                    `;
                    labelWrap.appendChild(badge);
                }
                // "Fallback" warning badge: the bundled copy failed to load its
                // routes, so the server fell back to this older user-installed
                // copy.  Warn users so they know the bundled build is broken and
                // can check the server startup log for the root cause.
                if (plugin.fallback) {
                    const fbBadge = document.createElement('span');
                    fbBadge.className = 'inline-flex items-center gap-1 text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded border border-yellow-400/40 bg-yellow-500/10 text-yellow-300';
                    fbBadge.setAttribute('aria-hidden', 'true');
                    fbBadge.innerHTML = '<svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/></svg> Fallback';
                    labelWrap.appendChild(fbBadge);
                }
                // Assemble inner header row: [name/badges (left)] [chevron (right)].
                // Both are placed in headerRow so the fallback note (if any)
                // can sit below the entire row as a second flex-col child of
                // summary, rather than being squeezed inline beside the chevron.
                headerRow.appendChild(labelWrap);
                // Chevron icon — built via setAttributeNS so the SVG sits in
                // the SVG namespace and renders correctly. Plugin label is
                // appended as text above so manifest values can't inject HTML.
                const svgNS = 'http://www.w3.org/2000/svg';
                const svg = document.createElementNS(svgNS, 'svg');
                svg.setAttribute('class', 'w-4 h-4 text-gray-500 transition-transform group-open:rotate-180');
                svg.setAttribute('fill', 'none');
                svg.setAttribute('stroke', 'currentColor');
                svg.setAttribute('viewBox', '0 0 24 24');
                const svgPath = document.createElementNS(svgNS, 'path');
                svgPath.setAttribute('stroke-linecap', 'round');
                svgPath.setAttribute('stroke-linejoin', 'round');
                svgPath.setAttribute('stroke-width', '2');
                svgPath.setAttribute('d', 'M19 9l-7 7-7-7');
                svg.appendChild(svgPath);
                headerRow.appendChild(svg);
                summary.appendChild(headerRow);
                // Fallback explanation note: a visible <p> below the header row,
                // accessible to touch/keyboard users (browser tooltip via title/
                // aria-label alone is hover-only and insufficient). Appended to
                // summary (not labelWrap) so it renders as the second child in
                // summary's flex-col layout, appearing below the name+badges row.
                if (plugin.fallback) {
                    const fbNote = document.createElement('span');
                    fbNote.className = 'block text-xs text-yellow-300/80 mt-1';
                    fbNote.textContent = 'The bundled version failed to start. This user-installed copy is serving as a fallback. Check the server startup log for details.';
                    summary.appendChild(fbNote);
                }
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
        // Restore nav so a failed re-hydration call doesn't leave it blank.
        if (_savedNav !== null && navContainer) navContainer.innerHTML = _savedNav;
        if (_savedMobileNav !== null && mobileNavContainer) mobileNavContainer.innerHTML = _savedMobileNav;
        _loadPluginsInFlight = false;
        return null;
    }
    _loadPluginsInFlight = false;
    return plugins;
}

async function _scheduleStartupRehydration() {
    // Continue polling until the backend startup completes (or a long deadline).
    // Used when the initial waitForPluginStartupComplete() window expired before
    // LOADED_PLUGINS was populated — re-hydrates plugins + viz picker once done.
    const REHYDRATE_TIMEOUT_MS = 10 * 60 * 1000; // 10 min additional window
    const start = Date.now();
    console.log('[slopsmith] _scheduleStartupRehydration: started');
    while (Date.now() - start < REHYDRATE_TIMEOUT_MS) {
        await new Promise((r) => setTimeout(r, 5000));
        try {
            const resp = await fetch('/api/startup-status');
            if (!resp.ok) continue;
            const status = await resp.json();
            console.log('[slopsmith] _scheduleStartupRehydration: poll —', status.phase, 'running:', status.running);
            if (!status.running) {
                if (status.phase === 'complete') {
                    console.log('[slopsmith] Background startup complete — re-hydrating plugins');
                    const plugins = await loadPlugins();
                    _populateVizPicker(plugins);
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
    // Treat the initial page load the same as a screen entry so the
    // restored selection scrolls into view exactly once on hard
    // reload. Without this, the scroll-on-screen-entry flag only
    // ever triggered when the user navigated away and back via
    // showScreen — a hard refresh in tree mode would land on the
    // top of the tree and force the user to scroll back to find
    // their selection.
    _libScrollOnNextRender.home = true;
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
