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
let libView = 'grid';
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

function setLibView(view) {
    libView = view;
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
        loadLibrary(0);
    }, 250);
}

function sortLibrary() {
    _libEpoch++;
    currentPage = 0;
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
        const resp = await fetch('/api/library/stats');
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

    for (const artist of artists) {
        const openClass = q || artists.length <= 5 ? ' open' : '';
        html += `<div class="artist-row${openClass}">`;
        html += `<div class="artist-header" onclick="this.parentElement.classList.toggle('open')">`;
        html += chevron;
        html += `<span class="text-white font-semibold text-sm flex-1">${esc(artist.name)}</span>`;
        html += `<span class="text-xs text-gray-600">${artist.song_count} song${artist.song_count !== 1 ? 's' : ''} · ${artist.album_count} album${artist.album_count !== 1 ? 's' : ''}</span>`;
        html += `</div><div class="artist-body">`;

        for (const album of artist.albums) {
            const artUrl = `/api/song/${encodeURIComponent(album.songs[0].filename)}/art${album.songs[0].mtime ? `?v=${Math.floor(album.songs[0].mtime)}` : ''}`;
            const albumOpen = q || artist.albums.length === 1 ? ' open' : '';
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

function toggleAllArtists(expand) {
    document.querySelectorAll('.artist-row').forEach(el => el.classList.toggle('open', expand));
    document.querySelectorAll('.album-group').forEach(el => el.classList.toggle('open', expand));
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
    // Native folder picker — only present when running inside slopsmith-desktop.
    if (window.slopsmithDesktop && typeof window.slopsmithDesktop.pickDirectory === 'function') {
        document.getElementById('btn-pick-dlc')?.classList.remove('hidden');
    }
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
        }),
    });
    const data = await resp.json();
    document.getElementById('settings-status').textContent = data.message || data.error;
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
// renderScale / showLyrics convention). Falls back to the slider's default.
(function _initVolume() {
    const slider = document.getElementById('volume');
    const label = document.getElementById('vol-label');
    const stored = parseFloat(localStorage.getItem('volume'));
    const v = Number.isFinite(stored) ? stored : parseFloat(slider.value);
    slider.value = v;
    label.textContent = v + '%';
    audio.volume = v / 100;
})();

// Re-sync audio volume from the slider every time a new source finishes
// loading metadata. Belt + suspenders — some combinations of plugin audio-
// graph routing and media-element swaps reset audio.volume to 1.0, which
// would leave the slider showing one value while audio plays at another
// (see slopsmith#54).
audio.addEventListener('loadedmetadata', () => {
    const slider = document.getElementById('volume');
    if (slider) audio.volume = parseFloat(slider.value) / 100;
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
function setVolume(v) {
    audio.volume = v / 100;
    document.getElementById('vol-label').textContent = v + '%';
    localStorage.setItem('volume', String(v));
}
function setSpeed(v) {
    audio.playbackRate = parseFloat(v);
    document.getElementById('speed-label').textContent = parseFloat(v).toFixed(2) + 'x';
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
async function loadPlugins() {
    try {
        const resp = await fetch('/api/plugins');
        const plugins = await resp.json();

        const navContainer = document.getElementById('nav-plugins');
        const mobileNavContainer = document.getElementById('mobile-nav-plugins');
        const settingsContainer = document.getElementById('plugin-settings');

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
                item.onclick = (e) => { e.preventDefault(); ddMenu.classList.add('hidden'); showScreen(screenId); };
                ddMenu.appendChild(item);

                // Mobile nav — flat list
                const ma = document.createElement('a');
                ma.href = '#';
                ma.className = 'text-gray-400 hover:text-white pl-4 text-sm';
                ma.textContent = plugin.nav.label;
                ma.onclick = (e) => { e.preventDefault(); showScreen(screenId); ma.closest('#mobile-menu').classList.add('hidden'); };
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

            // Inject settings section
            if (plugin.has_settings && settingsContainer) {
                const settingsDiv = document.createElement('div');
                settingsDiv.id = `plugin-settings-${plugin.id}`;
                settingsContainer.appendChild(settingsDiv);

                const settingsResp = await fetch(`/api/plugins/${plugin.id}/settings.html`);
                settingsDiv.innerHTML = await settingsResp.text();
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
    }
}

// Load library on start
loadPlugins().then(() => {
    setLibView('grid');
    checkScanAndLoad();
});
