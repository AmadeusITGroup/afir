"""
The Configuration tab: read, change, import and export the app's YAML config.

Three surfaces, in increasing order of how much damage they can do:

1. **The form** — one typed, range-checked control per ``config_store.SECTIONS`` descriptor. A
   click cannot introduce a key that does not exist or a value outside its bounds.
2. **The raw editor** — the whole file as text, for the sections the form does not model
   (backends, log sources, plugins). Parses the candidate, keeps the previous as ``.bak``.
3. **Import** — every file is validated before *any* is written: importing a matched pair where
   only the first is valid leaves the app running half of somebody else's environment.

Three things the UI states rather than implies. A redacted secret means "keep what's on disk",
so saving a field still holding ``__redacted__`` is a server-side no-op and a form that PUTs its
own read back cannot overwrite a password; an ``${ENV_VAR}`` reference reads back verbatim,
because the operator needs to see which variable a backend uses. Live vs. restart is labelled
per field, some keys being re-read on every use and others consumed once at startup. And unset
is not blank: a key absent from the file is still in force through ``.get(key, default)``, so
those fields render the default with an "unset" tag and saving one inserts the key.

There is no authentication on this API (see ``docs/API.md``), so redaction lives server-side in
``config_store`` and covers the structured view and the raw text alike.
"""

CONFIGURE_HTML = r"""    <div class="panel subbar">
      <h2>Configuration <span class="sub" id="cfgDir"></span></h2>
      <div class="banner" id="cfgSecurityNote">
        Literal secrets are redacted as <code id="cfgPlaceholder">__redacted__</code> and never leave the
        server. Leaving that placeholder in a field keeps whatever is on disk. <code>${ENV_VAR}</code>
        references are shown as-is — that is a variable <em>name</em>, not a credential.
      </div>
      <div class="row" style="margin-top:0">
        <div class="seg">
          <label><input type="radio" name="cfgview" value="form" checked/> Form</label>
          <label><input type="radio" name="cfgview" value="raw"/> Raw YAML</label>
          <label><input type="radio" name="cfgview" value="io"/> Import / export</label>
        </div>
        <button class="btn small" id="cfgReload" title="Re-read the files from disk, discarding unsaved edits"><svg class="ico"><use href="#i-refresh"/></svg> Reload from disk</button>
        <div class="spacer"></div>
        <span class="statusline" id="cfgStatus"></span>
      </div>
    </div>

    <div class="panel" id="cfgFormView">
      <div class="banner ok" id="cfgResult" hidden></div>
      <div class="cfggrid">
        <nav class="cfgnav" id="cfgNav"></nav>
        <div id="cfgSections"><div class="empty">Loading configuration…</div></div>
      </div>
      <div class="savebar">
        <button class="btn primary" id="cfgSave" disabled><svg class="ico"><use href="#i-upload"/></svg> Save changes</button>
        <button class="btn" id="cfgRevert" disabled><svg class="ico"><use href="#i-refresh"/></svg> Discard edits</button>
        <span class="statusline" id="cfgDirtyCount"></span>
        <div class="spacer"></div>
        <span class="statusline">
          <span class="hbadge tag-live">live</span> takes effect on the next stage that reads it ·
          <span class="hbadge tag-restart">restart</span> needs a process restart
        </span>
      </div>
    </div>

    <div class="panel" id="cfgRawView" hidden>
      <div class="banner">
        The whole file, for the sections the form does not model — backends, log sources, RAG
        source registry, plugins. The candidate text is parsed before it replaces anything and
        the previous version is kept as <code>&lt;name&gt;.bak</code>. A whole-file save is
        always treated as restart-required.
      </div>
      <div class="row" style="margin-top:0">
        <select id="cfgRawFile"></select>
        <button class="btn" id="cfgRawSave"><svg class="ico"><use href="#i-upload"/></svg> Save file</button>
        <button class="btn small" id="cfgRawExport" title="Download this file, redacted — safe to share or commit as a template"><svg class="ico"><use href="#i-download"/></svg> Export</button>
        <div class="spacer"></div>
        <span class="statusline" id="cfgRawStatus"></span>
      </div>
      <textarea id="cfgRaw" spellcheck="false"></textarea>
    </div>

    <div class="panel" id="cfgIoView" hidden>
      <div class="banner">
        Every file is validated before any is written. An export is redacted, and the importer
        refuses text still carrying the placeholder — so a redacted export cannot be re-imported
        over real credentials by accident.
      </div>
      <h2>Export</h2>
      <div class="row" id="cfgExportRows"></div>
      <h2 style="margin-top:1.2rem">Import</h2>
      <div class="row">
        <input type="file" id="cfgImportFiles" multiple accept=".yaml,.yml"/>
        <button class="btn primary" id="cfgImport" disabled><svg class="ico"><use href="#i-upload"/></svg> Import selected</button>
        <span class="statusline" id="cfgImportStatus"></span>
      </div>
      <div id="cfgImportList"></div>
    </div>
"""
