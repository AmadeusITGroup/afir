"""
The Knowledge tab: browse, edit, create and check a domain knowledge pack.

Every domain-specific fact lives in a pack, the engine being deliberately generic, so this tab
is what makes authoring one possible without shell access to the deployment. Five surfaces, in
increasing order of how much they can change:

1. **Files** — a tree on the left, one file in a text area on the right. Save is a whole-file
   replace and keeps the previous version.
2. **Check** — the diagnostics, at three severities. Errors are the ones that make a pack
   silently do nothing.
3. **History** — every earlier version, with restore.
4. **New / import** — scaffold from the checked-in template, or drop in a set of files.
5. **Assist** — describe a change and get a proposed set of edits to approve.

Four things this tab states rather than implies. A broken file does not announce itself at run
time: the loader swallows a parse failure and hands the engine an empty document, so a pack with
a broken catalogue loads with zero sources and every check resolves to "unknown", which the
report reads as *the data had nothing to say*. A save that would produce that state is refused,
and every write reports the checker's findings plus ``restart_required``, editing not reloading
the running process. Nothing is deleted unless asked: saving keeps the prior version, deleting
needs its own confirmed action, restoring is itself undoable. And there is no authentication on
this API (see ``docs/API.md``), so the bounds are the server's path guard and suffix allowlist,
which refuse a path leaving the pack rather than rewriting it.

The size limit is load-bearing: a pack can hold generated inventory files of several hundred
kilobytes and round-tripping one through a browser text area loses it, so those render read-only
with a download button and a pointer to the assistant's line-range edit.
"""

KNOWLEDGE_HTML = r"""    <div class="panel subbar">
      <h2>Knowledge <span class="sub" id="pkRoot"></span></h2>
      <div class="banner" id="pkSafetyNote">
        Every save keeps the previous version, and nothing is deleted unless you ask.
        A file that would load as empty is refused — that state is invisible at run time,
        because the loader reports it as data having nothing to say. Changes need a
        <strong>restart</strong> to take effect. This API has no authentication.
      </div>
      <div class="row" style="margin-top:0">
        <select id="pkPack" title="Which pack to edit — the one in use is marked"></select>
        <div class="seg">
          <label><input type="radio" name="pkview" value="files" checked/> Files</label>
          <label><input type="radio" name="pkview" value="check"/> Check</label>
          <label><input type="radio" name="pkview" value="history"/> History</label>
          <label><input type="radio" name="pkview" value="io"/> New / import</label>
          <label><input type="radio" name="pkview" value="assist"/> Assist</label>
        </div>
        <button class="btn small" id="pkReload" title="Re-read this pack from disk, discarding unsaved edits"><svg class="ico"><use href="#i-refresh"/></svg> Reload from disk</button>
        <div class="spacer"></div>
        <span class="statusline" id="pkStatus"></span>
      </div>
      <div class="row" style="margin-top:.4rem">
        <span class="statusline" id="pkCounts"></span>
        <div class="spacer"></div>
        <span class="statusline" id="pkHealth"></span>
      </div>
    </div>

    <div class="panel" id="pkFilesView">
      <div class="banner ok" id="pkResult" hidden></div>
      <div class="cfggrid">
        <nav class="cfgnav pktree" id="pkTree"><div class="empty">Loading&#8230;</div></nav>
        <div>
          <div class="row" style="margin-top:0">
            <span class="statusline mono" id="pkFileMeta">Select a file on the left.</span>
            <div class="spacer"></div>
            <input id="pkNewPath" placeholder="new file path, e.g. shared/checks/mine.yaml" style="flex:1;min-width:14rem"/>
            <button class="btn small" id="pkNewFile" title="Create a new file at the path on the left">+ New file</button>
            <button class="btn small" id="pkDownload" title="Download this file as it is on disk" disabled><svg class="ico"><use href="#i-download"/></svg> Download</button>
          </div>
          <div id="pkFileNote"></div>
          <textarea id="pkEditor" spellcheck="false" placeholder="Select a file on the left to edit it."></textarea>
          <div id="pkFileDiags"></div>
        </div>
      </div>
      <div class="savebar">
        <button class="btn primary" id="pkSave" disabled><svg class="ico"><use href="#i-upload"/></svg> Save file</button>
        <button class="btn" id="pkRevert" disabled><svg class="ico"><use href="#i-refresh"/></svg> Discard edits</button>
        <button class="btn danger" id="pkDelete" disabled title="Removes the file. Its content stays in History and can be restored."><svg class="ico"><use href="#i-x"/></svg> Delete file</button>
        <span class="statusline" id="pkDirty">no changes</span>
        <div class="spacer"></div>
        <span class="statusline">Saving keeps the previous version &#183; a restart is needed to use the change</span>
      </div>
    </div>

    <div class="panel" id="pkCheckView" hidden>
      <div class="banner">
        <strong>error</strong> — the pack does not work, or works while telling you something
        untrue. <strong>warning</strong> — it works, but a human is misled or a declaration is
        doing nothing at all. <strong>note</strong> — a fact about the pack, not a defect.
        A warning never blocks a save.
      </div>
      <div class="row" style="margin-top:0">
        <button class="btn small" id="pkRecheck"><svg class="ico"><use href="#i-refresh"/></svg> Check again</button>
        <div class="spacer"></div>
        <span class="statusline" id="pkCheckStatus"></span>
      </div>
      <div id="pkDiags"><div class="empty">Nothing checked yet.</div></div>
    </div>

    <div class="panel" id="pkHistoryView" hidden>
      <div class="banner">
        Every version this editor replaced, newest first. Restoring is itself recorded, so
        undoing an undo is possible &#8212; and a restore is allowed even when the stored
        version does not parse, because that is exactly the moment you need it.
      </div>
      <div class="row" style="margin-top:0">
        <select id="pkHistFilter" title="Show every version, or only one file's"></select>
        <button class="btn small" id="pkHistRefresh"><svg class="ico"><use href="#i-refresh"/></svg> Refresh</button>
        <div class="spacer"></div>
        <span class="statusline" id="pkHistStatus"></span>
      </div>
      <div id="pkHistRows"><div class="empty">No earlier versions yet.</div></div>
      <div id="pkHistPreview"></div>
    </div>

    <div class="panel" id="pkIoView" hidden>
      <h2>New pack</h2>
      <div class="banner">
        A new pack is a copy of the checked-in template, which already passes every check &#8212;
        so your first edit has a working baseline to differ from. The word list is required: it
        is what proves the engine itself never names this domain, and a pack without one breaks
        that guarantee for every pack installed.
      </div>
      <div class="banner" data-user-only hidden>
        Creating a pack is an administrator's action. Editing one is not &#8212; every file in an
        existing pack is yours to change, saved as your own draft. What is refused here is a
        <em>new</em> pack, because a pack is what a run loads and there is one of those.
      </div>
      <div class="row" data-admin-only>
        <input id="pkNewName" placeholder="pack name (letters, digits, _ and -)" style="flex:1"/>
        <input id="pkNewVocab" placeholder="the domain's own nouns, comma separated" style="flex:2"/>
        <button class="btn primary" id="pkScaffold">+ Create pack</button>
        <span class="statusline" id="pkScaffoldStatus"></span>
      </div>
      <h2 style="margin-top:1.2rem">Import files</h2>
      <div class="banner">
        Every file is checked before any is written. An import that would break one file writes
        none of them &#8212; a half-applied change leaves a pack broken in a way neither version
        would explain. Existing files are replaced, with their previous version kept.
      </div>
      <div class="banner" data-user-only hidden>
        Importing writes files into the shared pack, so it is an administrator's action. To bring
        in your own version of a file, open it in the editor and paste the text: that lands in
        your drafts.
      </div>
      <div class="row" data-admin-only>
        <input type="file" id="pkImportFiles" multiple accept=".yaml,.yml,.md,.txt"/>
        <input id="pkImportPrefix" placeholder="destination folder inside the pack (optional)" style="flex:1"/>
        <button class="btn primary" id="pkImport" disabled><svg class="ico"><use href="#i-upload"/></svg> Import selected</button>
        <span class="statusline" id="pkImportStatus"></span>
      </div>
      <div id="pkImportList"></div>
    </div>

    <div class="panel" id="pkAssistView" hidden>
      <div class="banner" id="pkAssistNote">
        Describe the change you want. The assistant reads this pack &#8212; and any files or
        images you attach &#8212; then <strong>proposes</strong> a set of edits with a diff.
        Nothing is written until you approve it, and you can edit the proposal first or send it
        back with a correction.
      </div>
      <textarea id="pkAsk" placeholder="e.g. add a check that the same operator did not act twice within an hour, and wire it into the existing ruleset"></textarea>
      <div class="row">
        <select id="pkFocus" multiple size="4" title="Files worth reading first — optional, the assistant can find its own"></select>
        <div>
          <input type="file" id="pkAttach" multiple/>
          <div class="statusline" id="pkAttachList">Attach documents or images: notes, a specification, a diagram.</div>
        </div>
        <button class="btn primary" id="pkRun"><svg class="ico"><use href="#i-play"/></svg> Ask</button>
        <button class="btn" id="pkAssistStop" hidden><svg class="ico"><use href="#i-stop"/></svg> Stop</button>
      </div>
      <div class="row">
        <span class="hbadge" id="pkToolMode" hidden></span>
        <span class="hbadge" id="pkImageMode" hidden></span>
        <div class="spacer"></div>
        <span class="statusline" id="pkAssistStatus"></span>
      </div>
      <div id="pkTrail"></div>
      <div id="pkProposal"></div>
      <div class="banner" data-user-only hidden>
        Anyone may ask, read the proposal and read its diff. Only an administrator can have the
        assistant write it: an assistant plan touches several files at once, and a partly applied
        one is the state neither version explains. Copy a proposed file into the editor and save
        it there to keep it as your own draft.
      </div>
      <div class="row" id="pkProposalBar" hidden>
        <button class="btn primary" id="pkApply" data-admin-only><svg class="ico"><use href="#i-check"/></svg> Apply these edits</button>
        <label class="toggle" data-admin-only><input type="checkbox" id="pkAllowDelete"/> also allow deletions</label>
        <button class="btn" id="pkEditFirst" title="Change the proposed text before it is written"><svg class="ico"><use href="#i-edit"/></svg> Edit first</button>
        <button class="btn danger" id="pkRejectPlan"><svg class="ico"><use href="#i-x"/></svg> Send back</button>
        <span class="statusline" id="pkApplyStatus"></span>
      </div>
      <textarea id="pkGuidance" hidden placeholder="What is wrong with this proposal and what to do instead. This is added to the request on the retry — sending it back with no correction produces the same proposal again."></textarea>
    </div>
"""
