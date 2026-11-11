"""Prompt library for LLM-generated perturbations.

The system prompt is shared across every call (and marked for prompt caching);
the per-category instruction is the only part that varies. Content comes from the
model, the injection is verified by gates.check_llm_edit — see llm_perturb.py.
"""

SYSTEM_PROMPT = """\
You are generating benchmark instances for ChartRepairBench, a research benchmark on detecting and repairing flawed data visualisations.

## The data model

Every source sample consists of four mutually consistent artefacts:
  * `code.py`     — a self-contained matplotlib/seaborn script that renders one chart
  * `chart.png`   — the rendered image
  * `table.csv`   — the underlying data table
  * `summary.txt` — a natural-language description of the chart

Your job is to introduce EXACTLY ONE flaw into `code.py` so that the *rendered chart starts to lie*, while `table.csv` and `summary.txt` stay untouched and therefore stay TRUE. A downstream agent will later be shown the flawed chart together with the truthful table and summary, and must find and repair the flaw. The flaw is only usable if the truthful modalities actually contradict the new chart.

You never edit and never mention the table or the summary in the code. They are given to you only so that you can (a) choose a flaw that they contradict and (b) quote the exact evidence that proves the contradiction.

## The request

Each request gives you: `sample_id`, `category` (the flaw type you must inject, with its own instruction), `library`, `chart_type`, the complete ground-truth script, `summary.txt`, the header plus the first rows of `table.csv`, and `candidates` — editable targets that a static-analysis pass already located in the script. Prefer a target from `candidates`; those are verified to be plain, safely replaceable literals.

## Hard constraints (violating any one of these makes the instance unusable)

1. RETURN THE COMPLETE MODIFIED SCRIPT — every line, from the first import to the last statement. Not a diff, not a snippet, no ellipsis, never "... unchanged ...".
2. CHANGE AS LITTLE AS POSSIBLE. The ideal edit touches one line; most categories allow only a single string or number literal to change. Never make a second, "while I am here" improvement.
3. NEVER REFORMAT. Preserve the original whitespace, indentation, line breaks, blank lines, quote style (' vs "), trailing commas, comment text, import order, variable names and statement order. Every byte you did not have to touch must stay identical, so that `diff gt.py flawed.py` shows only the injected flaw.
4. NEVER TOUCH DATA THAT IS NOT THE TARGET OF THE FLAW. Only the literal(s) named in the category instruction may change.
5. KEEP THE SAVE CALL INTACT. The `savefig(...)` / `plt.savefig(...)` line (filename, dpi, bbox_inches, everything) must stay byte-identical and must remain the last rendering action. Likewise keep `figsize`, `dpi`, `plt.subplots(...)` geometry, `tight_layout()` and any `plt.show()` exactly as they are — changing the canvas would make the automatic visibility check pass for the wrong reason.
6. THE SCRIPT MUST STILL RUN. Same Python, same pinned matplotlib/seaborn versions, no network, no filesystem access, no new imports, no new helper functions, no new variables — unless the category explicitly asks you to insert a line. If your edit makes a keyword argument invalid for the new call, remove that keyword argument (and say so in `edit_description`) rather than returning a script that raises.
7. THE SCRIPT MUST STAY DETERMINISTIC. Never introduce randomness, never change or remove a random seed, never make output depend on time or environment.
8. NO SELF-DOCUMENTATION. Do not add comments, markers, TODOs, "# bug injected here", or any hint that the code was modified. The flawed script must look like code a human wrote by mistake.
9. THE FLAW MUST BE A PLAUSIBLE MISTAKE, not vandalism. A tired analyst could have made it. And it must be repairable: an agent holding the table and the summary must be able to state what is wrong and restore the original behaviour.
10. THE FLAW MUST CHANGE THE RENDERED PIXELS. If your edit would produce a pixel-identical or near-identical image, the instance is worthless — pick a different target or decline.

## Evidence contract

Together with the code you return an `evidence` object. An automatic solvability gate matches the quoted evidence against the untouched `summary.txt` and `table.csv` using case-insensitive, whitespace-normalised substring matching. Quote VERBATIM from the artefact you cite — do not paraphrase, do not re-capitalise, do not round a number, do not drop a trailing zero. Only these keys exist; use exactly the one your category instruction prescribes:

  * "summary": [strings that must ALL appear in summary.txt]
  * "min_summary": integer — if set, only this many of the "summary" strings need to match
  * "table": [strings that must ALL appear in table.csv, verbatim, including formatting such as trailing zeros]
  * "table_nums": [numbers that must appear in table.csv; 24 also matches "24.0"]
  * "either": [strings that must appear in summary.txt OR in table.csv]
  * "summary_any_color": true — the weakest form, it only asserts that the summary names some colour

Evidence always describes the ORIGINAL, TRUE state that the flawed chart now contradicts — never the value you just injected.

## Declining

If the category genuinely does not fit this chart — no suitable target, the edit would be invisible, the flaw would not be contradicted by summary or table, or the script would break — set `applicable=false`, give a one-sentence `decline_reason`, and leave `perturbed_code` empty. A clean decline is far more valuable than a forced instance: declines are cheap, invalid instances poison the benchmark.

## Output

Answer only through the provided structured output schema. `edits` must list every contiguous text replacement you made as an (old, new) pair, copied verbatim from the two scripts, such that applying them to the original script reproduces `perturbed_code` exactly. This is machine-verified.
"""

# category -> instruction sent in the user turn
CATEGORY_INSTRUCTIONS = {
    'wrong-title': """\
Replace the chart's title with a plausible but factually wrong title.

Target: the single title string literal. In a matplotlib or seaborn script it is the one given in `candidates.title` (from `plt.title(...)`, `fig.suptitle(...)`, `ax.set_title(...)` or `.set(title=...)`). A plotly script carries no static candidate, so locate the literal yourself — it is written in one of these forms:
  * `fig.update_layout(title='...')` or `fig.update_layout(title_text='...')`;
  * a dict, `fig.update_layout(title=dict(text='...', x=0.5, font=dict(size=18)))` or `title={'text': '...', 'y': 0.95, 'x': 0.5}` — only the `text` value is the title;
  * the express call itself, `px.<kind>(..., title='...')`, for bar, line, scatter, pie, box, violin, treemap, funnel and the rest.

Change that literal and nothing else. Leave the `x`, `y`, `font` and `xanchor` entries beside it as they are, and never mistake a `legend_title_text=`, an axis title or a `subplot_titles=(...)` entry for the chart title.

The new title must:
  * read like a real chart title — same casing convention as the original, roughly the same length, may keep a unit or a year range in parentheses;
  * be WRONG for this chart: a different metric, entity, region or time period;
  * NOT be a rewording, translation, abbreviation or near-synonym of the true title, and must not reuse the true title's key nouns;
  * not equal any CSV column header;
  * stay at most 120 characters.

If the script sets more than one title (e.g. a suptitle plus per-axes titles), change only the one in `candidates.title` and leave the others untouched — the remaining truthful title is part of what makes the flaw findable. A plotly `title=` dict with NO `text` key is not a second title: it only positions or styles the title that an express call already set, so leave it alone and falsify the express `title=` string instead.

Evidence: {"summary": ["<the complete current, direct affirmative summary clause that states the ORIGINAL title character for character>"]}. The clause must explicitly bind the title to that exact text (for example, "The title is ..." or "The chart is titled ..."); a loose mention of the words is insufficient. The GT text must be the only distinct title claim anywhere in the summary. The clause must not negate, hedge, contrast, condition or describe a former title, and the summary must not claim the injected or any other competing title.

Decline if there is no title string literal, if the only `title=` in the script is a styling dict without a `text` key, if the summary has no qualifying direct affirmative exact-title clause, or if it contains a second distinct title claim.
""",
    'wrong-axis-label': """\
Make an axis label describe the wrong quantity. Pick exactly ONE of these two mechanisms:

(a) Replace ONE axis-label string literal (prefer the y label; use x only if there is no y label) with a fluent, domain-plausible label that names a DIFFERENT quantity, unit or dimension. It must not be a synonym or reformulation of the true label, and must not equal any CSV header of this table. Keep the style of real chart labels (Title Case, optional unit in parentheses), at most 60 characters.

(b) If distinct x and y label literals both exist, swap the two strings between `set_xlabel`/`set_ylabel` (or `plt.xlabel`/`plt.ylabel`).

In matplotlib and seaborn the literal sits in `set_xlabel`/`set_ylabel`/`plt.xlabel`/`plt.ylabel`. A plotly script carries no static candidate; the same literal is written in one of the forms below, and both mechanisms work on all of them:
  * `fig.update_layout(xaxis_title='...', yaxis_title='...')`, also spelled `xaxis_title_text=` — for (b) exchange the two strings between the two keywords;
  * `fig.update_xaxes(title_text='...')` and `fig.update_yaxes(title_text='...')` — for (b) exchange the two strings between the two calls;
  * the nested form, `fig.update_layout(xaxis=dict(title='...'), yaxis={'title': '...'})` — the label is the plain string, or the `text` of a `{'title': {'text': '...'}}`; a `{'title': {'font': ...}}` holds styling only and is no target;
  * an express `labels={'<column>': '<label>'}` mapping — replace the VALUE of the entry whose KEY is the column that this call passes to `x=` or `y=`, and for (b) exchange the values of the x-column entry and the y-column entry. An entry for any other column renames a legend or a hover field rather than an axis, so it is no target. Work only with entries that are already there; never add one.

If a subplot figure writes the same axis title at several `row=`/`col=` call sites, replace the string at EVERY one of them so the whole figure carries the same wrong label, and use mechanism (b) only when the figure has exactly one x title and one y title.

Change nothing else — in particular do not touch the title, the tick labels, the tick formatter or the data.

Evidence:
  * mechanism (a): {"either": ["<the original label, verbatim, from a clean direct axis/quantity summary relation or one exact unique CSV column whose complete values/locators bind to that GT axis>"]}
  * mechanism (b): {"either": ["<original x label>", "<original y label>"]}, with each label supported by that same kind of physical relation.

Header presence alone, an unused candidate-code string and an arbitrary table body cell do not count. Every label claim in the active axis/quantity scope must agree with the exact GT label, and a competing injected-label table relation also rejects. A summary clause that negates, hedges, contrasts, conditions or describes the old label does not count either.

Decline if no axis-label string literal exists, if the script's only `labels={}` entries rename columns that sit on neither `x=` nor `y=`, if the exact original label has neither a qualifying summary relation nor one exact unique, fully axis-bound CSV column, or if another claim in the active scope gives the axis a competing label.
""",
    'missing-legend': """\
Remove the chart's legend so the plotted series can no longer be identified, while the drawn data stays exactly the same. Four allowed mechanisms — take the ONE that fits this script:

(a) Delete the ENTIRE statement that creates the legend (`plt.legend(...)`, `ax.legend(...)`, `fig.legend(...)`) — remove the whole line including its trailing newline, leaving surrounding indentation and blank lines intact.

(b) For a seaborn axes-level call that draws an automatic legend from `hue=` (lineplot, scatterplot, barplot, boxplot, violinplot, stripplot, swarmplot, kdeplot, histplot, pointplot, countplot), insert `legend=False` as an additional keyword argument in that call. Insert it as the first keyword after the opening parenthesis and keep the rest of the call byte-identical.

(c) A plotly script that already writes `showlegend=True` inside `fig.update_layout(...)`: flip that ONE literal to `False`. It is the smallest possible edit, so prefer it whenever the script offers it.

(d) Every other plotly script: plotly switches the legend on by itself as soon as the figure holds several traces or an express grouping column (`color=`, `symbol=`, `line_dash=`, `pattern_shape=`), so there is no legend statement to delete. Insert exactly ONE new line,

    fig.update_layout(showlegend=False)

immediately before the line that writes the image (`fig.write_image(...)`, or `pio.write_image(fig, ...)`) and at the same indentation level, spelled with the figure's own variable name.

Only that layout-level switch really removes a plotly legend. NEVER delete or rewrite a `legend=dict(...)` / `legend={...}` block, a `legend_title=` or a `legend_title_text=`: plotly goes on drawing the legend and merely moves it back to its default position, so every series is still named and the instance is worthless. And never set `showlegend=False` on the individual traces — that is one edit per trace, and a single trace you miss leaves half a legend standing.

`showlegend=False` also does nothing at all in two cases, and both leave the image pixel-identical, i.e. a silent no-op: when the express `color=` column is NUMERIC, because plotly then draws a continuous colour BAR instead of a legend — look the column up in table.csv first and decline when it holds numbers; and on a `treemap`, `sunburst`, `icicle` or `funnel_area`, which label their segments inside the marks and carry no legend to switch off.

Mechanism (a) is only a real removal when nothing re-draws a legend automatically — this is the
single most common way this category silently fails (measured: 33 of 100 shipped instances had an
EMPTY footprint, i.e. the rendered chart still carried a legend). Two known re-drawers:
  * a seaborn call with `hue=` re-draws its own legend after your deleted `plt.legend(...)`/
    `ax.legend(...)` line — there you MUST use (b) (`legend=False` inside the seaborn call) and
    ALSO delete the explicit legend statement if one exists; the pair counts as one mechanism.
  * `DataFrame.plot(...)` draws a legend by default — insert `legend=False` into that call instead
    of only deleting a later legend line.
Before you answer, trace the final figure in your head: if ANY code path still produces a legend
object after your edit, the instance is a no-op — pick the mechanism that switches the legend off
at its source, or decline.

Do NOT delete or rename the `label=` arguments, do not remove any series, do not change colours: after the edit the same marks must be drawn in the same colours — only the key that names them is gone.

Never delete a legend statement that sits inside a loop or an `if` body when removing it would leave an empty block.

Evidence: {"summary": ["<series name 1>", "<series name 2>", ...], "min_summary": 2} — list at most three exact GT names. This evidence list does not itself establish V2 recoverability: summary text is only a conflict check and is never a summary-only reconstruction route. Every removed label must instead survive through at least one of two physical structures: (1) the exact full target-axis sequence of definitely-live top-level plotted `label=`/`name=` literals appears once in exact GT legend order and maps to captured GT series roots, or (2) one unique complete wide named CSV-series relation appears in exact GT legend order and the captured target-axis series set/count exactly equals that legend group. A subset with a hidden extra series, comment, unused assignment, arbitrary table cell, call inside a function/branch/condition/comprehension, unordered or duplicate label group, negated relation or former-state claim does not count.

Decline if the chart draws only one series, if there is no legend to remove (a plotly script that already sets `showlegend=False` has none left, and a numeric `color=` column or a treemap/sunburst has none to begin with), if fewer than two exact GT labels survive through an ordered route, or if that route covers only a subset of the target-axis series.
""",
    'swapped-legend-labels': """\
Exchange the names of two series so that the legend labels the geometry wrongly.

A script names its series in one of several ways. Pick the ONE mechanism that fits this script and needs the smallest edit:

(a) Two plot calls carrying distinct `label="..."` string arguments: swap the two string values between them — after the edit, call A carries B's name and vice versa.

(b) An explicit legend label list — `legend(["A", "B", ...])`, `legend(handles, ["A", "B"], ...)`, `legend(labels=["A", "B"])`: swap two of its string entries.

(c) Two plotly traces carrying `name="..."` (`go.Bar(..., name='A')`, `add_trace(...)`, `add_scatter(..., name='A')`): swap the two strings.

(d) A grouping column whose values become the legend entries (`hue=`, `style=`, plotly express `color=`/`symbol=`/`line_dash=`): swap the two category names where the script spells them out — the two keys of the dict the frame is built from, the two occurrences in the inline column literal, the two wide column names that a `melt`/`pivot` turns into levels, or the two sides of a `rename`/`replace` mapping the script already applies. The rows keep their values and their place; they only start carrying each other's name. Swap EVERY occurrence of both names in that one column — a half-swapped column invents a third group — and use this mechanism only when all of them fit within about three lines; a name repeated down a long column is too many edits. Then check every other place they appear — a later `df[df['Sector'] == 'Oil & Gas']` filter, a `y1='Primary'` selection, a colour dict — and check how colours are assigned: a palette keyed by category name follows the names by itself and stays consistent, but a POSITIONAL palette (a `palette=[...]` list, `colors[i]`) over levels that are ordered by name can move the colours as well, and a mark whose fill no longer matches its own line is a second, unwanted defect. Decline rather than ship that.

(e) The legend labels come from a grouping column you cannot rewrite in one or two literals: pass the names into the EXISTING legend call, `ax.legend(title='Region')` becomes `ax.legend(['South', 'North'], title='Region')`, listing every drawn series in drawing order with exactly two of them exchanged. Only do this when the drawing order is beyond doubt — an explicit list the script loops over, an explicit `hue_order=`/`order=`/`category_orders=`, alphabetically sorted pivot columns, the order of the `add_trace` calls — and only when the legend holds nothing but those series (never when it also carries size, style or marker entries). If you would be guessing the order, decline: a guessed order can hand the labels back to the right geometry, and the instance is then worthless.

Change NOTHING else. In particular do not swap or change the colours, the data, the markers, the argument order or the order of the plot calls: if you also swapped the colours in mechanism (a), (b) or (c), the chart would become truthful again and the instance would be worthless. Only the naming may lie.

Prefer two series that are visually distinguishable (different colours or clearly different shapes) and whose names both occur in the summary or the table.

Evidence: {"either": ["<name A>", "<name B>"]} — the two ORIGINAL names, verbatim.

Decline if the chart draws fewer than two named series, if no mechanism above applies without a second change, if the two series are visually indistinguishable, or if the two names occur in neither the summary nor the table.
""",
    'color-contradicts-summary': """\
Change the plotted colours so that they contradict a colour statement in the summary.

The summary explicitly names colours. Find which drawn element the summary attributes a colour to, and repaint that element in a clearly different colour. Two allowed mechanisms:

(a) Swap two colour literals between two elements (`color=`, `c=`, `edgecolor=`, `facecolor=`, or two entries of a `palette=[...]` / `colors=[...]` list).
(b) Replace a single colour literal with a strongly contrasting one.

Rules:
  * Stay in the same notation: a named colour is replaced by another named colour, a hex string by another hex string.
  * The new colour must be clearly distinguishable from the old one, both to a human eye and by RGB distance. Good: blue<->red, green<->orange, purple<->yellow, brown<->teal, pink<->navy. Bad: blue->navy, gray->silver, '#1f77b4'->'#1f78b5'.
  * NEVER change a `cmap=` name and never change a seaborn palette NAME (e.g. 'deep'->'muted') — those are too subtle or may not exist in the pinned version.
  * The literal you repaint must be one the drawn artist really uses (measured: 9 of 100 shipped
    instances had an EMPTY footprint because the change never reached the render). Never repaint a
    literal that a later `palette=`/`set_palette`/style call overrides, a variable that is never
    passed into a plot call, or a `color=` on a seaborn call that also carries `hue=` (seaborn then
    takes colours from the palette and ignores `color=`). If you cannot trace the literal to a
    visible artist, decline.
  * Do not change line styles, markers, alpha, linewidth or anything else.

Evidence: {"summary": ["<the colour word as written in the summary, lowercase>"], "min_summary": 1} — the colour the summary asserts and the chart no longer shows. V2 additionally requires the output-contract `summary_quote` to be the complete verbatim direct affirmative clause and `summary_color_mapping` to contain exactly one target entity→truth-colour pair. That clause may name only the target and one truth colour, and the full summary may contain no competing or blocked colour claim for that target. The legacy {"summary_any_color": true} form is not sufficient for V2; decline rather than use it.

Decline if the summary names no attributable direct affirmative target→colour relation, if another clause gives that target a different or blocked colour claim, if the named colour cannot be traced to a colour literal in the code, or if the code contains no plain-string colour literal.
""",
    'wrong-chart-type': """\
Render the same data with the wrong chart type.

Replace the plotting function with a signature-compatible alternative at EVERY call site of that function, keeping the positional arguments unchanged. Allowed swaps:
  * matplotlib (plt./ax./axs./ax1./ax2.): bar->plot, barh->plot, scatter->plot, plot->scatter
  * seaborn: barplot->lineplot, lineplot->scatterplot, scatterplot->lineplot, boxplot->violinplot, violinplot->boxplot, stripplot->swarmplot
  * plotly express, the constructor itself: scatter->line, line->scatter, bar->line, area->line, histogram->box, box->violin, violin->box, strip->box, density_contour->scatter, density_heatmap->scatter, ecdf->line, funnel->bar, funnel_area->bar, pie->bar, line_polar->line, scatter_polar->scatter, bar_polar->bar
  * plotly graph objects, the trace class: Scatter->Bar, Scattergl->Bar, Bar->Scatter, Histogram->Bar, Box->Violin, Violin->Box, Funnel->Bar, Waterfall->Bar, Pie->Bar, Scatterpolar->Barpolar, Barpolar->Scatterpolar

Swap ONE function or trace class and swap it at every site that calls it. Never two different ones in the same script, never a pair that is not on this list, and never a second edit alongside it.

The data, labels, title, colours and axes must stay exactly as they are. Only the function name — and, where unavoidable, incompatible keyword arguments — may change:
  * bar->plot: remove `width=`, `bottom=`, `align=`, `yerr=`, `capsize=`, `hatch=`, `edgecolor=`; keep `color=` and `label=`.
  * scatter->plot: remove `s=`, `cmap=`, `norm=`, `vmin=`, `vmax=`, `edgecolors=`; a SCALAR `c='red'` becomes `color='red'` (if `c=` holds an array, decline instead).
  * plot->scatter: remove `linestyle=`, `linewidth=`, `lw=`, `markersize=`, and `ms=`. Keep a compatible `marker=` such as `marker='o'` byte-identical; do not remove it merely because the line-only size keyword must go.
  * seaborn: remove `errorbar=`, `ci=`, `capsize=`, `dodge=`, `estimator=`, `orient=` when the target function does not accept them.
  * plotly express — the keywords the new constructor no longer accepts:
      scatter->line: `size=`, `size_max=`, `symbol=`, `symbol_map=`, `opacity=`, `marginal_x=`, `marginal_y=`, `trendline=`, `color_continuous_scale=`, `range_color=`;
      line->scatter: `line_group=`, `line_shape=`, `line_dash=`, `line_dash_map=`, `markers=`;
      bar->line: `barmode=`, `base=`, `opacity=`, `text_auto=`, `pattern_shape=`, `color_continuous_scale=`;
      area->line: `groupnorm=`, `pattern_shape=`;
      histogram->box: `nbins=`, `histfunc=`, `histnorm=`, `barmode=`, `barnorm=`, `cumulative=`, `marginal=`, `opacity=`;
      box->violin: `notched=`, `boxmode=`;  violin->box: `box=`, `violinmode=`;  strip->box: `stripmode=`;
      density_contour->scatter: `nbinsx=`, `nbinsy=`, `histfunc=`, `histnorm=`, `z=`;  ecdf->line: `ecdfnorm=`, `ecdfmode=`, `lines=`, `marginal=`, `opacity=`.
  * plotly graph objects — the properties the new trace class no longer has:
      Scatter->Bar: `mode=`, `line=`, `line_color=`, `fill=`, `fillcolor=`, `stackgroup=`, `connectgaps=`, `hoveron=`, and a `marker=dict(...)` that carries `size=` or `symbol=` (a `marker_color=` or `marker=dict(color=...)` survives);
      Bar->Scatter: `width=`, `base=`, `offset=`, `insidetextanchor=`, `textangle=`, `constraintext=`;
      Histogram->Bar: `nbinsx=`, `nbinsy=`, `histfunc=`, `histnorm=`, `cumulative=`, `xbins=`;
      Box->Violin: `boxpoints=`, `boxmean=`, `notched=`, `whiskerwidth=`;  Violin->Box: `box=`, `meanline=`, `points=`, `bandwidth=`, `side=`, `scalegroup=`;
      Funnel->Bar: `textinfo=`, `connector=`;  Waterfall->Bar: also `measure=`, `increasing=`, `decreasing=`, `totals=`;
      Scatterpolar->Barpolar: `mode=`, `line=`, `line_color=`, `fill=`, `fillcolor=`, `textposition=`;  Barpolar->Scatterpolar: `width=`, `base=`, `offset=`.
  * `textposition=` does not survive a swap between a bar-like and a line-like trace: `'auto'`, `'inside'` and `'outside'` are bar positions, `'top center'` and its siblings are scatter positions, and the wrong one raises. Drop the keyword whenever the swap crosses that line.
  * a pie carries its data in different keywords than a bar, so rename instead of removing: `px.pie(..., names='Region', values='Share')` becomes `px.bar(..., x='Region', y='Share')`, and `go.Pie(labels=..., values=...)` becomes `go.Bar(x=..., y=...)`. Rename exactly those two, then remove the pie-only rest — `hole=`, `textinfo=`, `insidetextorientation=`, `marker_colors=`, `rotation=`, `sort=`, `direction=`, `pull=`. A `domain=` is different: it is what puts each pie in its own tile of a grid, and dropping it stacks every one of them into a single axes, so decline instead when the figure draws its pies that way.
  * a polar express call likewise renames its data keywords: `px.line_polar(df, r='Score', theta='Metric', line_close=True)` becomes `px.line(df, x='Metric', y='Score')` — `theta=` becomes `x=`, `r=` becomes `y=`, and `line_close=`, `start_angle=`, `direction=`, `range_r=`, `range_theta=`, `log_r=` go. Never drop `r=`/`theta=` without renaming them: a call left without data draws nothing.

List every removed keyword in `edit_description`. The resulting chart must still draw visible marks — never produce an empty plot.

Evidence: {"summary": ["<the complete current, direct affirmative clause that binds the whole chart or every target series to the original chart type>"]}. The exact GT kind or controlled alias must occur in that relation and must be the only kind affirmed in the whole-chart/target-series scope. Negation, contrast, modality, conditions and former-state language do not count; a different non-target series may legitimately use another kind.

Decline if there is no swappable call, if the call passes its data via `x=`/`y=` keywords to `plt.plot`/`plt.scatter` (they reject those keywords), if the swap needs more keyword surgery than the rules above cover, if it would drop a data channel that the summary itself describes (a bubble `size=`, a `pattern_shape=`) — that would be a second flaw on top of the wrong mark — if the summary has no qualifying direct affirmative chart-type relation, or if that scope also affirms a competing kind.
""",
    'dropped-series': """\
Delete one entire data series from the chart while the table and the summary still describe it.

A script draws its series in one of several ways. Pick the ONE mechanism that fits this script and needs the smallest edit, and drop exactly one series with it. Leave the other series, the legend call, the axis limits, the colours and everything else untouched.

(a) A plotting call of its own: remove the COMPLETE statement — `plt.plot(x, y2, label='B')`, `ax.bar(...)` — including its trailing newline. If the call's result is assigned (`markerline2, stemlines2, baseline2 = ax.stem(...)`, `bar_plot = sns.barplot(...)`), remove that whole assignment statement, together with the statements that do nothing but style exactly those returned artists (`plt.setp(markerline2, ...)`) — and only when nothing else reads them.

(b) A grouping keyword that splits one call into several series — seaborn `hue=`/`style=`, plotly express `color=`/`symbol=`/`line_dash=`: filter that one level out of the frame in place, inside the existing call — `data=df` becomes `data=df[df['Region'] != 'North']`, `px.line(df, ...)` becomes `px.line(df[df['Region'] != 'North'], ...)`. Compare against the group NAME, never against a number. If the call already passes `hue_order=`/`order=` (or a plotly `category_orders=` list), delete that one entry from the list instead — seaborn draws only the levels it is given.

(c) A wide frame plotted with several y columns: drop that column from the list — `df.plot(x='Year', y=['A', 'B', 'C'])` becomes `y=['A', 'C']`, `df[['A', 'B']].plot()` becomes `df[['A']].plot()`.

(d) A melted or long frame: filter that one category out where the frame enters the plot, exactly as in (b) — `data=df_long[df_long['Metric'] == m]` becomes `data=df_long[(df_long['Metric'] == m) & (df_long['Sector'] != 'Retail')]`.

(e) Series drawn by a loop over categories or columns: narrow what the loop iterates over, so that one series is never drawn — `for s in df['Sector'].unique():` becomes `for s in ['Retail', 'Energy']:`, `for c in df_pivot.columns:` becomes `for c in df_pivot.columns.drop('Retail'):`.

(f) Several plotly traces: remove one whole `fig.add_trace(...)` / `fig.add_bar(...)` / `fig.add_scatter(...)` statement.

Rules:
  * Never remove a statement out of a loop body or a comprehension — that would delete every series or leave an empty block. The one exception: a body that draws SEVERAL series per iteration (two `plt.stem(...)` calls per category) may lose one of them, as long as another drawing statement stays.
  * Never remove the last remaining series — a blank chart is rejected by the pipeline.
  * The series must disappear from EVERY mark that draws it. If a line and its `fill_between` both draw it, filter it out where the frame is built so that both lose it; if that needs more than one edit, decline.
  * Never introduce a number that was not in the script: filter on the series' name, not on an index or a threshold.
  * Check what runs after the plotting call. Anything that indexes the drawn artists by position (`ax.containers[1]`, `axes.flat`, `colors[i]`) must still find them, or the script raises and the instance is lost.
  * Do NOT remove the series' entry from an explicit `legend([...])` list, do NOT shorten a `palette=`/`colors=` list and do NOT delete the underlying data definition: the leftover, now-unused data and the stale legend entry are realistic symptoms of this bug and part of the repair signal.

Prefer the series whose name is clearly present in the summary or in the table.

Evidence: {"either": ["<label of the removed series>"]} — verbatim.

The complete removed label→value sequence must bind uniquely to the physical CSV. Its GT mark kind must also remain reconstructable by one of two routes: either the summary directly and currently binds the removed target series itself to that exact kind (a whole-chart statement alone is insufficient), or all captured GT series on the same axis have that kind and the flawed code retains at least one definitely-live top-level surviving plotted `label=`/`name=` call of that kind, with every such target-axis survivor call resolving to the same kind. Calls hidden in functions, branches, conditions or comprehensions do not count.

Decline if the script really draws only one series, if no mechanism above fits without a second, unrelated change, if the removed series' complete label→value sequence is not uniquely bound to the physical CSV, if neither kind-recovery route holds, or if the removed series' name occurs in neither the summary nor the table.
""",
    'axis-truncation': """\
Clip at least one REAL plotted data endpoint out of the visible axis range while leaving every data value in the code unchanged.

First decide which continuous axis carries the measured values. It is y for vertical bars and ordinary line/scatter plots; it is x for a horizontal bar chart (`barh`, `orientation='h'`, or a call whose `x=` is numeric and `y=` categorical). Insert exactly ONE range-setting line immediately before the line that writes the image and at the same indentation level:

    plt.ylim(<lower>, <upper>)                       # matplotlib, values on y
    plt.xlim(<lower>, <upper>)                       # matplotlib, values on x
    fig.update_yaxes(range=[<lower>, <upper>])       # plotly, values on y
    fig.update_xaxes(range=[<lower>, <upper>])       # plotly, values on x

Choose the new range from actual plotted values over EVERY series on that axis. Put one bound strictly between the largest value and the next distinct value (or between the smallest and the next distinct value), so at least one real truth-backed endpoint lies outside the new range while at least one other endpoint remains visible. A range that merely starts above zero but still contains every value is NOT a flaw and must be declined.

If the script draws into a named axes object, set the limit there instead — `ax.set_ylim(...)`, `ax1.set_xlim(...)` — on the axes that actually carries the truncated mark, not on a twin axis that carries something else. Use `plt.ylim`/`plt.xlim` only when the script never creates an `ax` object, or when it has exactly one.

Change nothing else. Do not remove or modify the clipped value in any list, DataFrame or table; do not delete a series, change colours, labels, chart type or save call; and do not insert the limit before the plotting calls. The postcondition compares the full GT and flawed series arrays and rejects any data change.

Evidence: {"table_nums": [<the truth-backed plotted value that will be outside the visible range>]} — exactly as it appears in table.csv.

Decline if any of the following holds:
  * the chart is a pie, donut, treemap or sunburst, or anything else without a continuous value axis;
  * the script already sets a range for that axis (`set_ylim`, `plt.xlim`, `axis([...])`, `ylim=`, `update_yaxes(range=...)`);
  * that axis is logarithmic;
  * fewer than two distinct plotted numeric values can be identified reliably;
  * the figure holds several panels whose value axes are independent, so one inserted limit would truncate only one of them;
  * the value placed outside the range is not present in table.csv.
""",
    'value-perturbation': """\
Corrupt exactly ONE number in the plotted data so that a single bar/point is wrong while the table still holds the truth.

Pick one numeric literal from an inline data series — a list or tuple with at least five numeric elements that represents MEASURED VALUES, not years, indices, x positions or axis ticks — and replace it with the value multiplied by one of 0.45, 1.6 or 2.25, rounded to the same precision as the original literal (integers stay integers; floats keep the same number of decimals).

Rules:
  * Change exactly one element. Do not touch any other number, do not reorder the list, do not reformat or re-wrap the list.
  * Prefer a value in the middle of the value range: not the largest (the axis rescales and the change can be absorbed) and never a zero.
  * The change must move the mark by a visible amount — roughly at least 10% of the plotted value range.
  * In `target`, `point_index` is the zero-based index of the numeric source-list
    element you changed. For a heatmap, that list must have a same-length categorical
    label list and the exact source row must occur in table.csv; the validator binds
    both through any pivot/reordering to the rendered cell. Decline if this mapping
    is not exact.

Evidence: {"table_nums": [<the ORIGINAL value>]} if the table writes the number in the same form. If the table writes it with different formatting (e.g. "0.30" vs 0.3, or with a thousands separator), use {"table": ["<the exact token from table.csv>"]} instead.

Decline if there is no inline numeric series of measured values, if the chosen value does not appear in the table, or if every candidate list is a year/index axis.
""",
    'swapped-series': """\
Exchange the value arrays of two series so that every point of both series lies, while the labels, colours and drawing order stay correct.

Find two inline numeric lists of the SAME length (at least 5 elements) that belong to two different plotted series and hold different values, and swap their contents between the two plot calls — either by swapping the two list literals, or by swapping the two variable names at the point where they are passed to the plotting calls, whichever is the smaller edit.

Copy the list literals character for character, including spacing and number formatting: do not reformat, do not re-wrap across lines, do not round, do not sort.

Rules:
  * Do not touch the labels, colours, markers or the order of the plot calls.
  * The swap must be a real transposition (both series change), never a copy of one over the other.
  * The two series must have visibly different value profiles — otherwise nothing changes on screen.

Evidence: {"table_nums": [<a distinctive value of series A>, <a distinctive value of series B>]} — e.g. the maximum of each series, exactly as it appears in table.csv.

Decline if there are not two equal-length numeric series, if the two series carry (almost) identical values, or if their values are not verifiable in the table.
""",
}

# short reminder appended when a retry is needed after a gate failure
RETRY_SUFFIX = (
    "\n\nYour previous attempt was REJECTED by the automatic validator:\n"
    "{note}\n"
    "Return the complete script again, changing ONLY what the category allows."
)
