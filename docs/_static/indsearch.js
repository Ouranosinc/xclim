/* Array of indicator objects */
let indicators = [];
let defModules = ["atmos", "generic", "land", "seaIce"];
/* MiniSearch object defining search mechanism */
let miniSearch = new MiniSearch({
  fields: ['title', 'abstract', 'variables', 'keywords', 'id'], // fields to index for full-text search
  storeFields: ['title', 'abstract', 'vars', 'realm', 'module', 'name', 'keywords'], // fields to return with search results
  searchOptions: {
    boost: {'title': 3, 'variables': 2},
    fuzzy: 0.1,
    prefix: true,
    boostDocument: (docID, term, storedFields) => {
      if (defModules.indexOf(storedFields['module']) > -1) {
        return 2;
      } else {
        return 1;
      }
    },
  },
  extractField: (doc, field) => {
    if (field === 'variables') {
      return Object.keys(doc['vars']).join(' ');
    }
    return MiniSearch.getDefault('extractField')(doc, field);
  }
});

// Populate search object with complete list of indicators
fetch('indicators.json')
  .then(data => data.json())
  .then(data => {
    indicators = Object.entries(data).map(([k, v]) => {
      return {id: k.toLowerCase(), ...v}
    });
    miniSearch.addAll(indicators);
    indFilter();
  });


function escapeHTML(str){
    /* Escape HTML characters in a string. */
    var map =
    {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return str.replace(/[&<>"']/g, function(m) {return map[m];});
}

function makeKeywordLabel(ind) {
    /* Print list of keywords only if there is at least one. */
    if (ind.keywords[0].length > 0) {
        const keywords = ind.keywords.map(v => `<code class="keywordlabel">${v.trim()}</code>`).join('');
        return `<div class="keywords">Keywords: ${keywords}</div>`;
        }
    else {
        return "";
        }
}


function makeVariableList(ind) {
    /* Print list of variables and include mouse-hover tooltip with variable description. */
    return Object.entries(ind.vars).map((kv) => {
        /* kv[0] is the variable name, kv[1] is the variable description. */
        /* Convert kv[1] to a string literal */
        const text = escapeHTML(kv[1]);
        const tooltip = `<button class="indVarname" title="${text}" alt="${text}">${kv[0]}</button>`;
        return tooltip
    }).join('');
}

function indTemplate(ind) {
  // const varlist = Object.entries(ind.vars).map((kv) => `<code class="indVarname">${kv[0]}</code>`).join('');
  const varlist = makeVariableList(ind);
  return `
    <div class="indElem" id="${ind.id}">
      <div class="indHeader">
        <b class="indTitle">${escapeHTML(ind.title)}</b>
        <a class="reference_internal indName" href="api_indicators.html#xclim.indicators.${ind.module}.${ind.name}" title="${ind.name}">
          <code>${ind.module}.${ind.name}</code>
        </a>
      </div>
      <div class="indVars">Uses: ${varlist}</div>
      <div class="indDesc"><p>${escapeHTML(ind.abstract)}</p></div>
      ${makeKeywordLabel(ind)}
      <div class="indID">Yaml ID: <code>${ind.id}</code></div>
    </div>
  `;
}

function indFilter() {
  const input = document.getElementById("queryInput").value;
  const incVirt = document.getElementById("incVirtMod").checked;
  let opts = {};
  if (!incVirt) {
    opts["filter"] = (result) => (defModules.indexOf(result.module) > -1);
  }
  let inds = [];
  if (input === "") { //Search wildcard so that boostDocument rules are applied.
    inds = miniSearch.search(MiniSearch.wildcard, opts);
  } else {
    inds = miniSearch.search(input, opts);
  }

  const newTable = inds.map(indTemplate).join('');
  const tableElem = document.getElementById("indTable");
  tableElem.innerHTML = newTable;
  return newTable;
}
