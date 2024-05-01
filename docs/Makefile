# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXPROJ    = xclim
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

livehtml:
	sphinx-autobuild --port 54345 --open-browser --delay 3 --re-ignore "$(BUILDDIR)|apidoc|Makefile|_dynamic/indicators.json|variables.json|__pycache__" "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
