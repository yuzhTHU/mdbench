# Minimal makefile for Sphinx documentation

# Re-read the toctree and rebuild every page so sidebar navigation cannot retain
# stale entries after pages are added, removed, or reordered.
SPHINXOPTS    ?= -E -a
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
