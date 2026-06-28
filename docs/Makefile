SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
# Documentation lives alongside the code inside the package, so the source
# tree is the package itself while conf.py stays here in ``source`` (CONFDIR).
SOURCEDIR     = ../icegraph
CONFDIR       = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help"
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" -c "$(CONFDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" -c "$(CONFDIR)" $(SPHINXOPTS) $(O)
