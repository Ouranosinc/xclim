"""
Sphinx extension that acts as a autodoc patch for documenting Indicator instances.

By default, indicator instances are skipped by autodoc because their subclass is not a builtin type of python.

Based on https://github.com/powerline/powerline/blob/83d855d3d73498c47553afeba212415990d95c54/docs/source/powerline_autodoc.py
"""

# TODO: Use the new version instead of the legacy class-based autodoc.
from __future__ import annotations

from sphinx.domains.python import PyFunction, PyVariable, PyXRefRole
from sphinx.ext import autodoc

from xclim import __version__
from xclim.core.collection import IndicatorCollection
from xclim.core.indicator import Indicator


class IndicatorDocumenter(autodoc.FunctionDocumenter):
    objtype = "indicator"
    directivetype = "indicator"

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return isinstance(member, Indicator)


class IndicatorDirective(PyFunction):
    pass


# This is quite hacky
class IndicatorCollectionDocumenter(autodoc.DataDocumenter):
    objtype = "indicatorcollection"
    directivetype = "indicatorcollection"

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return isinstance(member, IndicatorCollection)

    def should_suppress_value_header(self):
        return True

    def document_members(self, all_members: bool = False) -> None:
        """
        Generate reST for member documentation.

        If *all_members* is True, document all members, else those given by
        *self.options.members*.
        """
        # set current namespace for finding members
        self._current_document.autodoc_module = self.modname
        if self.objpath:
            self._current_document.autodoc_class = self.objpath[0]

        # document non-skipped members
        member_documenters = []
        for mname in self.object:
            full_mname = f"{self.modname}::" + ".".join((*self.objpath, mname))
            documenter = IndicatorDocumenter(self.directive, full_mname, self.indent)
            member_documenters.append(documenter)

        member_documenters = [
            documenter for documenter in member_documenters if documenter.parse_name() and documenter.import_object()
        ]
        member_documenters = sorted(member_documenters, key=lambda doc: doc.object.identifier)

        for documenter in member_documenters:
            documenter._generate(all_members=True, real_modname=self.real_modname, check_module=False)

        # reset current objects
        self._current_document.autodoc_module = ""
        self._current_document.autodoc_class = ""


class IndicatorCollectionDirective(PyVariable):
    pass


def setup(app):
    app.setup_extension("sphinx.ext.autodoc")
    app.add_autodocumenter(IndicatorDocumenter)
    app.add_autodocumenter(IndicatorCollectionDocumenter)
    app.add_directive_to_domain("py", "indicator", IndicatorDirective)
    app.add_role_to_domain("py", "indicator", PyXRefRole())
    app.add_directive_to_domain("py", "indicatorcollection", IndicatorCollectionDirective)
    app.add_role_to_domain("py", "indicatorcollection", PyXRefRole())
    return {"version": __version__, "parallel_read_safe": True}
