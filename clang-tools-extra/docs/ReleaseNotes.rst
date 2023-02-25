====================================================
Extra Clang Tools |release| |ReleaseNotesTitle|
====================================================

.. contents::
   :local:
   :depth: 3

Written by the `LLVM Team <https://llvm.org/>`_

Introduction
============

This document contains the release notes for the Extra Clang Tools, part of the
Clang release |release|. Here we describe the status of the Extra Clang Tools in
some detail, including major improvements from the previous release and new
feature work. All LLVM releases may be downloaded from the `LLVM releases web
site <https://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about
the latest release, please see the `Clang Web Site <https://clang.llvm.org>`_ or
the `LLVM Web Site <https://llvm.org>`_.

Note that if you are reading this file from a Git checkout or the
main Clang web page, this document applies to the *next* release, not
the current one. To see the release notes for a specific release, please
see the `releases page <https://llvm.org/releases/>`_.

What's New in Extra Clang Tools |release|?
==========================================

Some of the major new features and improvements to Extra Clang Tools are listed
here. Generic improvements to Extra Clang Tools as a whole or to its underlying
infrastructure are described first, followed by tool-specific sections.

Major New Features
------------------

...

Improvements to clangd
----------------------

Inlay hints
^^^^^^^^^^^

- Provide hints for:
    - Lambda return types.
    - Forwarding functions using the underlying function call.
- Support for standard LSP 3.17 inlay hints protocol.
- Designator inlay hints are enabled by default.

Diagnostics
^^^^^^^^^^^

- Improved Fix-its of some clang-tidy checks when applied with clangd.
- Clangd now produces diagnostics for forwarding functions like make_unique.
- Include cleaner analysis can be disabled with the ``Diagnostics.Includes.IgnoreHeader`` config option.
- Include cleaner doesn’t diagnose exporting headers.
- clang-tidy and include cleaner diagnostics have links to their documentation.

Semantic Highlighting
^^^^^^^^^^^^^^^^^^^^^

- Semantic highlighting works for tokens that span multiple lines.
- Mutable reference parameters in function calls receive ``usedAsMutableReference`` modifier.

Hover
^^^^^

- Hover displays desugared types by default now.

Code completion
^^^^^^^^^^^^^^^

- Improved ranking/filtering for ObjC method selectors.
- Support for C++20 concepts and requires expressions.

Signature help
^^^^^^^^^^^^^^

- Signature help for function pointers.
- Provides hints using underlying functions in forwarded calls.

Cross-references
^^^^^^^^^^^^^^^^

Code Actions
^^^^^^^^^^^^

- New code action to generate ObjC initializers.
- New code action to generate move/copy constructors/assignments.
- Extract to function works for methods in addition to free functions.
- Related diagnostics are attached to code actions response, if any.
- Extract variable works in C and ObjC files.
- Fix to define outline when the parameter has a braced initializer.

Miscellaneous
^^^^^^^^^^^^^

- Include fixer supports symbols inside macro arguments.
- Dependent autos are now deduced when there’s a single instantiation.
- Support for symbols exported with using declarations in all features.
- Fixed background-indexing priority for M1 chips.
- Indexing for standard library symbols.
- ObjC framework includes are spelled properly during include insertion operations.

Improvements to clang-doc
-------------------------

Improvements to clang-query
---------------------------

The improvements are...

Improvements to clang-rename
----------------------------

The improvements are...

Improvements to clang-tidy
--------------------------

New checks
^^^^^^^^^^

New check aliases
^^^^^^^^^^^^^^^^^

Changes in existing checks
^^^^^^^^^^^^^^^^^^^^^^^^^^

Removed checks
^^^^^^^^^^^^^^

Improvements to include-fixer
-----------------------------

The improvements are...

Improvements to clang-include-fixer
-----------------------------------

The improvements are...

Improvements to modularize
--------------------------

The improvements are...

Improvements to pp-trace
------------------------

Clang-tidy Visual Studio plugin
-------------------------------
