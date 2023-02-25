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

- The default executor was changed to standalone to match other tools.

Improvements to clang-query
---------------------------

The improvements are...

Improvements to clang-rename
----------------------------

The improvements are...

Improvements to clang-tidy
--------------------------

- Change to Python 3 in the shebang of `add_new_check.py` and `rename_check.py`,
  as the existing code is not compatible with Python 2.

- Fix a minor bug in `add_new_check.py` to only traverse subdirectories
  when updating the list of checks in the documentation.

- Deprecate the global configuration file option `AnalyzeTemporaryDtors`,
  which is no longer in use. The option will be fully removed in
  :program:`clang-tidy` version 18.

New checks
^^^^^^^^^^

- New :doc:`bugprone-suspicious-realloc-usage
  <clang-tidy/checks/bugprone/suspicious-realloc-usage>` check.

  Finds usages of ``realloc`` where the return value is assigned to the
  same expression as passed to the first argument.

- New :doc:`cppcoreguidelines-avoid-const-or-ref-data-members
  <clang-tidy/checks/cppcoreguidelines/avoid-const-or-ref-data-members>` check.

  Warns when a struct or class uses const or reference (lvalue or rvalue) data members.

- New :doc:`cppcoreguidelines-avoid-do-while
  <clang-tidy/checks/cppcoreguidelines/avoid-do-while>` check.

  Warns when using ``do-while`` loops.

- New :doc:`cppcoreguidelines-avoid-reference-coroutine-parameters
  <clang-tidy/checks/cppcoreguidelines/avoid-reference-coroutine-parameters>` check.

  Warns on coroutines that accept reference parameters.

- New :doc:`misc-use-anonymous-namespace
  <clang-tidy/checks/misc/use-anonymous-namespace>` check.

  Warns when using ``static`` function or variables at global scope, and suggests
  moving them into an anonymous namespace.

- New :doc:`bugprone-standalone-empty <clang-tidy/checks/bugprone/standalone-empty>` check.

  Warns when `empty()` is used on a range and the result is ignored. Suggests `clear()`
  if it is an existing member function.

New check aliases
^^^^^^^^^^^^^^^^^

- New alias :doc:`cert-msc54-cpp
  <clang-tidy/checks/cert/msc54-cpp>` to
  :doc:`bugprone-signal-handler
  <clang-tidy/checks/bugprone/signal-handler>` was added.


Changes in existing checks
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Fixed a false positive in :doc:`bugprone-assignment-in-if-condition
  <clang-tidy/checks/bugprone/assignment-in-if-condition>` check when there
  was an assignement in a lambda found in the condition of an ``if``.

- Improved :doc:`bugprone-signal-handler
  <clang-tidy/checks/bugprone/signal-handler>` check. Partial
  support for C++14 signal handler rules was added. Bug report generation was
  improved.

- Fixed a false positive in :doc:`cppcoreguidelines-pro-type-member-init
  <clang-tidy/checks/cppcoreguidelines/pro-type-member-init>` when warnings
  would be emitted for uninitialized members of an anonymous union despite
  there being an initializer for one of the other members.

- Fixed false positives in :doc:`google-objc-avoid-throwing-exception
  <clang-tidy/checks/google/objc-avoid-throwing-exception>` check for exceptions
  thrown by code emitted from macros in system headers.

- Improved :doc:`misc-redundant-expression <clang-tidy/checks/misc/redundant-expression>`
  check.

  The check now skips concept definitions since redundant expressions still make sense
  inside them.

- Improved :doc:`modernize-loop-convert <clang-tidy/checks/modernize/loop-convert>`
  to check for container functions ``begin``/``end`` etc on base classes of the container
  type, instead of only as direct members of the container type itself.

- Improved :doc:`modernize-use-emplace <clang-tidy/checks/modernize/use-emplace>`
  check.

- Fixed a false positive in :doc:`bugprone-branch-clone
  <clang-tidy/checks/bugprone/branch-clone>` when the branches
  involve unknown expressions.

- Fixed some false positives in :doc:`bugprone-infinite-loop
  <clang-tidy/checks/bugprone/infinite-loop>` involving dependent expressions.

- Fixed a crash in :doc:`bugprone-sizeof-expression
  <clang-tidy/checks/bugprone/sizeof-expression>` when `sizeof(...)` is
  compared against a `__int128_t`.

- Fixed bugs in :doc:`bugprone-use-after-move
  <clang-tidy/checks/bugprone/use-after-move>`:

  - Treat a move in a lambda capture as happening in the function that defines
    the lambda, not within the body of the lambda (as we were previously doing
    erroneously).

  - Don't emit an erroneous warning on self-moves.

- Improved :doc:`cert-dcl58-cpp
  <clang-tidy/checks/cert/dcl58-cpp>` check.

  The check now detects explicit template specializations that are handled specially.

- Made :doc:`cert-oop57-cpp <clang-tidy/checks/cert/oop57-cpp>` more sensitive
  by checking for an arbitrary expression in the second argument of ``memset``.

- Made the fix-it of :doc:`cppcoreguidelines-init-variables
  <clang-tidy/checks/cppcoreguidelines/init-variables>` use ``false`` to initialize
  boolean variables.

- Improved :doc:`cppcoreguidelines-prefer-member-initializer
  <clang-tidy/checks/cppcoreguidelines/prefer-member-initializer>` check.

  Fixed an issue when there was already an initializer in the constructor and
  the check would try to create another initializer for the same member.

- Fixed a false positive in :doc:`cppcoreguidelines-virtual-class-destructor
  <clang-tidy/checks/cppcoreguidelines/virtual-class-destructor>` involving
  ``final`` classes. The check will not diagnose classes marked ``final``, since
  those cannot be used as base classes, consequently, they can not violate the
  rule.

- Fixed a crash in :doc:`llvmlibc-callee-namespace
  <clang-tidy/checks/llvmlibc/callee-namespace>` when executing for C++ code
  that contain calls to advanced constructs, e.g. overloaded operators.

- Fixed false positives in :doc:`misc-redundant-expression
  <clang-tidy/checks/misc/redundant-expression>`:

  - Fixed a false positive involving overloaded comparison operators.

  - Fixed a false positive involving assignments in
    conditions. This fixes `Issue 35853 <https://github.com/llvm/llvm-project/issues/35853>`_.

- Fixed a false positive in :doc:`misc-unused-parameters
  <clang-tidy/checks/misc/unused-parameters>`
  where invalid parameters were implicitly being treated as being unused.
  This fixes `Issue 56152 <https://github.com/llvm/llvm-project/issues/56152>`_.

- Fixed false positives in :doc:`misc-unused-using-decls
  <clang-tidy/checks/misc/unused-using-decls>` where `using` statements bringing
  operators into the scope where incorrectly marked as unused.
  This fixes `issue 55095 <https://github.com/llvm/llvm-project/issues/55095>`_.

- Fixed a false positive in :doc:`modernize-deprecated-headers
  <clang-tidy/checks/modernize/deprecated-headers>` involving including
  C header files from C++ files wrapped by ``extern "C" { ... }`` blocks.
  Such includes will be ignored by now.
  By default now it doesn't warn for including deprecated headers from header
  files, since that header file might be used from C source files. By passing
  the ``CheckHeaderFile=true`` option if header files of the project only
  included by C++ source files.

- Improved :doc:`performance-inefficient-vector-operation
  <clang-tidy/checks/performance/inefficient-vector-operation>` to work when
  the vector is a member of a structure.

- Fixed a crash in :doc:`performance-unnecessary-value-param
  <clang-tidy/checks/performance/unnecessary-value-param>` when the specialization
  template has an unnecessary value parameter. Removed the fix for a template.

- Fixed a crash in :doc:`readability-const-return-type
  <clang-tidy/checks/readability/const-return-type>` when a pure virtual function
  overrided has a const return type. Removed the fix for a virtual function.

- Skipped addition of extra parentheses around member accesses (``a.b``) in fix-it for
  :doc:`readability-container-data-pointer <clang-tidy/checks/readability/container-data-pointer>`.

- Fixed incorrect suggestions for :doc:`readability-container-size-empty
  <clang-tidy/checks/readability/container-size-empty>` when smart pointers are involved.

- Fixed a false positive in :doc:`readability-non-const-parameter
  <clang-tidy/checks/readability/non-const-parameter>` when the parameter is
  referenced by an lvalue.

- Expanded :doc:`readability-simplify-boolean-expr
  <clang-tidy/checks/readability/simplify-boolean-expr>` to simplify expressions
  using DeMorgan's Theorem.

  The check now supports detecting alias cases of ``push_back`` ``push`` and
  ``push_front`` on STL-style containers and replacing them with ``emplace_back``,
  ``emplace`` or ``emplace_front``.

- Improved :doc:`modernize-use-equals-default <clang-tidy/checks/modernize/use-equals-default>`
  check.

  The check now skips unions/union-like classes since in this case a default constructor
  with empty body is not equivalent to the explicitly defaulted one, variadic constructors
  since they cannot be explicitly defaulted. The check also skips copy assignment operators
  with nonstandard return types, template constructors, private/protected default constructors
  for C++17 or earlier. The automatic fixit has been adjusted to avoid adding superfluous
  semicolon. The check is restricted to C++11 or later.

- Change the default behavior of :doc:`readability-avoid-const-params-in-decls
  <clang-tidy/checks/readability/avoid-const-params-in-decls>` to not
  warn about `const` value parameters of declarations inside macros.

- Fixed crashes in :doc:`readability-braces-around-statements
  <clang-tidy/checks/readability/braces-around-statements>` and
  :doc:`readability-simplify-boolean-expr <clang-tidy/checks/readability/simplify-boolean-expr>`
  when using a C++23 ``if consteval`` statement.

- Change the behavior of :doc:`readability-const-return-type
  <clang-tidy/checks/readability/const-return-type>` to not
  warn about `const` return types in overridden functions since the derived
  class cannot always choose to change the function signature.

- Change the default behavior of :doc:`readability-const-return-type
  <clang-tidy/checks/readability/const-return-type>` to not
  warn about `const` value parameters of declarations inside macros.

- Support removing ``c_str`` calls from ``std::string_view`` constructor calls in
  :doc:`readability-redundant-string-cstr <clang-tidy/checks/readability/redundant-string-cstr>`
  check.

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
