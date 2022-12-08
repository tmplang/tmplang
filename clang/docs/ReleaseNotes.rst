===========================================
Clang |release| |ReleaseNotesTitle|
===========================================

.. contents::
   :local:
   :depth: 2

Written by the `LLVM Team <https://llvm.org/>`_

Introduction
============

This document contains the release notes for the Clang C/C++/Objective-C
frontend, part of the LLVM Compiler Infrastructure, release |release|. Here we
describe the status of Clang in some detail, including major
improvements from the previous release and new feature work. For the
general LLVM release notes, see `the LLVM
documentation <https://llvm.org/docs/ReleaseNotes.html>`_. All LLVM
releases may be downloaded from the `LLVM releases web
site <https://llvm.org/releases/>`_.

For more information about Clang or LLVM, including information about the
latest release, please see the `Clang Web Site <https://clang.llvm.org>`_ or the
`LLVM Web Site <https://llvm.org>`_.

Note that if you are reading this file from a Git checkout or the
main Clang web page, this document applies to the *next* release, not
the current one. To see the release notes for a specific release, please
see the `releases page <https://llvm.org/releases/>`_.

What's New in Clang |release|?
==============================

Some of the major new features and improvements to Clang are listed
here. Generic improvements to Clang as a whole or to its underlying
infrastructure are described first, followed by language-specific
sections with improvements to Clang's support for those languages.

Potentially Breaking Changes
============================
These changes are ones which we think may surprise users when upgrading to
Clang |release| because of the opportunity they pose for disruption to existing
code bases.

- The ``-Wimplicit-function-declaration`` and ``-Wimplicit-int`` warning
  diagnostics are now enabled by default in C99, C11, and C17. As of C2x,
  support for implicit function declarations and implicit int has been removed,
  and the warning options will have no effect. Specifying ``-Wimplicit-int`` in
  C89 mode will now issue warnings instead of being a noop.
  *NOTE* these warnings are expected to default to an error in Clang 16. We
  recommend that projects using configure scripts verify the results do not
  change before/after setting ``-Werror=implicit-function-declarations`` or
  ``-Wimplicit-int`` to avoid incompatibility with Clang 16.

Major New Features
------------------

Bug Fixes
---------
- ``CXXNewExpr::getArraySize()`` previously returned a ``llvm::Optional``
  wrapping a ``nullptr`` when the ``CXXNewExpr`` did not have an array
  size expression. This was fixed and ``::getArraySize()`` will now always
  either return ``None`` or a ``llvm::Optional`` wrapping a valid ``Expr*``.
  This fixes `Issue 53742 <https://github.com/llvm/llvm-project/issues/53742>`_.
- We now ignore full expressions when traversing cast subexpressions. This
  fixes `Issue 53044 <https://github.com/llvm/llvm-project/issues/53044>`_.
- Allow ``-Wno-gnu`` to silence GNU extension diagnostics for pointer
  arithmetic diagnostics. Fixes `Issue 54444
  <https://github.com/llvm/llvm-project/issues/54444>`_.
- Placeholder constraints, as in ``Concept auto x = f();``, were not checked
  when modifiers like ``auto&`` or ``auto**`` were added. These constraints are
  now checked.
  This fixes  `Issue 53911 <https://github.com/llvm/llvm-project/issues/53911>`_
  and  `Issue 54443 <https://github.com/llvm/llvm-project/issues/54443>`_.
- Previously invalid member variables with template parameters would crash clang.
  Now fixed by setting identifiers for them.
  This fixes `Issue 28475 (PR28101) <https://github.com/llvm/llvm-project/issues/28475>`_.
- Now allow the ``restrict`` and ``_Atomic`` qualifiers to be used in
  conjunction with ``__auto_type`` to match the behavior in GCC. This fixes
  `Issue 53652 <https://github.com/llvm/llvm-project/issues/53652>`_.
- No longer crash when specifying a variably-modified parameter type in a
  function with the ``naked`` attribute. This fixes
  `Issue 50541 <https://github.com/llvm/llvm-project/issues/50541>`_.
- Allow multiple ``#pragma weak`` directives to name the same undeclared (if an
  alias, target) identifier instead of only processing one such ``#pragma weak``
  per identifier.
  Fixes `Issue 28985 <https://github.com/llvm/llvm-project/issues/28985>`_.
- Assignment expressions in C11 and later mode now properly strip the _Atomic
  qualifier when determining the type of the assignment expression. Fixes
  `Issue 48742 <https://github.com/llvm/llvm-project/issues/48742>`_.
- Improved the diagnostic when accessing a member of an atomic structure or
  union object in C; was previously an unhelpful error, but now issues a
  ``-Watomic-access`` warning which defaults to an error. Fixes
  `Issue 54563 <https://github.com/llvm/llvm-project/issues/54563>`_.
- Unevaluated lambdas in dependant contexts no longer result in clang crashing.
  This fixes Issues `50376 <https://github.com/llvm/llvm-project/issues/50376>`_,
  `51414 <https://github.com/llvm/llvm-project/issues/51414>`_,
  `51416 <https://github.com/llvm/llvm-project/issues/51416>`_,
  and `51641 <https://github.com/llvm/llvm-project/issues/51641>`_.
- The builtin function __builtin_dump_struct would crash clang when the target
  struct contains a bitfield. It now correctly handles bitfields.
  This fixes Issue `Issue 54462 <https://github.com/llvm/llvm-project/issues/54462>`_.
- Statement expressions are now disabled in default arguments in general.
  This fixes Issue `Issue 53488 <https://github.com/llvm/llvm-project/issues/53488>`_.
- According to `CWG 1394 <https://wg21.link/cwg1394>`_ and
  `C++20 [dcl.fct.def.general]p2 <https://timsong-cpp.github.io/cppwp/n4868/dcl.fct.def#general-2.sentence-3>`_,
  Clang should not diagnose incomplete types in function definitions if the function body is ``= delete;``.
  This fixes Issue `Issue 52802 <https://github.com/llvm/llvm-project/issues/52802>`_.
- Unknown type attributes with a ``[[]]`` spelling are no longer diagnosed twice.
  This fixes Issue `Issue 54817 <https://github.com/llvm/llvm-project/issues/54817>`_.
- Clang should no longer incorrectly diagnose a variable declaration inside of
  a lambda expression that shares the name of a variable in a containing
  if/while/for/switch init statement as a redeclaration.
  This fixes `Issue 54913 <https://github.com/llvm/llvm-project/issues/54913>`_.
- Overload resolution for constrained function templates could use the partial
  order of constraints to select an overload, even if the parameter types of
  the functions were different. It now diagnoses this case correctly as an
  ambiguous call and an error. Fixes
  `Issue 53640 <https://github.com/llvm/llvm-project/issues/53640>`_.
- No longer crash when trying to determine whether the controlling expression
  argument to a generic selection expression has side effects in the case where
  the expression is result dependent. This fixes
  `Issue 50227 <https://github.com/llvm/llvm-project/issues/50227>`_.
- Fixed an assertion when constant evaluating an initializer for a GCC/Clang
  floating-point vector type when the width of the initialization is exactly
  the same as the elements of the vector being initialized.
  Fixes `Issue 50216 <https://github.com/llvm/llvm-project/issues/50216>`_.
- Fixed a crash when the ``__bf16`` type is used such that its size or
  alignment is calculated on a target which does not support that type. This
  fixes `Issue 50171 <https://github.com/llvm/llvm-project/issues/50171>`_.
- Fixed a false positive diagnostic about an unevaluated expression having no
  side effects when the expression is of VLA type and is an operand of the
  ``sizeof`` operator. Fixes `Issue 48010 <https://github.com/llvm/llvm-project/issues/48010>`_.
- Fixed a false positive diagnostic about scoped enumerations being a C++11
  extension in C mode. A scoped enumeration's enumerators cannot be named in C
  because there is no way to fully qualify the enumerator name, so this
  "extension" was unintentional and useless. This fixes
  `Issue 42372 <https://github.com/llvm/llvm-project/issues/42372>`_.
- Clang will now find and emit a call to an allocation function in a
  promise_type body for coroutines if there is any allocation function
  declaration in the scope of promise_type. Additionally, to implement CWG2585,
  a coroutine will no longer generate a call to a global allocation function
  with the signature ``(std::size_t, p0, ..., pn)``.
  This fixes Issue `Issue 54881 <https://github.com/llvm/llvm-project/issues/54881>`_.
- Implement `CWG 2394 <https://wg21.link/cwg2394>`_: Const class members
  may be initialized with a defaulted default constructor under the same
  conditions it would be allowed for a const object elsewhere.
- ``__has_unique_object_representations`` no longer reports that ``_BitInt`` types
  have unique object representations if they have padding bits.
- Unscoped and scoped enumeration types can no longer be initialized from a
  brace-init-list containing a single element of a different scoped enumeration
  type.
- Allow use of an elaborated type specifier as a ``_Generic`` selection
  association in C++ mode. This fixes
  `Issue 55562 <https://github.com/llvm/llvm-project/issues/55562>`_.
- Clang will allow calling a ``consteval`` function in a default argument. This
  fixes `Issue 48230 <https://github.com/llvm/llvm-project/issues/48230>`_.
- Fixed memory leak due to ``VarTemplateSpecializationDecl`` using
  ``TemplateArgumentListInfo`` instead of ``ASTTemplateArgumentListInfo``.
- An initializer for a static variable declaration, which is nested
  inside a statement expression in an aggregate initializer, is now
  emitted as a dynamic initializer. Previously the variable would
  incorrectly be zero-initialized. In contexts where a dynamic
  initializer is not allowed this is now diagnosed as an error.
- Clang now correctly emits symbols for implicitly instantiated constexpr
  template function. Fixes `Issue 55560 <https://github.com/llvm/llvm-project/issues/55560>`_.
- Clang now checks ODR violations when merging concepts from different modules.
  Note that this may possibly break existing code, and is done so intentionally.
  Fixes `Issue 56310 <https://github.com/llvm/llvm-project/issues/56310>`_.
- Clang will now look through type sugar when checking a member function is a
  move assignment operator. Fixes `Issue 56456 <https://github.com/llvm/llvm-project/issues/56456>`_.
- Fixed a crash when a variable with a bool enum type that has no definition
  used in comparison operators. Fixes `Issue 56560 <https://github.com/llvm/llvm-project/issues/56560>`_.
- Fix that ``if consteval`` could evaluate to ``true`` at runtime because it was incorrectly
  constant folded. Fixes `Issue 55638 <https://github.com/llvm/llvm-project/issues/55638>`_.
- Fixed incompatibility of Clang's ``<stdatomic.h>`` with MSVC ``<atomic>``.
  Fixes `MSVC STL Issue 2862 <https://github.com/microsoft/STL/issues/2862>`_.
- Empty enums and enums with a single enumerator with value zero will be
  considered to have one positive bit in order to represent the underlying
  value. This effects whether we consider the store of the value one to be well
  defined.
- An operator introduced to the scope via a ``using`` statement now correctly references this
  statement in clangd (hover over the symbol, jump to definition) as well as in the AST dump.
  This also fixes `issue 55095 <https://github.com/llvm/llvm-project/issues/#55095>`_ as a
  side-effect.
- When including a PCH from a GCC style directory with multiple alternative PCH
  files, Clang now requires all defines set on the command line while generating
  the PCH and when including it to match. This matches GCC's behaviour.
  Previously Clang would tolerate defines to be set when creating the PCH but
  missing when used, or vice versa. This makes sure that Clang picks the
  correct one, where it previously would consider multiple ones as potentially
  acceptable (and erroneously use whichever one is tried first).
- Fix a crash when generating code coverage information for an
  ``if consteval`` statement. This fixes
  `Issue 57377 <https://github.com/llvm/llvm-project/issues/57377>`_.
- Fix a crash when a ``btf_type_tag`` attribute is applied to the pointee of
  a function pointer.
- Clang 14 predeclared some builtin POSIX library functions in ``gnu2x`` mode,
  and Clang 15 accidentally stopped predeclaring those functions in that
  language mode. Clang 16 now predeclares those functions again. This fixes
  `Issue 56607 <https://github.com/llvm/llvm-project/issues/56607>`_.

Improvements to Clang's diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- ``-Wliteral-range`` will warn on floating-point equality comparisons with
  constants that are not representable in a casted value. For example,
  ``(float) f == 0.1`` is always false.
- ``-Winline-namespace-reopened-noninline`` now takes into account that the
  ``inline`` keyword must appear on the original but not necessarily all
  extension definitions of an inline namespace and therefore points its note
  at the original definition. This fixes `Issue 50794 (PR51452)
  <https://github.com/llvm/llvm-project/issues/50794>`_.
- ``-Wunused-but-set-variable`` now also warns if the variable is only used
  by unary operators.
- ``-Wunused-variable`` no longer warn for references extending the lifetime
  of temporaries with side effects. This fixes `Issue 54489
  <https://github.com/llvm/llvm-project/issues/54489>`_.
- Modified the behavior of ``-Wstrict-prototypes`` and added a new, related
  diagnostic ``-Wdeprecated-non-prototype``. The strict prototypes warning will
  now only diagnose deprecated declarations and definitions of functions
  without a prototype where the behavior in C2x will remain correct. This
  diagnostic remains off by default but is now enabled via ``-pedantic`` due to
  it being a deprecation warning. ``-Wstrict-prototypes`` has no effect in C2x
  or when ``-fno-knr-functions`` is enabled. ``-Wdeprecated-non-prototype``
  will diagnose cases where the deprecated declarations or definitions of a
  function without a prototype will change behavior in C2x. Additionally, it
  will diagnose calls which pass arguments to a function without a prototype.
  This warning is enabled only when the ``-Wdeprecated-non-prototype`` option
  is enabled at the function declaration site, which allows a developer to
  disable the diagnostic for all callers at the point of declaration. This
  diagnostic is grouped under the ``-Wstrict-prototypes`` warning group, but is
  enabled by default. ``-Wdeprecated-non-prototype`` has no effect in C2x or
  when ``-fno-knr-functions`` is enabled.
- Clang now appropriately issues an error in C when a definition of a function
  without a prototype and with no arguments is an invalid redeclaration of a
  function with a prototype. e.g., ``void f(int); void f() {}`` is now properly
  diagnosed.
- No longer issue a "declaration specifiers missing, defaulting to int"
  diagnostic in C89 mode because it is not an extension in C89, it was valid
  code. The diagnostic has been removed entirely as it did not have a
  diagnostic group to disable it, but it can be covered wholly by
  ``-Wimplicit-int``.
- ``-Wmisexpect`` warns when the branch weights collected during profiling
  conflict with those added by ``llvm.expect``.
- ``-Wthread-safety-analysis`` now considers overloaded compound assignment and
  increment/decrement operators as writing to their first argument, thus
  requiring an exclusive lock if the argument is guarded.
- ``-Wenum-conversion`` now warns on converting a signed enum of one type to an
  unsigned enum of a different type (or vice versa) rather than
  ``-Wsign-conversion``.
- Added the ``-Wunreachable-code-generic-assoc`` diagnostic flag (grouped under
  the ``-Wunreachable-code`` flag) which is enabled by default and warns the
  user about ``_Generic`` selection associations which are unreachable because
  the type specified is an array type or a qualified type.
- Added the ``-Wgnu-line-marker`` diagnostic flag (grouped under the ``-Wgnu``
  flag) which is a portability warning about use of GNU linemarker preprocessor
  directives. Fixes `Issue 55067 <https://github.com/llvm/llvm-project/issues/55067>`_.
- Using ``#warning``, ``#elifdef`` and ``#elifndef`` that are incompatible with C/C++
  standards before C2x/C++2b are now warned via ``-pedantic``. Additionally,
  on such language mode, ``-Wpre-c2x-compat`` and ``-Wpre-c++2b-compat``
  diagnostic flags report a compatibility issue.
  Fixes `Issue 55306 <https://github.com/llvm/llvm-project/issues/55306>`_.
- Clang now checks for stack resource exhaustion when recursively parsing
  declarators in order to give a diagnostic before we run out of stack space.
  This fixes `Issue 51642 <https://github.com/llvm/llvm-project/issues/51642>`_.
- Unknown preprocessor directives in a skipped conditional block are now given
  a typo correction suggestion if the given directive is sufficiently similar
  to another preprocessor conditional directive. For example, if ``#esle``
  appears in a skipped block, we will warn about the unknown directive and
  suggest ``#else`` as an alternative. ``#elifdef`` and ``#elifndef`` are only
  suggested when in C2x or C++2b mode. Fixes
  `Issue 51598 <https://github.com/llvm/llvm-project/issues/51598>`_.
- The ``-Wdeprecated`` diagnostic will now warn on out-of-line ``constexpr``
  declarations downgraded to definitions in C++1z, in addition to the
  existing warning on out-of-line ``const`` declarations.
- ``-Wshift-overflow`` will not warn for signed left shifts in C++20 mode
  (and newer), as it will always wrap and never overflow. This fixes
  `Issue 52873 <https://github.com/llvm/llvm-project/issues/52873>`_.
- When using class templates without arguments, clang now tells developers
  that template arguments are missing in certain contexts.
  This fixes `Issue 55962 <https://github.com/llvm/llvm-project/issues/55962>`_.
- Printable Unicode characters within ``static_assert`` messages are no longer
  escaped.
- The ``-Winfinite-recursion`` diagnostic no longer warns about
  unevaluated operands of a ``typeid`` expression, as they are now
  modeled correctly in the CFG. This fixes
  `Issue 21668 <https://github.com/llvm/llvm-project/issues/21668>`_.
- ``-Wself-assign``, ``-Wself-assign-overloaded`` and ``-Wself-move`` will
  suggest a fix if the decl being assigned is a parameter that shadows a data
  member of the contained class.
- Added ``-Winvalid-utf8`` which diagnoses invalid UTF-8 code unit sequences in
  comments.
- The ``-Wint-conversion`` warning diagnostic for implicit int <-> pointer
  conversions now defaults to an error in all C language modes. It may be
  downgraded to a warning with ``-Wno-error=int-conversion``, or disabled
  entirely with ``-Wno-int-conversion``.
- Deprecated lax vector conversions for Altivec vectors.
  The default behaviour with respect to these conversions
  will switch to disable them in an upcoming release.
- On AIX, only emit XL compatibility warning when 16 byte aligned structs are
  pass-by-value function arguments.


Non-comprehensive list of changes in this release
-------------------------------------------------

- Improve ``__builtin_dump_struct``:

  - Support bitfields in struct and union.
  - Improve the dump format, dump both bitwidth (if its a bitfield) and field
    value.
  - Remove anonymous tag locations and flatten anonymous struct members.
  - Beautify dump format, add indent for struct members.
  - Support passing additional arguments to the formatting function, allowing
    use with ``fprintf`` and similar formatting functions.
  - Support use within constant evaluation in C++, if a ``constexpr``
    formatting function is provided.
  - Support formatting of base classes in C++.
  - Support calling a formatting function template in C++, which can provide
    custom formatting for non-aggregate types.

- Previously disabled sanitizer options now enabled by default:

  - ``ASAN_OPTIONS=detect_stack_use_after_return=1`` (only on Linux).
  - ``MSAN_OPTIONS=poison_in_dtor=1``.

- Some type-trait builtins, such as ``__has_trivial_assign``, have been documented
  as deprecated for a while because their semantics don't mix well with post-C++11 type-traits.
  Clang now emits deprecation warnings for them under the flag ``-Wdeprecated-builtins``.

New Compiler Flags
------------------

Deprecated Compiler Flags
-------------------------

Modified Compiler Flags
-----------------------

Removed Compiler Flags
-------------------------

New Pragmas in Clang
--------------------
- ...

Attribute Changes in Clang
--------------------------

- Added support for parameter pack expansion in ``clang::annotate``.

- The ``overloadable`` attribute can now be written in all of the syntactic
  locations a declaration attribute may appear.
  This fixes `Issue 53805 <https://github.com/llvm/llvm-project/issues/53805>`_.

- Improved namespace attributes handling:

  - Handle GNU attributes before a namespace identifier and subsequent
    attributes of different kinds.
  - Emit error on GNU attributes for a nested namespace definition.

- Statement attributes ``[[clang::noinline]]`` and  ``[[clang::always_inline]]``
  can be used to control inlining decisions at callsites.

- ``#pragma clang attribute push`` now supports multiple attributes within a single directive.

- The ``__declspec(naked)`` attribute can no longer be written on a member
  function in Microsoft compatibility mode, matching the behavior of cl.exe.

- Attribute ``no_builtin`` should now affect the generated code. It now disables
  builtins (corresponding to the specific names listed in the attribute) in the
  body of the function the attribute is on.

- When the ``weak`` attribute is applied to a const qualified variable clang no longer
  tells the backend it is allowed to optimize based on initializer value.

- Added the ``clang::annotate_type`` attribute, which can be used to add
  annotations to types (see documentation for details).

- Added half float to types that can be represented by ``__attribute__((mode(XX)))``.

- The ``format`` attribute can now be applied to non-variadic functions. The
  format string must correctly format the fixed parameter types of the function.
  Using the attribute this way emits a GCC compatibility diagnostic.

- Support was added for ``__attribute__((function_return("thunk-extern")))``
  to X86 to replace ``ret`` instructions with ``jmp __x86_return_thunk``. The
  corresponding attribute to disable this,
  ``__attribute__((function_return("keep")))`` was added. This is intended to
  be used by the Linux kernel to mitigate RETBLEED.

- Ignore the ``__preferred_name__`` attribute when writing for C++20 module interfaces.
  This is a short-term workaround intentionally since clang doesn't take care of the
  serialization and deserialization of ``__preferred_name__``.  See
  https://github.com/llvm/llvm-project/issues/56490 for example.

Windows Support
---------------

AIX Support
-----------

C Language Changes in Clang
---------------------------

C2x Feature Support
-------------------

C++ Language Changes in Clang
-----------------------------

- Improved ``-O0`` code generation for calls to ``std::move``, ``std::forward``,
  ``std::move_if_noexcept``, ``std::addressof``, and ``std::as_const``. These
  are now treated as compiler builtins and implemented directly, rather than
  instantiating the definition from the standard library.
- Fixed mangling of nested dependent names such as ``T::a::b``, where ``T`` is a
  template parameter, to conform to the Itanium C++ ABI and be compatible with
  GCC. This breaks binary compatibility with code compiled with earlier versions
  of clang; use the ``-fclang-abi-compat=14`` option to get the old mangling.
- Preprocessor character literals with a ``u8`` prefix are now correctly treated as
  unsigned character literals. This fixes `Issue 54886 <https://github.com/llvm/llvm-project/issues/54886>`_.
- Stopped allowing constraints on non-template functions to be compliant with
  dcl.decl.general p4.
- Improved `copy elision` optimization. It's possible to apply `NRVO` for an object if at the moment when
  any return statement of this object is executed, the `return slot` won't be occupied by another object.


C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^

- Support capturing structured bindings in lambdas
  (`P1091R3 <https://wg21.link/p1091r3>`_ and `P1381R1 <https://wg21.link/P1381R1>`).
  This fixes issues `GH52720 <https://github.com/llvm/llvm-project/issues/52720>`_,
  `GH54300 <https://github.com/llvm/llvm-project/issues/54300>`_,
  `GH54301 <https://github.com/llvm/llvm-project/issues/54301>`_,
  and `GH49430 <https://github.com/llvm/llvm-project/issues/49430>`_.
- Consider explicitly defaulted constexpr/consteval special member function
  template instantiation to be constexpr/consteval even though a call to such
  a function cannot appear in a constant expression.
  (C++14 [dcl.constexpr]p6 (CWG DR647/CWG DR1358))
- Correctly defer dependent immediate function invocations until template instantiation.
  This fixes `GH55601 <https://github.com/llvm/llvm-project/issues/55601>`_.


- Enhanced the support for C++20 Modules, including: Partitions,
  Reachability, Header Unit and ``extern "C++"`` semantics.

- Implemented `P1103R3: Merging Modules <https://wg21.link/P1103R3>`_.
- Implemented `P1779R3: ABI isolation for member functions <https://wg21.link/P1779R3>`_.
- Implemented `P1874R1: Dynamic Initialization Order of Non-Local Variables in Modules <https://wg21.link/P1874R1>`_.
- Partially implemented `P1815R2: Translation-unit-local entities <https://wg21.link/P1815R2>`_.

- As per "Conditionally Trivial Special Member Functions" (P0848), it is
  now possible to overload destructors using concepts. Note that the rest
  of the paper about other special member functions is not yet implemented.
- Skip rebuilding lambda expressions in arguments of immediate invocations.
  This fixes `GH56183 <https://github.com/llvm/llvm-project/issues/56183>`_,
  `GH51695 <https://github.com/llvm/llvm-project/issues/51695>`_,
  `GH50455 <https://github.com/llvm/llvm-project/issues/50455>`_,
  `GH54872 <https://github.com/llvm/llvm-project/issues/54872>`_,
  `GH54587 <https://github.com/llvm/llvm-project/issues/54587>`_.

C++2b Feature Support
^^^^^^^^^^^^^^^^^^^^^

CUDA/HIP Language Changes in Clang
----------------------------------

- Added ``__noinline__`` as a keyword to avoid diagnostics due to usage of
  ``__attribute__((__noinline__))`` in CUDA/HIP programs.

Objective-C Language Changes in Clang
-------------------------------------

OpenCL Kernel Language Changes in Clang
---------------------------------------

- Improved/fixed misc issues in the builtin function support and diagnostics.
- Improved diagnostics for unknown extension pragma, subgroup functions and
  implicit function prototype.
- Added ``-cl-ext`` flag to the Clang driver to toggle extensions/features
  compiled for.
- Added ``cl_khr_subgroup_rotate`` extension.
- Removed some ``printf`` and ``hostcall`` related diagnostics when compiling
  for AMDGPU.
- Fixed alignment of pointer types in kernel arguments.

ABI Changes in Clang
--------------------

- When compiling C for ARM or AArch64, a zero-length bitfield in a ``struct``
  (e.g. ``int : 0``) no longer prevents the structure from being considered a
  homogeneous floating-point or vector aggregate. The new behavior agrees with
  the AAPCS specification, and matches the similar bug fix in GCC 12.1.
- Targeting AArch64, since D127209 LLVM now only preserves the z8-z23
  and p4-p15 registers across a call if the registers z0-z7 or p0-p3 are
  used to pass data into or out of a subroutine. The new behavior
  matches the AAPCS. Previously LLVM preserved z8-z23 and p4-p15 across
  a call if the callee had an SVE type anywhere in its signature. This
  would cause an incorrect use of the caller-preserved z8-z23 and p4-p15
  ABI for example if the 9th argument or greater were the first SVE type
  in the signature of a function.
- All copy constructors can now be trivial if they are not user-provided,
  regardless of the type qualifiers of the argument of the defaulted constructor,
  fixing dr2171.
  You can switch back to the old ABI behavior with the flag:
  ``-fclang-abi-compat=14.0``.

OpenMP Support in Clang
-----------------------

...

CUDA Support in Clang
---------------------

- ...

RISC-V Support in Clang
-----------------------

- ``sifive-7-rv32`` and ``sifive-7-rv64`` are no longer supported for `-mcpu`.
  Use `sifive-e76`, `sifive-s76`, or `sifive-u74` instead.

X86 Support in Clang
--------------------

- Support ``-mindirect-branch-cs-prefix`` for call and jmp to indirect thunk.

The ``_Float16`` type requires SSE2 feature and above due to the instruction
limitations. When using it on i386 targets, you need to specify ``-msse2``
explicitly.

For targets without F16C feature or above, please make sure:

- Use GCC 12.0 and above if you are using libgcc.
- If you are using compiler-rt, use the same version with the compiler.
  Early versions provided FP16 builtins in a different ABI. A workaround is to use
  a small code snippet to check the ABI if you cannot make sure of it.
- If you are using downstream runtimes that provide FP16 conversions, update
  them with the new ABI.

DWARF Support in Clang
----------------------

Arm and AArch64 Support in Clang
--------------------------------

- clang now supports the Cortex-M85 CPU, which can be chosen with
  ``-mcpu=cortex-m85``. By default, this has PACBTI turned on, but it can be
  disabled with ``-mcpu=cortex-m85+nopacbti``.
- clang now supports using C/C++ operators on sizeless SVE vectors such as
  ``svint32_t``. The set of supported operators is shown in the table Vector
  Operations found in the :ref:`Clang Language Extensions <Vector Operations>`
  document.

RISC-V Support in Clang
-----------------------

- Updates to the RISC-V vector intrinsics to align with ongoing additions to
  the RISC-V Vector intrinsics specification. Additionally, these intrinsics
  are now generated lazily, resulting a substantial improvement in
  compile-time for code including the vector intrinsics header.
- Intrinsics added for the RISC-V scalar crypto ('K') extensions.
- Intrinsics added for the RISC-V CLZ and CTZ instructions in the Zbb
  extension.
- An ABI lowering bug (resulting in incorrect LLVM IR generation) was fixed.
  The bug could be triggered in particular circumstances in C++ when passing a
  data-only struct that inherits from another struct.

SPIR-V Support in Clang
-----------------------

- Added flag ``-fintegrated-objemitter`` to enable use of experimental
  integrated LLVM backend when generating SPIR-V binary.
- The SPIR-V generator continues to produce typed pointers in this release
  despite the general switch of LLVM to opaque pointers.

Floating Point Support in Clang
-------------------------------

Internal API Changes
--------------------

Build System Changes
--------------------

AST Matchers
------------

clang-format
------------

clang-extdef-mapping
--------------------

libclang
--------

- Introduce new option ``CLANG_FORCE_MATCHING_LIBCLANG_SOVERSION`` that defaults to ON.
  This means that by default libclang's SOVERSION matches the major version of LLVM.
  Setting this to OFF makes the SOVERSION be the ABI compatible version (currently 13).
  See `discussion <https://discourse.llvm.org/t/rationale-for-removing-versioned-libclang-middle-ground-to-keep-it-behind-option/64410>`_
  here.

Static Analyzer
---------------
- `New CTU implementation
  <https://discourse.llvm.org/t/rfc-much-faster-cross-translation-unit-ctu-analysis-implementation/61728>`_
  that keeps the slow-down around 2x compared to the single-TU analysis, even
  in case of complex C++ projects. Still, it finds the majority of the "old"
  CTU findings. Besides, not more than ~3% of the bug reports are lost compared
  to single-TU analysis, the lost reports are highly likely to be false
  positives.

- Added a new checker ``alpha.unix.cstring.UninitializedRead`` this will check for uninitialized reads
  from common memory copy/manipulation functions such as ``memcpy``, ``mempcpy``, ``memmove``, ``memcmp``,
  ``strcmp``, ``strncmp``, ``strcpy``, ``strlen``, ``strsep`` and many more. Although
  this checker currently is in list of alpha checkers due to a false positive.

- Added a new checker ``alpha.unix.Errno``. This can find the first read
  of ``errno`` after successful standard function calls, such use of ``errno``
  could be unsafe.

- Deprecate the ``-analyzer-store region`` and
  ``-analyzer-opt-analyze-nested-blocks`` analyzer flags.
  These flags are still accepted, but a warning will be displayed.
  These flags will be rejected, thus turned into a hard error starting with
  ``clang-16``.

.. _release-notes-ubsan:

Undefined Behavior Sanitizer (UBSan)
------------------------------------

Core Analysis Improvements
==========================

- ...

New Issues Found
================

- ...

Python Binding Changes
----------------------

The following methods have been added:

-  ...

Significant Known Problems
==========================

Additional Information
======================

A wide variety of additional information is available on the `Clang web
page <https://clang.llvm.org/>`_. The web page contains versions of the
API documentation which are up-to-date with the Git version of
the source code. You can access versions of these documents specific to
this release by going into the "``clang/docs/``" directory in the Clang
tree.

If you have any questions or comments about Clang, please feel free to
contact us on the `Discourse forums (Clang Frontend category)
<https://discourse.llvm.org/c/clang/6>`_.
