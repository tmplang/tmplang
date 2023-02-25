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

Potentially Breaking Changes
============================
These changes are ones which we think may surprise users when upgrading to
Clang |release| because of the opportunity they pose for disruption to existing
code bases.

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

Improvements to Clang's diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

C++20 Feature Support
^^^^^^^^^^^^^^^^^^^^^

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

LoongArch Support in Clang
--------------------------

RISC-V Support in Clang
-----------------------

X86 Support in Clang
--------------------

WebAssembly Support in Clang
----------------------------

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

.. _release-notes-sanitizers:

Sanitizers
----------

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
