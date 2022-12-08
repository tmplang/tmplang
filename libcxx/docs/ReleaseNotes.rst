=========================================
Libc++ 15.0.0 Release Notes
=========================================

.. contents::
   :local:
   :depth: 2

Written by the `Libc++ Team <https://libcxx.llvm.org>`_

Introduction
============

This document contains the release notes for the libc++ C++ Standard Library,
part of the LLVM Compiler Infrastructure, release 16.0.0. Here we describe the
status of libc++ in some detail, including major improvements from the previous
release and new feature work. For the general LLVM release notes, see `the LLVM
documentation <https://llvm.org/docs/ReleaseNotes.html>`_. All LLVM releases may
be downloaded from the `LLVM releases web site <https://llvm.org/releases/>`_.

For more information about libc++, please see the `Libc++ Web Site
<https://libcxx.llvm.org>`_ or the `LLVM Web Site <https://llvm.org>`_.

Note that if you are reading this file from a Git checkout or the
main Libc++ web page, this document applies to the *next* release, not
the current one. To see the release notes for a specific release, please
see the `releases page <https://llvm.org/releases/>`_.

What's New in Libc++ 16.0.0?
============================

Implemented Papers
------------------

- P1004R2 - Making ``std::vector`` constexpr
- P0627R6 - Function to mark unreachable code
- P1165R1 - Make stateful allocator propagation more consistent for ``operator+(basic_string)``
- P0674R1 - Support arrays in ``make_shared`` and ``allocate_shared``
- P0980R1 - Making ``std::string`` constexpr
- P2216R3 - ``std::format`` improvements
- P0174R2 - Deprecating Vestigial Library Parts in C++17
- N4190 - Removing ``auto_ptr``, ``random_shuffle()``, And Old ``<functional>`` Stuff
- P0154R1 - Hardware inference size
- P0618R0 - Deprecating ``<codecvt>``
- P2418R2 - Add support for ``std::generator``-like types to ``std::format``
- LWG3659 - Consider ``ATOMIC_FLAG_INIT`` undeprecation
- P1423R3 - ``char8_t`` backward compatibility remediation
- P2508R1 - Exposing ``std::basic-format-string``

- Marked the following papers as "Complete" (note that some of those might have
  been implemented in a previous release but not marked as such):

    - P1207R4 - Movability of Single-pass Iterators
    - P1474R1 - Helpful pointers for ``ContiguousIterator``
    - P1522R1 - Iterator Difference Type and Integer Overflow
    - P1523R1 - Views and Size Types
    - P1456R1 - Move-only views
    - P1870R1 - ``forwarding-range`` is too subtle
    - P1878R1 - Constraining Readable Types
    - P1970R2 - Consistency for ``size()`` functions: Add ``ranges::ssize``
    - P1983R0 - Wording for GB301, US296, US292, US291, and US283

Improvements and New Features
-----------------------------

Deprecations and Removals
-------------------------

- The header ``<experimental/filesystem>`` has been removed. Instead, use
  ``<filesystem>`` header. The associated macro
  ``_LIBCPP_DEPRECATED_EXPERIMENTAL_FILESYSTEM`` has been removed too.

- The C++14 function ``std::quoted(const char*)`` is no longer supported in
  C++03 or C++11 modes.

- Setting a custom debug handler with ``std::__libcpp_debug_function`` is not
  supported anymore. Please migrate to using the new support for
  :ref:`assertions <assertions-mode>` instead.

- ``std::function`` has been removed in C++03. If you are using it, please remove usages
  or upgrade to C++11 or later. It is possible to re-enable ``std::function`` in C++03 by defining
  ``_LIBCPP_ENABLE_CXX03_FUNCTION``. This option will be removed in LLVM 16.

- ``unary_function`` and ``binary_function`` are now marked as ``[[deprecated]]`` in C++11 and later.
  Deprecation warnings can be disabled by defining ``_LIBCPP_DISABLE_DEPRECATION_WARNINGS``, however
  this disables all deprecation warnings, not only those for ``unary_function`` and ``binary_function``.
  Also note that starting in LLVM 16, ``unary_function`` and ``binary_function`` will be removed entirely
  (not only deprecated) in C++17 and above, as mandated by the Standard.

- The contents of ``<codecvt>``, ``wstring_convert`` and ``wbuffer_convert`` have been marked as deprecated.
  To disable deprecation warnings you have to define ``_LIBCPP_DISABLE_DEPRECATION_WARNINGS``. Note that this
  disables all deprecation warnings.

- The ``_LIBCPP_DISABLE_EXTERN_TEMPLATE`` macro is not honored anymore when defined by
  users of libc++. Instead, users not wishing to take a dependency on libc++ should link
  against the static version of libc++, which will result in no dependency being
  taken against the shared library.

- The ``_LIBCPP_ABI_UNSTABLE`` macro has been removed in favour of setting
  ``_LIBCPP_ABI_VERSION=2``. This should not have any impact on users because
  they were not supposed to set ``_LIBCPP_ABI_UNSTABLE`` manually, however we
  still feel that it is worth mentioning in the release notes in case some users
  had been doing it.

- The integer distributions ``binomial_distribution``, ``discrete_distribution``,
  ``geometric_distribution``, ``negative_binomial_distribution``, ``poisson_distribution``,
  and ``uniform_int_distribution`` now conform to the Standard by rejecting
  template parameter types other than ``short``, ``int``, ``long``, ``long long``,
  and the unsigned versions thereof. As an extension, ``int8_t``, ``__int128_t`` and
  their unsigned versions are supported too. In particular, instantiating these
  distributions with non-integer types like ``bool`` and ``char`` will not compile
  anymore.

Upcoming Deprecations and Removals
----------------------------------

API Changes
-----------
- The comparison operators on ``thread::id`` are now defined as free-standing
  functions instead of as hidden friends, in conformance with the C++ standard.
  Also see `issue 56187 <https://github.com/llvm/llvm-project/issues/56187>`_.

ABI Affecting Changes
---------------------
- In freestanding mode, ``atomic<small enum class>`` does not contain a lock byte anymore if the platform
  can implement lockfree atomics for that size. More specifically, in LLVM <= 11.0.1, an ``atomic<small enum class>``
  would not contain a lock byte. This was broken in LLVM >= 12.0.0, where it started including a lock byte despite
  the platform supporting lockfree atomics for that size. Starting in LLVM 15.0.1, the ABI for these types has been
  restored to what it used to be (no lock byte), which is the most efficient implementation.

  This ABI break only affects users that compile with ``-ffreestanding``, and only for ``atomic<T>`` where ``T``
  is a non-builtin type that could be lockfree on the platform. See https://llvm.org/D133377 for more details.

Build System Changes
--------------------

- Support for standalone builds have been entirely removed from libc++, libc++abi and
  libunwind. Please use :ref:`these instructions <build instructions>` for building
  libc++, libc++abi and/or libunwind.

- The ``{LIBCXX,LIBCXXABI,LIBUNWIND}_TARGET_TRIPLE``, ``{LIBCXX,LIBCXXABI,LIBUNWIND}_SYSROOT`` and
  ``{LIBCXX,LIBCXXABI,LIBUNWIND}_GCC_TOOLCHAIN`` CMake variables have been removed. Instead, please
  use the ``CMAKE_CXX_COMPILER_TARGET``, ``CMAKE_SYSROOT`` and ``CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN``
  variables provided by CMake.

- Previously, the C++ ABI library headers would be installed inside ``<prefix>/include/c++/v1``
  alongside the libc++ headers as part of building libc++. This is not the case anymore -- the
  ABI library is expected to install its headers where it wants them as part of its own build.
  Note that no action is required for most users, who build libc++ against libc++abi, since
  libc++abi already installs its headers in the right location. However, vendors building
  libc++ against alternate ABI libraries should make sure that their ABI library installs
  its own headers.

- The legacy testing configuration is now deprecated and will be removed in LLVM 16. For
  most users, this should not have any impact. However, if you are testing libc++, libc++abi, or
  libunwind in a configuration or on a platform that used to be supported by the legacy testing
  configuration and isn't supported by one of the configurations in ``libcxx/test/configs``,
  ``libcxxabi/test/configs``, or ``libunwind/test/configs``, please move to one of those
  configurations or define your own.

- MinGW DLL builds of libc++ no longer use dllimport in their headers, which
  means that the same set of installed headers works for both DLL and static
  linkage. This means that distributors finally can build both library
  versions with a single CMake invocation.

- The ``LIBCXX_HIDE_FROM_ABI_PER_TU_BY_DEFAULT`` configuration option has been removed. Indeed,
  the risk of ODR violations from mixing different versions of libc++ in the same program has
  been mitigated with a different technique that is simpler and does not have the drawbacks of
  using internal linkage.
