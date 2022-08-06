# Adding new unit tests

- To add a new unit test to an existing folder, just add it's source file to the corresponding `CMakeLists.txt`.

- To add a new unit test folder, create a `CMakeLists.txt` file and use the `add_tmplang_unittest` macro. The first argument
is the name of the test target, which **must** end in `Tests` in order for `lit` to find them. After that, you can list as many source files containing tests as you need.
