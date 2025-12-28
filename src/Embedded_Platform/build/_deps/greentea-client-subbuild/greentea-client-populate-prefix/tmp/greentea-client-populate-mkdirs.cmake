# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/dinh-kiet/bfmc_embed/Embedded_Platform/build/_deps/greentea-client-src"
  "/home/dinh-kiet/bfmc_embed/Embedded_Platform/build/_deps/greentea-client-build"
  "/home/dinh-kiet/bfmc_embed/Embedded_Platform/build/_deps/greentea-client-subbuild/greentea-client-populate-prefix"
  "/home/dinh-kiet/bfmc_embed/Embedded_Platform/build/_deps/greentea-client-subbuild/greentea-client-populate-prefix/tmp"
  "/home/dinh-kiet/bfmc_embed/Embedded_Platform/build/_deps/greentea-client-subbuild/greentea-client-populate-prefix/src/greentea-client-populate-stamp"
  "/home/dinh-kiet/bfmc_embed/Embedded_Platform/build/_deps/greentea-client-subbuild/greentea-client-populate-prefix/src"
  "/home/dinh-kiet/bfmc_embed/Embedded_Platform/build/_deps/greentea-client-subbuild/greentea-client-populate-prefix/src/greentea-client-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/dinh-kiet/bfmc_embed/Embedded_Platform/build/_deps/greentea-client-subbuild/greentea-client-populate-prefix/src/greentea-client-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/dinh-kiet/bfmc_embed/Embedded_Platform/build/_deps/greentea-client-subbuild/greentea-client-populate-prefix/src/greentea-client-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
