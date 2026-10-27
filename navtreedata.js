/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var NAVTREE =
[
  [ "myl7/fss", "index.html", [
    [ "Introduction", "index.html#autotoc_md1", null ],
    [ "Get Started", "index.html#autotoc_md2", [
      [ "Prerequisites", "index.html#autotoc_md3", null ],
      [ "Build", "index.html#autotoc_md4", null ],
      [ "CPU", "index.html#autotoc_md5", null ],
      [ "CPU Point Evaluation Fast Path", "index.html#autotoc_md6", null ],
      [ "GPU", "index.html#autotoc_md7", null ],
      [ "Samples", "index.html#autotoc_md8", null ],
      [ "Python", "index.html#autotoc_md9", null ],
      [ "Compiler Warnings", "index.html#autotoc_md10", null ],
      [ "nvcc 12.8: <tt>Uint</tt> as a <tt>__global__</tt> kernel template argument", "index.html#autotoc_md11", null ]
    ] ],
    [ "Benchmarks", "index.html#autotoc_md12", [
      [ "Comparison Curves", "index.html#autotoc_md13", null ],
      [ "Flamegraph", "index.html#autotoc_md14", null ]
    ] ],
    [ "License", "index.html#autotoc_md15", null ],
    [ "API capabilities for v1.3.0", "md_doc_2api-capabilities.html", [
      [ "C++ operations", "md_doc_2api-capabilities.html#autotoc_md51", null ],
      [ "Python operations", "md_doc_2api-capabilities.html#autotoc_md52", null ],
      [ "Host parallel execution", "md_doc_2api-capabilities.html#autotoc_md53", null ],
      [ "Backends and limits", "md_doc_2api-capabilities.html#autotoc_md54", null ],
      [ "Released interfaces and retry behavior", "md_doc_2api-capabilities.html#autotoc_md55", null ],
      [ "GPU scratch lifetime", "md_doc_2api-capabilities.html#autotoc_md56", null ],
      [ "JIT and packaging", "md_doc_2api-capabilities.html#autotoc_md57", null ]
    ] ],
    [ "Benchmark curve methodology", "md_doc_2bench__figures.html", [
      [ "Work and timing", "md_doc_2bench__figures.html#autotoc_md59", null ],
      [ "Native configurations and validation", "md_doc_2bench__figures.html#autotoc_md60", null ],
      [ "GPU block-size sensitivity", "md_doc_2bench__figures.html#autotoc_md61", null ],
      [ "Run and export", "md_doc_2bench__figures.html#autotoc_md62", null ]
    ] ],
    [ "Benchmark coverage", "md_doc_2benchmarks.html", [
      [ "Coverage", "md_doc_2benchmarks.html#autotoc_md64", null ],
      [ "GPU kernel register usage", "md_doc_2benchmarks.html#autotoc_md65", null ]
    ] ],
    [ "Third-party DPF/DCF Library Benchmarks", "md_doc_2bench__third__parties.html", [
      [ "Libraries", "md_doc_2bench__third__parties.html#autotoc_md67", null ],
      [ "Benchmarked Operations", "md_doc_2bench__third__parties.html#autotoc_md68", null ],
      [ "Settings", "md_doc_2bench__third__parties.html#autotoc_md69", [
        [ "libdpf (Rust)", "md_doc_2bench__third__parties.html#autotoc_md70", null ],
        [ "libfss (C++)", "md_doc_2bench__third__parties.html#autotoc_md71", null ],
        [ "google_dpf (C++)", "md_doc_2bench__third__parties.html#autotoc_md72", null ],
        [ "GPU-DPF CPU (C++)", "md_doc_2bench__third__parties.html#autotoc_md73", null ],
        [ "GPU-DPF GPU (Python/CUDA)", "md_doc_2bench__third__parties.html#autotoc_md74", null ],
        [ "EzPC (CUDA C++)", "md_doc_2bench__third__parties.html#autotoc_md75", null ],
        [ "fss-rs 0.6.0 (Rust)", "md_doc_2bench__third__parties.html#autotoc_md76", null ],
        [ "fss 0.7.0 CPU (C)", "md_doc_2bench__third__parties.html#autotoc_md77", null ],
        [ "fss 0.7.0 GPU (C/CUDA)", "md_doc_2bench__third__parties.html#autotoc_md78", null ],
        [ "fss CPU (C++/CUDA)", "md_doc_2bench__third__parties.html#autotoc_md79", null ],
        [ "fss GPU (C++/CUDA)", "md_doc_2bench__third__parties.html#autotoc_md80", null ],
        [ "mastic (Rust)", "md_doc_2bench__third__parties.html#autotoc_md81", null ],
        [ "plasma (Rust)", "md_doc_2bench__third__parties.html#autotoc_md82", null ],
        [ "sycret (Rust)", "md_doc_2bench__third__parties.html#autotoc_md83", null ],
        [ "dpf-cpp (C++)", "md_doc_2bench__third__parties.html#autotoc_md84", null ],
        [ "dpfplusplus (C++)", "md_doc_2bench__third__parties.html#autotoc_md85", null ],
        [ "torchcsprng (C++/CUDA)", "md_doc_2bench__third__parties.html#autotoc_md86", null ]
      ] ],
      [ "Running", "md_doc_2bench__third__parties.html#autotoc_md87", null ],
      [ "Results", "md_doc_2bench__third__parties.html#autotoc_md88", [
        [ "Hardware", "md_doc_2bench__third__parties.html#autotoc_md89", null ],
        [ "CPU", "md_doc_2bench__third__parties.html#autotoc_md90", null ],
        [ "GPU", "md_doc_2bench__third__parties.html#autotoc_md91", null ]
      ] ]
    ] ],
    [ "dpf++ (rh3nry/dpfplusplus) CPU Benchmark", "md_doc_2bench-dpfplusplus.html", [
      [ "Provenance", "md_doc_2bench-dpfplusplus.html#autotoc_md93", null ],
      [ "License status and why nothing is vendored", "md_doc_2bench-dpfplusplus.html#autotoc_md94", null ],
      [ "How the bench acquires the code", "md_doc_2bench-dpfplusplus.html#autotoc_md95", null ],
      [ "The three text fixes", "md_doc_2bench-dpfplusplus.html#autotoc_md96", null ],
      [ "Build and benchmark shape", "md_doc_2bench-dpfplusplus.html#autotoc_md97", null ],
      [ "Reproducing", "md_doc_2bench-dpfplusplus.html#autotoc_md98", null ]
    ] ],
    [ "AES-128 MMO Software PRG: T-table vs Textbook", "md_doc_2bench__aes128__soft.html", [
      [ "Settings", "md_doc_2bench__aes128__soft.html#autotoc_md100", null ],
      [ "Hardware", "md_doc_2bench__aes128__soft.html#autotoc_md101", null ],
      [ "GPU Results", "md_doc_2bench__aes128__soft.html#autotoc_md102", [
        [ "GPU Resource Usage (sm_120)", "md_doc_2bench__aes128__soft.html#autotoc_md103", null ]
      ] ],
      [ "CPU Results", "md_doc_2bench__aes128__soft.html#autotoc_md104", null ],
      [ "Software-AES DPF Evaluation", "md_doc_2bench__aes128__soft.html#autotoc_md105", null ],
      [ "Analysis", "md_doc_2bench__aes128__soft.html#autotoc_md106", null ]
    ] ],
    [ "VDPF and VDMPF Merkle-fold proof accumulation: security analysis", "md_doc_2vdpf__merkle__fold__security.html", [
      [ "1. Overview", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md108", null ],
      [ "2. The original construction", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md109", null ],
      [ "3. The Merkle-fold Prove", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md110", null ],
      [ "4. Correctness", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md111", null ],
      [ "5. Soundness", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md112", [
        [ "Assumptions", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md113", null ],
        [ "The reduction", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md114", null ]
      ] ],
      [ "6. Required hash properties", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md115", null ],
      [ "7. Discussion", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md116", null ],
      [ "8. The VDMPF Merkle-fold accumulation", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md117", [
        [ "Construction", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md118", null ],
        [ "Correctness", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md119", null ],
        [ "Soundness", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md120", null ]
      ] ],
      [ "9. References", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md121", null ]
    ] ],
    [ "Concepts", "concepts.html", "concepts" ],
    [ "Classes", "annotated.html", [
      [ "Class List", "annotated.html", "annotated_dup" ],
      [ "Class Index", "classes.html", null ],
      [ "Class Members", "functions.html", [
        [ "All", "functions.html", null ],
        [ "Functions", "functions_func.html", null ],
        [ "Variables", "functions_vars.html", null ],
        [ "Typedefs", "functions_type.html", null ]
      ] ]
    ] ],
    [ "Files", "files.html", [
      [ "File List", "files.html", "files_dup" ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"aes128__feistel_8cuh.html",
"structfss_1_1HalfTreeDpf_1_1Key.html"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';