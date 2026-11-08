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
    [ "Build and install", "index.html#autotoc_md1", null ],
    [ "Generate and evaluate a point function", "index.html#autotoc_md2", null ],
    [ "Evaluate on CUDA", "index.html#autotoc_md3", null ],
    [ "Use Python", "index.html#autotoc_md4", null ],
    [ "Run the samples", "index.html#autotoc_md5", null ],
    [ "Measure performance", "index.html#autotoc_md6", null ],
    [ "License", "index.html#autotoc_md7", null ],
    [ "Benchmark curve methodology", "md_doc_2bench__figures.html", [
      [ "Work and timing", "md_doc_2bench__figures.html#autotoc_md42", null ],
      [ "Native configurations and validation", "md_doc_2bench__figures.html#autotoc_md43", null ],
      [ "GPU block-size sensitivity", "md_doc_2bench__figures.html#autotoc_md44", null ],
      [ "Run and export", "md_doc_2bench__figures.html#autotoc_md45", null ]
    ] ],
    [ "Benchmark coverage", "md_doc_2benchmarks.html", [
      [ "Coverage", "md_doc_2benchmarks.html#autotoc_md47", null ],
      [ "Historical GPU kernel register usage", "md_doc_2benchmarks.html#autotoc_md48", null ]
    ] ],
    [ "Third-party DPF/DCF Library Benchmarks", "md_doc_2bench__third__parties.html", [
      [ "Libraries", "md_doc_2bench__third__parties.html#autotoc_md50", null ],
      [ "Benchmarked Operations", "md_doc_2bench__third__parties.html#autotoc_md51", null ],
      [ "Settings", "md_doc_2bench__third__parties.html#autotoc_md52", [
        [ "libdpf (Rust)", "md_doc_2bench__third__parties.html#autotoc_md53", null ],
        [ "libfss (C++)", "md_doc_2bench__third__parties.html#autotoc_md54", null ],
        [ "google_dpf (C++)", "md_doc_2bench__third__parties.html#autotoc_md55", null ],
        [ "GPU-DPF CPU (C++)", "md_doc_2bench__third__parties.html#autotoc_md56", null ],
        [ "GPU-DPF GPU (Python/CUDA)", "md_doc_2bench__third__parties.html#autotoc_md57", null ],
        [ "EzPC (CUDA C++)", "md_doc_2bench__third__parties.html#autotoc_md58", null ],
        [ "fss-rs 0.6.0 (Rust)", "md_doc_2bench__third__parties.html#autotoc_md59", null ],
        [ "fss 0.7.0 CPU (C)", "md_doc_2bench__third__parties.html#autotoc_md60", null ],
        [ "fss 0.7.0 GPU (C/CUDA)", "md_doc_2bench__third__parties.html#autotoc_md61", null ],
        [ "fss CPU (C++/CUDA)", "md_doc_2bench__third__parties.html#autotoc_md62", null ],
        [ "fss GPU (C++/CUDA)", "md_doc_2bench__third__parties.html#autotoc_md63", null ],
        [ "mastic (Rust)", "md_doc_2bench__third__parties.html#autotoc_md64", null ],
        [ "plasma (Rust)", "md_doc_2bench__third__parties.html#autotoc_md65", null ],
        [ "sycret (Rust)", "md_doc_2bench__third__parties.html#autotoc_md66", null ],
        [ "dpf-cpp (C++)", "md_doc_2bench__third__parties.html#autotoc_md67", null ],
        [ "dpfplusplus (C++)", "md_doc_2bench__third__parties.html#autotoc_md68", null ],
        [ "torchcsprng (C++/CUDA)", "md_doc_2bench__third__parties.html#autotoc_md69", null ]
      ] ],
      [ "Running", "md_doc_2bench__third__parties.html#autotoc_md70", null ],
      [ "Results", "md_doc_2bench__third__parties.html#autotoc_md71", [
        [ "Hardware", "md_doc_2bench__third__parties.html#autotoc_md72", null ],
        [ "CPU", "md_doc_2bench__third__parties.html#autotoc_md73", null ],
        [ "GPU", "md_doc_2bench__third__parties.html#autotoc_md74", null ]
      ] ]
    ] ],
    [ "dpf++ (rh3nry/dpfplusplus) CPU Benchmark", "md_doc_2bench-dpfplusplus.html", [
      [ "Provenance", "md_doc_2bench-dpfplusplus.html#autotoc_md76", null ],
      [ "License status and why nothing is vendored", "md_doc_2bench-dpfplusplus.html#autotoc_md77", null ],
      [ "How the bench acquires the code", "md_doc_2bench-dpfplusplus.html#autotoc_md78", null ],
      [ "The three text fixes", "md_doc_2bench-dpfplusplus.html#autotoc_md79", null ],
      [ "Build and benchmark shape", "md_doc_2bench-dpfplusplus.html#autotoc_md80", null ],
      [ "Reproducing", "md_doc_2bench-dpfplusplus.html#autotoc_md81", null ]
    ] ],
    [ "AES-128 MMO Software PRG: T-table vs Textbook", "md_doc_2bench__aes128__soft.html", [
      [ "Settings", "md_doc_2bench__aes128__soft.html#autotoc_md83", null ],
      [ "Hardware", "md_doc_2bench__aes128__soft.html#autotoc_md84", null ],
      [ "GPU Results", "md_doc_2bench__aes128__soft.html#autotoc_md85", [
        [ "GPU Resource Usage (sm_120)", "md_doc_2bench__aes128__soft.html#autotoc_md86", null ]
      ] ],
      [ "CPU Results", "md_doc_2bench__aes128__soft.html#autotoc_md87", null ],
      [ "Software-AES DPF Evaluation", "md_doc_2bench__aes128__soft.html#autotoc_md88", null ],
      [ "Analysis", "md_doc_2bench__aes128__soft.html#autotoc_md89", null ]
    ] ],
    [ "VDPF and VDMPF Merkle-fold proof accumulation: security analysis", "md_doc_2vdpf__merkle__fold__security.html", [
      [ "1. Overview", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md91", null ],
      [ "2. The original construction", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md92", null ],
      [ "3. The Merkle-fold Prove", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md93", null ],
      [ "4. Correctness", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md94", null ],
      [ "5. Soundness", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md95", [
        [ "Assumptions", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md96", null ],
        [ "The reduction", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md97", null ]
      ] ],
      [ "6. Required hash properties", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md98", null ],
      [ "7. Discussion", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md99", null ],
      [ "8. The VDMPF Merkle-fold accumulation", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md100", [
        [ "Construction", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md101", null ],
        [ "Correctness", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md102", null ],
        [ "Soundness", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md103", null ]
      ] ],
      [ "9. References", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md104", null ]
    ] ],
    [ "Topics", "topics.html", "topics" ],
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
"aes128__ccr_8cuh.html",
"md_doc_2bench__figures.html#autotoc_md42"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';