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
    [ "Get started", "index.html#autotoc_md2", [
      [ "Prerequisites", "index.html#autotoc_md3", null ],
      [ "Build", "index.html#autotoc_md4", null ],
      [ "CPU", "index.html#autotoc_md5", null ],
      [ "GPU", "index.html#autotoc_md6", null ],
      [ "Samples", "index.html#autotoc_md7", null ],
      [ "Python", "index.html#autotoc_md8", null ]
    ] ],
    [ "Benchmarks", "index.html#autotoc_md9", [
      [ "Comparison curves", "index.html#autotoc_md10", null ],
      [ "Run benchmarks", "index.html#autotoc_md11", null ],
      [ "Flamegraph", "index.html#autotoc_md12", null ]
    ] ],
    [ "License", "index.html#autotoc_md13", null ],
    [ "Benchmark curve methodology", "md_doc_2bench__figures.html", [
      [ "Work and timing", "md_doc_2bench__figures.html#autotoc_md48", null ],
      [ "Native configurations and validation", "md_doc_2bench__figures.html#autotoc_md49", null ],
      [ "GPU block-size sensitivity", "md_doc_2bench__figures.html#autotoc_md50", null ],
      [ "Run and export", "md_doc_2bench__figures.html#autotoc_md51", null ]
    ] ],
    [ "Benchmark runner and third-party comparisons", "md_doc_2bench__third__parties.html", [
      [ "Libraries", "md_doc_2bench__third__parties.html#autotoc_md53", null ],
      [ "Benchmarked Operations", "md_doc_2bench__third__parties.html#autotoc_md54", null ],
      [ "Settings", "md_doc_2bench__third__parties.html#autotoc_md55", [
        [ "libdpf (Rust)", "md_doc_2bench__third__parties.html#autotoc_md56", null ],
        [ "libfss (C++)", "md_doc_2bench__third__parties.html#autotoc_md57", null ],
        [ "google_dpf (C++)", "md_doc_2bench__third__parties.html#autotoc_md58", null ],
        [ "GPU-DPF CPU (C++)", "md_doc_2bench__third__parties.html#autotoc_md59", null ],
        [ "GPU-DPF GPU (Python/CUDA)", "md_doc_2bench__third__parties.html#autotoc_md60", null ],
        [ "EzPC (CUDA C++)", "md_doc_2bench__third__parties.html#autotoc_md61", null ],
        [ "fss-rs 0.6.0 (Rust)", "md_doc_2bench__third__parties.html#autotoc_md62", null ],
        [ "fss 0.7.0 CPU (C)", "md_doc_2bench__third__parties.html#autotoc_md63", null ],
        [ "fss 0.7.0 GPU (C/CUDA)", "md_doc_2bench__third__parties.html#autotoc_md64", null ],
        [ "fss CPU (C++/CUDA)", "md_doc_2bench__third__parties.html#autotoc_md65", null ],
        [ "fss GPU (C++/CUDA)", "md_doc_2bench__third__parties.html#autotoc_md66", null ],
        [ "mastic (Rust)", "md_doc_2bench__third__parties.html#autotoc_md67", null ],
        [ "plasma (Rust)", "md_doc_2bench__third__parties.html#autotoc_md68", null ],
        [ "sycret (Rust)", "md_doc_2bench__third__parties.html#autotoc_md69", null ],
        [ "dpf-cpp (C++)", "md_doc_2bench__third__parties.html#autotoc_md70", null ],
        [ "dpfplusplus (C++)", "md_doc_2bench__third__parties.html#autotoc_md71", null ],
        [ "torchcsprng (C++/CUDA)", "md_doc_2bench__third__parties.html#autotoc_md72", null ]
      ] ],
      [ "Running", "md_doc_2bench__third__parties.html#autotoc_md73", [
        [ "Run the native suite", "md_doc_2bench__third__parties.html#autotoc_md74", null ],
        [ "Native benchmark coverage", "md_doc_2bench__third__parties.html#autotoc_md75", null ],
        [ "Run profiles", "md_doc_2bench__third__parties.html#autotoc_md76", [
          [ "Full-domain launch control", "md_doc_2bench__third__parties.html#autotoc_md77", null ]
        ] ],
        [ "Run third-party comparisons", "md_doc_2bench__third__parties.html#autotoc_md78", null ]
      ] ],
      [ "Results", "md_doc_2bench__third__parties.html#autotoc_md79", null ]
    ] ],
    [ "dpf++ (rh3nry/dpfplusplus) CPU Benchmark", "md_doc_2bench-dpfplusplus.html", [
      [ "Provenance", "md_doc_2bench-dpfplusplus.html#autotoc_md81", null ],
      [ "License status and why nothing is vendored", "md_doc_2bench-dpfplusplus.html#autotoc_md82", null ],
      [ "How the bench acquires the code", "md_doc_2bench-dpfplusplus.html#autotoc_md83", null ],
      [ "The three text fixes", "md_doc_2bench-dpfplusplus.html#autotoc_md84", null ],
      [ "Build and benchmark shape", "md_doc_2bench-dpfplusplus.html#autotoc_md85", null ],
      [ "Reproducing", "md_doc_2bench-dpfplusplus.html#autotoc_md86", null ]
    ] ],
    [ "Software AES-128 MMO: T-table and textbook implementations", "md_doc_2bench__aes128__soft.html", [
      [ "Settings", "md_doc_2bench__aes128__soft.html#autotoc_md88", null ],
      [ "Hardware", "md_doc_2bench__aes128__soft.html#autotoc_md89", null ],
      [ "GPU Results", "md_doc_2bench__aes128__soft.html#autotoc_md90", [
        [ "GPU Resource Usage (sm_120)", "md_doc_2bench__aes128__soft.html#autotoc_md91", null ]
      ] ],
      [ "CPU Results", "md_doc_2bench__aes128__soft.html#autotoc_md92", null ],
      [ "Software-AES DPF Evaluation", "md_doc_2bench__aes128__soft.html#autotoc_md93", null ],
      [ "Analysis", "md_doc_2bench__aes128__soft.html#autotoc_md94", null ]
    ] ],
    [ "Migration guide", "md_doc_2migration.html", [
      [ "v1.3.0 to v2.0.0", "md_doc_2migration.html#autotoc_md96", [
        [ "Keep compiling with the compatibility switch", "md_doc_2migration.html#autotoc_md97", null ],
        [ "Regenerate persisted keys and proofs", "md_doc_2migration.html#autotoc_md98", null ],
        [ "Construct schemes through their constructors", "md_doc_2migration.html#autotoc_md99", null ],
        [ "Drop removed template parameters", "md_doc_2migration.html#autotoc_md100", null ],
        [ "Own the generation retry", "md_doc_2migration.html#autotoc_md101", null ],
        [ "Multi-point keys store wider data", "md_doc_2migration.html#autotoc_md102", null ],
        [ "Move wide groups to Uint128", "md_doc_2migration.html#autotoc_md103", null ],
        [ "Update primitive backends", "md_doc_2migration.html#autotoc_md104", null ],
        [ "Check errors where the library now throws", "md_doc_2migration.html#autotoc_md105", null ],
        [ "Move off the deprecated entries", "md_doc_2migration.html#autotoc_md106", null ]
      ] ],
      [ "v1.2.0 to v1.3.0", "md_doc_2migration.html#autotoc_md107", [
        [ "Keep your v1.2.0 calls", "md_doc_2migration.html#autotoc_md108", null ],
        [ "Review the behavior changes", "md_doc_2migration.html#autotoc_md109", null ],
        [ "Keys and proofs", "md_doc_2migration.html#autotoc_md110", null ],
        [ "Adopt the v1.3.0 additions when useful", "md_doc_2migration.html#autotoc_md111", null ]
      ] ]
    ] ],
    [ "The VDPF and VDMPF proof-hash replacement attempt: a complete record", "md_doc_2vdpf__proof__hash__exploration.html", [
      [ "Overview", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md113", null ],
      [ "Timeline", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md114", null ],
      [ "Part I — The requirement", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md116", null ],
      [ "1. Required property and code scope", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md117", null ],
      [ "2. What one-way MMO does and does not establish", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md118", null ],
      [ "Part II — First proposal and rejection", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md119", null ],
      [ "3. Earlier candidates: valid attacks and corrected claims", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md120", [
        [ "3.1 Fixed-key XOR-injection chains", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md121", null ],
        [ "3.2 CBC-style compression", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md122", null ],
        [ "3.3 Independent block hashing with symmetric combination", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md123", null ]
      ] ],
      [ "4. The proposed two-lane construction is broken at the target level", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md124", [
        [ "Attack 1: keep one lane unchanged", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md125", null ],
        [ "Attack 2: make both lane inputs vary together", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md126", null ]
      ] ],
      [ "5. A precise ideal-cipher bound", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md127", [
        [ "5.1 What an invertible mixing layer cannot repair", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md128", null ]
      ] ],
      [ "Part III — Survey and the selected constructions", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md129", null ],
      [ "6. Survey of suitable and unsuitable AES construction families", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md130", null ],
      [ "7. Selected construction: two Hirose compression steps", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md131", [
        [ "7.1 Exact function and encoding", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md132", null ],
        [ "7.2 Ideal-cipher theorem and local composition proof", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md133", null ],
        [ "7.3 Concrete AES and implementation approval", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md134", null ],
        [ "7.4 Third review: truncated ChaCha20 instantiation", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md135", [
          [ "Exact primitive and scope", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md136", null ],
          [ "Where the AES proof uses a permutation", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md137", null ],
          [ "A direct random-function bound", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md138", null ],
          [ "Truncation and concrete assumptions", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md139", null ]
        ] ]
      ] ],
      [ "Part IV — Roads not taken", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md140", null ],
      [ "8. Protocol-specific weakening of H'", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md141", null ],
      [ "9. What the cited reference implementation actually computes", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md142", null ],
      [ "Part V — Performance: estimates and measurements", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md143", null ],
      [ "10. Performance budget and measurement limits", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md144", [
        [ "10.1 CPU costs", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md145", null ],
        [ "10.2 GPU costs and a conditional speed budget", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md146", null ],
        [ "10.3 ChaCha20 cost estimate, without new measurements", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md147", null ],
        [ "10.4 Measured primitive costs (2026-10-17)", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md148", null ]
      ] ],
      [ "Part VI — Deployment and implementation requirements", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md149", null ],
      [ "11. Deployment and implementation requirements", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md150", [
        [ "11.1 ChaCha implementation checks across native CPU, CUDA, and Python/JIT", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md151", null ]
      ] ],
      [ "Part VII — Decision record", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md152", null ],
      [ "12. Decision record", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md153", null ],
      [ "13. References", "md_doc_2vdpf__proof__hash__exploration.html#autotoc_md154", null ]
    ] ],
    [ "VDPF and VDMPF Merkle-fold proof accumulation: security analysis", "md_doc_2vdpf__merkle__fold__security.html", [
      [ "1. Overview", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md156", null ],
      [ "2. The original construction", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md157", null ],
      [ "3. The Merkle-fold Prove", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md158", null ],
      [ "4. Correctness", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md159", null ],
      [ "5. Soundness", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md160", [
        [ "Assumptions", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md161", null ],
        [ "The reduction", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md162", null ]
      ] ],
      [ "6. Required hash properties", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md163", null ],
      [ "7. Discussion", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md164", null ],
      [ "8. The VDMPF Merkle-fold accumulation", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md165", [
        [ "Construction", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md166", null ],
        [ "Correctness", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md167", null ],
        [ "Soundness", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md168", null ]
      ] ],
      [ "9. References", "md_doc_2vdpf__merkle__fold__security.html#autotoc_md169", null ]
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
"md_doc_2bench__figures.html#autotoc_md51"
];

var SYNCONMSG = 'click to disable panel synchronization';
var SYNCOFFMSG = 'click to enable panel synchronization';
var LISTOFALLMEMBERS = 'List of all members';