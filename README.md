# codonMNV
in-codon Multiple Nucleotide Variant detector for GATK derived Multisample VCF files.

This tool is part of the master thesis:
> Large-scale characterization of intraspecific variation in outer membrane proteins of *Treponema pallidum*

Written by N. M. Gerbes
at the Institute for Bioinformatics and Medical Informatics, University Tübingen

A detailed description of the tool can be found in section 3.2: "Implementation of codonMNV for in-
codon MNV detection"


## Overview

Standard variant annotation tools may fail to identify in-codon MNVs in multi-sample VCF files when variants are represented as separate SNV entries. `codonMNV` addresses this by evaluating sample-specific genotype information directly from the VCF genotype fields.

For each codon, the tool collects all SNVs located within the three codon positions and tests whether at least two affected codon positions are present in the same sample. Valid combinations are emitted as additional cMNV entries while preserving all original VCF entries.

## cMNV Definition

An in-codon MNV (cMNV) is defined as a set of two or three SNVs located at distinct positions within the same codon and therefore present in the same chromosome and sample.

For each sample, a variant is considered present if the genotype value indicates at least one alternative allele. Genotypes equal to `0` or non-numeric genotype values are treated as absent.

## Features

- Detects 2-base and 3-base in-codon MNVs.
- Processes multi-sample VCF files.
- Preserves all original VCF entries.
- Adds newly generated MNV entries to the output VCF.
- Aggregates sample-specific genotype fields conservatively.
- Supports multiple alternative bases at the same codon position.
- Adds metadata headers describing the command and the generated `cMNV` INFO field.

## Input requirements

`codonMNV` requires two input files:

1. A gene-specific VCF file containing variant calls.
2. The reference sequence of the chromosome or contig in FASTA format.

The VCF file must fulfill the following requirements:

- It must contain variants from a single coding sequence.
- Variants should be normalized with respect to the reference genome.
- Multiallelic records should be split.
- Per-sample genotype fields must be present from column 9 onward.
- The genotype format is expected to contain GATK-style fields such as:

```text
GT:AD:AF:DP:GQ:PL
```


## Planned Features:
- Remove necessity for gene specific vcf files -> work on complete vcf
- Replace reference sequence input with genebank or gff file
- Dynamically detect VCF genotype-fields
- update INFO field to match exact VCF standards


