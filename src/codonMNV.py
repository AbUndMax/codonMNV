# Copyright (c) 2026 Max G
# Licensed under the MIT License.
# See LICENSE file in the project root for details.

# codonMNV: codon Multiple Nucleotide Variant
# finds and adds codon based Multiple Nucleotide Variants to a VCF file
# TODO: exact USAGE and README pending!

import argparse
import sys
import logging
import re
from pathlib import Path

import numpy as np
from Bio import SeqIO


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s |")


def collect_samples_with_variant(line_split):
    """
    Takes a single VCF entry line and processes the genotype fields of each sample.

    The following assumptions are made in order to correctly parse:
    - the FORMAT is always the same and has 6 fields (GT:AD:AF:DP:GQ:PL)
    - GT is only one integer or missing (i.e ".") formats like 1|1 ./. or similar are not handled! (later update)
    - AD is either "." or a comma-separated list of integers (i.e. "0,1") with len = 2
    - AF is either "." or a floating point number (i.e. "0.5")
    - DP and GQ are either "." or integers (i.e. "100" or "0")
    - PL is either "." or a comma-separated list of integers (i.e. "0,100") with len = 2

    If the any of those assumptions are not met, the program will crash and terminate with ValueError.
    Which is intentional in the current early stage version of the tool

    Returns:
        tuple of: (numpy array, SampleStats)
            - the numpy array is parallel to the samples in the VCF header starting at column index 9
            and has boolean values (True = variant present, False = no variant present)
            - the SampleStats object contains all genotype field informations (GT, AD, AF, DP, GQ, PL).
            Each stat is organized in a parallel numpy array. (see @SampleStats class)
    """

    # generate list of "per sample" stats: [[GT, AD, AF, DP, GQ, PL], [GT, AD, AF, DP, GQ, PL], ...]
    per_sample_stats = SampleStats(line_split[9:])
    per_sample_gt = np.array(per_sample_stats.id_to_stats_array["GT"][:]) > 0 # does also cover nan vals (np.nan > 0 = False)

    return per_sample_gt, per_sample_stats



def start_end_of_gene(vcf_file):
    """
    Reads the header of a VCF file in order to find the genomic region of the gene.

    This is only possible if the VCF file was processed with bcftools view
    in order to truncate a full VCF only to the CDS region of a specific gene.
    Thus resulting in a header entry starting with: ##bcftools_viewCommand=view -r <chromosome>:<start>-<end>
    from which the CDS region can be extracted.

    Returns:
         The start and end CDS region of the gene in the gene-specific VCF file.
    """

    with open(vcf_file, "r") as f:
        for line in f:
            if line.startswith("##bcftools_viewCommand=view -r"):
                startstop = line.split(" ")[2]
                startstop_split = startstop.split(":")[1].split("-")
                start = startstop_split[0]
                end = startstop_split[1]
                return int(start), int(end)

        raise ValueError(f"Start Codon not found in VCF header: {vcf_file.name}.")



def parse_reference_fasta(fasta_file, start_of_gene, end_of_gene):
    """slice the respective gene region from the reference fasta file"""
    reference = SeqIO.read(fasta_file, "fasta")
    # start -1 to account for 0-based indexing (VCF counts 1-based!)
    return str(reference.seq)[start_of_gene-1:end_of_gene]



# possoble improvements: Dynamic type setting (via format_meta)
class SampleStats:
    """
    This class is used to store the sample stats (GT, AD, AF, DP, GQ, PL) of all genotype fields in a VCF entry.

    Each stat is organized in a parallel numpy array with its respective type.

    MISSING_VALUE is set to -1 in order to maintain type consistency. (np.nan is always a float!)
    """

    MISSING_VALUE = -1

    def __init__(self, sample_stats_list=None):
        self.ids = ["GT", "AD", "AF", "DP", "GQ", "PL"]
        self.id_to_stats_array: dict[str, np.ndarray | None] = {
            "GT": None,
            "AD": None,
            "AF": None,
            "DP": None,
            "GQ": None,
            "PL": None
        }

        if sample_stats_list:
            self.parse_sample_stats(sample_stats_list)

    @staticmethod
    def parse_int(value):
        if value == ".":
            return SampleStats.MISSING_VALUE
        return int(value)

    @staticmethod
    def parse_float(value):
        if value == ".":
            return SampleStats.MISSING_VALUE
        return float(value)

    @staticmethod
    def parse_int_list(value, shape):
        if value == ".":
            return [SampleStats.MISSING_VALUE] * shape #Improvment possible: dynamicly calculate shape mathcing other lists
        return [int(x) if x != "." else SampleStats.MISSING_VALUE for x in value.split(",")]

    def parse_sample_stats(self, sample_stats_list):
        """
        Awaits a list of genotype fields i.e. ["GT:AD:AF:DP:GQ:PL", "GT:AD:AF:DP:GQ:PL", ...]

        handles the genotype field for each sample by splitting the genotype fields
        and storing the individual components in their respective numpy arrays.

        Arguments:
            sample_stats_list (list[str]): List of genotype field strings to be parsed. Each
                string is expected to follow the format "GT:AD:AF:DP:GQ:PL".

        Raises:
            ValueError: If the number of components in a genotype field string does not match
                the expected FORMAT fields.
        """
        GT, AD, AF, DP, GQ, PL = [], [], [], [], [], []

        for sample in sample_stats_list:
            stat_split = sample.split(":")

            if len(stat_split) != len(self.ids):
                raise ValueError(
                    f"Expected 6 FORMAT fields (GT:AD:AF:DP:GQ:PL), got {len(stat_split)}"
                )

            gt, ad, af, dp, gq, pl = stat_split

            GT.append(self.parse_int(gt))
            AD.append(self.parse_int_list(ad, shape=2))
            AF.append(self.parse_float(af))
            DP.append(self.parse_int(dp))
            GQ.append(self.parse_int(gq))
            PL.append(self.parse_int_list(pl, shape=2))

        self.id_to_stats_array["GT"] = np.array(GT, dtype=int)
        self.id_to_stats_array["AD"] = np.array(AD, dtype=int)
        self.id_to_stats_array["AF"] = np.array(AF, dtype=float)
        self.id_to_stats_array["DP"] = np.array(DP, dtype=int)
        self.id_to_stats_array["GQ"] = np.array(GQ, dtype=int)
        self.id_to_stats_array["PL"] = np.array(PL, dtype=int)


    @staticmethod
    def combine_sample_stats(*sample_stats):
        """
        Combines two or more SampleStats objects into a single SampleStats object.

        The combined SampleStats object will contain the minimum of the stats for each stat type. (conservative)

        GT  -> Minimum pro Sample
        AD  -> elementweises Minimum pro Sample
        AF  -> Minimum pro Sample
        DP  -> Minimum pro Sample
        GQ  -> Minimum pro Sample
        PL  -> elementweises Maximum pro Sample

        None values within sample_stats are ignored.

        Returns a new SampleStats object with the combined stats.
        """

        if len(sample_stats) == 1 and isinstance(sample_stats[0], (list, tuple)):
            sample_stats = sample_stats[0]

        if len(sample_stats) == 0:
            raise ValueError("No SampleStats objects provided")

        # remove None values
        valid_sample_stats = [s for s in sample_stats if s is not None]

        for s in valid_sample_stats:

            if s.ids != ["GT", "AD", "AF", "DP", "GQ", "PL"]:
                raise ValueError("Unexpected ids in SampleStats")

        combined = SampleStats()

        # GT: scalar -> min
        gt_arrays = [s.id_to_stats_array["GT"] for s in valid_sample_stats]
        combined.id_to_stats_array["GT"] = np.min(gt_arrays, axis=0)

        # AD: vector -> elementwise min
        ad_arrays = [s.id_to_stats_array["AD"] for s in valid_sample_stats]
        combined.id_to_stats_array["AD"] = np.min(ad_arrays, axis=0)

        # AF: scalar -> min
        af_arrays = [s.id_to_stats_array["AF"] for s in valid_sample_stats]
        combined.id_to_stats_array["AF"] = np.min(af_arrays, axis=0)

        # DP: scalar -> min
        dp_arrays = [s.id_to_stats_array["DP"] for s in valid_sample_stats]
        combined.id_to_stats_array["DP"] = np.min(dp_arrays, axis=0)

        # GQ: scalar -> min
        gq_arrays = [s.id_to_stats_array["GQ"] for s in valid_sample_stats]
        combined.id_to_stats_array["GQ"] = np.min(gq_arrays, axis=0)

        # PL: vector -> elementwise max
        pl_arrays = [s.id_to_stats_array["PL"] for s in valid_sample_stats]
        combined.id_to_stats_array["PL"] = np.max(pl_arrays, axis=0)

        return combined


    def __str__(self):
        """
        Generates the genotype field string for the SampleStats object.
        """

        arrays = [self.id_to_stats_array[id] for id in self.ids]

        full_line = []
        for z in zip(*arrays):
            sample_stats = []
            for arr in z:
                if isinstance(arr, np.ndarray):
                    sample_stats.append(",".join(map(str, arr)))
                else:
                    sample_stats.append(str(arr))

            sample_str = ":".join(sample_stats)
            # replace missing float values
            sample_str = sample_str.replace(str(SampleStats.MISSING_VALUE)+ ".0", ".")
            # replace missing int values
            sample_str = sample_str.replace(str(SampleStats.MISSING_VALUE), ".")
            full_line.append(sample_str)

        return "\t".join(full_line).replace(str(SampleStats.MISSING_VALUE), ".")



class Codon:
    """
    Represents a codon at a specific position on a reference sequence.

    This class stores all variants observed within that codon and provides methods
    to iterate over all possible codon-level variant combinations across samples.
    These combinations are later used to identify candidate cMNVs.
    """

    # initilaize
    def __init__(self, ref_seq, chromosome=None, format_string=None,
                 codon_index=0, pos=None, relative_pos=None, alt=None, per_sample_gt=None, per_sample_stats=None):
        """
        Represents a codon with informations, such as reference sequence, chromosome,
        format string, and various codon-related attributes, with support for modeling multiple variants
        at the same position within the codon.

        Class fields:
            ref_seq: str
                The reference nucleotide sequence.
            chromosome: str or None
                The name of the chromosome. Default is None.
            format_string: str or None
                An optional format string for variant representation. Default is None.
            codon_index: int
                The index of the codon in the sequence. Default is 0.
            codon_sample_stats_per_position: list[dict]
                A list containing three dictionaries, each corresponding to one position in the codon.
                Each dictionary models multiple variants at the same position in the codon, with keys
                representing alternative variants and values as tuples containing per-sample genotype and
                genotype fields per sample.
            affected_positions: list or None
                A list of three elements representing whether each position in the codon is affected by variants.
            reference_codon_seq: list[str]
                A list of nucleotides representing the reference codon sequence.
            codon_seq: list[list[str]]
                A list of lists where each sublist contains possible letters (e.g., nucleotides) for
                each position in the codon, based on modeled variants.

        Arguments:
            ref_seq (str): Reference nucleotide sequence of the CDS or analyzed region.
            chromosome (str | None): Chromosome or contig name. Default is None.
            format_string (str | None): VCF FORMAT string used for the genotype fields. Default is None.
            codon_index (int): Zero-based index of the codon within ref_seq. Default is 0.
            pos (int | str | None): Genomic position of the variant. Default is None.
            relative_pos (int | None): Zero-based variant position relative to the start of ref_seq. Default is None.
            alt (str | None): Alternate nucleotide observed at this position. Default is None.
            per_sample_gt (np.ndarray | None): Boolean array indicating which samples carry the variant. Default is None.
            per_sample_stats (SampleStats | None): Parsed genotype field statistics for all samples. Default is None.
        """

        self.ref_seq = ref_seq
        self.chromosome = chromosome
        self.format_string = format_string
        self.codon_index = codon_index

        # use of dictionary to model multiple variants at the same position in the codon
        # key = alt variant in this position
        #       value= (per_sample_gt, per_sample_stats)
        self.codon_sample_stats_per_position = [
            {},  # pos 0
            {},  # pos 1
            {}   # pos 2
        ]
        self.affected_positions = [None, None, None]
        self. reference_codon_seq = list(ref_seq[codon_index*3:codon_index*3+3])

        #using a list of list to model multiple variants at the same position in the codon
        # possible letters per position: ie.: [[A, C], [T], [G, T]]
        self.codon_seq = [[letter] for letter in self.reference_codon_seq]

        if all(x is not None for x in (pos, relative_pos, alt, per_sample_gt, per_sample_stats)):
            self.add_new_alt(pos, relative_pos, alt, per_sample_gt, per_sample_stats)


    def add_new_alt(self, pos, relative_pos, alt, per_sample_gt, per_sample_stats):
        """
        Adds a new alternate variant to the codon sequence at the specified position, saves
        informations associated with the variant and records the genomic position.

        Arguments:
            pos (int): Genomic position of the alternate variant.
            relative_pos (int): Position relative to the codon sequence.
            alt (str): Alternate variant to be added.
            per_sample_gt (Any): GT boolean array related to the variant, for each
                sample.
            per_sample_stats (Any): Genotype fields associated with the variant, for each
                sample.
        """

        pos_in_codon = relative_pos % 3

        # save the samples that have that variant in pos_in_cdn position
        self.codon_sample_stats_per_position[pos_in_codon][alt] = (per_sample_gt, per_sample_stats)

        # note the genomic position at its respective pos_in_cdn
        self.affected_positions[pos_in_codon] = int(pos)

        # If current letter in pos_in_codon is the same as ref letter, replace with variant,
        # else append to already existing variants
        if self.codon_seq[pos_in_codon][-1] == self.reference_codon_seq[pos_in_codon]:
            self.codon_seq[pos_in_codon] = [alt]
        else:
            # replace in the current codon seq the letter in pos_in_cdn with the alt letter
            self.codon_seq[pos_in_codon].append(alt)


    def __eq__(self, other):
        return self.codon_index == other.codon_index and self.chromosome == other.chromosome

    def __str__(self):
        return f"codon_idx={self.codon_index}, REF_CODON={self.reference_codon_seq}, ALT_CODON={self.codon_seq}"


    def is_possible_mnv(self):
        """Checks if the codon is a possible mnv. (i.e. more than one alt variant in one position)"""
        return sum(bool(pos_dict) for pos_dict in self.codon_sample_stats_per_position) >= 2


    def _check_double_alt_in_same_pos_in_one_sample(self):
        """
        Checks rather a single sample shows two alt variants in the exact same position of the same codon.
        """
        # for each position in this codon
        for position_dic in self.codon_sample_stats_per_position:
            if not position_dic:
                continue

            # check if for all alt variants in this position, any sample shows at least two or more of them
            # i.e. the sample sets for each alt variant at one of each position are distinct!
            #[1, 0, 1, 0] -> gts for alt 1 in pos x
            #[0, 1, 0, 0] -> gts for alt 2 in pos x
            #[0, 0, 0, 0] -> gts for alt 3 in pos x
            #----------------
            #[1, 1, 1, 0] -> sum over all gts -> if each position is at least <= 1, means that one sample has at max
            #                                        1 alt in this position!
            gts_per_variant = np.array([tup[0] for tup in position_dic.values()])
            counts = gts_per_variant.sum(axis=0)
            if np.any(counts > 1):
                    raise Warning(f"""
                    POS={self.affected_positions}, CODON={self.codon_seq}
                    at least one sample shows two or more alt variants in same pos: {counts}""")


    def iter_base_combinations(self):
        """
        Generates all possible base combinations for a codon sequence
        while including genotype field informations of the respective variant.

        Yields:
            tuple: A tuple containing:
                - A list of three characters representing the current combination of bases for
                  the codon sequence.
                - boolean GT np array.
                - Genotype Fields for each sample.
        """

        self._check_double_alt_in_same_pos_in_one_sample()

        for letter0 in self.codon_seq[0]:
            if letter0 != self.reference_codon_seq[0]:
                per_sample_gts_0, per_sample_stats_0 = self.codon_sample_stats_per_position[0][letter0]
            else:
                per_sample_gts_0 = per_sample_stats_0 = None

            for letter1 in self.codon_seq[1]:
                if letter1 != self.reference_codon_seq[1]:
                    per_sample_gts_1, per_sample_stats_1 = self.codon_sample_stats_per_position[1][letter1]
                else:
                    per_sample_gts_1 = per_sample_stats_1 = None

                for letter2 in self.codon_seq[2]:
                    if letter2 != self.reference_codon_seq[2]:
                        per_sample_gts_2, per_sample_stats_2 = self.codon_sample_stats_per_position[2][letter2]
                    else:
                        per_sample_gts_2 = per_sample_stats_2 = None


                    yield ([letter0, letter1, letter2],
                           [per_sample_gts_0, per_sample_gts_1, per_sample_gts_2],
                           [per_sample_stats_0, per_sample_stats_1, per_sample_stats_2])







def gt_intersection(gts_per_position):
    """
    Determines the intersection of ground-truth arrays for a given set of positions.

    This function processes a list of arrays, removing any `None` values, and evaluates
    whether all valid arrays have a value of 1 at each corresponding position.

    ie.:
    [[0, 1, 1, 0, 1]   | gts for pos 1
     [1, 1, 0, 1, 1]   | gts for pos 2
     [0, 1, 1, 1, 1]]  | gts for pos 3
    -------------------
     [0, 1, 0, 0, 1] <- Intersection (returned array)

    (False interpreted as 0, True interpreted as 1)

    Arguments:
    gts_per_position : list
        A list where each element is either a numpy array or None. Each non-None
        array is expected to have the same shape.

    Returns:
    np array
        A boolean array indicating positions where all valid arrays have a value of 1.
    """
    valid = np.array([arr for arr in gts_per_position if arr is not None])
    return np.all(valid == 1, axis=0)




def mnv_vcf_entry_string(codon,
                         current_codon_seq,
                         gts_per_position,
                         sample_stats_per_position,
                         num_alt_in_codon,
                         intersection_per_position):
    """
    Generates a cMNV VCF entry string for a codon with multiple nucleotide variants.
    """

    if codon.affected_positions[2] is None:
        start = 0
        stop = 1
    elif codon.affected_positions[1] is None:
        start = 0
        stop = 2
    else:
        start = 1
        stop = 2

    pos = codon.affected_positions[start]
    ref = "".join(codon.reference_codon_seq[start:stop+1])
    alt = "".join(current_codon_seq[start:stop+1])
    combined_sample_stats = SampleStats.combine_sample_stats(sample_stats_per_position)

    return (f"{codon.chromosome}\t{pos}\t.\t{ref}\t{alt}\t.\t.\t"
            f"cMNV={num_alt_in_codon}|{codon.codon_index}|[{"".join(codon.reference_codon_seq)}]|"
            f"[{"".join(current_codon_seq)}]|[{",".join(map(str, codon.affected_positions))}]|"
            f"[{",".join([str(gts.sum()) if gts is not None else "None" for gts in gts_per_position])}]|"
            f"[{",".join(map(str, intersection_per_position))}]\t"
            f"{codon.format_string}\t{combined_sample_stats}")


def prepare_mnv_vcf_entry_string(codon,
                                 current_codon_seq,
                                 gts_per_position,
                                 sample_stats_per_position):
    """
    Prepares a partially applied cMNV VCF entry string generator.
    
    This helper stores the codon-specific arguments that stay constant for one
    codon variant combination and returns a nested function that only requires
    the number of altered positions and the sample intersection information.
    
    Arguments:
        codon (Codon): Codon object containing reference sequence, position,
            chromosome and FORMAT information.
        current_codon_seq (list[str]): Current codon sequence after applying
            the selected alternate bases.
        gts_per_position (list[np.ndarray | None]): Per-position genotype arrays
            indicating which samples carry the respective variant.
        sample_stats_per_position (list[SampleStats | None]): Per-position sample
            genotype field statistics for the respective variant.
    
    Returns:
        Callable: Function that takes num_alt_in_codon and intersection_per_position
            and returns the final cMNV VCF entry string.
    """

    def rest(num_alt_in_codon, intersection_per_position):
        return mnv_vcf_entry_string(codon,
                                    current_codon_seq,
                                    gts_per_position,
                                    sample_stats_per_position,
                                    num_alt_in_codon,
                                    intersection_per_position)

    return rest



def report_codon_variants(codon: Codon):
    """
    Reports all cMNV entries detected within a given codon.

    The function iterates over all possible alternate base combinations of the codon
    and checks whether at least two altered positions are shared by the same samples.

    For codons with two altered positions, one cMNV entry is reported if the two
    genotype arrays have a non-empty intersection.

    For codons with three altered positions, the function first checks for samples
    carrying all three variants. It then checks all pairwise combinations while
    excluding samples already counted in the three-position overlap.

    Arguments:
        codon (Codon): Codon object containing the observed variants, genotype arrays
            and genotype field statistics.

    Returns:
        str: String containing one or more cMNV VCF entry lines, each ending with a
            newline. Returns an empty string if no cMNV is detected.
    """

    mnv_entries = []

    # since it is possible to have multiple alt entries for one position,
    # iterate over all possible combinations of alt letters within that codon
    for letter_combination, gts_per_position, sample_stats_per_position in codon.iter_base_combinations():

        entry_string = prepare_mnv_vcf_entry_string(codon,
                                                    letter_combination,
                                                    gts_per_position,
                                                    sample_stats_per_position)

        # if two positions have alts and there are samples that show both of them
        if (sum(x is None for x in sample_stats_per_position) == 1
                and np.any(intersectn := gt_intersection(gts_per_position))):
            mnv_entries.append(entry_string(2, [intersectn.sum() if s is not None else None for s in sample_stats_per_position]))


        # if all three positions have altered samples
        elif sum(x is None for x in sample_stats_per_position) == 0:

            intersectn_case_1 = gt_intersection(gts_per_position)

            # four cases:
            # 1: there are samples that have all three variants
            if np.any(intersectn_case_1):
                mnv_entries.append(entry_string(3, [intersectn_case_1.sum()]*3))

            # for the next cases, only two positions have variances that are shared within a sample.

            # exclude the triple-overlap from the pairwise overlap so that samples with
            # all three variants are not counted again as two-variant combinations

            # 2: there are samples that have the first two variants
            if np.any(intersectn := gts_per_position[0] & gts_per_position[1] & ~intersectn_case_1):
                mnv_entries.append(entry_string(2, [intersectn.sum(), intersectn.sum(), None]))

            # 3: there are samples that have the second and third variant:
            if np.any(intersectn := gts_per_position[1] & gts_per_position[2] & ~intersectn_case_1):
                mnv_entries.append(entry_string(2, [None, intersectn.sum(), intersectn.sum()]))

            # 4: there are samples that have the first and the third variant:
            if np.any(intersectn := gts_per_position[0] & gts_per_position[2] & ~intersectn_case_1):
                mnv_entries.append(entry_string(2, [intersectn.sum(), None, intersectn.sum()]))

    if mnv_entries:
        return "\n".join(mnv_entries) + "\n"
    else:
        return ""



def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Find and add Codon based Multiple Nucleotide Variants to a VCF file")
    parser.add_argument("vcf_file", type=Path, help="VCF file for a single gene to process")
    parser.add_argument("reference_sequence", type=Path, help="Reference FASTA file")
    parser.add_argument("output_vcf", type=Path, help="Output VCF file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print debug information")
    return parser.parse_args()



def main():
    """
    Runs the codonMNV command-line workflow.

    The function reads the input VCF and reference FASTA, extracts the CDS region,
    scans variants codon by codon and writes the original VCF entries together
    with newly detected cMNV entries to the output VCF file.

    The main() function also includes one of the two algorithms for finding cMNVs:
    it groups consecutive VCF entries by codon, filters unsupported variant types
    and reports a codon only after all variants belonging to that codon have been collected.
    """

    args = parse_args()
    genomic_start, genomic_end = start_end_of_gene(args.vcf_file)
    ref_seq = parse_reference_fasta(args.reference_sequence, genomic_start, genomic_end)

    # header strings
    CALL_STRING = f"##codonMNVCommand={" ".join(sys.argv)}\n"
    INFO_STRING = ('##INFO=<ID=cMNV,Number=1,Type=String,Description="'
                         'Computed Multiple Nucleotide Variant within one Codon. '
                         'Format: Num_alt_in_codon | Codon_idx_in_CDS | Original_codon '
                         '| Variant_codon | affected_positions | samples_with_alt_per_position '
                         '| intersection_of_samples_with_alt">\n')

    format_meta = {} # saves type and id for each field: key=id, value=dict(Number, Type)

    # find mnvs and write them to output file
    with open(args.vcf_file, "r") as vcf_file, open(args.output_vcf, "w") as out_file:

        current_codon = Codon(ref_seq, codon_index=-1)

        for line in vcf_file:

            if line.startswith("##"):
                if line.startswith("##FILTER=<ID") and "Number" in line and "Type" in line:
                    match = re.search(r"<(.*?)>", line)
                    if match:
                        content = match.group(1)
                        fields = content.split(",")
                        for field in fields:
                            if "Number" in field:
                                field_split = field.strip().split("=")
                                format_meta["Number"] = field_split[1]
                            elif "Type" in field:
                                field_split = field.strip().split("=")
                                format_meta["Type"] = field_split[1]

                out_file.write(line)
                continue

            if line.startswith("#"):
                out_file.write(CALL_STRING)
                out_file.write(INFO_STRING)
                out_file.write(line)
                continue

            line_split = line.strip().split("\t")
            per_sample_gt, per_sample_stats = collect_samples_with_variant(line_split)
            chrom = line_split[0]
            pos =  line_split[1]
            ref = line_split[3]
            alt = line_split[4]
            format_string = line_split[8]
            relative_pos = int(pos) - genomic_start
            codon_idx = relative_pos // 3

            # Only look for codon changes and not frame shifts or continued deletions (ie. *)
            if len(alt := alt) > 1 or alt == "*" or len(ref) > 1:
                out_file.write(line)
                continue

            # if no sample carries this variant (i.e all gt values are 0), skip it
            if np.all(per_sample_gt == 0):
                out_file.write(line)
                continue

            # if in a new codon
            if codon_idx != current_codon.codon_index:

                # report codon if current_codon has at least two alts (i.e. possible mnv)
                if current_codon.is_possible_mnv():
                    out_file.write(report_codon_variants(current_codon))

                # initiate new codon with current alt
                current_codon = Codon(ref_seq, chrom, format_string,
                                      codon_idx, pos, relative_pos, alt, per_sample_gt, per_sample_stats)

            # if in same codon, add the new alt to the current codon
            else:
                current_codon.add_new_alt(pos, relative_pos, alt, per_sample_gt, per_sample_stats)

            out_file.write(line)

            if args.verbose: print(f"POS={pos}, pos_in_codon={codon_idx % 3}, {current_codon}") # debug line

        # report possible final mnv
        if current_codon.is_possible_mnv():
            out_file.write(report_codon_variants(current_codon))



if __name__ == "__main__":
    main()