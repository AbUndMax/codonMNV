import unittest
import numpy as np
import tempfile
from pathlib import Path
import sys

import codonMNV


# -----------------------------------------
# test Helpers
# -----------------------------------------

class TestCollectSamplesWithVariant(unittest.TestCase):

    # Test if GT arrays are correctly parsed and converted to boolean arrays
    def test_collect_samples_with_variant(self):
        line_split_dummy = ["chr1", "100", ".", "A", "G", ".", ".", ".",
                            "GT:AD:AF:DP:GQ:PL",
                            "0:10,0:0.0:10:99:0,99",
                            "1:5,5:0.5:10:80:20,0",
                            ".:.:.:.:.:.,."]

        per_sample_gt, per_sample_stats = codonMNV.collect_samples_with_variant(line_split_dummy)

        np.testing.assert_array_equal(per_sample_gt,
                                      np.array([False, True, False]))

        self.assertIsInstance(per_sample_stats, codonMNV.SampleStats)
        # SampleStats instance itself is tested in TestSampleStats class below



class TestStartEndOfGene(unittest.TestCase):

    # test if the start and end of the gene are correctly extracted from the VCF header
    def test_start_end_of_gene_with_realistic_header(self):
        vcf_header = ("##fileformat=VCFv4.2\n"
                      "##FILTER=<ID=PASS,Description=\"All filters passed\">\n"
                      "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n"
                      "##...\n"
                      "##bcftools_viewCommand=view -r NC_021508:134911-136707 -o TP0117-snv-indel-normalized.vcf variants-snv-indel-normalized.vcf.gz; Date=Tue Apr 14 21:36:16 2026\n"
                      "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tCP001752\n")

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_file:
            tmp_file.write(vcf_header)
            tmp_path = Path(tmp_file.name)

        try:
            start, end = codonMNV.start_end_of_gene(tmp_path)

            self.assertEqual(start, 134911)
            self.assertEqual(end, 136707)

        finally:
            tmp_path.unlink()





# -----------------------------------------
# SampleStats class tests
# -----------------------------------------

class TestSampleStats(unittest.TestCase):

    # test if the sample stats are correctly parsed and converted to numpy arrays
    def test_parse_sample_stats(self):
        sample_stats = codonMNV.SampleStats(["0:10,0:0.0:10:99:0,99",
                                             "1:5,5:0.5:10:80:20,0",
                                             ".:.:.:.:.:."])

        np.testing.assert_array_equal(sample_stats.id_to_stats_array["GT"],
                                      np.array([0, 1, -1]))

        np.testing.assert_array_equal(sample_stats.id_to_stats_array["AD"],
                                      np.array([[10, 0], [5, 5], [-1, -1]]))

        np.testing.assert_array_equal(sample_stats.id_to_stats_array["AF"],
                                      np.array([0.0, 0.5, -1.0]))

        np.testing.assert_array_equal(sample_stats.id_to_stats_array["DP"],
                                      np.array([10, 10, -1]))

        np.testing.assert_array_equal(sample_stats.id_to_stats_array["GQ"],
                                      np.array([99, 80, -1]))

        np.testing.assert_array_equal(sample_stats.id_to_stats_array["PL"],
                                      np.array([[0, 99], [20, 0], [-1, -1]]))


    # test if a SampleStats object is correctly converted to a valid genotype field string
    def test_sample_stats_string_conversion(self):
        sample_stats = codonMNV.SampleStats(["0:10,0:0.0:10:99:0,99",
                                             "1:5,5:0.5:10:80:20,0",
                                             ".:.:.:.:.:."])

        self.assertEqual(str(sample_stats),
                         "0:10,0:0.0:10:99:0,99\t"
                         "1:5,5:0.5:10:80:20,0\t"
                         ".:.,.:.:.:.:.,.")


    # test if multiple SampleStats objects are correctly combined (conservative)
    def test_combine_sample_stats(self):
        sample_stats_1 = codonMNV.SampleStats(["1:10,5:0.5:15:99:10,0",
                                               "0:20,0:0.0:20:80:0,90"])

        sample_stats_2 = codonMNV.SampleStats(["1:8,4:0.4:12:70:20,5",
                                               "1:18,2:0.1:18:60:5,100"])

        combined = codonMNV.SampleStats.combine_sample_stats(sample_stats_1, sample_stats_2)

        np.testing.assert_array_equal(combined.id_to_stats_array["GT"],
                                      np.array([1, 0]))

        np.testing.assert_array_equal(combined.id_to_stats_array["AD"],
                                      np.array([[8, 4], [18, 0]]))

        np.testing.assert_array_equal(combined.id_to_stats_array["AF"],
                                      np.array([0.4, 0.0]))

        np.testing.assert_array_equal(combined.id_to_stats_array["DP"],
                                      np.array([12, 18]))

        np.testing.assert_array_equal(combined.id_to_stats_array["GQ"],
                                      np.array([70, 60]))

        np.testing.assert_array_equal(combined.id_to_stats_array["PL"],
                                      np.array([[20, 5], [5, 100]]))


    # test for correct handling of lists as argument
    def test_combine_sample_stats_accepts_list(self):
        sample_stats_1 = codonMNV.SampleStats(["1:10,5:0.5:15:99:10,0"])

        sample_stats_2 = codonMNV.SampleStats(["1:8,4:0.4:12:70:20,5"])

        combined = codonMNV.SampleStats.combine_sample_stats([sample_stats_1, sample_stats_2])

        np.testing.assert_array_equal(combined.id_to_stats_array["GT"],
                                      np.array([1]))


    # test if None values are correctly ignored
    def test_combine_sample_stats_ignores_none(self):
        sample_stats = codonMNV.SampleStats(["1:10,5:0.5:15:99:10,0"])

        combined = codonMNV.SampleStats.combine_sample_stats(None, sample_stats, None)

        np.testing.assert_array_equal(combined.id_to_stats_array["GT"],
                                      np.array([1]))





# -----------------------------------------
# Codon class tests
# -----------------------------------------

class TestCodon(unittest.TestCase):

    # test if the codon is correctly initialized using only the reference sequence
    def test_codon_init_ref_arg(self):
        codon = codonMNV.Codon("ATGCCC", codon_index=0)

        self.assertEqual(codon.reference_codon_seq, ["A", "T", "G"])
        self.assertEqual(codon.codon_seq, [["A"], ["T"], ["G"]])
        self.assertEqual(codon.affected_positions, [None, None, None])
        self.assertFalse(codon.is_possible_mnv())

        codon = codonMNV.Codon("ATGCCC", codon_index=1)

        self.assertEqual(codon.reference_codon_seq, ["C", "C", "C"])
        self.assertEqual(codon.codon_seq, [["C"], ["C"], ["C"]])
        self.assertEqual(codon.affected_positions, [None, None, None])
        self.assertFalse(codon.is_possible_mnv())


    def test_codon_init_alternative(self):
        stats = codonMNV.SampleStats(["1:5,5:0.5:10:80:20,0"])
        gt = np.array([True])

        codon = codonMNV.Codon(ref_seq="ATGCCC", chromosome="chr1", format_string="GT:AD:AF:DP:GQ:PL",
            codon_index=0, pos=101, relative_pos=1, alt="C", per_sample_gt=gt, per_sample_stats=stats)

        self.assertEqual(codon.codon_seq, [["A"], ["C"], ["G"]])
        self.assertEqual(codon.affected_positions, [None, 101, None])
        self.assertIs(codon.codon_sample_stats_per_position[1]["C"][1], stats)


    def test_add_new_alt(self):
        codon = codonMNV.Codon("ATGCCC", codon_index=0)
        stats_1 = codonMNV.SampleStats(["1:5,5:0.5:10:80:20,0"])
        stats_2 = codonMNV.SampleStats(["1:4,6:0.6:10:70:30,0"])

        codon.add_new_alt(101, 1, "C", np.array([True]), stats_1)
        codon.add_new_alt(101, 1, "G", np.array([False]), stats_2)

        self.assertEqual(codon.codon_seq, [["A"], ["C", "G"], ["G"]])
        self.assertEqual(codon.affected_positions, [None, 101, None])
        self.assertIn("C", codon.codon_sample_stats_per_position[1])
        self.assertIn("G", codon.codon_sample_stats_per_position[1])


    def test_is_possible_mnv(self):
        codon = codonMNV.Codon("ATGCCC", codon_index=0)
        stats = codonMNV.SampleStats(["1:5,5:0.5:10:80:20,0"])

        codon.add_new_alt(100, 0, "C", np.array([True]), stats)
        self.assertFalse(codon.is_possible_mnv())

        codon.add_new_alt(101, 1, "C", np.array([True]), stats)
        self.assertTrue(codon.is_possible_mnv())


    def test_base_combinations(self):
        codon = codonMNV.Codon("ATGCCC", codon_index=0)
        stats = codonMNV.SampleStats(["1:5,5:0.5:10:80:20,0"])

        gt = np.array([True])

        codon.add_new_alt(100, 0, "C", gt, stats)
        codon.add_new_alt(101, 1, "A", gt, stats)

        combinations = list(codon.iter_base_combinations())

        self.assertEqual(len(combinations), 1)
        self.assertEqual(combinations[0][0], ["C", "A", "G"])
        np.testing.assert_array_equal(combinations[0][1][0], gt)
        np.testing.assert_array_equal(combinations[0][1][1], gt)
        self.assertIsNone(combinations[0][1][2])


    def test_base_combinations_multiple_alts(self):
        codon = codonMNV.Codon("ATGCCC", codon_index=0)
        stats = codonMNV.SampleStats(["1:5,5:0.5:10:80:20,0"])

        codon.add_new_alt(100, 0, "C", np.array([True, False]), stats)
        codon.add_new_alt(100, 0, "G", np.array([False, True]), stats)
        codon.add_new_alt(101, 1, "A", np.array([True, True]), stats)

        combinations = list(codon.iter_base_combinations())

        self.assertEqual([combo[0] for combo in combinations],
                         [["C", "A", "G"], ["G", "A", "G"]])


    def test_same_alt_in_same_pos_in_same_sample_warning(self):
        codon = codonMNV.Codon("ATGCCC", codon_index=0)
        stats = codonMNV.SampleStats(["1:5,5:0.5:10:80:20,0"])

        codon.add_new_alt(100, 0, "C", np.array([True, False]), stats)
        codon.add_new_alt(100, 0, "G", np.array([True, False]), stats)

        with self.assertRaises(Warning):
            list(codon.iter_base_combinations())



# -----------------------------------------
# gt_intersection tests
# -----------------------------------------

class TestGtIntersection(unittest.TestCase):

    def test_gt_intersection_with_three_arrays(self):
        gts_per_position = [np.array([0, 1, 1, 0, 1]),
                            np.array([1, 1, 0, 1, 1]),
                            np.array([0, 1, 1, 1, 1])]

        result = codonMNV.gt_intersection(gts_per_position)

        np.testing.assert_array_equal(result,
                                      np.array([False, True, False, False, True]))

    def test_gt_intersection_ignores_none(self):
        gts_per_position = [np.array([1, 1, 0, 1]),
                            None,
                            np.array([1, 0, 0, 1])]

        result = codonMNV.gt_intersection(gts_per_position)

        np.testing.assert_array_equal(result,
                                      np.array([True, False, False, True]))

    def test_gt_intersection_with_boolean_arrays(self):
        gts_per_position = [np.array([True, True, False]),
                            np.array([True, False, False])]

        result = codonMNV.gt_intersection(gts_per_position)

        np.testing.assert_array_equal(result,
                                      np.array([True, False, False]))




# -----------------------------------------
# report_codon_variants tests
# -----------------------------------------

class TestReportCodonVariants(unittest.TestCase):

    def test_report_codon_variants_returns_empty_string_without_overlap(self):
        codon = codonMNV.Codon("ATGCCC", chromosome="chr1",
                               format_string="GT:AD:AF:DP:GQ:PL",
                               codon_index=0)

        stats_1 = codonMNV.SampleStats(["1:5,5:0.5:10:80:20,0", "0:10,0:0.0:10:99:0,99"])

        stats_2 = codonMNV.SampleStats(["0:10,0:0.0:10:99:0,99", "1:5,5:0.5:10:80:20,0"])

        codon.add_new_alt(100, 0, "C", np.array([True, False]), stats_1)
        codon.add_new_alt(101, 1, "A", np.array([False, True]), stats_2)

        result = codonMNV.report_codon_variants(codon)

        self.assertEqual(result, "")

    def test_report_codon_variants_two_positions_with_overlap(self):
        codon = codonMNV.Codon("ATGCCC", chromosome="chr1",
                               format_string="GT:AD:AF:DP:GQ:PL",
                               codon_index=0)

        stats_1 = codonMNV.SampleStats(["1:5,5:0.5:10:80:20,0", "1:4,6:0.6:10:70:30,0"])

        stats_2 = codonMNV.SampleStats(["1:6,4:0.4:10:75:25,0", "0:10,0:0.0:10:99:0,99"])

        codon.add_new_alt(100, 0, "C", np.array([True, True]), stats_1)
        codon.add_new_alt(101, 1, "A", np.array([True, False]), stats_2)

        result = codonMNV.report_codon_variants(codon)

        self.assertIn("chr1\t100\t.\tAT\tCA", result)
        self.assertIn("cMNV=2|0|[ATG]|[CAG]|[100,101,None]", result)
        self.assertIn("[2,1,None]|[1,1,None]", result)
        self.assertTrue(result.endswith("\n"))

    def test_report_codon_variants_three_positions_with_triple_overlap(self):
        codon = codonMNV.Codon("ATGCCC", chromosome="chr1",
                               format_string="GT:AD:AF:DP:GQ:PL",
                               codon_index=0)

        stats = codonMNV.SampleStats(["1:5,5:0.5:10:80:20,0", "1:4,6:0.6:10:70:30,0"])

        codon.add_new_alt(100, 0, "C", np.array([True, True]), stats)
        codon.add_new_alt(101, 1, "A", np.array([True, False]), stats)
        codon.add_new_alt(102, 2, "T", np.array([True, False]), stats)

        result = codonMNV.report_codon_variants(codon)

        self.assertIn("cMNV=3|0|[ATG]|[CAT]|[100,101,102]", result)
        self.assertIn("[2,1,1]|[1,1,1]", result)
        self.assertEqual(result.count("\n"), 1)

    def test_report_codon_variants_three_positions_with_pairwise_overlap(self):
        codon = codonMNV.Codon("ATGCCC", chromosome="chr1",
                               format_string="GT:AD:AF:DP:GQ:PL",
                               codon_index=0)

        stats = codonMNV.SampleStats(["1:5,5:0.5:10:80:20,0", "1:4,6:0.6:10:70:30,0"])

        codon.add_new_alt(100, 0, "C", np.array([True, False]), stats)
        codon.add_new_alt(101, 1, "A", np.array([True, True]), stats)
        codon.add_new_alt(102, 2, "T", np.array([False, True]), stats)

        result = codonMNV.report_codon_variants(codon)

        self.assertIn("cMNV=2|0|[ATG]|[CAT]|[100,101,102]", result)
        self.assertIn("[1,2,1]|[1,1,None]", result)
        self.assertIn("[1,2,1]|[None,1,1]", result)
        self.assertEqual(result.count("\n"), 2)







# -----------------------------------------
# Main testing -> all cMNVs correctly reported
# -----------------------------------------

class TestMain(unittest.TestCase):

    def test_main_creates_output_vcf(self):
        vcf_content = ("##fileformat=VCFv4.2\n"
                       "##bcftools_viewCommand=view -r chr1:1-6 input.vcf; Date=test\n"
                       "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n"
                       "chr1\t1\t.\tA\tC\t.\t.\t.\tGT:AD:AF:DP:GQ:PL\t1:5,5:0.5:10:80:20,0\t1:6,4:0.4:10:70:30,0\n"
                       "chr1\t2\t.\tT\tA\t.\t.\t.\tGT:AD:AF:DP:GQ:PL\t1:5,5:0.5:10:80:20,0\t0:10,0:0.0:10:99:0,99\n")

        fasta_content = (">chr1\n"
                         "ATGCCC\n")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)

            vcf_path = tmp_dir / "input.vcf"
            fasta_path = tmp_dir / "reference.fasta"
            output_path = tmp_dir / "output.vcf"

            vcf_path.write_text(vcf_content)
            fasta_path.write_text(fasta_content)

            old_argv = sys.argv
            try:
                sys.argv = ["codonMNV.py",
                            str(vcf_path),
                            str(fasta_path),
                            str(output_path)]

                codonMNV.main()

            finally:
                sys.argv = old_argv

            result = output_path.read_text()

        self.assertIn("##codonMNVCommand=", result)
        self.assertIn("##INFO=<ID=cMNV", result)
        self.assertIn("chr1\t1\t.\tAT\tCA", result)
        self.assertIn("cMNV=2|0|[ATG]|[CAG]|[1,2,None]", result)




if __name__ == "__main__":
    unittest.main()