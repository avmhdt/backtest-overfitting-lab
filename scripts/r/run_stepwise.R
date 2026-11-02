# Run StepwiseTest::FWERkControl on one exported case.
#
# usage: Rscript run_stepwise.R CASE_DIR
#
# The statistics come from this library, so this checks the stepdown logic, not the studentising.

args <- commandArgs(trailingOnly = TRUE)
case_dir <- args[1]

meta <- scan(file.path(case_dir, "meta.csv"), sep = ",", quiet = TRUE)
k <- meta[2]
b <- meta[3]

test_stat <- readBin(file.path(case_dir, "tstats.bin"), "double", n = k, endian = "little")
boot_stat <- matrix(
  readBin(file.path(case_dir, "zstats.bin"), "double", n = k * b, endian = "little"),
  nrow = k,
  ncol = b
)

result <- StepwiseTest::FWERkControl(test_stat, boot_stat, 1, 0.05)
writeLines(
  paste(which(as.vector(result$Reject) == 1) - 1, collapse = " "),
  file.path(case_dir, "stepwise.txt")
)
